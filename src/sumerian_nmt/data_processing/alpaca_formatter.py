#!/usr/bin/env python3
"""
Consolidate training data into LLM-ready Alpaca format for fine-tuning.

Creates:
  - sft_train.json: Master instruction dataset from train_substitution.jsonl
  - sft_test.json: Gold validation set from valid.jsonl

Output format (Alpaca-style):
{
    "instruction": "Translate this Sumerian text into English.",
    "input": "<sumerian text>",
    "output": "<english translation>",
    "meta": {
        "source_id": "<composition_id or 'synthetic'>",
        "quality": "gold|augmented|synthetic",
        "method": "<augmentation method if applicable>"
    }
}
"""

import argparse
import json
import random
import os
import re
from pathlib import Path

from sumerian_nmt.utils.io import ChunkedParquetWriter, parquet_to_json
from sumerian_nmt.utils.text import clean_source_text
from sumerian_nmt.config import ControlTokens, Paths

# Configuration - use centralized path resolution
BASE_DIR = Paths.ROOT
INPUT_TRAIN = BASE_DIR / 'output_training_v2_clean/finetune/train_substitution.jsonl'
INPUT_VALID = BASE_DIR / 'output_training_v2_clean/finetune/valid.jsonl'
OUTPUT_DIR = BASE_DIR / 'data/final_llm_ready'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Diversity prompts to prevent overfitting to single phrasing
PROMPTS = [
    "Translate this Sumerian text into English.",
    "Convert the following cuneiform transliteration to English:",
    "Provide an English translation of this Sumerian passage:",
    "As an expert Sumerologist, translate the following:",
    "Render this ancient Sumerian text in modern English:",
]

# Control token patterns (centralized in config.py)
CONTROL_TOKEN_PATTERN = ControlTokens.PATTERN
CONTROL_TOKEN_VARIANTS = ControlTokens.VARIANTS


def validate_no_control_tokens(text: str) -> list:
    """
    P3-1 fix: Validate that no control tokens remain in text.

    Args:
        text: Text to validate

    Returns:
        List of matched control tokens (empty if clean)
    """
    matches = []
    for pattern in CONTROL_TOKEN_VARIANTS:
        found = pattern.findall(text)
        matches.extend(found)
    return matches


def determine_quality(record: dict) -> tuple[str, str]:
    """
    Determine the quality tier and method of a record.

    Returns:
        (quality_tier, method)
    """
    src_text = record['source'].get('text_normalized', '')

    # Check for control tokens
    if '<gold>' in src_text:
        # Gold data that may have been augmented
        if record.get('quality', {}).get('synthetic'):
            return 'augmented', record.get('quality', {}).get('method', 'unknown')
        return 'gold', 'original'
    elif '<aug>' in src_text:
        return 'augmented', record.get('quality', {}).get('method', 'unknown')
    elif '<silver>' in src_text:
        return 'silver', record.get('quality', {}).get('method', 'unknown')

    # Check quality flags (original gold records without prefix)
    if 'quality_flags' in record:
        return 'gold', 'original'

    # Check explicit quality marker
    if record.get('quality', {}).get('synthetic'):
        return 'synthetic', record.get('quality', {}).get('method', 'unknown')

    return 'unknown', 'unknown'


def parse_record(record: dict, use_diverse_prompts: bool = True) -> dict:
    """
    Parse a single record into Alpaca format.
    """
    # Extract source text
    src_text = record['source'].get('text_normalized', '')

    # Extract target text (handle both formats)
    target = record.get('target', {})
    if isinstance(target, dict):
        tgt_text = target.get('text', '')
    else:
        tgt_text = str(target)

    # Determine quality
    quality, method = determine_quality(record)

    # Clean source (keep placeholders for now - model should learn to handle them)
    clean_src = clean_source_text(src_text, remove_placeholders=False)

    # Get source ID with fallback chain
    source_id = record.get('composition_id')
    if not source_id:
        source_id = record.get('alignment_id')
    if not source_id:
        # For augmented records, use template_line_id from metadata
        source_id = record.get('metadata', {}).get('template_line_id', 'synthetic')

    # Select instruction
    instruction = random.choice(PROMPTS) if use_diverse_prompts else PROMPTS[0]

    return {
        "instruction": instruction,
        "input": clean_src,
        "output": tgt_text,
        "meta": {
            "source_id": source_id,
            "quality": quality,
            "method": method
        }
    }


def convert_file(
    input_path: Path,
    output_path: Path,
    use_diverse_prompts: bool = True,
    output_format: str = "json",
    chunk_size: int = 10000,
) -> dict:
    """
    Convert a JSONL file to Alpaca-format JSON or chunked parquet.

    Args:
        input_path: Input JSONL file
        output_path: Output JSON file path
        use_diverse_prompts: Use diverse instruction prompts
        output_format: "json", "parquet", or "both"
        chunk_size: Records per parquet chunk

    Returns:
        Statistics dict
    """
    dataset = []
    stats = {'total': 0, 'gold': 0, 'augmented': 0, 'synthetic': 0, 'silver': 0, 'unknown': 0}
    source_ids = set()

    # P3-1 fix: Track control token leakage
    control_token_leaks = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                parsed = parse_record(record, use_diverse_prompts)

                # Skip empty outputs
                if not parsed['output'].strip():
                    continue

                # P3-1 fix: Validate no control tokens in final output
                input_leaks = validate_no_control_tokens(parsed['input'])
                output_leaks = validate_no_control_tokens(parsed['output'])

                if input_leaks or output_leaks:
                    control_token_leaks.append({
                        'line': line_num,
                        'input_leaks': input_leaks,
                        'output_leaks': output_leaks,
                        'source_id': parsed['meta']['source_id']
                    })

                dataset.append(parsed)
                stats['total'] += 1
                stats[parsed['meta']['quality']] = stats.get(parsed['meta']['quality'], 0) + 1
                source_ids.add(parsed['meta']['source_id'])

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")

    # Shuffle training data
    if 'train' in str(output_path):
        random.shuffle(dataset)

    # Determine output paths
    parquet_dir = output_path.parent / f"{output_path.stem}_parquet"
    prefix = output_path.stem

    # Write parquet chunks (GitHub-friendly)
    if output_format in ("parquet", "both"):
        # Flatten for parquet (meta dict becomes separate columns)
        flat_records = []
        for rec in dataset:
            flat_rec = {
                "instruction": rec["instruction"],
                "input": rec["input"],
                "output": rec["output"],
                "source_id": rec["meta"]["source_id"],
                "quality": rec["meta"]["quality"],
                "method": rec["meta"]["method"],
            }
            flat_records.append(flat_rec)

        writer = ChunkedParquetWriter(
            output_dir=parquet_dir,
            prefix=prefix,
            chunk_size=chunk_size,
            compression="snappy",
            generator="consolidate_for_llm.py",
        )
        writer.add_records(flat_records)
        writer.finalize()

    # Write JSON (for Axolotl)
    if output_format in ("json", "both"):
        if output_format == "both" and parquet_dir.exists():
            # For Axolotl, we need the nested format, so write directly
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"Saved {stats['total']} records to {output_path}")
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"Saved {stats['total']} records to {output_path}")

    stats['unique_sources'] = len(source_ids)
    print(f"  Quality distribution: gold={stats['gold']}, augmented={stats['augmented']}, "
          f"synthetic={stats['synthetic']}, silver={stats['silver']}, unknown={stats['unknown']}")

    # P3-1 fix: Report control token leakage
    if control_token_leaks:
        print(f"  WARNING: Found {len(control_token_leaks)} records with control token leakage!")
        # Save detailed leak report
        leak_report_path = output_path.parent / f"{output_path.stem}_control_token_leaks.json"
        with open(leak_report_path, 'w', encoding='utf-8') as f:
            json.dump(control_token_leaks, f, indent=2)
        print(f"  Leak report saved to: {leak_report_path}")
    else:
        print(f"  Control token validation: PASSED (no leaks detected)")

    return stats, source_ids


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate training data into LLM-ready Alpaca format"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "parquet", "both"],
        default="both",
        help="Output format: json (Axolotl), parquet (GitHub-friendly), or both (default)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Records per parquet chunk (default: 10000, ~5-30MB per chunk)"
    )
    args = parser.parse_args()

    random.seed(42)  # Reproducibility

    print("=" * 60)
    print("LLM Dataset Consolidation")
    print("=" * 60)
    print(f"Output format: {args.output_format}")
    if args.output_format in ("parquet", "both"):
        print(f"Chunk size: {args.chunk_size}")

    # Convert training data
    print(f"\nProcessing training data: {INPUT_TRAIN}")
    train_stats, train_ids = convert_file(
        INPUT_TRAIN,
        OUTPUT_DIR / 'sft_train.json',
        output_format=args.output_format,
        chunk_size=args.chunk_size,
    )

    # Convert validation data (use consistent prompt for evaluation)
    print(f"\nProcessing validation data: {INPUT_VALID}")
    valid_stats, valid_ids = convert_file(
        INPUT_VALID,
        OUTPUT_DIR / 'sft_test.json',
        use_diverse_prompts=False,  # Consistent prompt for evaluation
        output_format=args.output_format,
        chunk_size=args.chunk_size,
    )

    # Check for leakage
    overlap = train_ids & valid_ids
    if overlap:
        print(f"\nWARNING: Found {len(overlap)} overlapping source IDs between train and test!")
        print(f"  Sample overlaps: {list(overlap)[:5]}")
    else:
        print("\nNo data leakage detected between train and test sets.")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Training set: {train_stats['total']} records")
    print(f"Test set: {valid_stats['total']} records")
    print(f"Output directory: {OUTPUT_DIR}")
    if args.output_format in ("parquet", "both"):
        print(f"Parquet chunks: {OUTPUT_DIR}/sft_train_parquet/, {OUTPUT_DIR}/sft_test_parquet/")


if __name__ == '__main__':
    main()
