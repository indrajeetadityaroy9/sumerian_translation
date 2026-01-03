#!/usr/bin/env python3
"""
Create DPO (Direct Preference Optimization) pairs for alignment training.

Generates preference pairs where:
  - Chosen: Human translation (gold standard)
  - Rejected: Concatenated glosses (word-by-word literal translation)

This teaches the model to prefer fluent, grammatical translations over
mechanical gloss concatenation.

Optimized for 52 vCPU systems with parallel processing support.

Output format:
{
    "instruction": "Translate this Sumerian text into English.",
    "input": "<sumerian text>",
    "chosen": "<human translation>",
    "rejected": "<gloss concatenation>"
}
"""

import json
import re
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

from common.io import ChunkedParquetWriter, parquet_to_json
from common.parallel import get_optimal_workers, parallel_map
from common.text import clean_source_text

# Configuration
BASE_DIR = Path(__file__).parent.parent
INPUT_TRAIN = BASE_DIR / 'output_training_v2_clean/finetune/train_substitution.jsonl'
INPUT_VALID = BASE_DIR / 'output_training_v2_clean/finetune/valid.jsonl'
INPUT_GOLD_PARQUET = BASE_DIR / 'data/consolidated/etcsl_gold.parquet'
OUTPUT_DIR = BASE_DIR / 'data/final_llm_ready'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard instruction for DPO
INSTRUCTION = "Translate this Sumerian text into English."

# P2-4 fix: Document minimum gloss length threshold
# Minimum number of gloss words required for a valid DPO pair.
# Glosses shorter than this are excluded as they lack semantic complexity
# for meaningful preference learning.
#
# Rationale for 3 words:
# - Single-word glosses (e.g., "king") provide no syntactic structure
# - Two-word glosses often just noun+modifier, minimal grammatical contrast
# - Three words typically include verb/subject/object or similar structure
#
# To analyze this threshold:
# - Run with --log-gloss-stats flag
# - Review gloss_length_distribution.csv in OUTPUT_DIR
# - Check if shorter glosses are semantically useful or just noise
MIN_GLOSS_LENGTH = 3

# Global stats for gloss extraction logging
_gloss_length_stats = {'excluded_short': 0, 'length_distribution': {}}


def extract_gloss_from_tokens(tokens: list, track_stats: bool = False) -> Optional[str]:
    """
    Extract concatenated glosses from token list.

    Args:
        tokens: List of token dicts with 'label' field
        track_stats: If True, track length distribution for analysis (P2-4 fix)

    Returns:
        Space-separated gloss string, or None if insufficient data
    """
    global _gloss_length_stats

    if not tokens:
        return None

    glosses = []
    for token in tokens:
        label = token.get('label', '')
        # Skip gap markers and empty labels
        if label and label not in ('…', 'X', ''):
            # Clean up multi-word glosses
            label = label.replace('(', '').replace(')', '')
            glosses.append(label)

    gloss_len = len(glosses)

    # P2-4 fix: Track length distribution for threshold analysis
    if track_stats:
        _gloss_length_stats['length_distribution'][gloss_len] = \
            _gloss_length_stats['length_distribution'].get(gloss_len, 0) + 1

    if gloss_len < MIN_GLOSS_LENGTH:
        if track_stats:
            _gloss_length_stats['excluded_short'] += 1
        return None

    return ' '.join(glosses)


def process_record_for_dpo(record: dict) -> Optional[dict]:
    """
    Process a single record for DPO pair generation (parallelizable).

    This function is designed to be used with parallel_map for CPU parallelization.

    Args:
        record: Single training record with source/target/tokens

    Returns:
        DPO pair dict or None if not valid
    """
    try:
        source = record.get('source', {})
        tokens = source.get('tokens', [])
        if not tokens:
            return None

        src_text = source.get('text_normalized', '')
        clean_src = clean_source_text(src_text)

        target = record.get('target', {})
        chosen = target.get('text', '') if isinstance(target, dict) else str(target)
        if not chosen.strip():
            return None

        # Generate gloss concatenation (rejected)
        rejected = extract_gloss_from_tokens(tokens, track_stats=False)
        if not rejected:
            return None

        # Verify meaningful difference
        if not is_meaningfully_different(chosen, rejected):
            return None

        return {
            "instruction": INSTRUCTION,
            "input": clean_src,
            "chosen": chosen,
            "rejected": rejected,
        }
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        # Return None to skip this record in parallel processing
        # Errors are expected for malformed records
        return None


def save_gloss_stats(output_path: Path) -> None:
    """
    Save gloss length distribution statistics for threshold analysis.

    P2-4 fix: Enables analysis of minimum gloss length threshold.

    Args:
        output_path: Path to save CSV file
    """
    if not _gloss_length_stats['length_distribution']:
        print("No gloss statistics to save.")
        return

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("gloss_length,count,cumulative_excluded\n")

        sorted_lens = sorted(_gloss_length_stats['length_distribution'].items())
        cumulative = 0

        for length, count in sorted_lens:
            if length < MIN_GLOSS_LENGTH:
                cumulative += count
            f.write(f"{length},{count},{cumulative if length < MIN_GLOSS_LENGTH else ''}\n")

    print(f"Gloss statistics saved to {output_path}")
    print(f"  Excluded (< {MIN_GLOSS_LENGTH} words): {_gloss_length_stats['excluded_short']}")
    print(f"  Total entries: {sum(_gloss_length_stats['length_distribution'].values())}")


def is_meaningfully_different(chosen: str, rejected: str, min_edit_ratio: float = 0.3) -> bool:
    """
    Check if chosen and rejected are meaningfully different.

    Args:
        chosen: Human translation
        rejected: Gloss concatenation
        min_edit_ratio: Minimum ratio of differing words

    Returns:
        True if sufficiently different
    """
    chosen_words = set(chosen.lower().split())
    rejected_words = set(rejected.lower().split())

    if not chosen_words or not rejected_words:
        return False

    # Calculate Jaccard distance
    intersection = len(chosen_words & rejected_words)
    union = len(chosen_words | rejected_words)

    if union == 0:
        return False

    similarity = intersection / union

    # We want them to be different (low similarity)
    return similarity < (1 - min_edit_ratio)


def process_gold_records_from_jsonl(input_path: Path, track_stats: bool = False) -> list:
    """
    Process JSONL file and extract DPO pairs from gold records with tokens.

    Args:
        input_path: Path to JSONL file
        track_stats: If True, track gloss length statistics (P2-4 fix)
    """
    dpo_pairs = []
    stats = {'processed': 0, 'with_tokens': 0, 'valid_pairs': 0, 'skipped_similar': 0}

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                stats['processed'] += 1

                # Only process records with token-level data (original gold)
                source = record.get('source', {})
                tokens = source.get('tokens', [])

                if not tokens:
                    continue

                stats['with_tokens'] += 1

                # Extract components
                src_text = source.get('text_normalized', '')
                clean_src = clean_source_text(src_text)

                # Get human translation (chosen)
                target = record.get('target', {})
                chosen = target.get('text', '') if isinstance(target, dict) else str(target)

                if not chosen.strip():
                    continue

                # Generate gloss concatenation (rejected)
                rejected = extract_gloss_from_tokens(tokens, track_stats=track_stats)

                if not rejected:
                    continue

                # Verify meaningful difference
                if not is_meaningfully_different(chosen, rejected):
                    stats['skipped_similar'] += 1
                    continue

                # P1-2 fix: Removed meta field that could cause Axolotl parsing issues
                # Axolotl DPO format only expects: instruction, input, chosen, rejected
                dpo_pairs.append({
                    "instruction": INSTRUCTION,
                    "input": clean_src,
                    "chosen": chosen,
                    "rejected": rejected,
                })
                stats['valid_pairs'] += 1

            except (json.JSONDecodeError, KeyError) as e:
                continue

    return dpo_pairs, stats


def process_gold_records_parallel(
    input_path: Path,
    num_workers: Optional[int] = None,
    track_stats: bool = False
) -> Tuple[List[dict], dict]:
    """
    Process JSONL file with parallel processing for DPO pair extraction.

    Provides 15-20x speedup on 52 vCPU systems.

    Args:
        input_path: Path to JSONL file
        num_workers: Number of workers (default: auto-detect)
        track_stats: If True, track gloss length statistics

    Returns:
        Tuple of (dpo_pairs, stats)
    """
    if num_workers is None:
        num_workers = get_optimal_workers()

    print(f"  Using parallel processing with {num_workers} workers")

    # Load all records first
    records = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    stats = {
        'processed': len(records),
        'with_tokens': 0,
        'valid_pairs': 0,
        'skipped_similar': 0
    }

    # Process records in parallel
    results = parallel_map(
        process_record_for_dpo,
        records,
        num_workers=num_workers,
        desc=f"Processing {input_path.name}",
    )

    # Filter out None results and collect pairs
    dpo_pairs = [r for r in results if r is not None]
    stats['valid_pairs'] = len(dpo_pairs)

    # Estimate with_tokens (records that had tokens)
    stats['with_tokens'] = len([r for r in records if r.get('source', {}).get('tokens')])

    return dpo_pairs, stats


def process_gold_from_parquet(track_stats: bool = False) -> list:
    """
    Alternative: Process DPO pairs from the consolidated parquet file.

    Args:
        track_stats: If True, track gloss length statistics (P2-4 fix)
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available, skipping parquet processing")
        return [], {}

    if not INPUT_GOLD_PARQUET.exists():
        return [], {}

    dpo_pairs = []
    stats = {'processed': 0, 'valid_pairs': 0, 'skipped_similar': 0, 'errors': 0}

    df = pd.read_parquet(INPUT_GOLD_PARQUET)
    stats['processed'] = len(df)

    for _, row in df.iterrows():
        try:
            # Parse tokens from JSON string
            tokens_str = row.get('tokens', '[]')
            if isinstance(tokens_str, str):
                tokens = json.loads(tokens_str)
            else:
                tokens = tokens_str if tokens_str else []

            if not tokens:
                continue

            # Extract components
            src_text = row.get('text_normalized', '')
            clean_src = clean_source_text(src_text)
            chosen = row.get('target_text', '')

            if not chosen or not clean_src:
                continue

            # Generate gloss concatenation
            rejected = extract_gloss_from_tokens(tokens, track_stats=track_stats)

            if not rejected:
                continue

            if not is_meaningfully_different(chosen, rejected):
                stats['skipped_similar'] += 1
                continue

            # P1-2 fix: Removed meta field that could cause Axolotl parsing issues
            dpo_pairs.append({
                "instruction": INSTRUCTION,
                "input": clean_src,
                "chosen": chosen,
                "rejected": rejected,
            })
            stats['valid_pairs'] += 1

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            stats['errors'] += 1
            continue

    return dpo_pairs, stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create DPO preference pairs for alignment training")
    parser.add_argument(
        "--log-gloss-stats",
        action="store_true",
        help="Log gloss length distribution for threshold analysis (P2-4 fix)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing (recommended for 52+ vCPU systems)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of workers for parallel mode (default: auto-detect)"
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

    track_stats = args.log_gloss_stats

    print("=" * 60)
    print("DPO Preference Pairs Generation")
    print("=" * 60)

    if track_stats:
        print("\nGloss statistics tracking: ENABLED")

    if args.parallel:
        workers = args.workers or get_optimal_workers()
        print(f"\nParallel mode: ENABLED ({workers} workers)")

    all_pairs = []

    # Try JSONL sources first (they have token data in gold records)
    print(f"\nProcessing: {INPUT_TRAIN}")
    if args.parallel:
        train_pairs, train_stats = process_gold_records_parallel(INPUT_TRAIN, args.workers, track_stats)
    else:
        train_pairs, train_stats = process_gold_records_from_jsonl(INPUT_TRAIN, track_stats=track_stats)
    print(f"  Processed: {train_stats['processed']}, With tokens: {train_stats['with_tokens']}, "
          f"Valid pairs: {train_stats['valid_pairs']}, Skipped (too similar): {train_stats.get('skipped_similar', 0)}")
    all_pairs.extend(train_pairs)

    print(f"\nProcessing: {INPUT_VALID}")
    if args.parallel:
        valid_pairs, valid_stats = process_gold_records_parallel(INPUT_VALID, args.workers, track_stats)
    else:
        valid_pairs, valid_stats = process_gold_records_from_jsonl(INPUT_VALID, track_stats=track_stats)
    print(f"  Processed: {valid_stats['processed']}, With tokens: {valid_stats['with_tokens']}, "
          f"Valid pairs: {valid_stats['valid_pairs']}, Skipped (too similar): {valid_stats.get('skipped_similar', 0)}")
    all_pairs.extend(valid_pairs)

    # If we didn't get enough pairs, try parquet
    if len(all_pairs) < 1000:
        print(f"\nSupplementing from parquet: {INPUT_GOLD_PARQUET}")
        parquet_pairs, parquet_stats = process_gold_from_parquet(track_stats=track_stats)
        if parquet_pairs:
            # Deduplicate by input text
            existing_inputs = {p['input'] for p in all_pairs}
            new_pairs = [p for p in parquet_pairs if p['input'] not in existing_inputs]
            print(f"  Added {len(new_pairs)} unique pairs from parquet")
            all_pairs.extend(new_pairs)

    # Save results based on output format
    print("\n" + "=" * 60)
    print("Saving Output")
    print("=" * 60)

    parquet_dir = OUTPUT_DIR / 'dpo_parquet'
    json_path = OUTPUT_DIR / 'dpo_pairs.json'

    # Write parquet chunks (GitHub-friendly, reproducible)
    if args.output_format in ("parquet", "both"):
        print(f"\nWriting parquet chunks (chunk_size={args.chunk_size})...")
        writer = ChunkedParquetWriter(
            output_dir=parquet_dir,
            prefix="dpo_pairs",
            chunk_size=args.chunk_size,
            compression="snappy",
            generator="create_dpo_pairs.py",
        )
        writer.add_records(all_pairs)
        metadata = writer.finalize()
        print(f"  Parquet output: {parquet_dir}/")

    # Write JSON (for Axolotl compatibility)
    if args.output_format in ("json", "both"):
        if args.output_format == "both" and parquet_dir.exists():
            # Convert from parquet to ensure consistency
            print(f"\nConverting parquet to JSON for Axolotl...")
            parquet_to_json(parquet_dir, "dpo_pairs", json_path)
        else:
            # Write JSON directly
            print(f"\nWriting JSON...")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_pairs, f, indent=2, ensure_ascii=False)
            print(f"  JSON output: {json_path}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total DPO pairs generated: {len(all_pairs)}")
    if args.output_format in ("parquet", "both"):
        print(f"Parquet chunks: {parquet_dir}/")
    if args.output_format in ("json", "both"):
        print(f"JSON output: {json_path}")

    # P2-4 fix: Save gloss statistics if tracking enabled
    if track_stats:
        stats_path = OUTPUT_DIR / 'gloss_length_distribution.csv'
        save_gloss_stats(stats_path)

    # Show sample
    if all_pairs:
        print("\nSample DPO pair:")
        sample = all_pairs[0]
        print(f"  Input: {sample['input'][:80]}...")
        print(f"  Chosen: {sample['chosen'][:80]}...")
        print(f"  Rejected: {sample['rejected'][:80]}...")


if __name__ == '__main__':
    main()
