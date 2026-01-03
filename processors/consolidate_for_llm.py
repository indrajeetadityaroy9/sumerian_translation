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

import json
import random
import os
import re
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent
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

# Control token pattern
CONTROL_TOKEN_PATTERN = re.compile(r'<(?:gold|aug|silver|gloss)>\s*')
# Bracketed gloss placeholders (e.g., [oil], [tree])
GLOSS_PLACEHOLDER_PATTERN = re.compile(r'\s*\[[^\]]+\]\s*')


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


def clean_source_text(text: str, remove_placeholders: bool = False) -> str:
    """
    Clean source text by removing control tokens and optionally gloss placeholders.
    """
    # Remove control tokens
    text = CONTROL_TOKEN_PATTERN.sub('', text)

    # Optionally remove bracketed placeholders like [oil], [tree]
    if remove_placeholders:
        text = GLOSS_PLACEHOLDER_PATTERN.sub(' ', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text.strip()


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

    # Get source ID
    source_id = record.get('composition_id', record.get('alignment_id', 'synthetic'))

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


def convert_file(input_path: Path, output_path: Path, use_diverse_prompts: bool = True) -> dict:
    """
    Convert a JSONL file to Alpaca-format JSON.

    Returns:
        Statistics dict
    """
    dataset = []
    stats = {'total': 0, 'gold': 0, 'augmented': 0, 'synthetic': 0, 'silver': 0, 'unknown': 0}
    source_ids = set()

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

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    stats['unique_sources'] = len(source_ids)
    print(f"Saved {stats['total']} records to {output_path}")
    print(f"  Quality distribution: gold={stats['gold']}, augmented={stats['augmented']}, "
          f"synthetic={stats['synthetic']}, silver={stats['silver']}, unknown={stats['unknown']}")

    return stats, source_ids


def main():
    random.seed(42)  # Reproducibility

    print("=" * 60)
    print("LLM Dataset Consolidation")
    print("=" * 60)

    # Convert training data
    print(f"\nProcessing training data: {INPUT_TRAIN}")
    train_stats, train_ids = convert_file(INPUT_TRAIN, OUTPUT_DIR / 'sft_train.json')

    # Convert validation data (use consistent prompt for evaluation)
    print(f"\nProcessing validation data: {INPUT_VALID}")
    valid_stats, valid_ids = convert_file(
        INPUT_VALID,
        OUTPUT_DIR / 'sft_test.json',
        use_diverse_prompts=False  # Consistent prompt for evaluation
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


if __name__ == '__main__':
    main()
