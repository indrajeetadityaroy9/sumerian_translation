#!/usr/bin/env python3
"""
Create DPO (Direct Preference Optimization) pairs for alignment training.

Generates preference pairs where:
  - Chosen: Human translation (gold standard)
  - Rejected: Concatenated glosses (word-by-word literal translation)

This teaches the model to prefer fluent, grammatical translations over
mechanical gloss concatenation.

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
from pathlib import Path
from typing import Optional

# Configuration
BASE_DIR = Path(__file__).parent.parent
INPUT_TRAIN = BASE_DIR / 'output_training_v2_clean/finetune/train_substitution.jsonl'
INPUT_VALID = BASE_DIR / 'output_training_v2_clean/finetune/valid.jsonl'
INPUT_GOLD_PARQUET = BASE_DIR / 'data/consolidated/etcsl_gold.parquet'
OUTPUT_DIR = BASE_DIR / 'data/final_llm_ready'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Control token pattern
CONTROL_TOKEN_PATTERN = re.compile(r'<(?:gold|aug|silver|gloss)>\s*')

# Standard instruction for DPO
INSTRUCTION = "Translate this Sumerian text into English."


def extract_gloss_from_tokens(tokens: list) -> Optional[str]:
    """
    Extract concatenated glosses from token list.

    Args:
        tokens: List of token dicts with 'label' field

    Returns:
        Space-separated gloss string, or None if insufficient data
    """
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

    if len(glosses) < 3:  # Too short to be meaningful
        return None

    return ' '.join(glosses)


def clean_source_text(text: str) -> str:
    """Remove control tokens and normalize whitespace."""
    text = CONTROL_TOKEN_PATTERN.sub('', text)
    return ' '.join(text.split()).strip()


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


def process_gold_records_from_jsonl(input_path: Path) -> list:
    """
    Process JSONL file and extract DPO pairs from gold records with tokens.
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
                rejected = extract_gloss_from_tokens(tokens)

                if not rejected:
                    continue

                # Verify meaningful difference
                if not is_meaningfully_different(chosen, rejected):
                    stats['skipped_similar'] += 1
                    continue

                dpo_pairs.append({
                    "instruction": INSTRUCTION,
                    "input": clean_src,
                    "chosen": chosen,
                    "rejected": rejected,
                    "meta": {
                        "source_id": record.get('composition_id', record.get('alignment_id', 'unknown')),
                        "chosen_len": len(chosen.split()),
                        "rejected_len": len(rejected.split())
                    }
                })
                stats['valid_pairs'] += 1

            except (json.JSONDecodeError, KeyError) as e:
                continue

    return dpo_pairs, stats


def process_gold_from_parquet() -> list:
    """
    Alternative: Process DPO pairs from the consolidated parquet file.
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available, skipping parquet processing")
        return [], {}

    if not INPUT_GOLD_PARQUET.exists():
        return [], {}

    dpo_pairs = []
    stats = {'processed': 0, 'valid_pairs': 0, 'skipped_similar': 0}

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
            rejected = extract_gloss_from_tokens(tokens)

            if not rejected:
                continue

            if not is_meaningfully_different(chosen, rejected):
                stats['skipped_similar'] += 1
                continue

            dpo_pairs.append({
                "instruction": INSTRUCTION,
                "input": clean_src,
                "chosen": chosen,
                "rejected": rejected,
                "meta": {
                    "source_id": row.get('composition_id', 'unknown'),
                    "chosen_len": len(chosen.split()),
                    "rejected_len": len(rejected.split())
                }
            })
            stats['valid_pairs'] += 1

        except Exception:
            continue

    return dpo_pairs, stats


def main():
    print("=" * 60)
    print("DPO Preference Pairs Generation")
    print("=" * 60)

    all_pairs = []

    # Try JSONL sources first (they have token data in gold records)
    print(f"\nProcessing: {INPUT_TRAIN}")
    train_pairs, train_stats = process_gold_records_from_jsonl(INPUT_TRAIN)
    print(f"  Processed: {train_stats['processed']}, With tokens: {train_stats['with_tokens']}, "
          f"Valid pairs: {train_stats['valid_pairs']}, Skipped (too similar): {train_stats['skipped_similar']}")
    all_pairs.extend(train_pairs)

    print(f"\nProcessing: {INPUT_VALID}")
    valid_pairs, valid_stats = process_gold_records_from_jsonl(INPUT_VALID)
    print(f"  Processed: {valid_stats['processed']}, With tokens: {valid_stats['with_tokens']}, "
          f"Valid pairs: {valid_stats['valid_pairs']}, Skipped (too similar): {valid_stats['skipped_similar']}")
    all_pairs.extend(valid_pairs)

    # If we didn't get enough pairs, try parquet
    if len(all_pairs) < 1000:
        print(f"\nSupplementing from parquet: {INPUT_GOLD_PARQUET}")
        parquet_pairs, parquet_stats = process_gold_from_parquet()
        if parquet_pairs:
            # Deduplicate by input text
            existing_inputs = {p['input'] for p in all_pairs}
            new_pairs = [p for p in parquet_pairs if p['input'] not in existing_inputs]
            print(f"  Added {len(new_pairs)} unique pairs from parquet")
            all_pairs.extend(new_pairs)

    # Save results
    output_path = OUTPUT_DIR / 'dpo_pairs.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total DPO pairs generated: {len(all_pairs)}")
    print(f"Output: {output_path}")

    # Show sample
    if all_pairs:
        print("\nSample DPO pair:")
        sample = all_pairs[0]
        print(f"  Input: {sample['input'][:80]}...")
        print(f"  Chosen: {sample['chosen'][:80]}...")
        print(f"  Rejected: {sample['rejected'][:80]}...")


if __name__ == '__main__':
    main()
