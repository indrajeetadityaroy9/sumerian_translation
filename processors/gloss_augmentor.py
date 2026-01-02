"""
Gloss Augmentation for Sumerian-English NMT

Generates synthetic training pairs by applying glossary-based word substitutions.
This expands the training data to reduce mode collapse in low-resource settings.

Usage:
    python3 processors/gloss_augmentor.py \
        --input output_training_v2_clean/finetune/train.jsonl \
        --output output_training_v2_clean/finetune/train_augmented.jsonl \
        --glossary data/consolidated/glossary_sux.parquet \
        --min-frequency 3
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


def load_glossary(path: Path, min_freq: int = 3) -> Dict[str, List[str]]:
    """
    Load glossary and filter by frequency.

    Returns:
        Dict mapping Sumerian citation forms to list of English glosses
    """
    df = pd.read_parquet(path)

    # Filter by minimum frequency
    df = df[df['instances'] >= min_freq]

    # Build lookup: citation_form -> list of guide_words
    glossary = {}
    for _, row in df.iterrows():
        form = row['citation_form'].lower().strip()
        gloss = row['guide_word']
        if form and gloss and isinstance(gloss, str):
            if form not in glossary:
                glossary[form] = []
            glossary[form].append(gloss.strip())

    return glossary


def normalize_sumerian(text: str) -> str:
    """Normalize Sumerian text for matching."""
    # Remove subscript numbers and special markers
    text = re.sub(r'[₀₁₂₃₄₅₆₇₈₉]', '', text)
    text = re.sub(r'[0-9]', '', text)
    # Normalize special characters
    text = text.replace('ĝ', 'g').replace('ŋ', 'g')
    text = text.replace('š', 's').replace('ṣ', 's')
    text = text.replace('ṭ', 't').replace('ḫ', 'h')
    return text.lower().strip()


def tokenize_sumerian(text: str) -> List[str]:
    """Split Sumerian text into tokens."""
    # Split on whitespace and common separators
    tokens = re.split(r'[\s\-\.]+', text)
    return [t for t in tokens if t]


def create_synthetic_pair(
    source: str,
    target: str,
    glossary: Dict[str, List[str]],
    max_substitutions: int = 2
) -> List[Tuple[str, str]]:
    """
    Create synthetic training pairs by substituting glossary entries.

    Returns:
        List of (source, target) pairs
    """
    pairs = []
    tokens = tokenize_sumerian(source)

    # Find tokens that have glossary entries
    substitutable = []
    for i, token in enumerate(tokens):
        norm = normalize_sumerian(token)
        if norm in glossary:
            substitutable.append((i, token, glossary[norm]))

    if not substitutable:
        return pairs

    # Generate variations
    num_subs = min(len(substitutable), max_substitutions)

    for _ in range(min(3, len(substitutable))):  # Up to 3 variations per sentence
        # Randomly select tokens to substitute
        selected = random.sample(substitutable, min(num_subs, len(substitutable)))

        new_tokens = tokens.copy()
        modifications = []

        for idx, orig_token, glosses in selected:
            # Pick a random gloss
            gloss = random.choice(glosses)
            new_tokens[idx] = f"[{gloss}]"
            modifications.append(f"{orig_token}→{gloss}")

        if modifications:
            new_source = ' '.join(new_tokens)
            # Keep original target - the model should still produce correct English
            pairs.append((new_source, target))

    return pairs


def augment_dataset(
    input_path: Path,
    output_path: Path,
    glossary: Dict[str, List[str]],
    augmentation_ratio: float = 0.5
) -> Tuple[int, int]:
    """
    Augment training dataset with synthetic pairs.

    Args:
        input_path: Input JSONL file
        output_path: Output JSONL file
        glossary: Word-to-gloss mapping
        augmentation_ratio: Fraction of original data to augment

    Returns:
        Tuple of (original_count, augmented_count)
    """
    original_data = []
    with open(input_path) as f:
        for line in f:
            original_data.append(json.loads(line))

    original_count = len(original_data)
    augmented_pairs = []

    # Select subset for augmentation
    to_augment = random.sample(
        original_data,
        int(len(original_data) * augmentation_ratio)
    )

    for item in to_augment:
        source = item.get('source', {}).get('text_normalized', '')
        target = item.get('target', {}).get('text', '')

        if source and target:
            synthetic = create_synthetic_pair(source, target, glossary)
            for new_source, new_target in synthetic:
                augmented_pairs.append({
                    'source': {
                        'text_normalized': new_source,
                        'text_raw': new_source,
                        'augmented': True
                    },
                    'target': {
                        'text': new_target
                    },
                    'quality': {
                        'synthetic': True,
                        'method': 'gloss_substitution'
                    }
                })

    # Combine original + augmented
    all_data = original_data + augmented_pairs
    random.shuffle(all_data)

    # Write output
    with open(output_path, 'w') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return original_count, len(augmented_pairs)


def main():
    parser = argparse.ArgumentParser(
        description="Augment Sumerian training data with glossary substitutions"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input JSONL training file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output JSONL file with augmented data'
    )
    parser.add_argument(
        '--glossary', '-g',
        type=str,
        default='data/consolidated/glossary_sux.parquet',
        help='Path to glossary parquet file'
    )
    parser.add_argument(
        '--min-frequency',
        type=int,
        default=3,
        help='Minimum word frequency to include in glossary'
    )
    parser.add_argument(
        '--augmentation-ratio',
        type=float,
        default=0.5,
        help='Fraction of data to augment (0.0-1.0)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("Gloss Augmentation for Sumerian NMT")
    print("=" * 60)

    # Load glossary
    print(f"\n[1/3] Loading glossary from {args.glossary}...")
    glossary = load_glossary(Path(args.glossary), args.min_frequency)
    print(f"  Loaded {len(glossary)} unique Sumerian words (freq >= {args.min_frequency})")

    # Sample glossary entries
    sample_entries = list(glossary.items())[:5]
    for form, glosses in sample_entries:
        print(f"    {form}: {', '.join(glosses[:3])}")

    # Augment dataset
    print(f"\n[2/3] Augmenting dataset...")
    print(f"  Input: {args.input}")
    print(f"  Augmentation ratio: {args.augmentation_ratio:.0%}")

    orig_count, aug_count = augment_dataset(
        Path(args.input),
        Path(args.output),
        glossary,
        args.augmentation_ratio
    )

    total_count = orig_count + aug_count

    # Report
    print(f"\n[3/3] Results:")
    print(f"  Original pairs:  {orig_count:,}")
    print(f"  Synthetic pairs: {aug_count:,}")
    print(f"  Total pairs:     {total_count:,}")
    print(f"  Expansion:       {total_count / orig_count:.1%}")
    print(f"\n  Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
