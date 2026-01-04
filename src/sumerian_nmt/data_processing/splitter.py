"""
ML Training Data Preparation Pipeline

Prepares ETCSL parallel corpus for mT5 fine-tuning:
- train.jsonl / valid.jsonl: For fine-tuning (Seq2Seq)
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from .normalization import normalize_etcsl


def split_etcsl_finetune(
    parallel_corpus_path: Path,
    train_output: Path,
    valid_output: Path,
    train_ratio: float = 0.9,
    seed: int = 42,
    verbose: bool = True
) -> dict:
    """
    Split ETCSL parallel corpus into train/valid sets.

    CRITICAL: Splits by composition_id to prevent data leakage.
    Lines from the same composition stay together.

    Args:
        parallel_corpus_path: Path to ETCSL parallel_corpus.jsonl
        train_output: Path to training output
        valid_output: Path to validation output
        train_ratio: Proportion for training (default 90%)
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        Statistics dict
    """
    random.seed(seed)

    # Load all alignments
    alignments = []
    with open(parallel_corpus_path, encoding="utf-8") as f:
        for line in f:
            alignments.append(json.loads(line))

    if verbose:
        print(f"Loaded {len(alignments)} alignments")

    # Group by composition_id
    by_comp = defaultdict(list)
    for a in alignments:
        by_comp[a["composition_id"]].append(a)

    if verbose:
        print(f"Found {len(by_comp)} unique compositions")

    # Shuffle and split compositions
    comp_ids = list(by_comp.keys())
    random.shuffle(comp_ids)
    split_idx = int(len(comp_ids) * train_ratio)

    train_comps = set(comp_ids[:split_idx])
    valid_comps = set(comp_ids[split_idx:])

    stats = {
        "train_compositions": len(train_comps),
        "valid_compositions": len(valid_comps),
        "train_alignments": 0,
        "valid_alignments": 0,
        "train_words": 0,
        "valid_words": 0,
    }

    # Ensure output directories exist
    train_output.parent.mkdir(parents=True, exist_ok=True)
    valid_output.parent.mkdir(parents=True, exist_ok=True)

    # Write train set
    with open(train_output, "w", encoding="utf-8") as f:
        for cid in train_comps:
            for a in by_comp[cid]:
                # Normalize source text
                if "text_normalized" in a.get("source", {}):
                    a["source"]["text_normalized"] = normalize_etcsl(
                        a["source"]["text_normalized"]
                    )
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
                stats["train_alignments"] += 1
                stats["train_words"] += len(
                    a.get("source", {}).get("text_normalized", "").split()
                )

    # Write valid set
    with open(valid_output, "w", encoding="utf-8") as f:
        for cid in valid_comps:
            for a in by_comp[cid]:
                # Normalize source text
                if "text_normalized" in a.get("source", {}):
                    a["source"]["text_normalized"] = normalize_etcsl(
                        a["source"]["text_normalized"]
                    )
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
                stats["valid_alignments"] += 1
                stats["valid_words"] += len(
                    a.get("source", {}).get("text_normalized", "").split()
                )

    if verbose:
        print(f"\nTrain: {stats['train_compositions']} compositions, "
              f"{stats['train_alignments']} alignments, {stats['train_words']} words")
        print(f"Valid: {stats['valid_compositions']} compositions, "
              f"{stats['valid_alignments']} alignments, {stats['valid_words']} words")
        print(f"\nOutput:")
        print(f"  {train_output}")
        print(f"  {valid_output}")

    return stats


def main():
    """Prepare training data from ETCSL parallel corpus."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare ML training data from ETCSL parallel corpus"
    )
    parser.add_argument(
        "--etcsl-path",
        type=Path,
        default=Path("output/parallel_corpus.jsonl"),
        help="Path to ETCSL parallel corpus"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_training"),
        help="Output directory for training data"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    print("\n" + "=" * 60)
    print("Splitting Fine-tuning Data")
    print("=" * 60)

    ft_stats = split_etcsl_finetune(
        parallel_corpus_path=args.etcsl_path,
        train_output=args.output_dir / "finetune" / "train.jsonl",
        valid_output=args.output_dir / "finetune" / "valid.jsonl",
        train_ratio=0.9,
        verbose=verbose
    )

    # Save stats
    with open(args.output_dir / "finetune" / "stats.json", "w") as f:
        json.dump(ft_stats, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {args.output_dir}")


if __name__ == "__main__":
    main()
