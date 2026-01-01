"""
ML Training Data Preparation Pipeline

Merges ETCSL and ORACC datasets into ML-ready training files:
- corpus_monolingual.txt: For pre-training (MLM)
- train.jsonl / valid.jsonl: For fine-tuning (Seq2Seq)
- Tokenizer training data
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# Add parent dir for imports when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from processors.normalization_bridge import normalize_etcsl
from processors.oracc_core import ORACCParser
from common.quality import is_mostly_broken


def build_monolingual_corpus(
    output_path: Path,
    oracc_dirs: list[Path],
    verbose: bool = True
) -> dict:
    """
    Build monolingual Sumerian corpus for pre-training.

    Args:
        output_path: Path to output text file
        oracc_dirs: List of ORACC corpus directories (in priority order)
        verbose: Print progress

    Returns:
        Statistics dict
    """
    stats = {
        "total_files": 0,
        "total_lines": 0,
        "total_words": 0,
        "by_corpus": {}
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parser = ORACCParser(normalize=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for corpus_dir in oracc_dirs:
            if not corpus_dir.exists():
                if verbose:
                    print(f"Skipping {corpus_dir} (not found)")
                continue

            corpus_name = corpus_dir.name
            corpus_stats = {"files": 0, "lines": 0, "words": 0}

            if verbose:
                print(f"\nProcessing {corpus_name}...")

            # Find corpusjson directories
            json_files = list(corpus_dir.rglob("corpusjson/*.json"))

            for i, json_file in enumerate(json_files):
                try:
                    with open(json_file, encoding="utf-8") as f:
                        data = json.load(f)

                    file_lines = 0
                    for line in parser.extract_lines(data):
                        if is_mostly_broken(line):
                            continue
                        out.write(line + "\n")
                        corpus_stats["lines"] += 1
                        corpus_stats["words"] += len(line.split())
                        file_lines += 1

                    if file_lines > 0:
                        corpus_stats["files"] += 1
                except (json.JSONDecodeError, Exception):
                    pass

                if verbose and (i + 1) % 500 == 0:
                    print(f"  {i + 1}/{len(json_files)} files...")

            stats["by_corpus"][corpus_name] = corpus_stats
            stats["total_files"] += corpus_stats["files"]
            stats["total_lines"] += corpus_stats["lines"]
            stats["total_words"] += corpus_stats["words"]

            if verbose:
                print(f"  {corpus_name}: {corpus_stats['files']} files, "
                      f"{corpus_stats['lines']} lines, {corpus_stats['words']} words")

    if verbose:
        print(f"\n{'='*50}")
        print(f"Total: {stats['total_files']} files, "
              f"{stats['total_lines']} lines, {stats['total_words']} words")
        print(f"Output: {output_path}")

    return stats


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
    """Run the full data preparation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare ML training data from ETCSL and ORACC"
    )
    parser.add_argument(
        "--step",
        choices=["monolingual", "finetune", "all"],
        default="all",
        help="Which step to run (default: all)"
    )
    parser.add_argument(
        "--oracc-dir",
        type=Path,
        default=Path("oracc_data"),
        help="Path to ORACC data directory"
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

    # Define ORACC corpus directories in priority order
    oracc_dirs = [
        args.oracc_dir / "epsd2-literary",  # Primary
        args.oracc_dir / "epsd2-royal",      # Secondary
        # args.oracc_dir / "epsd2-admin-ur3",  # Tertiary (large, formulaic)
    ]

    if args.step in ("monolingual", "all"):
        print("\n" + "=" * 60)
        print("STEP 1: Building Monolingual Corpus")
        print("=" * 60)

        mono_stats = build_monolingual_corpus(
            output_path=args.output_dir / "pretrain" / "corpus_monolingual.txt",
            oracc_dirs=oracc_dirs,
            verbose=verbose
        )

        # Save stats
        with open(args.output_dir / "pretrain" / "stats.json", "w") as f:
            json.dump(mono_stats, f, indent=2)

    if args.step in ("finetune", "all"):
        print("\n" + "=" * 60)
        print("STEP 2: Splitting Fine-tuning Data")
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
