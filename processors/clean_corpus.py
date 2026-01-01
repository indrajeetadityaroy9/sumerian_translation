"""
Corpus Cleaning Script

Filters out low-quality data that would hurt ML training:
- Single-word lines (no syntax context for MLM)
- Excessive duplicates (causes model bias toward formulaic text)
- Empty targets in translation pairs
- Extreme length ratios (likely misalignments)
"""

import json
import argparse
from collections import Counter
from pathlib import Path


def clean_pretraining_data(
    input_file: Path,
    output_file: Path,
    min_words: int = 2,
    max_duplicates: int = 5,
    verbose: bool = True
) -> dict:
    """
    Clean monolingual corpus for MLM pre-training.

    Args:
        input_file: Path to corpus_monolingual.txt
        output_file: Path to cleaned output
        min_words: Minimum words per line (default: 2)
        max_duplicates: Max times a line can appear (default: 5)
        verbose: Print statistics

    Returns:
        Statistics dict
    """
    stats = {
        "input_lines": 0,
        "output_lines": 0,
        "filtered_short": 0,
        "filtered_duplicate": 0,
        "input_words": 0,
        "output_words": 0,
    }

    seen_lines = Counter()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        for line in f_in:
            text = line.strip()
            if not text:
                continue

            stats["input_lines"] += 1
            tokens = text.split()
            stats["input_words"] += len(tokens)

            # Filter 1: Single-word lines (no syntax context for MLM)
            if len(tokens) < min_words:
                stats["filtered_short"] += 1
                continue

            # Filter 2: Cap duplicates (prevent bias toward formulaic text)
            if seen_lines[text] >= max_duplicates:
                stats["filtered_duplicate"] += 1
                continue
            seen_lines[text] += 1

            f_out.write(text + "\n")
            stats["output_lines"] += 1
            stats["output_words"] += len(tokens)

    if verbose:
        print(f"\nPretraining Data Cleaning:")
        print(f"  Input:  {stats['input_lines']:,} lines, {stats['input_words']:,} words")
        print(f"  Output: {stats['output_lines']:,} lines, {stats['output_words']:,} words")
        print(f"  Filtered (short):     {stats['filtered_short']:,}")
        print(f"  Filtered (duplicate): {stats['filtered_duplicate']:,}")
        retention = stats['output_lines'] / stats['input_lines'] * 100
        print(f"  Retention: {retention:.1f}%")

    return stats


def clean_finetuning_data(
    input_file: Path,
    output_file: Path,
    min_ratio: float = 0.25,
    max_ratio: float = 4.0,
    verbose: bool = True
) -> dict:
    """
    Clean parallel corpus for fine-tuning.

    Args:
        input_file: Path to train.jsonl or valid.jsonl
        output_file: Path to cleaned output
        min_ratio: Minimum source/target length ratio
        max_ratio: Maximum source/target length ratio
        verbose: Print statistics

    Returns:
        Statistics dict
    """
    stats = {
        "input_pairs": 0,
        "output_pairs": 0,
        "filtered_empty_target": 0,
        "filtered_empty_source": 0,
        "filtered_ratio": 0,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:

        for line in f_in:
            entry = json.loads(line)
            stats["input_pairs"] += 1

            src = entry.get("source", {}).get("text_normalized", "")
            tgt = entry.get("target", {}).get("text", "")

            # Filter 1: Empty targets
            if not tgt or not tgt.strip():
                stats["filtered_empty_target"] += 1
                continue

            # Filter 2: Empty sources
            if not src or not src.strip():
                stats["filtered_empty_source"] += 1
                continue

            src_len = len(src.split())
            tgt_len = len(tgt.split())

            # Filter 3: Extreme length ratios (likely misalignment)
            if src_len == 0 or tgt_len == 0:
                stats["filtered_empty_source"] += 1
                continue

            ratio = src_len / tgt_len
            if ratio < min_ratio or ratio > max_ratio:
                stats["filtered_ratio"] += 1
                continue

            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stats["output_pairs"] += 1

    if verbose:
        print(f"\nFine-tuning Data Cleaning ({input_file.name}):")
        print(f"  Input:  {stats['input_pairs']:,} pairs")
        print(f"  Output: {stats['output_pairs']:,} pairs")
        print(f"  Filtered (empty target): {stats['filtered_empty_target']:,}")
        print(f"  Filtered (empty source): {stats['filtered_empty_source']:,}")
        print(f"  Filtered (ratio):        {stats['filtered_ratio']:,}")
        if stats['input_pairs'] > 0:
            retention = stats['output_pairs'] / stats['input_pairs'] * 100
            print(f"  Retention: {retention:.1f}%")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Clean training data for ML"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output_training"),
        help="Directory with training data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_training_clean"),
        help="Output directory for cleaned data"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=2,
        help="Minimum words per line for pretraining (default: 2)"
    )
    parser.add_argument(
        "--max-duplicates",
        type=int,
        default=5,
        help="Maximum duplicate lines (default: 5)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress output"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    all_stats = {}

    # Clean pretraining data
    pretrain_input = args.input_dir / "pretrain" / "corpus_monolingual.txt"
    if pretrain_input.exists():
        pretrain_output = args.output_dir / "pretrain" / "corpus_monolingual.txt"
        all_stats["pretrain"] = clean_pretraining_data(
            pretrain_input,
            pretrain_output,
            min_words=args.min_words,
            max_duplicates=args.max_duplicates,
            verbose=verbose
        )

    # Clean fine-tuning data
    for split in ["train", "valid"]:
        ft_input = args.input_dir / "finetune" / f"{split}.jsonl"
        if ft_input.exists():
            ft_output = args.output_dir / "finetune" / f"{split}.jsonl"
            all_stats[split] = clean_finetuning_data(
                ft_input,
                ft_output,
                verbose=verbose
            )

    # Save stats
    stats_file = args.output_dir / "cleaning_stats.json"
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "w") as f:
        json.dump(all_stats, f, indent=2)

    if verbose:
        print(f"\nCleaning complete. Output: {args.output_dir}")
        print(f"Stats saved to: {stats_file}")


if __name__ == "__main__":
    main()
