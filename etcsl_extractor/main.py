"""
ETCSL Dataset Extractor - Main CLI entry point.

Extracts the parallel corpus from the Electronic Text Corpus of Sumerian Literature:
- parallel_corpus.jsonl: Aligned Sumerian-English pairs (Gold Standard for NMT)
"""

import argparse
from pathlib import Path

from .config import OUTPUT_DIR
from .exporters.parallel_corpus_exporter import export_parallel_corpus


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract parallel corpus from ETCSL"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of compositions to process (for testing)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ETCSL Parallel Corpus Extractor")
    print("=" * 60)

    print("\nGenerating parallel corpus...")
    parallel_path = export_parallel_corpus(limit=args.limit)
    print(f"  -> {parallel_path}")

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
