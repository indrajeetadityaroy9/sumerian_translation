"""
ETCSL Dataset Extractor - Main CLI entry point.

Extracts unified datasets from the Electronic Text Corpus of Sumerian Literature:
- parallel_corpus.jsonl: Aligned Sumerian-English pairs
- linguistic_annotations.jsonl: Word-level annotations
- metadata_catalogue.json: Composition metadata
- vocabulary.json: Lemma inventory
- named_entities.json: Named entity index
"""

import argparse
import sys
from pathlib import Path

from .config import OUTPUT_DIR
from .exporters.parallel_corpus_exporter import export_parallel_corpus
from .exporters.annotation_exporter import export_annotations
from .exporters.metadata_exporter import export_metadata


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract unified datasets from ETCSL corpus"
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
    parser.add_argument(
        "--dataset",
        choices=["all", "parallel", "annotations", "metadata"],
        default="all",
        help="Which dataset(s) to generate"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ETCSL Dataset Extractor")
    print("=" * 60)

    if args.dataset in ["all", "parallel"]:
        print("\n[1/3] Generating parallel corpus...")
        parallel_path = export_parallel_corpus(limit=args.limit)
        print(f"  -> {parallel_path}")

    if args.dataset in ["all", "annotations"]:
        print("\n[2/3] Generating linguistic annotations...")
        annot_path = export_annotations(limit=args.limit)
        print(f"  -> {annot_path}")

    if args.dataset in ["all", "metadata"]:
        print("\n[3/3] Generating metadata catalogue...")
        meta_path = export_metadata(limit=args.limit)
        print(f"  -> {meta_path}")

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
