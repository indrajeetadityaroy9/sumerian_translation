"""
Unified ORACC Corpus Parser

Consolidates ORACC data extraction into a single interface:
- Monolingual line extraction (for MLM pre-training)
- Parallel pair extraction (for alignment pre-training)

Replaces: oracc_extractor.py, oracc_gloss_extractor.py
"""

import json
from pathlib import Path
from typing import Dict, List, Generator, Tuple, Optional

from sumerian_nmt.data_processing.normalization import normalize_oracc
from sumerian_nmt.utils.quality import is_mostly_broken


class ORACCParser:
    """
    Unified parser for ORACC CDL JSON format.

    ORACC uses Cuneiform Digital Library (CDL) JSON structure with:
    - 'line-start' markers (type="line-start") to delimit lines
    - Lemma nodes (node="l") containing word data
    - Forms in f.form, glosses in f.gw
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize parser.

        Args:
            normalize: Apply text normalization (default True)
        """
        self.normalize = normalize

    @staticmethod
    def flatten_cdl(node) -> List[dict]:
        """
        Flatten CDL tree into ordered list of nodes.

        Args:
            node: CDL node (dict or list)

        Returns:
            Flattened list of all nodes in traversal order
        """
        result = []
        if isinstance(node, dict):
            result.append(node)
            for v in node.values():
                if isinstance(v, (dict, list)):
                    result.extend(ORACCParser.flatten_cdl(v))
        elif isinstance(node, list):
            for item in node:
                result.extend(ORACCParser.flatten_cdl(item))
        return result

    def _extract_form(self, node: dict) -> Optional[str]:
        """Extract normalized form from a lemma node."""
        if node.get("node") == "l":
            f = node.get("f", {})
            form = f.get("form", "")
            if form:
                return normalize_oracc(form) if self.normalize else form
        return None

    def _extract_gloss(self, node: dict) -> Optional[str]:
        """Extract guide word (gloss) from a lemma node."""
        if node.get("node") == "l":
            f = node.get("f", {})
            gw = f.get("gw", "")
            if gw:
                gw_clean = gw.strip().lower()
                # Skip placeholder glosses
                if gw_clean and gw_clean not in {"x", "?", "n", "..."}:
                    return gw_clean
        return None

    def extract_lines(self, json_data: dict) -> Generator[str, None, None]:
        """
        Extract monolingual Sumerian lines from ORACC CDL JSON.

        Used for MLM pre-training - preserves sentence structure.

        Args:
            json_data: Parsed ORACC JSON

        Yields:
            Normalized line strings
        """
        all_nodes = self.flatten_cdl(json_data)
        current_line_words = []

        for node in all_nodes:
            # Line-start marker - emit current line and start new one
            if node.get("type") == "line-start":
                if current_line_words:
                    yield " ".join(current_line_words)
                    current_line_words = []

            # Lemma node - extract form
            form = self._extract_form(node)
            if form:
                current_line_words.append(form)

        # Don't forget the last line
        if current_line_words:
            yield " ".join(current_line_words)

    def extract_parallel_pairs(
        self, json_data: dict
    ) -> Generator[Tuple[str, str], None, None]:
        """
        Extract Sumerian lines paired with synthetic English from glosses.

        Used for alignment pre-training (Silver corpus).

        Args:
            json_data: Parsed ORACC JSON

        Yields:
            (sumerian_line, synthetic_english) tuples
        """
        all_nodes = self.flatten_cdl(json_data)
        current_sumerian = []
        current_glosses = []

        for node in all_nodes:
            # Line-start marker - emit current line and start new one
            if node.get("type") == "line-start":
                if current_sumerian and current_glosses:
                    yield (" ".join(current_sumerian), " ".join(current_glosses))
                current_sumerian = []
                current_glosses = []

            # Lemma node - extract form and gloss
            if node.get("node") == "l":
                form = self._extract_form(node)
                if form:
                    current_sumerian.append(form)

                gloss = self._extract_gloss(node)
                if gloss:
                    current_glosses.append(gloss)

        # Don't forget the last line
        if current_sumerian and current_glosses:
            yield (" ".join(current_sumerian), " ".join(current_glosses))


def process_monolingual_corpus(
    corpus_dir: Path,
    output_path: Path,
    filter_broken: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Process ORACC corpus for MLM pre-training.

    Args:
        corpus_dir: Path to corpus directory (contains corpusjson/)
        output_path: Path to output text file
        filter_broken: Filter mostly-broken lines
        verbose: Print progress

    Returns:
        Statistics dict
    """
    parser = ORACCParser(normalize=True)
    stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "total_lines": 0,
        "total_words": 0,
    }

    # Find all JSON files
    json_files = list(corpus_dir.rglob("corpusjson/*.json"))
    if verbose:
        print(f"Found {len(json_files)} JSON files in {corpus_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for i, json_file in enumerate(json_files):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                file_lines = 0
                for line in parser.extract_lines(data):
                    if filter_broken and is_mostly_broken(line):
                        continue
                    out.write(line + "\n")
                    stats["total_lines"] += 1
                    stats["total_words"] += len(line.split())
                    file_lines += 1

                if file_lines > 0:
                    stats["files_processed"] += 1
                else:
                    stats["files_skipped"] += 1

            except (json.JSONDecodeError, Exception):
                stats["files_skipped"] += 1

            if verbose and (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(json_files)} files...")

    if verbose:
        print(f"\nCompleted: {stats['files_processed']} files, "
              f"{stats['total_lines']} lines, {stats['total_words']} words")

    return stats


def process_parallel_corpus(
    corpus_dir: Path,
    output_path: Path,
    min_words: int = 2,
    max_words: int = 50,
    verbose: bool = True
) -> Dict:
    """
    Process ORACC corpus for parallel data (Silver corpus).

    Args:
        corpus_dir: Path to corpus directory (contains corpusjson/)
        output_path: Path to output JSONL file
        min_words: Minimum words per line
        max_words: Maximum words per line
        verbose: Print progress

    Returns:
        Statistics dict
    """
    parser = ORACCParser(normalize=True)
    stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "pairs_extracted": 0,
        "pairs_filtered": 0,
        "total_sumerian_words": 0,
        "total_english_words": 0,
        "unique_glosses": set(),
    }

    # Find all JSON files
    json_files = list(corpus_dir.rglob("corpusjson/*.json"))
    if verbose:
        print(f"Found {len(json_files)} JSON files in {corpus_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for i, json_file in enumerate(json_files):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                text_id = data.get("textid", json_file.stem)
                file_pairs = 0

                for sumerian, english in parser.extract_parallel_pairs(data):
                    sum_words = len(sumerian.split())
                    eng_words = len(english.split())

                    # Filter by length
                    if not (min_words <= sum_words <= max_words):
                        stats["pairs_filtered"] += 1
                        continue
                    if eng_words < min_words:
                        stats["pairs_filtered"] += 1
                        continue

                    # Write pair
                    record = {
                        "source": {
                            "text_normalized": sumerian,
                            "text_id": text_id,
                        },
                        "target": {
                            "text": english,
                            "type": "synthetic_gloss",
                        }
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")

                    stats["pairs_extracted"] += 1
                    stats["total_sumerian_words"] += sum_words
                    stats["total_english_words"] += eng_words
                    stats["unique_glosses"].update(english.split())
                    file_pairs += 1

                if file_pairs > 0:
                    stats["files_processed"] += 1
                else:
                    stats["files_skipped"] += 1

            except (json.JSONDecodeError, Exception):
                stats["files_skipped"] += 1

            if verbose and (i + 1) % 200 == 0:
                print(f"  Processed {i + 1}/{len(json_files)} files...")

    # Convert set to count for JSON serialization
    stats["unique_gloss_count"] = len(stats["unique_glosses"])
    del stats["unique_glosses"]

    if verbose:
        print(f"\nExtraction complete:")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Pairs extracted: {stats['pairs_extracted']}")
        print(f"  Unique glosses: {stats['unique_gloss_count']}")

    return stats


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract data from ORACC corpus"
    )
    parser.add_argument(
        "corpus_dir",
        type=Path,
        help="Path to ORACC corpus directory"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output file path"
    )
    parser.add_argument(
        "--mode",
        choices=["monolingual", "parallel"],
        default="parallel",
        help="Extraction mode (default: parallel)"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=2,
        help="Minimum words per line (default: 2)"
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=50,
        help="Maximum words per line (default: 50)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    if args.mode == "monolingual":
        stats = process_monolingual_corpus(
            corpus_dir=args.corpus_dir,
            output_path=args.output,
            verbose=not args.quiet
        )
    else:
        stats = process_parallel_corpus(
            corpus_dir=args.corpus_dir,
            output_path=args.output,
            min_words=args.min_words,
            max_words=args.max_words,
            verbose=not args.quiet
        )

    print(f"\nStats: {json.dumps(stats, indent=2)}")
