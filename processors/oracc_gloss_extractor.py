"""
ORACC Gloss-to-Text Extractor

Generates "Silver Standard" parallel corpus by extracting Sumerian lines
and building synthetic English from word-level glosses.

Usage:
    python oracc_gloss_extractor.py --corpus-dir epsd2-literary --output oracc_synthetic.jsonl
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
from collections import defaultdict


def normalize_sumerian(form: str) -> str:
    """
    Normalize Sumerian form for consistent tokenization.

    - Removes subscript numbers
    - Normalizes special characters
    """
    # Remove subscript numbers
    for i in range(10):
        form = form.replace(f"₀₁₂₃₄₅₆₇₈₉"[i], "")
        form = form.replace(str(i), "")

    # Basic normalization
    form = form.replace("š", "sz").replace("ŋ", "j").replace("ḫ", "h")

    return form.strip()


def extract_line_pairs(json_data: dict) -> Generator[Tuple[str, str], None, None]:
    """
    Extract Sumerian lines paired with synthetic English from glosses.

    Args:
        json_data: Parsed ORACC JSON

    Yields:
        (sumerian_line, synthetic_english) tuples
    """

    def flatten_cdl(node) -> list:
        """Flatten CDL tree into ordered list of nodes."""
        result = []
        if isinstance(node, dict):
            result.append(node)
            for v in node.values():
                if isinstance(v, (dict, list)):
                    result.extend(flatten_cdl(v))
        elif isinstance(node, list):
            for item in node:
                result.extend(flatten_cdl(item))
        return result

    # Flatten the entire CDL tree
    all_nodes = flatten_cdl(json_data)

    # Collect lemmas between line-start markers
    current_sumerian = []
    current_glosses = []

    for node in all_nodes:
        # Line-start marker - emit current line and start new one
        if node.get("type") == "line-start":
            if current_sumerian and current_glosses:
                sumerian = " ".join(current_sumerian)
                english = " ".join(current_glosses)
                yield (sumerian, english)
            current_sumerian = []
            current_glosses = []

        # Lemma node - extract form and gloss
        if node.get("node") == "l":
            f = node.get("f", {})
            form = f.get("form", "")
            gw = f.get("gw", "")

            if form:
                # Normalize Sumerian form
                normalized = normalize_sumerian(form)
                if normalized:
                    current_sumerian.append(normalized)

            if gw:
                # Clean up guide word
                gw_clean = gw.strip().lower()
                # Skip placeholder glosses
                if gw_clean and gw_clean not in {"x", "?", "n", "..."}:
                    current_glosses.append(gw_clean)

    # Don't forget the last line
    if current_sumerian and current_glosses:
        sumerian = " ".join(current_sumerian)
        english = " ".join(current_glosses)
        yield (sumerian, english)


def process_corpus(
    corpus_dir: Path,
    output_path: Path,
    min_words: int = 2,
    max_words: int = 50,
    verbose: bool = True
) -> Dict:
    """
    Process all JSON files in an ORACC corpus directory.

    Args:
        corpus_dir: Path to corpus directory (contains corpusjson/)
        output_path: Path to output JSONL file
        min_words: Minimum words per line
        max_words: Maximum words per line
        verbose: Print progress

    Returns:
        Statistics dict
    """
    stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "pairs_extracted": 0,
        "pairs_filtered": 0,
        "total_sumerian_words": 0,
        "total_english_words": 0,
        "unique_glosses": set(),
    }

    # Find corpusjson directory
    corpusjson_dirs = list(corpus_dir.rglob("corpusjson"))
    if not corpusjson_dirs:
        print(f"No corpusjson directory found in {corpus_dir}")
        return stats

    json_files = []
    for cdir in corpusjson_dirs:
        json_files.extend(cdir.glob("*.json"))

    if verbose:
        print(f"Found {len(json_files)} JSON files")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for i, json_file in enumerate(json_files):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)

                text_id = data.get("textid", json_file.stem)
                file_pairs = 0

                for sumerian, english in extract_line_pairs(data):
                    sum_words = len(sumerian.split())
                    eng_words = len(english.split())

                    # Filter by length
                    if sum_words < min_words or sum_words > max_words:
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

            except Exception as e:
                if verbose:
                    print(f"Error processing {json_file}: {e}")
                stats["files_skipped"] += 1

            if verbose and (i + 1) % 200 == 0:
                print(f"  Processed {i + 1}/{len(json_files)} files...")

    # Convert set to count for JSON serialization
    stats["unique_gloss_count"] = len(stats["unique_glosses"])
    del stats["unique_glosses"]

    if verbose:
        print(f"\nExtraction complete:")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Files skipped: {stats['files_skipped']}")
        print(f"  Pairs extracted: {stats['pairs_extracted']}")
        print(f"  Pairs filtered: {stats['pairs_filtered']}")
        print(f"  Unique glosses: {stats['unique_gloss_count']}")
        print(f"  Output: {output_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract synthetic parallel corpus from ORACC glosses"
    )
    parser.add_argument(
        "corpus_dir",
        type=Path,
        help="Path to ORACC corpus directory"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("oracc_synthetic.jsonl"),
        help="Output file path"
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

    stats = process_corpus(
        corpus_dir=args.corpus_dir,
        output_path=args.output,
        min_words=args.min_words,
        max_words=args.max_words,
        verbose=not args.quiet
    )

    # Save stats
    stats_path = args.output.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()
