"""
ORACC Text Extractor

Extracts reconstructed LINES (not individual words) from ORACC JSON files.
Critical for MLM pre-training: preserves sentence structure for syntax learning.
"""

import json
import sys
from pathlib import Path
from typing import Generator

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "oracc_data"))
from normalization_bridge import normalize_oracc


def is_mostly_broken(text: str, threshold: float = 0.5) -> bool:
    """
    Filter lines that are mostly illegible.

    Args:
        text: Line text
        threshold: Max ratio of broken tokens (default 50%)

    Returns:
        True if line should be filtered out
    """
    tokens = text.split()
    if not tokens:
        return True
    broken_markers = {"X", "x", "…", "...", "n", "ø"}
    broken_count = sum(1 for t in tokens if t in broken_markers)
    return (broken_count / len(tokens)) > threshold


def is_composite_text(json_data: dict) -> bool:
    """
    Check if this is a composite (reconstructed) text vs physical tablet.

    Composites (Q-numbers) give cleaner continuous text.
    Physical tablets (P-numbers) have gaps and duplicates.
    """
    text_id = json_data.get("textid", "")
    # Q-numbers are composites
    if text_id.startswith("Q"):
        return True
    # Check type field
    text_type = json_data.get("type", "").lower()
    if "composite" in text_type:
        return True
    # Default: accept all (literary corpus is mostly composites anyway)
    return True


def extract_lines_from_oracc(json_data: dict) -> Generator[str, None, None]:
    """
    Extract reconstructed lines from ORACC CDL JSON.

    IMPORTANT: We extract LINES not WORDS.
    MLM needs: "lugal-e e2 mu-na-du3" (full sentence)
    NOT: "lugal-e" (one word per line)

    ORACC uses 'line-start' markers (type="line-start", node="d") to delimit lines.
    Lemmas (node="l") appear between line-start markers.

    Args:
        json_data: Parsed ORACC JSON

    Yields:
        Normalized line strings
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

    def extract_form(node: dict) -> str | None:
        """Extract normalized form from a lemma node."""
        if node.get("node") == "l":
            f = node.get("f", {})
            form = f.get("form", "")
            if form:
                return normalize_oracc(form)
        return None

    # Flatten the entire CDL tree
    all_nodes = flatten_cdl(json_data)

    # Group lemmas between line-start markers
    current_line_words = []

    for node in all_nodes:
        # Line-start marker - emit current line and start new one
        if node.get("type") == "line-start":
            if current_line_words:
                yield " ".join(current_line_words)
                current_line_words = []

        # Lemma node - extract form
        form = extract_form(node)
        if form:
            current_line_words.append(form)

    # Don't forget the last line
    if current_line_words:
        yield " ".join(current_line_words)


def process_oracc_file(json_path: str | Path) -> list[str]:
    """
    Extract all normalized lines from an ORACC JSON file.

    Args:
        json_path: Path to ORACC JSON file

    Returns:
        List of normalized, filtered lines
    """
    with open(json_path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return []

    # Optional: filter non-composite texts
    # if not is_composite_text(data):
    #     return []

    lines = []
    for line in extract_lines_from_oracc(data):
        # Skip empty or mostly broken lines
        if line and not is_mostly_broken(line):
            lines.append(line)

    return lines


def process_corpus_directory(
    corpus_dir: str | Path,
    output_path: str | Path,
    verbose: bool = True
) -> dict:
    """
    Process all JSON files in an ORACC corpus directory.

    Args:
        corpus_dir: Path to corpusjson/ directory
        output_path: Path to output text file
        verbose: Print progress

    Returns:
        Statistics dict
    """
    corpus_dir = Path(corpus_dir)
    output_path = Path(output_path)

    # Find all JSON files
    json_files = list(corpus_dir.rglob("*.json"))
    if verbose:
        print(f"Found {len(json_files)} JSON files in {corpus_dir}")

    stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "total_lines": 0,
        "total_words": 0,
    }

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for i, json_file in enumerate(json_files):
            # Skip non-corpus files (glossary, catalogue, etc.)
            if json_file.parent.name != "corpusjson":
                continue

            lines = process_oracc_file(json_file)

            if lines:
                stats["files_processed"] += 1
                for line in lines:
                    out.write(line + "\n")
                    stats["total_lines"] += 1
                    stats["total_words"] += len(line.split())
            else:
                stats["files_skipped"] += 1

            if verbose and (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(json_files)} files...")

    if verbose:
        print(f"\nCompleted: {stats['files_processed']} files, "
              f"{stats['total_lines']} lines, {stats['total_words']} words")

    return stats


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract text from ORACC JSON corpus"
    )
    parser.add_argument(
        "corpus_dir",
        help="Path to ORACC corpus directory (contains corpusjson/)"
    )
    parser.add_argument(
        "-o", "--output",
        default="corpus.txt",
        help="Output file path (default: corpus.txt)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    stats = process_corpus_directory(
        args.corpus_dir,
        args.output,
        verbose=not args.quiet
    )

    print(f"\nStats: {json.dumps(stats, indent=2)}")
