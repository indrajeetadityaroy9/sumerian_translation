"""
JSONL file I/O utilities.

Provides consistent interface for reading/writing JSONL files
across all training and processing scripts.
"""

import json
from pathlib import Path
from typing import Iterator, List, Dict, Any, Union


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load all records from a JSONL file into memory.

    Args:
        path: Path to JSONL file

    Returns:
        List of parsed JSON objects
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """
    Iterate over JSONL file without loading all into memory.

    Args:
        path: Path to JSONL file

    Yields:
        Parsed JSON objects one at a time
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_jsonl(
    path: Union[str, Path],
    records: List[Dict[str, Any]],
    ensure_ascii: bool = False
) -> int:
    """
    Save records to a JSONL file.

    Args:
        path: Output file path
        records: List of JSON-serializable objects
        ensure_ascii: If True, escape non-ASCII characters

    Returns:
        Number of records written
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")

    return len(records)


def load_text_corpus(path: Union[str, Path]) -> List[str]:
    """
    Load a plain text corpus (one line per document).

    Args:
        path: Path to text file

    Returns:
        List of non-empty lines
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
