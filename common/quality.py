"""
Data quality filtering utilities.

Provides consistent quality checks for training data across
processing scripts.
"""

from collections import Counter
from typing import Set, Optional


# Common markers indicating broken/illegible text
BROKEN_MARKERS: Set[str] = {"X", "x", "...", "n", "ø"}


def is_mostly_broken(
    text: str,
    threshold: float = 0.5,
    broken_markers: Optional[Set[str]] = None
) -> bool:
    """
    Check if a text line is mostly illegible/broken.

    Args:
        text: Line text to check
        threshold: Maximum ratio of broken tokens (default 50%)
        broken_markers: Set of tokens indicating damage (uses default if None)

    Returns:
        True if line should be filtered out
    """
    if broken_markers is None:
        broken_markers = BROKEN_MARKERS

    tokens = text.split()
    if not tokens:
        return True

    broken_count = sum(1 for t in tokens if t in broken_markers)
    return (broken_count / len(tokens)) > threshold


def is_valid_length(
    text: str,
    min_words: int = 2,
    max_words: int = 50
) -> bool:
    """
    Check if text has valid length for training.

    Args:
        text: Text to check
        min_words: Minimum word count
        max_words: Maximum word count

    Returns:
        True if text length is valid
    """
    word_count = len(text.split())
    return min_words <= word_count <= max_words


def filter_duplicates(
    texts: list,
    max_duplicates: int = 5
) -> list:
    """
    Filter texts that appear too many times.

    Useful for removing overly formulaic patterns from training data.

    Args:
        texts: List of text strings
        max_duplicates: Maximum allowed occurrences of any text

    Returns:
        Filtered list with excessive duplicates removed
    """
    counts = Counter(texts)
    return [
        text for text in texts
        if counts[text] <= max_duplicates
    ]


def is_valid_pair(
    source: str,
    target: str,
    min_source_words: int = 2,
    max_source_words: int = 50,
    min_target_words: int = 1,
    max_target_words: int = 100,
    max_ratio: float = 5.0
) -> bool:
    """
    Validate a source-target pair for translation training.

    Args:
        source: Source (Sumerian) text
        target: Target (English) text
        min_source_words: Minimum source words
        max_source_words: Maximum source words
        min_target_words: Minimum target words
        max_target_words: Maximum target words
        max_ratio: Maximum length ratio between source and target

    Returns:
        True if pair is valid for training
    """
    src_words = len(source.split())
    tgt_words = len(target.split())

    # Check length bounds
    if not (min_source_words <= src_words <= max_source_words):
        return False
    if not (min_target_words <= tgt_words <= max_target_words):
        return False

    # Check length ratio (avoid very unbalanced pairs)
    if src_words > 0 and tgt_words > 0:
        ratio = max(src_words / tgt_words, tgt_words / src_words)
        if ratio > max_ratio:
            return False

    return True
