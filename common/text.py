"""
Text processing utilities for Sumerian translation pipeline.

Shared functions for text cleaning, normalization, and control token handling.
"""

import re
from config import ControlTokens


# Bracketed gloss placeholders (e.g., [oil], [tree])
GLOSS_PLACEHOLDER_PATTERN = re.compile(r'\s*\[[^\]]+\]\s*')


def clean_source_text(
    text: str,
    remove_placeholders: bool = False,
    control_pattern: re.Pattern = None,
) -> str:
    """
    Clean source text by removing control tokens and normalizing whitespace.

    Args:
        text: Input text to clean
        remove_placeholders: If True, also remove bracketed placeholders like [oil]
        control_pattern: Custom regex pattern for control tokens (default: ControlTokens.PATTERN)

    Returns:
        Cleaned and normalized text
    """
    if control_pattern is None:
        control_pattern = ControlTokens.PATTERN

    # Remove control tokens
    text = control_pattern.sub('', text)

    # Optionally remove bracketed placeholders
    if remove_placeholders:
        text = GLOSS_PLACEHOLDER_PATTERN.sub(' ', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text.strip()
