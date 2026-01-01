"""
Text generation from tokens.

Generates multiple text representations (raw, display, normalized) from tokens,
guaranteeing alignment between token list and text.
"""

from typing import Dict, List, Any
import re

from ..config import DETERMINATIVES, NORMALIZATION_MAP


def placeholder_to_display(text: str) -> str:
    """
    Convert determinative placeholders to display format.

    Args:
        text: Text with {{DET_*}} placeholders

    Returns:
        Text with superscript-style determinatives like <d>
    """
    result = text
    for entity, (placeholder, _, display) in DETERMINATIVES.items():
        result = result.replace(placeholder, f"<{display}>")
    return result


def placeholder_to_normalized(text: str) -> str:
    """
    Remove all determinative placeholders for normalized text.

    Args:
        text: Text with {{DET_*}} placeholders

    Returns:
        Text with determinatives removed, suitable for NLP
    """
    result = text
    for entity, (placeholder, _, _) in DETERMINATIVES.items():
        result = result.replace(placeholder, "")
    return result


def normalize_form(form: str) -> str:
    """
    Normalize a Sumerian form for consistent tokenization.

    - Removes determinative placeholders
    - Applies character normalization

    Args:
        form: Raw form string

    Returns:
        Normalized form
    """
    result = placeholder_to_normalized(form)

    # Apply additional normalization
    for old, new in NORMALIZATION_MAP.items():
        result = result.replace(old, new)

    return result


def generate_texts_from_tokens(tokens: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Generate all text representations from a token list.

    This guarantees that len(tokens) == len(text_normalized.split())
    by constructing text FROM the tokens.

    Args:
        tokens: List of token dictionaries with 'form' key

    Returns:
        Dictionary with text_raw, text_display, text_normalized
    """
    forms_raw = []
    forms_display = []
    forms_normalized = []

    for token in tokens:
        form = token.get("form", "")
        if not form:
            continue

        forms_raw.append(form)
        forms_display.append(placeholder_to_display(form))
        forms_normalized.append(normalize_form(form))

    return {
        "text_raw": " ".join(forms_raw),
        "text_display": " ".join(forms_display),
        "text_normalized": " ".join(forms_normalized),
    }


def aggregate_line_texts(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate multiple lines into a single text block with proper spacing.

    Args:
        lines: List of line dictionaries

    Returns:
        Aggregated text representations and flattened tokens
    """
    all_tokens = []

    for line in lines:
        for word in line.get("words", []):
            all_tokens.append(word)

    texts = generate_texts_from_tokens(all_tokens)
    texts["tokens"] = all_tokens
    texts["token_count"] = len(all_tokens)

    return texts


def clean_translation_text(text: str) -> str:
    """
    Clean translation text for export.

    - Normalizes whitespace
    - Removes excessive line breaks

    Args:
        text: Raw translation text

    Returns:
        Cleaned text
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
