"""
Normalization Bridge for ETCSL and ORACC Text Conventions

This module unifies the different character conventions used by ETCSL and ORACC
corpora for consistent tokenization in the training pipeline.

Key differences:
    ETCSL: Uses ĝ (U+011E), š (U+0161), determinative placeholders {{DET_*}}
    ORACC: Uses ŋ (U+014B), š (U+0161), inline determinatives

Both use Unicode subscript numerals (₀-₉) which are typically removed for NLP.
"""

import re
import sys
from pathlib import Path

# Import ETCSL config for determinative mappings
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from etcsl_extractor.config import DETERMINATIVES, NORMALIZATION_MAP
except ImportError:
    # Fallback definitions if config not available
    DETERMINATIVES = {}
    NORMALIZATION_MAP = {
        "sz": "š", "SZ": "Š",
        "j": "ĝ", "J": "Ĝ",
        "₀": "", "₁": "", "₂": "", "₃": "", "₄": "",
        "₅": "", "₆": "", "₇": "", "₈": "", "₉": "",
    }

# Unicode subscript digits
SUBSCRIPT_DIGITS = "₀₁₂₃₄₅₆₇₈₉"

# ORACC-specific character mappings (ŋ -> ĝ for ETCSL compatibility)
ORACC_TO_ETCSL = {
    "ŋ": "ĝ",
    "Ŋ": "Ĝ",
}

# ASCII-safe conversions (for models that don't handle Unicode well)
UNICODE_TO_ASCII = {
    "š": "sz",
    "Š": "SZ",
    "ĝ": "j",
    "Ĝ": "J",
    "ŋ": "j",
    "Ŋ": "J",
    "ḫ": "h",
    "Ḫ": "H",
    "ṣ": "s",
    "Ṣ": "S",
    "ṭ": "t",
    "Ṭ": "T",
}


def remove_subscripts(text: str) -> str:
    """
    Remove all subscript numerals from text.

    Handles both Unicode subscripts (₀-₉) and trailing digits.

    Args:
        text: Input text

    Returns:
        Text with subscripts removed
    """
    # Remove Unicode subscripts
    for digit in SUBSCRIPT_DIGITS:
        text = text.replace(digit, "")

    # Remove trailing digits from tokens (e.g., dug4 -> dug)
    # But be careful not to remove standalone numbers
    text = re.sub(r'(\w+?)(\d+)(?=\s|$|-)', r'\1', text)

    return text


def remove_determinative_placeholders(text: str) -> str:
    """
    Remove ETCSL-style determinative placeholders ({{DET_*}}).

    Args:
        text: Text with {{DET_*}} placeholders

    Returns:
        Text with placeholders removed
    """
    # Remove all {{DET_*}} patterns
    text = re.sub(r'\{\{DET_\w+\}\}', '', text)

    # Also handle individual determinative mappings from config
    for entity, (placeholder, _, _) in DETERMINATIVES.items():
        text = text.replace(placeholder, "")

    return text


def normalize_etcsl(text: str, keep_unicode: bool = True) -> str:
    """
    Normalize ETCSL text for consistent tokenization.

    Processing steps:
    1. Remove determinative placeholders ({{DET_DIVINE}}, etc.)
    2. Apply NORMALIZATION_MAP (sz->š, j->ĝ, remove subscripts)
    3. Optionally convert to ASCII (š->sz, ĝ->j)
    4. Normalize whitespace

    Args:
        text: ETCSL text (may contain {{DET_*}} placeholders)
        keep_unicode: If True, keep Unicode chars (š, ĝ). If False, use ASCII.

    Returns:
        Normalized text ready for tokenization
    """
    if not text:
        return ""

    # Step 1: Remove determinative placeholders
    result = remove_determinative_placeholders(text)

    # Step 2: Apply NORMALIZATION_MAP (normalizes ASCII variants to Unicode)
    for old, new in NORMALIZATION_MAP.items():
        result = result.replace(old, new)

    # Step 3: Optionally convert to ASCII
    if not keep_unicode:
        for unicode_char, ascii_char in UNICODE_TO_ASCII.items():
            result = result.replace(unicode_char, ascii_char)

    # Step 4: Normalize whitespace
    result = re.sub(r'\s+', ' ', result).strip()

    return result


def normalize_oracc(form: str, keep_unicode: bool = True) -> str:
    """
    Normalize ORACC form for consistent tokenization.

    Processing steps:
    1. Convert ŋ -> ĝ (ORACC to ETCSL convention)
    2. Remove subscript numerals
    3. Optionally convert to ASCII
    4. Normalize whitespace

    Args:
        form: ORACC form string
        keep_unicode: If True, keep Unicode chars. If False, use ASCII.

    Returns:
        Normalized form ready for tokenization
    """
    if not form:
        return ""

    result = form

    # Step 1: Convert ORACC ŋ to ETCSL ĝ
    for oracc_char, etcsl_char in ORACC_TO_ETCSL.items():
        result = result.replace(oracc_char, etcsl_char)

    # Step 2: Remove subscripts
    result = remove_subscripts(result)

    # Step 3: Optionally convert to ASCII
    if not keep_unicode:
        for unicode_char, ascii_char in UNICODE_TO_ASCII.items():
            result = result.replace(unicode_char, ascii_char)

    # Step 4: Normalize whitespace
    result = re.sub(r'\s+', ' ', result).strip()

    return result


def unify_conventions(text: str, source: str = "auto") -> str:
    """
    Unify text to a common convention regardless of source.

    Auto-detection heuristics:
    - Contains {{DET_*}} -> ETCSL
    - Contains ŋ -> ORACC
    - Default -> treat as ETCSL

    Args:
        text: Input text
        source: "etcsl", "oracc", or "auto" for auto-detection

    Returns:
        Normalized text in unified convention (Unicode, no determinatives)
    """
    if not text:
        return ""

    # Auto-detect source
    if source == "auto":
        if "{{DET_" in text:
            source = "etcsl"
        elif "ŋ" in text or "Ŋ" in text:
            source = "oracc"
        else:
            source = "etcsl"  # Default

    # Apply appropriate normalization
    if source == "etcsl":
        return normalize_etcsl(text, keep_unicode=True)
    elif source == "oracc":
        return normalize_oracc(text, keep_unicode=True)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'etcsl', 'oracc', or 'auto'.")


def to_ascii(text: str) -> str:
    """
    Convert normalized text to ASCII-safe representation.

    Useful for models/tokenizers that don't handle Unicode well.

    Args:
        text: Normalized text (may contain š, ĝ, ḫ, etc.)

    Returns:
        ASCII-safe text (sz, j, h, etc.)
    """
    result = text
    for unicode_char, ascii_char in UNICODE_TO_ASCII.items():
        result = result.replace(unicode_char, ascii_char)
    return result


# Convenience aliases for backward compatibility
def normalize(text: str) -> str:
    """Alias for unify_conventions with auto-detection."""
    return unify_conventions(text, source="auto")
