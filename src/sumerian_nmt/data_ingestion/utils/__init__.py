"""Utilities for XML parsing and corpus processing."""

from .xml_utils import (
    parse_xml_string,
    extract_text_content,
    check_damage_context,
    extract_word_data,
    extract_line_data,
    extract_paragraph_data,
    extract_header_metadata,
)

__all__ = [
    "parse_xml_string",
    "extract_text_content",
    "check_damage_context",
    "extract_word_data",
    "extract_line_data",
    "extract_paragraph_data",
    "extract_header_metadata",
]
