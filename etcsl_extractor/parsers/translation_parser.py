"""
Parser for ETCSL translation files (t.*.xml).

Extracts English prose translations with named entities and quoted speech.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from lxml import etree

from .preprocessor import XMLPreprocessor
from ..utils.xml_utils import (
    parse_xml_string,
    extract_paragraph_data,
    extract_header_metadata,
    extract_text_content,
)
from ..config import TRANSLATIONS_DIR


class TranslationParser:
    """
    Parser for ETCSL translation files.

    Handles:
    - XML entity preprocessing
    - Paragraph extraction with corresp links
    - Named entity extraction from translations
    - Quoted speech with speaker attribution
    """

    def __init__(self):
        self.preprocessor = XMLPreprocessor()

    def parse_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a complete translation file.

        Args:
            file_path: Path to t.*.xml file

        Returns:
            Dictionary with composition_id, metadata, paragraphs
        """
        file_path = Path(file_path)

        # Preprocess to handle entities
        xml_content = self.preprocessor.preprocess_file(file_path)

        # Parse XML
        root = parse_xml_string(xml_content)

        # Extract composition ID
        composition_id = root.get("id", "").replace("t.", "")

        # Extract header metadata
        header = root.find(".//teiHeader")
        metadata = extract_header_metadata(header) if header is not None else {}

        # Extract paragraphs from body
        paragraphs = []
        body = root.find(".//body")
        if body is not None:
            for p_elem in body.findall(".//p"):
                para_data = extract_paragraph_data(p_elem)
                paragraphs.append(para_data)

        # Build corresp mapping for alignment
        corresp_map = {}
        for para in paragraphs:
            corresp = para.get("corresp")
            if corresp:
                corresp_map[corresp] = para

        # Calculate statistics
        stats = self._calculate_stats(paragraphs)

        return {
            "composition_id": composition_id,
            "file_name": file_path.name,
            "metadata": metadata,
            "paragraphs": paragraphs,
            "corresp_map": corresp_map,
            "statistics": stats,
        }

    def _calculate_stats(self, paragraphs: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics for the translation."""
        total_chars = 0
        total_words = 0
        named_entities = {
            "DN": set(),
            "GN": set(),
            "PN": set(),
            "SN": set(),
            "TN": set(),
            "WN": set(),
            "RN": set(),
        }
        gap_count = 0
        quote_count = 0

        for para in paragraphs:
            text = para.get("text", "")
            total_chars += len(text)
            total_words += len(text.split())

            gap_count += len(para.get("gaps", []))
            quote_count += len(para.get("quotes", []))

            for ne in para.get("named_entities", []):
                ne_type = ne.get("type")
                ne_text = ne.get("text")
                if ne_type in named_entities and ne_text:
                    named_entities[ne_type].add(ne_text)

        return {
            "paragraph_count": len(paragraphs),
            "character_count": total_chars,
            "word_count": total_words,
            "gap_count": gap_count,
            "quote_count": quote_count,
            "named_entities": {k: list(v) for k, v in named_entities.items() if v},
        }

    def parse_all(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Parse all translation files.

        Args:
            limit: Optional limit on number of files to parse

        Returns:
            List of parsed translations
        """
        results = []
        files = sorted(TRANSLATIONS_DIR.glob("t.*.xml"))

        if limit:
            files = files[:limit]

        for file_path in files:
            try:
                result = self.parse_file(file_path)
                results.append(result)
            except Exception as e:
                print(f"Error parsing {file_path.name}: {e}")

        return results


if __name__ == "__main__":
    # Test parsing
    parser = TranslationParser()

    test_file = TRANSLATIONS_DIR / "t.1.1.1.xml"
    if test_file.exists():
        result = parser.parse_file(test_file)

        print(f"Composition: {result['composition_id']}")
        print(f"Title: {result['metadata'].get('title', 'N/A')}")
        print(f"Paragraphs: {result['statistics']['paragraph_count']}")
        print(f"Words: {result['statistics']['word_count']}")
        print(f"Gaps: {result['statistics']['gap_count']}")
        print(f"Quotes: {result['statistics']['quote_count']}")

        # Show sample paragraph
        if result["paragraphs"]:
            print("\nSample paragraph:")
            para = result["paragraphs"][0]
            print(f"  id={para['id']}, n={para['n']}, corresp={para['corresp']}")
            print(f"  Text: {para['text'][:200]}...")
            if para['named_entities']:
                print(f"  Named entities: {para['named_entities'][:3]}")
    else:
        print(f"Test file not found: {test_file}")
