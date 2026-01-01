"""
Parser for ETCSL transliteration files (c.*.xml).

Extracts complete linguistic annotations from Sumerian composite texts.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from lxml import etree

from .preprocessor import XMLPreprocessor
from ..utils.xml_utils import (
    parse_xml_string,
    extract_line_data,
    extract_header_metadata,
)
from ..config import TRANSLITERATIONS_DIR


class TransliterationParser:
    """
    Parser for ETCSL transliteration (composite) files.

    Handles:
    - XML entity preprocessing
    - TEI header extraction
    - Line-by-line parsing with word-level annotations
    - Damage/supplied/unclear flag detection
    """

    def __init__(self):
        self.preprocessor = XMLPreprocessor()

    def parse_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a complete transliteration file.

        Args:
            file_path: Path to c.*.xml file

        Returns:
            Dictionary with composition_id, metadata, lines, and statistics
        """
        file_path = Path(file_path)

        # Preprocess to handle entities
        xml_content = self.preprocessor.preprocess_file(file_path)

        # Parse XML
        root = parse_xml_string(xml_content)

        # Extract composition ID from root element
        composition_id = root.get("id", "").replace("c.", "")

        # Extract header metadata
        header = root.find(".//teiHeader")
        metadata = extract_header_metadata(header) if header is not None else {}

        # Extract all lines from body
        lines = []
        body = root.find(".//body")
        if body is not None:
            for l_elem in body.findall(".//l"):
                line_data = extract_line_data(l_elem)
                lines.append(line_data)

        # Calculate statistics
        stats = self._calculate_stats(lines)

        return {
            "composition_id": composition_id,
            "file_name": file_path.name,
            "metadata": metadata,
            "lines": lines,
            "statistics": stats,
        }

    def _calculate_stats(self, lines: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics for the composition."""
        total_words = 0
        damaged_words = 0
        supplied_words = 0
        unclear_words = 0
        unique_lemmas = set()
        named_entities = {
            "DN": set(),
            "GN": set(),
            "PN": set(),
            "SN": set(),
            "TN": set(),
            "WN": set(),
            "RN": set(),
        }
        pos_counts = {}
        gap_count = 0

        for line in lines:
            gap_count += len(line.get("gaps", []))

            for word in line.get("words", []):
                total_words += 1

                # Count damage flags
                flags = word.get("flags", {})
                if flags.get("damaged"):
                    damaged_words += 1
                if flags.get("supplied"):
                    supplied_words += 1
                if flags.get("unclear"):
                    unclear_words += 1

                # Track lemmas
                lemma = word.get("lemma")
                if lemma:
                    unique_lemmas.add(lemma)

                # Track named entities
                word_type = word.get("type")
                label = word.get("label")
                if word_type in named_entities and label:
                    named_entities[word_type].add(label)

                # Track POS
                pos = word.get("pos")
                if pos:
                    pos_counts[pos] = pos_counts.get(pos, 0) + 1

        return {
            "line_count": len(lines),
            "word_count": total_words,
            "unique_lemmas": len(unique_lemmas),
            "damaged_words": damaged_words,
            "supplied_words": supplied_words,
            "unclear_words": unclear_words,
            "gap_count": gap_count,
            "pos_distribution": pos_counts,
            "named_entities": {k: list(v) for k, v in named_entities.items() if v},
        }

    def parse_all(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Parse all transliteration files.

        Args:
            limit: Optional limit on number of files to parse

        Returns:
            List of parsed compositions
        """
        results = []
        files = sorted(TRANSLITERATIONS_DIR.glob("c.*.xml"))

        if limit:
            files = files[:limit]

        for file_path in files:
            try:
                result = self.parse_file(file_path)
                results.append(result)
            except Exception as e:
                print(f"Error parsing {file_path.name}: {e}")

        return results


def extract_tokens_for_export(parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten parsed data into token-level records for linguistic annotation export.

    Args:
        parsed_data: Output from TransliterationParser.parse_file()

    Returns:
        List of token records
    """
    tokens = []
    composition_id = parsed_data["composition_id"]

    for line in parsed_data["lines"]:
        line_id = line.get("id", "")
        line_n = line.get("n", "")

        for word in line.get("words", []):
            token = {
                "word_id": f"{line_id}.{word['position']}",
                "composition_id": composition_id,
                "line_id": line_id,
                "line_number": line_n,
                "position_in_line": word["position"],
                "form": word["form"],
                "lemma": word.get("lemma"),
                "pos": word.get("pos"),
                "type": word.get("type"),
                "label": word.get("label"),
                "det": word.get("det"),
                "form_type": word.get("form_type"),
                "emesal": word.get("emesal"),
                "flags": word.get("flags", {}),
            }
            tokens.append(token)

    return tokens


if __name__ == "__main__":
    # Test parsing
    parser = TransliterationParser()

    test_file = TRANSLITERATIONS_DIR / "c.1.1.1.xml"
    if test_file.exists():
        result = parser.parse_file(test_file)

        print(f"Composition: {result['composition_id']}")
        print(f"Title: {result['metadata'].get('title', 'N/A')}")
        print(f"Lines: {result['statistics']['line_count']}")
        print(f"Words: {result['statistics']['word_count']}")
        print(f"Unique lemmas: {result['statistics']['unique_lemmas']}")
        print(f"Damaged words: {result['statistics']['damaged_words']}")
        print(f"POS distribution: {result['statistics']['pos_distribution']}")
        print(f"Named entities: {result['statistics']['named_entities']}")

        # Show sample line
        if result["lines"]:
            print("\nSample line (first with words):")
            for line in result["lines"]:
                if line["words"]:
                    print(f"  Line {line['n']} (id={line['id']}):")
                    for w in line["words"][:3]:
                        print(f"    {w['form']} -> lemma={w['lemma']}, pos={w['pos']}, type={w['type']}")
                    break
    else:
        print(f"Test file not found: {test_file}")
