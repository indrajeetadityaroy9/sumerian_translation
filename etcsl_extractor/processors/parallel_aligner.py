"""
Parallel corpus aligner for ETCSL.

Aligns transliteration lines with translation paragraphs using corresp attributes.
Uses LUT-based range resolution for non-sequential line IDs.
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


class LineAligner:
    """
    Aligns transliteration lines using Look-Up Table (LUT) approach.

    Handles non-sequential line IDs like "12A", "12B" which math-based
    range parsing cannot handle.
    """

    def __init__(self, lines: List[Dict[str, Any]]):
        """
        Initialize with ordered list of lines.

        Args:
            lines: List of line dictionaries from transliteration parser
        """
        # Build ordered list of all line IDs
        self.all_line_ids = [line.get("id") for line in lines if line.get("id")]
        self.id_to_index = {lid: i for i, lid in enumerate(self.all_line_ids)}
        self.id_to_line = {line.get("id"): line for line in lines if line.get("id")}

    def resolve_range(self, corresp: str) -> List[str]:
        """
        Resolve a corresp reference to a list of line IDs.

        Handles:
        - Single ID: "c111.1"
        - Full range: "c111.1-c111.4"
        - Abbreviated range: "c111.1-4"

        Args:
            corresp: Corresp attribute value

        Returns:
            List of resolved line IDs
        """
        if not corresp:
            return []

        if "-" not in corresp:
            return [corresp] if corresp in self.id_to_index else []

        # Split on hyphen
        parts = corresp.split("-", 1)
        start_id = parts[0]
        end_part = parts[1]

        # Handle abbreviated form: "c111.1-4" → "c111.1-c111.4"
        if "." not in end_part:
            # Get prefix from start_id
            prefix = start_id.rsplit(".", 1)[0]
            end_id = f"{prefix}.{end_part}"
        else:
            end_id = end_part

        # Look up indices
        start_idx = self.id_to_index.get(start_id)
        end_idx = self.id_to_index.get(end_id)

        if start_idx is None or end_idx is None:
            # Fallback: return just the start if it exists
            if start_id in self.id_to_index:
                return [start_id]
            return []

        return self.all_line_ids[start_idx:end_idx + 1]

    def get_lines(self, line_ids: List[str]) -> List[Dict[str, Any]]:
        """Get line data for a list of line IDs."""
        return [self.id_to_line[lid] for lid in line_ids if lid in self.id_to_line]


class ParallelAligner:
    """
    Creates aligned parallel corpus entries from transliteration and translation.
    """

    def __init__(
        self,
        transliteration_data: Dict[str, Any],
        translation_data: Dict[str, Any]
    ):
        """
        Initialize with parsed transliteration and translation data.

        Args:
            transliteration_data: Output from TransliterationParser.parse_file()
            translation_data: Output from TranslationParser.parse_file()
        """
        self.translit = transliteration_data
        self.translation = translation_data

        # Build line aligner
        self.line_aligner = LineAligner(transliteration_data.get("lines", []))

        # Build paragraph lookup by ID
        self.para_by_id = {
            p.get("id"): p for p in translation_data.get("paragraphs", [])
        }

    def create_alignments(self) -> List[Dict[str, Any]]:
        """
        Create all parallel alignments.

        Returns:
            List of alignment records
        """
        alignments = []
        used_lines = set()

        # Strategy 1: Use translation paragraphs as anchor
        # Each paragraph has corresp pointing to source lines
        for para in self.translation.get("paragraphs", []):
            corresp = para.get("corresp")
            if not corresp:
                continue

            # Resolve which lines this paragraph corresponds to
            line_ids = self.line_aligner.resolve_range(corresp)
            lines = self.line_aligner.get_lines(line_ids)

            if not lines:
                continue

            # Mark lines as used
            for lid in line_ids:
                used_lines.add(lid)

            # Create alignment record
            alignment = self._create_alignment_record(
                lines=lines,
                paragraph=para,
                alignment_type="exact"
            )
            alignments.append(alignment)

        # Strategy 2: Check lines with corresp pointing to translation
        # Some lines have corresp attributes pointing to paragraph IDs
        for line in self.translit.get("lines", []):
            line_id = line.get("id")
            if line_id in used_lines:
                continue

            corresp = line.get("corresp")
            if corresp and corresp in self.para_by_id:
                para = self.para_by_id[corresp]

                alignment = self._create_alignment_record(
                    lines=[line],
                    paragraph=para,
                    alignment_type="reverse_corresp"
                )
                alignments.append(alignment)
                used_lines.add(line_id)

        return alignments

    def _create_alignment_record(
        self,
        lines: List[Dict[str, Any]],
        paragraph: Dict[str, Any],
        alignment_type: str
    ) -> Dict[str, Any]:
        """
        Create a single alignment record.

        Args:
            lines: Source lines from transliteration
            paragraph: Target paragraph from translation
            alignment_type: Type of alignment (exact, reverse_corresp, etc.)

        Returns:
            Alignment record dictionary
        """
        composition_id = self.translit.get("composition_id", "")

        # Collect all tokens from lines
        tokens = []
        for line in lines:
            for word in line.get("words", []):
                tokens.append(word)

        # Collect line IDs
        line_ids = [line.get("id") for line in lines]

        # Determine quality flags
        has_damage = any(
            t.get("flags", {}).get("damaged", False) for t in tokens
        )
        has_supplied = any(
            t.get("flags", {}).get("supplied", False) for t in tokens
        )
        has_gaps = any(len(line.get("gaps", [])) > 0 for line in lines)
        has_gaps = has_gaps or len(paragraph.get("gaps", [])) > 0

        return {
            "composition_id": composition_id,
            "alignment_id": f"{line_ids[0]}_{paragraph.get('id', '')}",
            "alignment_type": alignment_type,
            "source": {
                "line_ids": line_ids,
                "line_numbers": [line.get("n") for line in lines],
                "tokens": tokens,
            },
            "target": {
                "paragraph_id": paragraph.get("id"),
                "line_range": paragraph.get("n"),
                "text": paragraph.get("text", ""),
                "named_entities": paragraph.get("named_entities", []),
                "quotes": paragraph.get("quotes", []),
            },
            "quality_flags": {
                "has_damage": has_damage,
                "has_supplied": has_supplied,
                "has_gaps": has_gaps,
                "alignment_confidence": alignment_type,
            }
        }


def align_composition(
    translit_path: str,
    translation_path: str
) -> List[Dict[str, Any]]:
    """
    Convenience function to align a single composition.

    Args:
        translit_path: Path to transliteration file
        translation_path: Path to translation file

    Returns:
        List of alignment records
    """
    from ..parsers.transliteration_parser import TransliterationParser
    from ..parsers.translation_parser import TranslationParser

    translit_parser = TransliterationParser()
    translation_parser = TranslationParser()

    translit_data = translit_parser.parse_file(translit_path)
    translation_data = translation_parser.parse_file(translation_path)

    aligner = ParallelAligner(translit_data, translation_data)
    return aligner.create_alignments()


if __name__ == "__main__":
    from ..config import TRANSLITERATIONS_DIR, TRANSLATIONS_DIR

    # Test alignment
    translit_file = TRANSLITERATIONS_DIR / "c.1.1.1.xml"
    translation_file = TRANSLATIONS_DIR / "t.1.1.1.xml"

    if translit_file.exists() and translation_file.exists():
        alignments = align_composition(str(translit_file), str(translation_file))

        print(f"Created {len(alignments)} alignments")

        if alignments:
            print("\nSample alignment:")
            a = alignments[0]
            print(f"  ID: {a['alignment_id']}")
            print(f"  Source lines: {a['source']['line_ids']}")
            print(f"  Target paragraph: {a['target']['paragraph_id']}")
            print(f"  Tokens: {len(a['source']['tokens'])}")
            print(f"  Translation text: {a['target']['text'][:150]}...")
            print(f"  Quality: {a['quality_flags']}")
    else:
        print("Test files not found")
