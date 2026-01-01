"""
Parallel corpus exporter.

Generates aligned Sumerian-English parallel corpus in JSONL format.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .text_generator import (
    generate_texts_from_tokens,
    aggregate_line_texts,
    clean_translation_text,
)
from ..parsers.transliteration_parser import TransliterationParser
from ..parsers.translation_parser import TranslationParser
from ..processors.parallel_aligner import ParallelAligner, LineAligner
from ..config import (
    TRANSLITERATIONS_DIR,
    TRANSLATIONS_DIR,
    OUTPUT_DIR,
    POS_TAGS,
    NE_TYPES,
)


class ParallelCorpusExporter:
    """
    Exports aligned parallel corpus to JSONL.

    Groups lines by their target paragraph to create proper many-to-one alignments.
    """

    def __init__(self):
        self.translit_parser = TransliterationParser()
        self.translation_parser = TranslationParser()
        self.seen_alignment_ids = set()
        self.stats = {
            "compositions_processed": 0,
            "alignments_created": 0,
            "tokens_extracted": 0,
            "errors": [],
        }

    def export_composition(
        self,
        translit_path: Path,
        translation_path: Path
    ) -> List[Dict[str, Any]]:
        """
        Export a single composition to alignment records.

        Groups lines that share the same corresp to create proper alignments.

        Args:
            translit_path: Path to transliteration file
            translation_path: Path to translation file

        Returns:
            List of alignment records
        """
        # Parse both files
        translit_data = self.translit_parser.parse_file(translit_path)
        translation_data = self.translation_parser.parse_file(translation_path)

        composition_id = translit_data["composition_id"]
        alignments = []

        # Build paragraph lookup
        para_by_id = {
            p.get("id"): p for p in translation_data.get("paragraphs", [])
        }

        # Group lines by their corresp (target paragraph)
        lines_by_corresp = defaultdict(list)
        for line in translit_data.get("lines", []):
            corresp = line.get("corresp")
            if corresp:
                lines_by_corresp[corresp].append(line)

        # Create alignments - one per paragraph
        for corresp, lines in lines_by_corresp.items():
            para = para_by_id.get(corresp)
            if not para:
                continue

            # Generate alignment
            alignment = self._create_alignment(
                composition_id=composition_id,
                lines=lines,
                paragraph=para,
            )

            if alignment:
                alignments.append(alignment)

        return alignments

    def _create_alignment(
        self,
        composition_id: str,
        lines: List[Dict[str, Any]],
        paragraph: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Create a single alignment record.

        Generates text FROM tokens to guarantee alignment.
        """
        # Generate source texts from tokens
        source_data = aggregate_line_texts(lines)

        if source_data["token_count"] == 0:
            return None

        # Build line info
        line_ids = [line.get("id") for line in lines]
        line_numbers = [line.get("n") for line in lines]

        # Create unique alignment ID
        alignment_id = f"{line_ids[0]}_{paragraph.get('id', '')}"

        # Check uniqueness
        if alignment_id in self.seen_alignment_ids:
            # Make unique by appending count
            count = 1
            while f"{alignment_id}_{count}" in self.seen_alignment_ids:
                count += 1
            alignment_id = f"{alignment_id}_{count}"
        self.seen_alignment_ids.add(alignment_id)

        # Clean translation text
        translation_text = clean_translation_text(paragraph.get("text", ""))

        # Determine quality flags
        tokens = source_data["tokens"]
        has_damage = any(t.get("flags", {}).get("damaged") for t in tokens)
        has_supplied = any(t.get("flags", {}).get("supplied") for t in tokens)
        has_unclear = any(t.get("flags", {}).get("unclear") for t in tokens)
        has_gaps = any(len(line.get("gaps", [])) > 0 for line in lines)
        has_gaps = has_gaps or len(paragraph.get("gaps", [])) > 0

        # Build token list for export (simplified)
        export_tokens = []
        for t in tokens:
            export_tokens.append({
                "form": t.get("form"),
                "form_normalized": t.get("form", "").replace("{{DET_", "").replace("}}", ""),
                "lemma": t.get("lemma"),
                "pos": t.get("pos"),
                "pos_full": POS_TAGS.get(t.get("pos")),
                "type": t.get("type"),
                "type_full": NE_TYPES.get(t.get("type")),
                "label": t.get("label"),
                "flags": t.get("flags", {}),
            })

        return {
            "composition_id": composition_id,
            "alignment_id": alignment_id,
            "source": {
                "line_ids": line_ids,
                "line_numbers": line_numbers,
                "text_raw": source_data["text_raw"],
                "text_display": source_data["text_display"],
                "text_normalized": source_data["text_normalized"],
                "tokens": export_tokens,
                "token_count": source_data["token_count"],
            },
            "target": {
                "paragraph_id": paragraph.get("id"),
                "line_range": paragraph.get("n"),
                "text": translation_text,
                "named_entities": paragraph.get("named_entities", []),
            },
            "quality_flags": {
                "has_damage": has_damage,
                "has_supplied": has_supplied,
                "has_unclear": has_unclear,
                "has_gaps": has_gaps,
            },
        }

    def export_all(
        self,
        output_path: Optional[Path] = None,
        limit: Optional[int] = None
    ) -> Path:
        """
        Export all compositions to JSONL file.

        Args:
            output_path: Output file path (default: output/parallel_corpus.jsonl)
            limit: Optional limit on number of compositions

        Returns:
            Path to output file
        """
        if output_path is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_DIR / "parallel_corpus.jsonl"

        # Find matching pairs
        translit_files = {f.stem.replace("c.", ""): f
                         for f in TRANSLITERATIONS_DIR.glob("c.*.xml")}
        translation_files = {f.stem.replace("t.", ""): f
                            for f in TRANSLATIONS_DIR.glob("t.*.xml")}

        # Find compositions with both files
        matched = set(translit_files.keys()) & set(translation_files.keys())
        matched = sorted(matched)

        if limit:
            matched = matched[:limit]

        print(f"Found {len(matched)} compositions with both transliteration and translation")

        # Export to JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for comp_id in matched:
                try:
                    translit_path = translit_files[comp_id]
                    translation_path = translation_files[comp_id]

                    alignments = self.export_composition(translit_path, translation_path)

                    for alignment in alignments:
                        f.write(json.dumps(alignment, ensure_ascii=False) + '\n')
                        self.stats["alignments_created"] += 1
                        self.stats["tokens_extracted"] += alignment["source"]["token_count"]

                    self.stats["compositions_processed"] += 1

                except Exception as e:
                    self.stats["errors"].append({
                        "composition": comp_id,
                        "error": str(e)
                    })
                    print(f"Error processing {comp_id}: {e}")

        print(f"\nExport complete:")
        print(f"  Compositions: {self.stats['compositions_processed']}")
        print(f"  Alignments: {self.stats['alignments_created']}")
        print(f"  Tokens: {self.stats['tokens_extracted']}")
        print(f"  Errors: {len(self.stats['errors'])}")
        print(f"  Output: {output_path}")

        return output_path


def export_parallel_corpus(
    output_path: Optional[str] = None,
    limit: Optional[int] = None
) -> Path:
    """
    Convenience function to export parallel corpus.

    Args:
        output_path: Optional output path
        limit: Optional limit on compositions

    Returns:
        Path to output file
    """
    exporter = ParallelCorpusExporter()
    return exporter.export_all(
        output_path=Path(output_path) if output_path else None,
        limit=limit
    )


if __name__ == "__main__":
    # Test with a few compositions
    output_path = export_parallel_corpus(limit=5)

    # Show sample
    print("\nSample output:")
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            record = json.loads(line)
            print(f"\n{record['composition_id']} - {record['alignment_id']}")
            print(f"  Source: {record['source']['text_display'][:80]}...")
            print(f"  Target: {record['target']['text'][:80]}...")
