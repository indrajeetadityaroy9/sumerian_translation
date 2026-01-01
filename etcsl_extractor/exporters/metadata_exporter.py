"""
Metadata catalogue exporter.

Generates composition metadata including bibliography and named entity indices.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from ..parsers.transliteration_parser import TransliterationParser
from ..parsers.translation_parser import TranslationParser
from ..config import (
    TRANSLITERATIONS_DIR,
    TRANSLATIONS_DIR,
    OUTPUT_DIR,
    CATEGORY_GROUPS,
)


class MetadataExporter:
    """
    Exports composition metadata to JSON.
    """

    def __init__(self):
        self.translit_parser = TransliterationParser()
        self.translation_parser = TranslationParser()
        self.all_named_entities = defaultdict(set)

    def export_composition_metadata(
        self,
        translit_path: Path,
        translation_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Export metadata for a single composition.
        """
        translit_data = self.translit_parser.parse_file(translit_path)

        translation_data = None
        if translation_path and translation_path.exists():
            translation_data = self.translation_parser.parse_file(translation_path)

        composition_id = translit_data["composition_id"]
        metadata = translit_data.get("metadata", {})
        stats = translit_data.get("statistics", {})

        # Determine category from composition ID
        parts = composition_id.split(".")
        group_num = int(parts[0]) if parts else 0
        category = {
            "group": group_num,
            "group_name": CATEGORY_GROUPS.get(group_num, "Unknown"),
        }

        # Track global named entities
        for ne_type, entities in stats.get("named_entities", {}).items():
            for entity in entities:
                self.all_named_entities[ne_type].add(entity)

        # Build bibliography with resolved citations
        bibliography = []
        for bib in metadata.get("bibliography", []):
            bibliography.append({
                "id": bib.get("ref_id"),
                "citation": bib.get("citation"),
            })

        # Build tablet sources
        tablets = [t.get("museum_number") for t in metadata.get("tablets", [])]

        result = {
            "composition_id": composition_id,
            "title": metadata.get("title"),
            "category": category,
            "files": {
                "transliteration": translit_path.name,
                "translation": translation_path.name if translation_path else None,
            },
            "statistics": {
                "lines": stats.get("line_count", 0),
                "words": stats.get("word_count", 0),
                "unique_lemmas": stats.get("unique_lemmas", 0),
                "damaged_words": stats.get("damaged_words", 0),
                "supplied_words": stats.get("supplied_words", 0),
                "gap_count": stats.get("gap_count", 0),
                "pos_distribution": stats.get("pos_distribution", {}),
            },
            "named_entities": stats.get("named_entities", {}),
            "sources": {
                "tablets": tablets,
                "bibliography": bibliography,
            },
            "editorial": {
                "editors": metadata.get("editors", []),
                "publication_date": metadata.get("publication_date"),
                "revision_count": len(metadata.get("revision_history", [])),
            },
        }

        # Add translation stats if available
        if translation_data:
            trans_stats = translation_data.get("statistics", {})
            result["translation_statistics"] = {
                "paragraphs": trans_stats.get("paragraph_count", 0),
                "words": trans_stats.get("word_count", 0),
                "quotes": trans_stats.get("quote_count", 0),
            }

        return result

    def export_all(
        self,
        output_path: Optional[Path] = None,
        limit: Optional[int] = None
    ) -> Path:
        """
        Export all composition metadata to JSON file.
        """
        if output_path is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_DIR / "metadata_catalogue.json"

        # Get all transliteration files
        translit_files = {f.stem.replace("c.", ""): f
                         for f in TRANSLITERATIONS_DIR.glob("c.*.xml")}
        translation_files = {f.stem.replace("t.", ""): f
                            for f in TRANSLATIONS_DIR.glob("t.*.xml")}

        comp_ids = sorted(translit_files.keys())
        if limit:
            comp_ids = comp_ids[:limit]

        print(f"Processing {len(comp_ids)} compositions...")

        compositions = []
        for comp_id in comp_ids:
            try:
                translit_path = translit_files[comp_id]
                translation_path = translation_files.get(comp_id)

                meta = self.export_composition_metadata(
                    translit_path,
                    translation_path
                )
                compositions.append(meta)

            except Exception as e:
                print(f"Error processing {comp_id}: {e}")

        # Build catalogue
        catalogue = {
            "total_compositions": len(compositions),
            "with_translations": sum(1 for c in compositions
                                     if c["files"]["translation"]),
            "total_lines": sum(c["statistics"]["lines"] for c in compositions),
            "total_words": sum(c["statistics"]["words"] for c in compositions),
            "compositions": compositions,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(catalogue, f, ensure_ascii=False, indent=2)

        print(f"\nExport complete:")
        print(f"  Compositions: {catalogue['total_compositions']}")
        print(f"  With translations: {catalogue['with_translations']}")
        print(f"  Total lines: {catalogue['total_lines']}")
        print(f"  Total words: {catalogue['total_words']}")
        print(f"  Output: {output_path}")

        # Export named entities index
        self._export_named_entities()

        return output_path

    def _export_named_entities(self):
        """Export global named entities index."""
        ne_path = OUTPUT_DIR / "named_entities.json"

        ne_index = {
            ne_type: sorted(entities)
            for ne_type, entities in self.all_named_entities.items()
        }

        # Add counts
        ne_summary = {
            "by_type": {
                ne_type: {
                    "count": len(entities),
                    "entities": sorted(entities),
                }
                for ne_type, entities in self.all_named_entities.items()
            },
            "total_unique": sum(len(e) for e in self.all_named_entities.values()),
        }

        with open(ne_path, 'w', encoding='utf-8') as f:
            json.dump(ne_summary, f, ensure_ascii=False, indent=2)

        print(f"  Named entities: {ne_path}")


def export_metadata(
    output_path: Optional[str] = None,
    limit: Optional[int] = None
) -> Path:
    """
    Convenience function to export metadata.
    """
    exporter = MetadataExporter()
    return exporter.export_all(
        output_path=Path(output_path) if output_path else None,
        limit=limit
    )


if __name__ == "__main__":
    output_path = export_metadata(limit=10)

    # Show sample
    print("\nSample metadata:")
    with open(output_path, 'r', encoding='utf-8') as f:
        catalogue = json.load(f)
        comp = catalogue["compositions"][0]
        print(f"\n{comp['composition_id']}: {comp['title']}")
        print(f"  Category: {comp['category']}")
        print(f"  Stats: {comp['statistics']['lines']} lines, {comp['statistics']['words']} words")
        print(f"  Named entities: {list(comp['named_entities'].keys())}")
