"""
Linguistic annotation exporter.

Generates word-level annotation dataset in JSONL format.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .text_generator import normalize_form, placeholder_to_display
from ..parsers.transliteration_parser import TransliterationParser
from ..config import (
    TRANSLITERATIONS_DIR,
    OUTPUT_DIR,
    POS_TAGS,
    NE_TYPES,
    DETERMINATIVES,
)


class AnnotationExporter:
    """
    Exports word-level linguistic annotations to JSONL.
    """

    def __init__(self):
        self.parser = TransliterationParser()
        self.stats = {
            "compositions_processed": 0,
            "tokens_exported": 0,
            "unique_lemmas": set(),
            "pos_distribution": defaultdict(int),
            "ne_distribution": defaultdict(int),
        }

    def export_composition(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Export annotations from a single composition.

        Args:
            file_path: Path to transliteration file

        Returns:
            List of token annotation records
        """
        parsed = self.parser.parse_file(file_path)
        composition_id = parsed["composition_id"]
        tokens = []

        for line in parsed.get("lines", []):
            line_id = line.get("id", "")
            line_n = line.get("n", "")
            corresp = line.get("corresp")

            # Get context: previous and next words in line
            words = line.get("words", [])

            for i, word in enumerate(words):
                # Build context
                prev_word = words[i - 1].get("form") if i > 0 else None
                next_word = words[i + 1].get("form") if i < len(words) - 1 else None

                form = word.get("form", "")
                lemma = word.get("lemma")
                pos = word.get("pos")
                word_type = word.get("type")
                det = word.get("det")

                # Resolve determinative type
                det_type = None
                if det:
                    for entity, (placeholder, d_type, _) in DETERMINATIVES.items():
                        if placeholder in det:
                            det_type = d_type
                            break

                # Track statistics
                if lemma:
                    self.stats["unique_lemmas"].add(lemma)
                if pos:
                    self.stats["pos_distribution"][pos] += 1
                if word_type:
                    self.stats["ne_distribution"][word_type] += 1

                token = {
                    "word_id": f"{line_id}.{word.get('position', i)}",
                    "composition_id": composition_id,
                    "line_id": line_id,
                    "line_number": line_n,
                    "position_in_line": word.get("position", i),
                    "corresp": corresp,

                    # Form variations
                    "form": form,
                    "form_display": placeholder_to_display(form),
                    "form_normalized": normalize_form(form),

                    # Linguistic annotation
                    "lemma": lemma,
                    "pos": pos,
                    "pos_full": POS_TAGS.get(pos),

                    # Named entity
                    "named_entity_type": word_type,
                    "named_entity_type_full": NE_TYPES.get(word_type),
                    "semantic_label": word.get("label"),

                    # Determinative
                    "determinative": det,
                    "determinative_type": det_type,

                    # Morphology
                    "form_type": word.get("form_type"),
                    "emesal": word.get("emesal"),
                    "bound": word.get("bound"),

                    # Text quality flags
                    "flags": word.get("flags", {}),

                    # Context
                    "context": {
                        "prev_word": prev_word,
                        "next_word": next_word,
                    },
                }
                tokens.append(token)

        return tokens

    def export_all(
        self,
        output_path: Optional[Path] = None,
        limit: Optional[int] = None
    ) -> Path:
        """
        Export all compositions to JSONL file.

        Args:
            output_path: Output file path
            limit: Optional limit on compositions

        Returns:
            Path to output file
        """
        if output_path is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_DIR / "linguistic_annotations.jsonl"

        files = sorted(TRANSLITERATIONS_DIR.glob("c.*.xml"))
        if limit:
            files = files[:limit]

        print(f"Processing {len(files)} transliteration files...")

        with open(output_path, 'w', encoding='utf-8') as f:
            for file_path in files:
                try:
                    tokens = self.export_composition(file_path)

                    for token in tokens:
                        f.write(json.dumps(token, ensure_ascii=False) + '\n')
                        self.stats["tokens_exported"] += 1

                    self.stats["compositions_processed"] += 1

                except Exception as e:
                    print(f"Error processing {file_path.name}: {e}")

        print(f"\nExport complete:")
        print(f"  Compositions: {self.stats['compositions_processed']}")
        print(f"  Tokens: {self.stats['tokens_exported']}")
        print(f"  Unique lemmas: {len(self.stats['unique_lemmas'])}")
        print(f"  Output: {output_path}")

        # Export vocabulary
        self._export_vocabulary()

        return output_path

    def _export_vocabulary(self):
        """Export vocabulary summary."""
        vocab_path = OUTPUT_DIR / "vocabulary.json"

        vocab = {
            "total_tokens": self.stats["tokens_exported"],
            "unique_lemmas": len(self.stats["unique_lemmas"]),
            "pos_distribution": dict(self.stats["pos_distribution"]),
            "ne_distribution": dict(self.stats["ne_distribution"]),
            "lemma_list": sorted(self.stats["unique_lemmas"]),
        }

        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        print(f"  Vocabulary: {vocab_path}")


def export_annotations(
    output_path: Optional[str] = None,
    limit: Optional[int] = None
) -> Path:
    """
    Convenience function to export annotations.
    """
    exporter = AnnotationExporter()
    return exporter.export_all(
        output_path=Path(output_path) if output_path else None,
        limit=limit
    )


if __name__ == "__main__":
    output_path = export_annotations(limit=5)

    # Show sample
    print("\nSample annotations:")
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            record = json.loads(line)
            print(f"\n{record['word_id']}: {record['form_display']}")
            print(f"  lemma={record['lemma']}, pos={record['pos_full']}")
            print(f"  NE={record['named_entity_type_full']}, label={record['semantic_label']}")
