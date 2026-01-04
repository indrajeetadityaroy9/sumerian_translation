"""
Data Ingestion: ETCSL and ORACC Corpus Extraction

Handles parsing of cuneiform XML corpora into structured parallel data:
- ETCSL (Electronic Text Corpus of Sumerian Literature): Primary bilingual corpus
- ORACC (Open Richly Annotated Cuneiform Corpus): Monolingual texts for augmentation

Components:
- extractor: Main entry point for corpus extraction
- parsers/: XML parsing for transliterations and translations
- exporters/: JSONL output generation
- processors/: Line alignment logic
- corpus_config: Determinatives, character mappings, entity types
- oracc_core: CDL JSON parsing for ORACC texts
"""

from .extractor import main as extract_corpus

__all__ = [
    "extract_corpus",
]
