"""
Entity Linker for Sumerian NMT Graph Augmentation

Maps Sumerian lemmas to entity types (DN, RN, GN) using glossary lookup
with fuzzy matching fallback for variant spellings.

Data sources:
- data/consolidated/glossary_sux.parquet: Entity types in 'pos' column
- ETCSL tokens already have 'type' and 'label' fields

Fuzzy matching fallback chain:
1. Exact match
2. Strip determinatives: {d}en-lil2 -> en-lil2
3. Remove subscripts: en-lil2 -> en-lil
4. Normalize unicode: ĝ -> g, š -> sz
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
import unicodedata

import pandas as pd


@dataclass
class Entity:
    """Represents a named entity from the glossary."""
    citation_form: str  # Canonical Sumerian form
    entity_type: str    # DN, RN, GN
    guide_word: str     # Disambiguator (often "1" for unique entries)
    instances: int      # Frequency in corpus
    english_label: Optional[str] = None  # English representation (usually same as citation_form for proper nouns)
    aliases: List[str] = field(default_factory=list)  # Variant forms that map to this entity

    def __hash__(self):
        return hash((self.citation_form, self.entity_type, self.guide_word))


class EntityLinker:
    """
    Links Sumerian lemmas to entity types using glossary lookup.

    Supports fuzzy matching for variant spellings common in cuneiform texts.
    """

    # Entity types for substitution
    # DN = Divine Name, RN = Royal Name, GN = Geographic Name, SN = Settlement Name
    ENTITY_TYPES = {'DN', 'RN', 'GN', 'SN'}

    # Common determinative patterns in Sumerian
    DETERMINATIVE_PATTERN = re.compile(r'\{[a-z0-9]+\}')

    # Subscript patterns (numeric suffixes like en-lil2)
    SUBSCRIPT_PATTERN = re.compile(r'[₀-₉0-9]+')

    # Unicode normalization map for Sumerian special characters
    UNICODE_MAP = {
        'ĝ': 'g',
        'Ĝ': 'G',
        'š': 'sz',
        'Š': 'SZ',
        'ŋ': 'ng',
        'ḫ': 'h',
        'Ḫ': 'H',
        'ṭ': 't',
        'ṣ': 's',
    }

    # Manual alias mappings: Surface Form -> Glossary Citation Form
    # Handles philological mismatches between text forms and dictionary headwords
    MANUAL_ALIASES = {
        # Inanna variants -> Inanak (the glossary form)
        'inanna': 'Inanak',
        '{d}inanna': 'Inanak',
        'inana': 'Inanak',
        '{d}inana': 'Inanak',
        # Nippur/Nibru (Settlement Name)
        'nibru': 'Nibru',
        'nibru{ki}': 'Nibru',
        '{ki}nibru': 'Nibru',
        'nippur': 'Nibru',
        # Enlil variants (backup)
        'en-lil': 'Enlil',
        'en-lil2': 'Enlil',
        '{d}en-lil': 'Enlil',
        '{d}en-lil2': 'Enlil',
        # Enki variants
        'en-ki': 'Enkik',
        '{d}en-ki': 'Enkik',
        'enki': 'Enkik',
        # Utu variants
        '{d}utu': 'Utu',
        # Nanna/Suen variants
        '{d}nanna': 'Nanna',
        '{d}suen': 'Suen',
        'su-en': 'Suen',
        '{d}su-en': 'Suen',
        # Common geographic names
        'unug': 'Unug',  # Uruk
        'urim': 'Urim',  # Ur
        'uri': 'Urim',
        '{ki}urim': 'Urim',
        'eridu': 'Eridug',
        '{ki}eridu': 'Eridug',
        'eridug': 'Eridug',
        # Akkad
        'agade': 'Akkad',
        '{ki}agade': 'Akkad',
    }

    def __init__(self, glossary_path: Optional[Path] = None):
        """
        Initialize entity linker with glossary data.

        Args:
            glossary_path: Path to glossary parquet file. If None, uses default from config.
        """
        if glossary_path is None:
            from config import Paths
            glossary_path = Paths.GLOSSARY_PARQUET

        self.glossary_path = Path(glossary_path)

        # Primary lookup: exact citation_form -> Entity
        self._exact_index: Dict[str, Entity] = {}

        # Secondary lookup: normalized form -> Entity (for fuzzy matching)
        self._normalized_index: Dict[str, Entity] = {}

        # Type-based index for substitution candidates
        self._by_type: Dict[str, List[Entity]] = {t: [] for t in self.ENTITY_TYPES}

        # Statistics
        self.stats = {
            'total_entities': 0,
            'by_type': {t: 0 for t in self.ENTITY_TYPES},
            'lookup_exact': 0,
            'lookup_stripped': 0,
            'lookup_normalized': 0,
            'lookup_failed': 0,
        }

        self._load_glossary()

    def _load_glossary(self):
        """Load and index glossary data."""
        if not self.glossary_path.exists():
            raise FileNotFoundError(f"Glossary not found: {self.glossary_path}")

        df = pd.read_parquet(self.glossary_path)

        # Filter to entity types only
        entity_df = df[df['pos'].isin(self.ENTITY_TYPES)]

        for _, row in entity_df.iterrows():
            cf = row['citation_form']
            entity_type = row['pos']
            guide_word = str(row['guide_word'])
            instances = int(row['instances']) if pd.notna(row['instances']) else 1

            # For proper nouns, citation_form IS the English representation
            english_label = cf

            entity = Entity(
                citation_form=cf,
                entity_type=entity_type,
                guide_word=guide_word,
                instances=instances,
                english_label=english_label,
            )

            # Index by exact form
            self._exact_index[cf.lower()] = entity

            # Index by normalized form (for fuzzy matching)
            normalized = self._normalize_form(cf)
            if normalized not in self._normalized_index:
                self._normalized_index[normalized] = entity

            # Add to type-based index
            self._by_type[entity_type].append(entity)

            # Update stats
            self.stats['total_entities'] += 1
            self.stats['by_type'][entity_type] += 1

        # Sort by frequency for substitution candidates
        for entity_type in self.ENTITY_TYPES:
            self._by_type[entity_type].sort(key=lambda e: e.instances, reverse=True)

    def _strip_determinatives(self, lemma: str) -> str:
        """Strip determinative markers like {d}, {m}, {ki}."""
        return self.DETERMINATIVE_PATTERN.sub('', lemma)

    def _strip_subscripts(self, lemma: str) -> str:
        """Remove numeric subscripts like en-lil2 -> en-lil."""
        return self.SUBSCRIPT_PATTERN.sub('', lemma)

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Sumerian special characters."""
        result = text
        for char, replacement in self.UNICODE_MAP.items():
            result = result.replace(char, replacement)
        return result

    def _normalize_form(self, lemma: str) -> str:
        """Fully normalize a lemma for fuzzy matching."""
        normalized = lemma.lower()
        normalized = self._strip_determinatives(normalized)
        normalized = self._strip_subscripts(normalized)
        normalized = self._normalize_unicode(normalized)
        normalized = normalized.replace('-', '')  # Remove hyphens
        return normalized.strip()

    def lookup(self, lemma: str) -> Optional[Entity]:
        """
        Look up entity by lemma with exact match only.

        Args:
            lemma: Sumerian lemma to look up

        Returns:
            Entity if found, None otherwise
        """
        lemma_lower = lemma.lower()

        # Check manual aliases first
        if lemma_lower in self.MANUAL_ALIASES:
            canonical = self.MANUAL_ALIASES[lemma_lower]
            if result := self._exact_index.get(canonical.lower()):
                return result

        return self._exact_index.get(lemma_lower)

    def fuzzy_lookup(self, lemma: str) -> Optional[Entity]:
        """
        Look up entity with fuzzy matching fallback chain.

        Fallback order:
        0. Check manual aliases (philological mismatches)
        1. Exact match
        2. Strip determinatives: {d}en-lil2 -> en-lil2
        3. Remove subscripts: en-lil2 -> en-lil
        4. Full normalization (unicode, hyphens)

        Args:
            lemma: Sumerian lemma to look up

        Returns:
            Entity if found, None otherwise
        """
        lemma_lower = lemma.lower()

        # 0. Check manual aliases first (handles philological mismatches)
        if lemma_lower in self.MANUAL_ALIASES:
            canonical = self.MANUAL_ALIASES[lemma_lower]
            if result := self._exact_index.get(canonical.lower()):
                self.stats['lookup_exact'] += 1
                return result

        # 1. Try exact match
        if result := self._exact_index.get(lemma_lower):
            self.stats['lookup_exact'] += 1
            return result

        # 2. Strip determinatives
        stripped = self._strip_determinatives(lemma_lower)
        if stripped != lemma_lower:
            if result := self._exact_index.get(stripped):
                self.stats['lookup_stripped'] += 1
                return result

        # 3. Remove subscripts
        no_subscript = self._strip_subscripts(stripped)
        if no_subscript != stripped:
            if result := self._exact_index.get(no_subscript):
                self.stats['lookup_stripped'] += 1
                return result

        # 4. Full normalization
        normalized = self._normalize_form(lemma)
        if result := self._normalized_index.get(normalized):
            self.stats['lookup_normalized'] += 1
            return result

        self.stats['lookup_failed'] += 1
        return None

    def get_type(self, lemma: str) -> Optional[str]:
        """Get entity type (DN, RN, GN) for a lemma."""
        entity = self.fuzzy_lookup(lemma)
        return entity.entity_type if entity else None

    def get_label(self, lemma: str) -> Optional[str]:
        """Get English label for a lemma."""
        entity = self.fuzzy_lookup(lemma)
        return entity.english_label if entity else None

    def get_substitution_candidates(
        self,
        entity_type: str,
        min_freq: int = 2,
        exclude: Optional[Set[str]] = None
    ) -> List[Entity]:
        """
        Get entities of a given type suitable for substitution.

        Args:
            entity_type: DN, RN, or GN
            min_freq: Minimum corpus frequency (default 2)
            exclude: Set of citation forms to exclude

        Returns:
            List of Entity objects sorted by frequency
        """
        if entity_type not in self.ENTITY_TYPES:
            return []

        exclude = exclude or set()

        return [
            e for e in self._by_type[entity_type]
            if e.instances >= min_freq and e.citation_form not in exclude
        ]

    def extract_entities_from_tokens(self, tokens: List[dict]) -> List[dict]:
        """
        Extract entities from ETCSL-style token list.

        ETCSL tokens already have 'type' and 'label' fields for entities.

        Args:
            tokens: List of token dicts from ETCSL parquet

        Returns:
            List of entity dicts with form, lemma, type, label, position
        """
        entities = []
        for i, token in enumerate(tokens):
            if not isinstance(token, dict):
                continue

            entity_type = token.get('type')
            if entity_type in self.ENTITY_TYPES:
                entities.append({
                    'form': token.get('form', ''),
                    'form_normalized': token.get('form_normalized', ''),
                    'lemma': token.get('lemma', ''),
                    'type': entity_type,
                    'label': token.get('label', ''),
                    'position': i,
                })

        return entities

    def link_entities_in_text(self, text_normalized: str) -> List[dict]:
        """
        Find and link entities in normalized Sumerian text using glossary.

        Used for ORACC data which lacks entity annotations.

        Args:
            text_normalized: Space-separated normalized Sumerian text

        Returns:
            List of entity dicts with form, type, label, position
        """
        entities = []
        words = text_normalized.split()

        for i, word in enumerate(words):
            entity = self.fuzzy_lookup(word)
            if entity:
                entities.append({
                    'form': word,
                    'lemma': word,
                    'type': entity.entity_type,
                    'label': entity.english_label,
                    'position': i,
                })

        return entities

    def get_statistics(self) -> dict:
        """Get linker statistics."""
        return {
            **self.stats,
            'lookup_success_rate': (
                (self.stats['lookup_exact'] + self.stats['lookup_stripped'] + self.stats['lookup_normalized']) /
                max(1, sum([
                    self.stats['lookup_exact'],
                    self.stats['lookup_stripped'],
                    self.stats['lookup_normalized'],
                    self.stats['lookup_failed']
                ]))
            )
        }

    def __repr__(self) -> str:
        return (
            f"EntityLinker(entities={self.stats['total_entities']}, "
            f"DN={self.stats['by_type']['DN']}, "
            f"RN={self.stats['by_type']['RN']}, "
            f"GN={self.stats['by_type']['GN']})"
        )


def main():
    """Test the entity linker."""
    from config import Paths

    print("=" * 60)
    print("Entity Linker Test")
    print("=" * 60)

    linker = EntityLinker()
    print(f"\nLoaded: {linker}")

    # Test lookups
    test_cases = [
        "Enlil",      # Divine name
        "en-lil2",    # With subscript
        "{d}en-lil2", # With determinative
        "Nippur",     # Geographic name
        "nibru",      # Sumerian form of Nippur
        "Akkad",      # Geographic name
        "Šulgi",      # Royal name (with special char)
        "unknown",    # Should fail
    ]

    print("\nLookup tests:")
    print("-" * 40)
    for lemma in test_cases:
        entity = linker.fuzzy_lookup(lemma)
        if entity:
            print(f"  {lemma:20} -> {entity.entity_type}: {entity.english_label} (freq={entity.instances})")
        else:
            print(f"  {lemma:20} -> NOT FOUND")

    # Test substitution candidates
    print("\nSubstitution candidates (DN, min_freq=5):")
    print("-" * 40)
    candidates = linker.get_substitution_candidates('DN', min_freq=5)[:10]
    for entity in candidates:
        print(f"  {entity.citation_form}: {entity.instances} occurrences")

    # Statistics
    print("\nStatistics:")
    print("-" * 40)
    stats = linker.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
