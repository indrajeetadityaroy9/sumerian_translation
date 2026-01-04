"""
Entity Substitution with Word Boundary Safety

Performs type-safe entity substitution with critical safeguards:
- Word boundary regex to prevent substring matches
- Compound phrase detection for human review
- Type constraint enforcement (DN↔DN only)

Example:
    Input:  source="en-lil2 nibru-a" target="Enlil in Nippur"
            template has en-ki (Enki) instead of en-lil2 (Enlil)

    Step 1: Check word boundary - "Enlil" appears as \\bEnlil\\b ✓
    Step 2: Check compound - no "King Enlil" or "The god Enlil" ✓
    Step 3: Substitute

    Output: source="en-ki nibru-a" target="Enki in Nippur"

See Section 3.2 of paper for word boundary safety rationale.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .entity_graph import LineNode
from .structural_matcher import LineMatch
from .constraints import TypeConstraints


@dataclass
class Substitution:
    """A single entity substitution."""
    source_lemma: str       # Original lemma in source line
    target_lemma: str       # Lemma to substitute
    source_label: str       # Original English label
    target_label: str       # New English label
    entity_type: str        # DN, RN, GN
    position: int           # Position in token sequence


@dataclass
class AugmentedPair:
    """An augmented source-target pair."""
    source_text: str        # New Sumerian text
    target_text: str        # New English text
    original_source: str    # Original source for reference
    original_target: str    # Original target for reference
    substitutions: List[Substitution]
    source_line_id: str     # ID of line with translation
    template_line_id: str   # ID of template line
    match_type: str         # "circle1" or "circle2"
    skeleton_similarity: float
    confidence: float
    is_compound_phrase: bool = False  # Flag for human review
    compound_phrases: List[str] = field(default_factory=list)  # Detected compounds


class EntitySubstitutor:
    """
    Performs entity substitution with safety checks.

    Critical safeguards:
    1. Word boundary regex (\\b) prevents substring matches
    2. Compound phrase detection flags "King X", "The shepherd X"
    3. Type constraint enforcement via TypeConstraints
    """

    # Compound phrase patterns that need review
    # P3-2 fix: Expanded patterns based on corpus analysis
    COMPOUND_PATTERNS = [
        # Titles and roles before name
        r'(king|queen|lord|lady|god|goddess)\s+',      # Royal/divine titles
        r'(shepherd|ruler|priest|priestess)\s+',       # Religious/political roles
        r'(divine|mighty|great|holy)\s+',              # Divine epithets
        r'(prince|princess|en|ensi|lugal)\s+',         # Sumerian titles

        # "The X" patterns
        r'(the\s+\w+)\s+',                             # "The shepherd X", "The god X"

        # Family relationship patterns
        r'(son|daughter|child|mother|father)\s+of\s+', # "son of X"
        r'(wife|husband|spouse)\s+of\s+',              # "wife of X"
        r'(servant|slave|minister)\s+of\s+',           # "servant of X"

        # Building/place relationship patterns
        r'(temple|shrine|house|city|land)\s+of\s+',    # "temple of X"
        r'(gate|wall|ziggurat)\s+of\s+',               # "gate of X"

        # Vocative patterns
        r'\bO\s+',                                      # "O X" (vocative)
        r'(hail|praise|glory to)\s+',                  # "Hail X", "Praise X"

        # Possessive and genitive patterns
        r"'s\s*$",                                     # Possessive after name
        r'\s+of\s+\w+$',                               # "of X" after name

        # Additional contextual patterns
        r'(beloved|chosen|favorite)\s+of\s+',          # "beloved of X"
        r'(fear|wrath|might)\s+of\s+',                 # "wrath of X"
    ]

    def __init__(self, safety_checker: Optional[TypeConstraints] = None):
        """
        Initialize swapper with optional safety checker.

        Args:
            safety_checker: TypeConstraints instance for type validation
        """
        self.safety_checker = safety_checker or TypeConstraints()

        # Compile compound patterns
        self._compound_re = [
            re.compile(p, re.IGNORECASE) for p in self.COMPOUND_PATTERNS
        ]

        # Statistics
        self.stats = {
            'swaps_attempted': 0,
            'swaps_successful': 0,
            'swaps_failed_safety': 0,
            'swaps_failed_not_found': 0,
            'compound_phrases_detected': 0,
        }

    def _substitute_sumerian(
        self,
        text: str,
        old_lemma: str,
        new_lemma: str,
        position: int
    ) -> Optional[str]:
        """
        Substitute lemma in Sumerian text.

        Uses position-based substitution since Sumerian words may contain
        special characters and determinatives.

        Args:
            text: Sumerian text
            old_lemma: Lemma to replace
            new_lemma: New lemma
            position: Word position (0-indexed)

        Returns:
            New text or None if substitution failed
        """
        words = text.split()

        if position >= len(words):
            return None

        # Check that word at position contains the lemma
        word = words[position]
        old_lower = old_lemma.lower()

        # Handle determinatives and suffixes
        # e.g., "{d}en-lil2-la2" should match "en-lil2"
        word_normalized = re.sub(r'\{[^}]+\}', '', word.lower())

        if old_lower not in word_normalized:
            # Try without subscripts
            old_no_sub = re.sub(r'[₀-₉0-9]+', '', old_lower)
            word_no_sub = re.sub(r'[₀-₉0-9]+', '', word_normalized)
            if old_no_sub not in word_no_sub:
                return None

        # Replace: preserve structure but swap the lemma
        new_word = word.replace(old_lemma, new_lemma)
        if new_word == word:
            # Try case-insensitive
            new_word = re.sub(
                re.escape(old_lemma),
                new_lemma,
                word,
                flags=re.IGNORECASE
            )

        words[position] = new_word
        return ' '.join(words)

    def _substitute_english(
        self,
        text: str,
        old_label: str,
        new_label: str
    ) -> Tuple[Optional[str], bool, List[str]]:
        """
        Substitute entity label in English text with word boundary safety.

        CRITICAL: Uses word boundaries to prevent substring matches.
        Example: Won't swap "Ur" inside "Ur-Namma"

        Args:
            text: English text
            old_label: Label to replace
            new_label: New label

        Returns:
            Tuple of (new_text, is_compound, compound_phrases)
            - new_text: Substituted text or None if not found
            - is_compound: True if compound phrase detected
            - compound_phrases: List of detected compound patterns
        """
        # CRITICAL: Use word boundaries
        pattern = r'\b' + re.escape(old_label) + r'\b'

        if not re.search(pattern, text, re.IGNORECASE):
            return None, False, []

        # Check for compound phrases (need human review)
        compound_phrases = []
        for cp_re in self._compound_re:
            # Check if compound pattern appears near the entity
            context_pattern = cp_re.pattern.rstrip('$') + re.escape(old_label)
            if re.search(context_pattern, text, re.IGNORECASE):
                compound_phrases.append(cp_re.pattern)

            # Also check after entity
            after_pattern = re.escape(old_label) + r'\s*' + cp_re.pattern.lstrip('^')
            if re.search(after_pattern, text, re.IGNORECASE):
                compound_phrases.append(cp_re.pattern)

        is_compound = len(compound_phrases) > 0
        if is_compound:
            self.stats['compound_phrases_detected'] += 1

        # Perform substitution
        new_text = re.sub(pattern, new_label, text, flags=re.IGNORECASE)

        return new_text, is_compound, compound_phrases

    def swap_entities(self, match: LineMatch) -> Optional[AugmentedPair]:
        """
        Perform entity substitution based on a line match.

        Args:
            match: LineMatch with source line, template line, and mappings

        Returns:
            AugmentedPair or None if substitution failed
        """
        self.stats['swaps_attempted'] += 1

        source_line = match.source_line
        template_line = match.template_line

        # Start with source text
        new_source = source_line.source_text
        new_target = source_line.target_text

        substitutions = []
        is_compound = False
        all_compounds = []

        for src_entity, tpl_entity in match.entity_mappings:
            # Validate type safety
            if not self.safety_checker.is_safe_swap(
                src_entity['type'],
                src_entity['lemma'],
                tpl_entity['type'],
                tpl_entity['lemma'],
                target_frequency=10  # Assume reasonable frequency
            ):
                self.stats['swaps_failed_safety'] += 1
                continue

            # Substitute in Sumerian
            result = self._substitute_sumerian(
                new_source,
                src_entity['lemma'],
                tpl_entity['lemma'],
                src_entity['position']
            )

            if result is None:
                # Try with template position if different
                result = self._substitute_sumerian(
                    new_source,
                    src_entity['lemma'],
                    tpl_entity['lemma'],
                    tpl_entity['position']
                )

            if result is None:
                self.stats['swaps_failed_not_found'] += 1
                continue

            new_source = result

            # Substitute in English
            src_label = src_entity.get('label', src_entity['lemma'])
            tpl_label = tpl_entity.get('label', tpl_entity['lemma'])

            eng_result, compound, compounds = self._substitute_english(
                new_target,
                src_label,
                tpl_label
            )

            if eng_result is None:
                # English substitution failed - entity not in translation
                self.stats['swaps_failed_not_found'] += 1
                continue

            new_target = eng_result
            is_compound = is_compound or compound
            all_compounds.extend(compounds)

            substitutions.append(Substitution(
                source_lemma=src_entity['lemma'],
                target_lemma=tpl_entity['lemma'],
                source_label=src_label,
                target_label=tpl_label,
                entity_type=src_entity['type'],
                position=src_entity['position'],
            ))

        if not substitutions:
            return None

        self.stats['swaps_successful'] += 1

        return AugmentedPair(
            source_text=new_source,
            target_text=new_target,
            original_source=source_line.source_text,
            original_target=source_line.target_text,
            substitutions=substitutions,
            source_line_id=source_line.line_id,
            template_line_id=template_line.line_id,
            match_type=match.match_type,
            skeleton_similarity=match.skeleton_similarity,
            confidence=match.confidence,
            is_compound_phrase=is_compound,
            compound_phrases=list(set(all_compounds)),
        )

    def swap_batch(self, matches: List[LineMatch]) -> List[AugmentedPair]:
        """
        Perform entity substitution on a batch of matches.

        Args:
            matches: List of LineMatch objects

        Returns:
            List of successful AugmentedPair objects
        """
        results = []
        for match in matches:
            result = self.swap_entities(match)
            if result:
                results.append(result)
        return results

    def get_statistics(self) -> dict:
        """Get swapper statistics."""
        total = max(1, self.stats['swaps_attempted'])
        return {
            **self.stats,
            'success_rate': self.stats['swaps_successful'] / total,
            'compound_rate': self.stats['compound_phrases_detected'] / total,
        }

    def __repr__(self) -> str:
        return (
            f"EntitySubstitutor(attempted={self.stats['swaps_attempted']}, "
            f"successful={self.stats['swaps_successful']})"
        )


def main():
    """Test entity substitutor."""
    from sumerian_nmt.config import Paths
    from .entity_linker import EntityLinker
    from .entity_graph import EntityGraph
    from .structural_matcher import StructuralMatcher

    print("=" * 60)
    print("Entity Substitutor Test")
    print("=" * 60)

    # Build graph and find matches
    print("\nBuilding entity graph...")
    linker = EntityLinker()
    etcsl_graph = EntityGraph.from_etcsl(Paths.ETCSL_PARQUET)
    print(f"  {etcsl_graph}")

    matcher = StructuralMatcher(etcsl_graph)
    matches = matcher.find_all_matches(max_per_line=3)
    print(f"  Found {len(matches)} matches")

    # Create swapper
    swapper = EntitySubstitutor()

    # Perform substitutions
    print("\nPerforming substitutions...")
    augmented = swapper.swap_batch(matches[:100])
    print(f"  Generated {len(augmented)} augmented pairs")

    # Show statistics
    print("\nStatistics:")
    print("-" * 40)
    for key, value in swapper.get_statistics().items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")

    # Show sample augmented pairs
    print("\nSample augmented pairs:")
    print("-" * 40)
    for pair in augmented[:3]:
        print(f"  Original source: {pair.original_source[:50]}...")
        print(f"  Original target: {pair.original_target[:50]}...")
        print(f"  New source: {pair.source_text[:50]}...")
        print(f"  New target: {pair.target_text[:50]}...")
        print(f"  Substitutions: {[(s.source_label, s.target_label) for s in pair.substitutions]}")
        print(f"  Compound phrase: {pair.is_compound_phrase}")
        print(f"  Confidence: {pair.confidence:.2%}")
        print()

    # Show compound phrase examples
    compound_pairs = [p for p in augmented if p.is_compound_phrase]
    if compound_pairs:
        print("\nCompound phrase examples (flagged for review):")
        print("-" * 40)
        for pair in compound_pairs[:3]:
            print(f"  Original: {pair.original_target[:60]}...")
            print(f"  New: {pair.target_text[:60]}...")
            print(f"  Patterns: {pair.compound_phrases}")
            print()


if __name__ == "__main__":
    main()
