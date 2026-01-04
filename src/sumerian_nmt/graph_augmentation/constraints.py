"""
Type Constraints for Entity Substitution

Enforces type constraints to prevent invalid substitutions:
- DN (Divine Name) can only swap with DN
- RN (Royal Name) can only swap with RN
- GN (Geographic Name) can only swap with GN

Also handles blacklisted entities and minimum frequency requirements.

This module is part of the novel graph-based entity substitution system.
See Section 3.1 of the paper for constraint rationale.
"""

from dataclasses import dataclass
from typing import Set, Optional


@dataclass
class SubstitutionCandidate:
    """A candidate entity for substitution."""
    lemma: str
    entity_type: str
    english_label: str
    frequency: int


class TypeConstraints:
    """
    Validates entity substitutions for type safety.

    Key constraints:
    1. Same-type only: DN↔DN, RN↔RN, GN↔GN
    2. Minimum frequency to avoid rare/uncertain entities
    3. Blacklist for context-sensitive words (an, ki)
    """

    VALID_TYPES = frozenset({'DN', 'RN', 'GN'})

    # Context-sensitive words that should not be substituted
    # These have meanings beyond proper nouns:
    # - 'an' = sky, heaven (common word)
    # - 'ki' = earth, place (common word)
    DEFAULT_BLACKLIST = frozenset({'an', 'ki'})

    # Minimum corpus frequency for substitution candidates
    DEFAULT_MIN_FREQUENCY = 2

    def __init__(
        self,
        min_frequency: int = DEFAULT_MIN_FREQUENCY,
        blacklist: Optional[Set[str]] = None
    ):
        """
        Initialize safety checker.

        Args:
            min_frequency: Minimum corpus frequency for substitution
            blacklist: Set of lemmas to never substitute (case-insensitive)
        """
        self.min_frequency = min_frequency
        self.blacklist = {b.lower() for b in (blacklist or self.DEFAULT_BLACKLIST)}

        # Statistics
        self.stats = {
            'checks_total': 0,
            'checks_passed': 0,
            'rejected_type_mismatch': 0,
            'rejected_low_frequency': 0,
            'rejected_blacklisted': 0,
            'rejected_same_entity': 0,
        }

    def is_valid_type(self, entity_type: str) -> bool:
        """Check if entity type is valid for substitution."""
        return entity_type in self.VALID_TYPES

    def is_blacklisted(self, lemma: str) -> bool:
        """Check if lemma is in blacklist."""
        return lemma.lower() in self.blacklist

    def is_safe_swap(
        self,
        source_type: str,
        source_lemma: str,
        target_type: str,
        target_lemma: str,
        target_frequency: int = 1
    ) -> bool:
        """
        Check if swapping source entity with target is safe.

        Args:
            source_type: Entity type of source (DN, RN, GN)
            source_lemma: Lemma being replaced
            target_type: Entity type of target
            target_lemma: Lemma to substitute in
            target_frequency: Corpus frequency of target entity

        Returns:
            True if swap is safe, False otherwise
        """
        self.stats['checks_total'] += 1

        # 1. Type must match exactly
        if source_type != target_type:
            self.stats['rejected_type_mismatch'] += 1
            return False

        # 2. Type must be valid
        if not self.is_valid_type(source_type):
            self.stats['rejected_type_mismatch'] += 1
            return False

        # 3. Cannot swap entity with itself
        if source_lemma.lower() == target_lemma.lower():
            self.stats['rejected_same_entity'] += 1
            return False

        # 4. Target must meet frequency threshold
        if target_frequency < self.min_frequency:
            self.stats['rejected_low_frequency'] += 1
            return False

        # 5. Neither entity should be blacklisted
        if self.is_blacklisted(source_lemma) or self.is_blacklisted(target_lemma):
            self.stats['rejected_blacklisted'] += 1
            return False

        self.stats['checks_passed'] += 1
        return True

    def filter_candidates(
        self,
        entity_type: str,
        candidates: list,
        exclude_lemma: Optional[str] = None
    ) -> list:
        """
        Filter substitution candidates by safety criteria.

        Args:
            entity_type: Required entity type (DN, RN, GN)
            candidates: List of Entity objects or dicts
            exclude_lemma: Lemma to exclude (the source being replaced)

        Returns:
            List of safe candidates
        """
        safe = []
        exclude_lower = exclude_lemma.lower() if exclude_lemma else None

        for c in candidates:
            # Handle both Entity objects and dicts
            if hasattr(c, 'entity_type'):
                c_type = c.entity_type
                c_lemma = c.citation_form
                c_freq = c.instances
            else:
                c_type = c.get('type') or c.get('entity_type')
                c_lemma = c.get('lemma') or c.get('citation_form')
                c_freq = c.get('frequency') or c.get('instances', 1)

            # Apply filters
            if c_type != entity_type:
                continue
            if c_lemma.lower() == exclude_lower:
                continue
            if c_freq < self.min_frequency:
                continue
            if self.is_blacklisted(c_lemma):
                continue

            safe.append(c)

        return safe

    def get_statistics(self) -> dict:
        """Get safety check statistics."""
        total = self.stats['checks_total']
        return {
            **self.stats,
            'pass_rate': self.stats['checks_passed'] / max(1, total),
        }

    def add_to_blacklist(self, lemma: str):
        """Add a lemma to the blacklist."""
        self.blacklist.add(lemma.lower())

    def remove_from_blacklist(self, lemma: str):
        """Remove a lemma from the blacklist."""
        self.blacklist.discard(lemma.lower())

    def __repr__(self) -> str:
        return (
            f"TypeConstraints(min_freq={self.min_frequency}, "
            f"blacklist_size={len(self.blacklist)})"
        )


def main():
    """Test type constraints."""
    checker = TypeConstraints(min_frequency=2)
    print(f"Initialized: {checker}")

    # Test cases
    tests = [
        # (source_type, source_lemma, target_type, target_lemma, freq, expected)
        ("DN", "Enlil", "DN", "Inanna", 100, True),   # Valid DN swap
        ("DN", "Enlil", "RN", "Shulgi", 100, False),  # Type mismatch
        ("RN", "Shulgi", "RN", "Shulgi", 100, False), # Same entity
        ("GN", "Nippur", "GN", "Ur", 1, False),       # Low frequency
        ("DN", "An", "DN", "Enlil", 100, False),      # Blacklisted
        ("XX", "foo", "XX", "bar", 100, False),       # Invalid type
    ]

    print("\nSafety checks:")
    print("-" * 60)
    for src_type, src_lemma, tgt_type, tgt_lemma, freq, expected in tests:
        result = checker.is_safe_swap(src_type, src_lemma, tgt_type, tgt_lemma, freq)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {src_lemma}({src_type}) → {tgt_lemma}({tgt_type}): {result}")

    print("\nStatistics:")
    for key, value in checker.get_statistics().items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
