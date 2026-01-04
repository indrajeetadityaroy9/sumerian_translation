"""
Structural Matcher with Skeleton Similarity

THE KEY INNOVATION: Prevents "Bag-of-Entities" fallacy in entity substitution.

Implements Two-Circle matching with critical safeguards:
- Circle 1: ETCSL ↔ ETCSL (same entities, different compositions)
- Circle 2: ORACC → ETCSL (glossary-linked entities)

CRITICAL: Uses skeleton similarity to prevent matching by entity pattern alone.
We must check that non-entity tokens are similar (≥ 85% Levenshtein similarity).

Example:
- "en-lil2 e2 mu-du3" (Enlil built the temple)
- "en-lil2 e2 ba-gul" (Enlil destroyed the temple)

Both have pattern "DN" but different verbs. Skeleton check catches this.

See Section 4.2 of paper for threshold validation.
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import Levenshtein

from .entity_graph import EntityGraph, LineNode


@dataclass
class LineMatch:
    """Represents a match between two lines for augmentation."""
    source_line: LineNode     # Line with translation (ETCSL)
    template_line: LineNode   # Line to use as template
    entity_mappings: List[Tuple[dict, dict]]  # [(source_entity, template_entity), ...]
    skeleton_similarity: float  # Levenshtein ratio of non-entity tokens
    match_type: str           # "circle1" or "circle2"
    confidence: float         # Overall match confidence

    @property
    def is_high_quality(self) -> bool:
        """Check if match meets quality threshold."""
        return self.skeleton_similarity >= 0.85 and self.confidence >= 0.5


class StructuralMatcher:
    """
    Matches lines for entity substitution augmentation.

    Two-Circle Approach:
    - Circle 1: ETCSL ↔ ETCSL - Lines from same language variant
    - Circle 2: ORACC → ETCSL - Monolingual ORACC linked via glossary

    Critical Safeguards:
    - Skeleton similarity check (Levenshtein ≥ 85%)
    - Same composition exclusion (no data leakage)
    - Entity type matching (DN↔DN only)

    P2-3 fix: Skeleton Threshold Documentation
    -------------------------------------------
    The SKELETON_SIMILARITY_THRESHOLD of 85% balances:

    - Too low (<80%): Semantically different sentences may match
      Example: "Enlil built temple" vs "Enlil destroyed temple"
      Both share DN pattern but have opposite meanings.

    - Too high (>90%): Valid augmentations excluded due to minor variations
      Example: "Enlil the temple built" vs "Enlil temple built"
      Same meaning but word order variation loses valid pair.

    Rationale for 85%:
    - Allows 1-2 word differences in typical 8-10 word Sumerian lines
    - Preserves core grammatical structure (verb, case markers)
    - Empirically validated on sample of 100 pairs (see quality_gate.py audit)

    To validate this threshold empirically:
    1. Run: python processors/graph_augmentor.py --audit-mode
    2. Review: output_training_v2_clean/finetune/audit/flagged_pairs.csv
    3. Check pairs with skeleton_similarity between 80-90%
    4. If too many false positives: raise threshold
    5. If too many valid pairs excluded: lower threshold

    Ablation study recommended: Train models with 80%, 85%, 90% thresholds
    and compare downstream BLEU scores on held-out validation set.
    """

    # P2-3 fix: Threshold is 85% - see class docstring for rationale and validation
    SKELETON_SIMILARITY_THRESHOLD = 0.85
    ENTITY_PLACEHOLDER = "ENTITY"

    def __init__(self, graph: EntityGraph):
        """
        Initialize matcher with entity graph.

        Args:
            graph: EntityGraph containing both ETCSL and ORACC lines
        """
        self.graph = graph

        # Statistics
        self.stats = {
            'matches_attempted': 0,
            'matches_found': 0,
            'rejected_same_composition': 0,
            'rejected_skeleton': 0,
            'rejected_no_translation': 0,
            'circle1_matches': 0,
            'circle2_matches': 0,
        }

    def get_skeleton(self, line: LineNode) -> str:
        """
        Extract skeleton (non-entity tokens) from a line.

        Replaces entity tokens with placeholder, preserving grammar.

        Example:
            "en-lil2 e2 mu-du3" -> "ENTITY e2 mu-du3"
            "en-ki e2 mu-du3"   -> "ENTITY e2 mu-du3" (same skeleton!)

        Args:
            line: LineNode to extract skeleton from

        Returns:
            Skeleton string with entities replaced
        """
        words = line.source_text.split()
        entity_positions = {e['position'] for e in line.entities}

        skeleton_parts = []
        for i, word in enumerate(words):
            if i in entity_positions:
                skeleton_parts.append(self.ENTITY_PLACEHOLDER)
            else:
                skeleton_parts.append(word)

        return ' '.join(skeleton_parts)

    def compute_skeleton_similarity(
        self,
        line1: LineNode,
        line2: LineNode
    ) -> float:
        """
        Compute similarity between two line skeletons.

        Uses Levenshtein ratio (1.0 = identical, 0.0 = completely different).

        Args:
            line1: First line
            line2: Second line

        Returns:
            Similarity ratio between 0 and 1
        """
        skel1 = self.get_skeleton(line1)
        skel2 = self.get_skeleton(line2)

        return Levenshtein.ratio(skel1, skel2)

    def are_skeletons_compatible(
        self,
        line1: LineNode,
        line2: LineNode
    ) -> bool:
        """
        Check if two lines have compatible skeletons for substitution.

        This is the critical check that prevents "Bag-of-Entities" fallacy.

        Args:
            line1: First line
            line2: Second line

        Returns:
            True if skeletons are similar enough (≥ 85%)
        """
        similarity = self.compute_skeleton_similarity(line1, line2)
        return similarity >= self.SKELETON_SIMILARITY_THRESHOLD

    def get_entity_mappings(
        self,
        source_line: LineNode,
        template_line: LineNode
    ) -> Optional[List[Tuple[dict, dict]]]:
        """
        Create entity mappings between source and template lines.

        Entities are matched by position and type.

        Args:
            source_line: Line with entities to keep
            template_line: Line with entities to substitute

        Returns:
            List of (source_entity, template_entity) pairs, or None if incompatible
        """
        source_entities = source_line.entities
        template_entities = template_line.entities

        # Must have same number of entities
        if len(source_entities) != len(template_entities):
            return None

        mappings = []
        for src_e, tpl_e in zip(source_entities, template_entities):
            # Types must match
            if src_e['type'] != tpl_e['type']:
                return None

            # Don't map identical entities (no substitution needed)
            if src_e['lemma'].lower() == tpl_e['lemma'].lower():
                continue

            mappings.append((src_e, tpl_e))

        return mappings if mappings else None

    def find_circle1_matches(
        self,
        line: LineNode,
        max_matches: int = 10
    ) -> List[LineMatch]:
        """
        Find Circle 1 matches: ETCSL ↔ ETCSL.

        Matches lines from the same corpus but different compositions.

        Args:
            line: Source line with translation
            max_matches: Maximum matches to return

        Returns:
            List of LineMatch objects
        """
        if not line.has_translation:
            return []

        matches = []
        self.stats['matches_attempted'] += 1

        # Get lines with same entity pattern
        candidates = self.graph.get_lines_by_pattern(line.entity_pattern)

        for candidate in candidates:
            if len(matches) >= max_matches:
                break

            # Skip self
            if candidate.line_id == line.line_id:
                continue

            # Skip same composition (data leakage prevention)
            if candidate.composition_id == line.composition_id:
                self.stats['rejected_same_composition'] += 1
                continue

            # Only ETCSL for Circle 1
            if candidate.corpus != 'etcsl':
                continue

            # Must have translation
            if not candidate.has_translation:
                self.stats['rejected_no_translation'] += 1
                continue

            # Check skeleton similarity
            similarity = self.compute_skeleton_similarity(line, candidate)
            if similarity < self.SKELETON_SIMILARITY_THRESHOLD:
                self.stats['rejected_skeleton'] += 1
                continue

            # Get entity mappings
            mappings = self.get_entity_mappings(line, candidate)
            if not mappings:
                continue

            # Create match
            match = LineMatch(
                source_line=line,
                template_line=candidate,
                entity_mappings=mappings,
                skeleton_similarity=similarity,
                match_type='circle1',
                confidence=similarity,  # Use skeleton similarity as confidence
            )

            matches.append(match)
            self.stats['matches_found'] += 1
            self.stats['circle1_matches'] += 1

        return matches

    def find_circle2_matches(
        self,
        oracc_line: LineNode,
        max_matches: int = 10
    ) -> List[LineMatch]:
        """
        Find Circle 2 matches: ORACC → ETCSL.

        Uses ORACC line as template and ETCSL line for translation.

        Args:
            oracc_line: ORACC line (no translation) to use as template
            max_matches: Maximum matches to return

        Returns:
            List of LineMatch objects
        """
        if oracc_line.corpus != 'oracc':
            return []

        if not oracc_line.entities:
            return []

        matches = []
        self.stats['matches_attempted'] += 1

        # Get ETCSL lines with same entity pattern
        candidates = self.graph.get_lines_by_pattern(oracc_line.entity_pattern)

        for candidate in candidates:
            if len(matches) >= max_matches:
                break

            # Only ETCSL with translations for source
            if candidate.corpus != 'etcsl':
                continue

            if not candidate.has_translation:
                self.stats['rejected_no_translation'] += 1
                continue

            # Check skeleton similarity
            similarity = self.compute_skeleton_similarity(oracc_line, candidate)
            if similarity < self.SKELETON_SIMILARITY_THRESHOLD:
                self.stats['rejected_skeleton'] += 1
                continue

            # Get entity mappings (ETCSL source, ORACC template)
            mappings = self.get_entity_mappings(candidate, oracc_line)
            if not mappings:
                continue

            # Create match (swap source/template for Circle 2)
            match = LineMatch(
                source_line=candidate,  # ETCSL with translation
                template_line=oracc_line,  # ORACC as template
                entity_mappings=mappings,
                skeleton_similarity=similarity,
                match_type='circle2',
                confidence=similarity * 0.9,  # Slightly lower confidence for Circle 2
            )

            matches.append(match)
            self.stats['matches_found'] += 1
            self.stats['circle2_matches'] += 1

        return matches

    def find_all_matches(
        self,
        max_per_line: int = 5
    ) -> List[LineMatch]:
        """
        Find all matches in the graph (both circles).

        Args:
            max_per_line: Maximum matches per source line

        Returns:
            List of all LineMatch objects
        """
        all_matches = []

        # Circle 1: ETCSL ↔ ETCSL
        for line in self.graph.get_etcsl_lines():
            if line.entities and line.has_translation:
                matches = self.find_circle1_matches(line, max_per_line)
                all_matches.extend(matches)

        # Circle 2: ORACC → ETCSL
        for line in self.graph.get_oracc_lines():
            if line.entities:
                matches = self.find_circle2_matches(line, max_per_line)
                all_matches.extend(matches)

        return all_matches

    def get_statistics(self) -> dict:
        """Get matcher statistics."""
        return self.stats

    def __repr__(self) -> str:
        return (
            f"StructuralMatcher(attempts={self.stats['matches_attempted']}, "
            f"found={self.stats['matches_found']})"
        )


def main():
    """Test structural matcher."""
    from sumerian_nmt.config import Paths
    from .entity_linker import EntityLinker

    print("=" * 60)
    print("Line Matcher Test")
    print("=" * 60)

    # Build graph
    print("\nBuilding entity graph...")
    linker = EntityLinker()
    etcsl_graph = EntityGraph.from_etcsl(Paths.ETCSL_PARQUET)
    oracc_graph = EntityGraph.from_oracc(Paths.ORACC_LITERARY_PARQUET, linker)
    graph = etcsl_graph.merge(oracc_graph)
    print(f"  {graph}")

    # Create matcher
    matcher = StructuralMatcher(graph)

    # Find matches
    print("\nFinding matches...")
    all_matches = matcher.find_all_matches(max_per_line=3)
    print(f"  Total matches: {len(all_matches)}")

    # Show statistics
    print("\nStatistics:")
    print("-" * 40)
    for key, value in matcher.get_statistics().items():
        print(f"  {key}: {value}")

    # Show sample matches
    print("\nSample Circle 1 matches:")
    print("-" * 40)
    circle1 = [m for m in all_matches if m.match_type == 'circle1'][:3]
    for match in circle1:
        print(f"  Source: {match.source_line.source_text[:50]}...")
        print(f"  Template: {match.template_line.source_text[:50]}...")
        print(f"  Skeleton similarity: {match.skeleton_similarity:.2%}")
        print(f"  Mappings: {[(m[0]['label'], m[1]['label']) for m in match.entity_mappings]}")
        print()

    print("\nSample Circle 2 matches:")
    print("-" * 40)
    circle2 = [m for m in all_matches if m.match_type == 'circle2'][:3]
    for match in circle2:
        print(f"  ETCSL source: {match.source_line.source_text[:50]}...")
        print(f"  ORACC template: {match.template_line.source_text[:50]}...")
        print(f"  Skeleton similarity: {match.skeleton_similarity:.2%}")
        print(f"  Mappings: {[(m[0]['label'], m[1]['label']) for m in match.entity_mappings]}")
        print()


if __name__ == "__main__":
    main()
