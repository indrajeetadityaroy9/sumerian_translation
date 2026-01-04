"""
Graph-Based Entity Substitution Augmentation

THE KEY TECHNICAL CONTRIBUTION: Two-Circle approach for low-resource NMT augmentation.

This module implements type-safe entity substitution using bipartite graphs to
augment limited parallel corpora while maintaining linguistic constraints.

Components:
- entity_graph: Bipartite graph connecting lines to named entities (DN/RN/GN)
- structural_matcher: Skeleton similarity matching to prevent Bag-of-Entities fallacy
- substitution: Type-safe entity swap with word boundary safety
- constraints: DN<->DN, RN<->RN, GN<->GN type enforcement
- entity_linker: Glossary-based entity resolution for ORACC
- pipeline: Main orchestration (GraphAugmentor class)

Two-Circle Approach:
- Circle 1: ETCSL <-> ETCSL (swap entities between compositions in same corpus)
- Circle 2: ORACC -> ETCSL (link monolingual ORACC texts via glossary)

References:
- See Section 4.2 of paper for skeleton similarity threshold validation
- See Section 3.1 for entity type constraint rationale
"""

from .constraints import TypeConstraints
from .entity_graph import EntityGraph, LineNode, build_combined_graph_parallel
from .structural_matcher import StructuralMatcher, LineMatch
from .substitution import EntitySubstitutor, AugmentedPair
from .entity_linker import EntityLinker
from .pipeline import GraphAugmentor

__all__ = [
    # Core classes
    "TypeConstraints",
    "EntityGraph",
    "LineNode",
    "StructuralMatcher",
    "LineMatch",
    "EntitySubstitutor",
    "AugmentedPair",
    "EntityLinker",
    "GraphAugmentor",
    # Factory functions
    "build_combined_graph_parallel",
]
