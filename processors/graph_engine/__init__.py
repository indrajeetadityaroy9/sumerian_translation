"""
Graph Engine for Sumerian NMT Data Augmentation

Uses NetworkX to build entity-line relationships and generate
new training pairs through type-safe entity substitution.

Components:
- builder: NetworkX bipartite graph construction
- matcher: Two-circle line matching with skeleton similarity
- swapper: Type-safe entity substitution
- safety: DN↔DN, RN↔RN, GN↔GN constraints
"""

from .safety import SafetyChecker
from .builder import EntityGraph, LineNode
from .matcher import LineMatcher, LineMatch
from .swapper import EntitySwapper, AugmentedPair

__all__ = [
    'SafetyChecker',
    'EntityGraph',
    'LineNode',
    'LineMatcher',
    'LineMatch',
    'EntitySwapper',
    'AugmentedPair',
]
