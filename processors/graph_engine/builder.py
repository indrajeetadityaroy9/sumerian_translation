"""
Graph Builder for Entity-Line Relationships

Constructs a NetworkX bipartite graph connecting:
- Line nodes: Individual text lines with source/target text
- Entity nodes: Named entities (DN, RN, GN) appearing in lines

This enables efficient lookup of:
- All lines containing a specific entity
- All entities in a specific line
- Lines sharing entity patterns for augmentation

Optimized for parallel loading on 52 vCPU systems.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

from processors.entity_linker import EntityLinker


@dataclass
class LineNode:
    """Represents a line in the corpus."""
    line_id: str           # Unique identifier
    source_text: str       # Normalized Sumerian text
    target_text: str       # English translation (empty for ORACC)
    tokens: List[dict]     # Token-level data
    entities: List[dict]   # Extracted entities with positions
    entity_pattern: str    # Pattern like "DN-GN-DN"
    corpus: str            # "etcsl" or "oracc"
    composition_id: str    # Parent composition/text ID

    @property
    def has_translation(self) -> bool:
        """Check if line has English translation."""
        return bool(self.target_text and self.target_text.strip())

    @property
    def entity_types(self) -> List[str]:
        """Get list of entity types in order."""
        return [e['type'] for e in self.entities]

    def __hash__(self):
        return hash(self.line_id)


@dataclass
class EntityNode:
    """Represents an entity in the graph."""
    lemma: str
    entity_type: str  # DN, RN, GN
    label: str        # English label
    line_ids: Set[str] = field(default_factory=set)

    def __hash__(self):
        return hash((self.lemma, self.entity_type))


class EntityGraph:
    """
    Bipartite graph connecting lines to entities.

    Enables efficient two-circle matching:
    - Circle 1: ETCSL ↔ ETCSL (same entities, different compositions)
    - Circle 2: ORACC → ETCSL (glossary-linked entities)
    """

    def __init__(self):
        self.graph = nx.Graph()

        # Indexes for fast lookup
        self._lines: Dict[str, LineNode] = {}
        self._entities: Dict[Tuple[str, str], EntityNode] = {}  # (lemma, type) -> EntityNode
        self._pattern_index: Dict[str, List[str]] = {}  # pattern -> [line_ids]
        self._by_corpus: Dict[str, List[str]] = {'etcsl': [], 'oracc': []}

        # Statistics
        self.stats = {
            'lines_total': 0,
            'lines_with_entities': 0,
            'lines_etcsl': 0,
            'lines_oracc': 0,
            'entities_unique': 0,
            'edges': 0,
        }

    @classmethod
    def from_etcsl(cls, parquet_path: Path) -> 'EntityGraph':
        """
        Build graph from ETCSL gold parquet.

        ETCSL tokens already have entity annotations (type, label fields).
        """
        graph = cls()

        df = pd.read_parquet(parquet_path)

        for idx, row in df.iterrows():
            # Parse tokens
            tokens = row.get('tokens', [])
            if isinstance(tokens, str):
                tokens = json.loads(tokens)

            # Extract entities from tokens
            entities = []
            for i, token in enumerate(tokens):
                if isinstance(token, dict) and token.get('type') in {'DN', 'RN', 'GN'}:
                    entities.append({
                        'form': token.get('form', ''),
                        'lemma': token.get('lemma', ''),
                        'type': token['type'],
                        'label': token.get('label', ''),
                        'position': i,
                    })

            # Create line node
            line_id = f"etcsl_{row.get('composition_id', 'unk')}_{idx}"
            entity_pattern = '-'.join(e['type'] for e in entities) or 'NONE'

            line_node = LineNode(
                line_id=line_id,
                source_text=row.get('text_normalized', ''),
                target_text=row.get('target_text', ''),
                tokens=tokens,
                entities=entities,
                entity_pattern=entity_pattern,
                corpus='etcsl',
                composition_id=str(row.get('composition_id', '')),
            )

            graph.add_line(line_node)

        return graph

    @classmethod
    def from_oracc(
        cls,
        parquet_path: Path,
        linker: EntityLinker,
        corpus_name: str = 'oracc'
    ) -> 'EntityGraph':
        """
        Build graph from ORACC parquet using glossary linking.

        ORACC data lacks entity annotations, so we use the EntityLinker
        to identify entities from the normalized text.
        """
        graph = cls()

        df = pd.read_parquet(parquet_path)

        for idx, row in df.iterrows():
            text = row.get('text_normalized', '')
            if not text:
                continue

            # Link entities using glossary
            entities = linker.link_entities_in_text(text)

            # Create line node (no translation for ORACC)
            text_id = row.get('text_id', 'unk')
            line_num = row.get('line_num', idx)
            line_id = f"{corpus_name}_{text_id}_{line_num}"
            entity_pattern = '-'.join(e['type'] for e in entities) or 'NONE'

            line_node = LineNode(
                line_id=line_id,
                source_text=text,
                target_text='',  # ORACC has no gold translations
                tokens=[],  # ORACC parquet doesn't store tokens
                entities=entities,
                entity_pattern=entity_pattern,
                corpus='oracc',
                composition_id=str(text_id),
            )

            graph.add_line(line_node)

        return graph

    def add_line(self, line: LineNode):
        """Add a line node and connect to entity nodes."""
        self._lines[line.line_id] = line
        self._by_corpus[line.corpus].append(line.line_id)

        # Add to pattern index
        if line.entity_pattern not in self._pattern_index:
            self._pattern_index[line.entity_pattern] = []
        self._pattern_index[line.entity_pattern].append(line.line_id)

        # Update stats
        self.stats['lines_total'] += 1
        if line.corpus == 'etcsl':
            self.stats['lines_etcsl'] += 1
        else:
            self.stats['lines_oracc'] += 1

        if line.entities:
            self.stats['lines_with_entities'] += 1

        # Add line node to graph
        self.graph.add_node(line.line_id, type='line', data=line)

        # Connect to entity nodes
        for entity in line.entities:
            entity_key = (entity['lemma'].lower(), entity['type'])

            # Create entity node if needed
            if entity_key not in self._entities:
                entity_node = EntityNode(
                    lemma=entity['lemma'],
                    entity_type=entity['type'],
                    label=entity.get('label', entity['lemma']),
                )
                self._entities[entity_key] = entity_node
                self.graph.add_node(
                    f"entity_{entity['lemma']}_{entity['type']}",
                    type='entity',
                    data=entity_node
                )
                self.stats['entities_unique'] += 1

            # Add edge
            entity_node = self._entities[entity_key]
            entity_node.line_ids.add(line.line_id)

            entity_graph_id = f"entity_{entity['lemma']}_{entity['type']}"
            if not self.graph.has_edge(line.line_id, entity_graph_id):
                self.graph.add_edge(line.line_id, entity_graph_id)
                self.stats['edges'] += 1

    def merge(self, other: 'EntityGraph') -> 'EntityGraph':
        """Merge another graph into this one."""
        for line_id, line in other._lines.items():
            if line_id not in self._lines:
                self.add_line(line)
        return self

    def get_line(self, line_id: str) -> Optional[LineNode]:
        """Get line by ID."""
        return self._lines.get(line_id)

    def get_lines_by_pattern(self, pattern: str) -> List[LineNode]:
        """Get all lines matching an entity pattern."""
        line_ids = self._pattern_index.get(pattern, [])
        return [self._lines[lid] for lid in line_ids]

    def get_lines_with_entity(
        self,
        lemma: str,
        entity_type: str
    ) -> List[LineNode]:
        """Get all lines containing a specific entity."""
        entity_key = (lemma.lower(), entity_type)
        entity_node = self._entities.get(entity_key)

        if not entity_node:
            return []

        return [self._lines[lid] for lid in entity_node.line_ids]

    def get_etcsl_lines(self) -> List[LineNode]:
        """Get all ETCSL lines."""
        return [self._lines[lid] for lid in self._by_corpus['etcsl']]

    def get_oracc_lines(self) -> List[LineNode]:
        """Get all ORACC lines."""
        return [self._lines[lid] for lid in self._by_corpus['oracc']]

    def get_lines_with_entities(self) -> List[LineNode]:
        """Get lines that have at least one entity."""
        return [line for line in self._lines.values() if line.entities]

    def get_statistics(self) -> dict:
        """Get graph statistics."""
        return {
            **self.stats,
            'patterns_unique': len(self._pattern_index),
        }

    def __repr__(self) -> str:
        return (
            f"EntityGraph(lines={self.stats['lines_total']}, "
            f"entities={self.stats['entities_unique']}, "
            f"edges={self.stats['edges']})"
        )


def build_combined_graph_parallel(
    etcsl_path: Path,
    oracc_literary_path: Path,
    oracc_royal_path: Path,
    linker: EntityLinker,
    verbose: bool = True
) -> EntityGraph:
    """
    Build combined entity graph with parallel loading.

    Loads ETCSL, ORACC literary, and ORACC royal corpora in parallel
    using ThreadPoolExecutor for 2-3x speedup on initialization.

    Args:
        etcsl_path: Path to ETCSL parquet
        oracc_literary_path: Path to ORACC literary parquet
        oracc_royal_path: Path to ORACC royal parquet
        linker: EntityLinker for ORACC entity resolution
        verbose: Print progress

    Returns:
        Merged EntityGraph containing all three corpora
    """
    if verbose:
        print("Building combined graph with parallel loading...")

    # Define loading tasks
    def load_etcsl():
        if verbose:
            print("  Loading ETCSL...")
        graph = EntityGraph.from_etcsl(etcsl_path)
        if verbose:
            print(f"    ETCSL: {graph.stats['lines_total']} lines")
        return graph

    def load_oracc_literary():
        if not oracc_literary_path.exists():
            return None
        if verbose:
            print("  Loading ORACC literary...")
        graph = EntityGraph.from_oracc(oracc_literary_path, linker, 'oracc_literary')
        if verbose:
            print(f"    ORACC literary: {graph.stats['lines_total']} lines")
        return graph

    def load_oracc_royal():
        if not oracc_royal_path.exists():
            return None
        if verbose:
            print("  Loading ORACC royal...")
        graph = EntityGraph.from_oracc(oracc_royal_path, linker, 'oracc_royal')
        if verbose:
            print(f"    ORACC royal: {graph.stats['lines_total']} lines")
        return graph

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        etcsl_future = executor.submit(load_etcsl)
        oracc_lit_future = executor.submit(load_oracc_literary)
        oracc_royal_future = executor.submit(load_oracc_royal)

        # Wait for all to complete
        etcsl_graph = etcsl_future.result()
        oracc_lit_graph = oracc_lit_future.result()
        oracc_royal_graph = oracc_royal_future.result()

    # Merge graphs
    if verbose:
        print("  Merging graphs...")

    if oracc_lit_graph:
        etcsl_graph.merge(oracc_lit_graph)
    if oracc_royal_graph:
        etcsl_graph.merge(oracc_royal_graph)

    if verbose:
        print(f"  Combined: {etcsl_graph}")

    return etcsl_graph


def main():
    """Test graph builder."""
    import sys
    sys.path.insert(0, '.')
    from config import Paths

    print("=" * 60)
    print("Entity Graph Builder Test")
    print("=" * 60)

    # Build from ETCSL
    print("\nLoading ETCSL...")
    etcsl_graph = EntityGraph.from_etcsl(Paths.ETCSL_PARQUET)
    print(f"  {etcsl_graph}")

    # Build from ORACC with linking
    print("\nLoading entity linker...")
    linker = EntityLinker()
    print(f"  {linker}")

    print("\nLoading ORACC literary...")
    oracc_graph = EntityGraph.from_oracc(
        Paths.ORACC_LITERARY_PARQUET,
        linker,
        corpus_name='oracc_literary'
    )
    print(f"  {oracc_graph}")

    # Merge graphs
    print("\nMerging graphs...")
    combined = etcsl_graph.merge(oracc_graph)
    print(f"  {combined}")

    # Show statistics
    print("\nStatistics:")
    print("-" * 40)
    for key, value in combined.get_statistics().items():
        print(f"  {key}: {value}")

    # Sample patterns
    print("\nTop entity patterns:")
    print("-" * 40)
    pattern_counts = [(p, len(lids)) for p, lids in combined._pattern_index.items()]
    pattern_counts.sort(key=lambda x: x[1], reverse=True)
    for pattern, count in pattern_counts[:10]:
        print(f"  {pattern}: {count} lines")

    # Sample lines with entities
    print("\nSample ETCSL lines with entities:")
    print("-" * 40)
    etcsl_with_entities = [l for l in combined.get_etcsl_lines() if l.entities][:3]
    for line in etcsl_with_entities:
        print(f"  ID: {line.line_id}")
        print(f"  Source: {line.source_text[:60]}...")
        print(f"  Target: {line.target_text[:60] if line.target_text else 'N/A'}...")
        print(f"  Entities: {[e['label'] for e in line.entities]}")
        print()


if __name__ == "__main__":
    main()
