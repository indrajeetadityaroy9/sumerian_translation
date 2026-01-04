"""
Graph-Enhanced Data Augmentation Pipeline

Main pipeline that orchestrates:
1. Entity linking (glossary-based)
2. Graph construction (ETCSL + ORACC)
3. Two-circle matching (with skeleton similarity)
4. Entity substitution (with word boundary safety)
5. Quality gate (audit CSV generation)

Control tokens injected:
- <gold>: Original ETCSL gold data
- <silver>: Exact matches (high confidence)
- <aug>: Entity substitution augmented
- <gloss>: Glossary-based augmented

Optimized for 52 vCPU systems with parallel processing.

Usage:
    python processors/graph_augmentor.py
    python processors/graph_augmentor.py --output train_graph_augmented.jsonl
    python processors/graph_augmentor.py --circle1-only
    python processors/graph_augmentor.py --parallel --workers 48  # Parallel mode
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from sumerian_nmt.utils.io import ChunkedParquetWriter
from sumerian_nmt.utils.parallel import get_optimal_workers, parallel_map

from sumerian_nmt.config import ControlTokens, Paths
from .entity_linker import EntityLinker
from .entity_graph import EntityGraph
from .structural_matcher import StructuralMatcher, LineMatch
from .substitution import EntitySubstitutor, AugmentedPair
from .constraints import TypeConstraints


class GraphAugmentor:
    """
    Main augmentation pipeline using graph-based entity substitution.

    Two-Circle Approach:
    - Circle 1: ETCSL ↔ ETCSL (same corpus, different compositions)
    - Circle 2: ORACC → ETCSL (monolingual linked via glossary)
    """

    SILVER_THRESHOLD = 0.95  # Skeleton similarity for <silver> tag

    def __init__(
        self,
        etcsl_path: Optional[Path] = None,
        oracc_literary_path: Optional[Path] = None,
        oracc_royal_path: Optional[Path] = None,
        glossary_path: Optional[Path] = None,
        min_frequency: int = 2,
    ):
        """
        Initialize augmentor with data paths.

        Args:
            etcsl_path: Path to ETCSL gold parquet
            oracc_literary_path: Path to ORACC literary parquet
            oracc_royal_path: Path to ORACC royal parquet
            glossary_path: Path to glossary parquet
            min_frequency: Minimum entity frequency for substitution
        """
        self.etcsl_path = etcsl_path or Paths.ETCSL_PARQUET
        self.oracc_literary_path = oracc_literary_path or Paths.ORACC_LITERARY_PARQUET
        self.oracc_royal_path = oracc_royal_path or Paths.ORACC_ROYAL_PARQUET
        self.glossary_path = glossary_path or Paths.GLOSSARY_PARQUET

        # Components
        self.linker: Optional[EntityLinker] = None
        self.graph: Optional[EntityGraph] = None
        self.matcher: Optional[StructuralMatcher] = None
        self.swapper: Optional[EntitySubstitutor] = None

        # Configuration
        self.min_frequency = min_frequency

        # Results
        self.augmented_pairs: List[AugmentedPair] = []

        # Statistics
        self.stats = {
            'gold_count': 0,
            'circle1_count': 0,
            'circle2_count': 0,
            'silver_count': 0,
            'aug_count': 0,
            'flagged_count': 0,
        }

    def initialize(self, verbose: bool = True):
        """Initialize all components."""
        if verbose:
            print("Initializing Graph Augmentor...")

        # 1. Entity Linker
        if verbose:
            print("  Loading entity linker...")
        self.linker = EntityLinker(self.glossary_path)
        if verbose:
            print(f"    {self.linker}")

        # 2. Type Constraints
        constraints = TypeConstraints(min_frequency=self.min_frequency)

        # 3. Build ETCSL graph
        if verbose:
            print("  Building ETCSL graph...")
        self.graph = EntityGraph.from_etcsl(self.etcsl_path)
        if verbose:
            print(f"    {self.graph}")

        # 4. Build ORACC graphs and merge
        if self.oracc_literary_path.exists():
            if verbose:
                print("  Building ORACC literary graph...")
            oracc_lit = EntityGraph.from_oracc(
                self.oracc_literary_path,
                self.linker,
                'oracc_literary'
            )
            self.graph.merge(oracc_lit)
            if verbose:
                print(f"    Merged: {self.graph}")

        if self.oracc_royal_path.exists():
            if verbose:
                print("  Building ORACC royal graph...")
            oracc_royal = EntityGraph.from_oracc(
                self.oracc_royal_path,
                self.linker,
                'oracc_royal'
            )
            self.graph.merge(oracc_royal)
            if verbose:
                print(f"    Merged: {self.graph}")

        # 5. Initialize matcher and swapper
        self.matcher = StructuralMatcher(self.graph)
        self.swapper = EntitySubstitutor(constraints)

        if verbose:
            print("  Initialization complete.")

    def run_circle1(
        self,
        max_per_line: int = 5,
        verbose: bool = True
    ) -> List[AugmentedPair]:
        """
        Run Circle 1 augmentation: ETCSL ↔ ETCSL.

        Args:
            max_per_line: Maximum matches per source line
            verbose: Print progress

        Returns:
            List of augmented pairs
        """
        if verbose:
            print("\nRunning Circle 1 (ETCSL ↔ ETCSL)...")

        etcsl_lines = [l for l in self.graph.get_etcsl_lines()
                       if l.entities and l.has_translation]

        pairs = []
        for line in tqdm(etcsl_lines, desc="Circle 1", disable=not verbose):
            matches = self.matcher.find_circle1_matches(line, max_per_line)
            for match in matches:
                result = self.swapper.swap_entities(match)
                if result:
                    pairs.append(result)

        self.stats['circle1_count'] = len(pairs)
        if verbose:
            print(f"  Generated {len(pairs)} Circle 1 pairs")

        return pairs

    def run_circle1_parallel(
        self,
        max_per_line: int = 5,
        num_workers: Optional[int] = None,
        verbose: bool = True
    ) -> List[AugmentedPair]:
        """
        Run Circle 1 augmentation with parallel processing.

        Uses ThreadPoolExecutor for parallel matching/swapping while
        sharing the entity graph across workers.

        Args:
            max_per_line: Maximum matches per source line
            num_workers: Number of workers (default: auto-detect for 52 vCPUs)
            verbose: Print progress

        Returns:
            List of augmented pairs
        """
        if num_workers is None:
            num_workers = get_optimal_workers()

        if verbose:
            print(f"\nRunning Circle 1 (ETCSL ↔ ETCSL) with {num_workers} workers...")

        etcsl_lines = [l for l in self.graph.get_etcsl_lines()
                       if l.entities and l.has_translation]

        def process_line(line):
            """Process a single line (matching + swapping)."""
            results = []
            matches = self.matcher.find_circle1_matches(line, max_per_line)
            for match in matches:
                result = self.swapper.swap_entities(match)
                if result:
                    results.append(result)
            return results

        # Use ThreadPoolExecutor to avoid pickling issues with shared graph
        pairs = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = list(tqdm(
                executor.map(process_line, etcsl_lines),
                total=len(etcsl_lines),
                desc="Circle 1 (parallel)",
                disable=not verbose
            ))
            for batch in futures:
                pairs.extend(batch)

        self.stats['circle1_count'] = len(pairs)
        if verbose:
            print(f"  Generated {len(pairs)} Circle 1 pairs")

        return pairs

    def run_circle2(
        self,
        max_per_line: int = 5,
        verbose: bool = True
    ) -> List[AugmentedPair]:
        """
        Run Circle 2 augmentation: ORACC → ETCSL.

        Args:
            max_per_line: Maximum matches per template line
            verbose: Print progress

        Returns:
            List of augmented pairs
        """
        if verbose:
            print("\nRunning Circle 2 (ORACC → ETCSL)...")

        oracc_lines = [l for l in self.graph.get_oracc_lines() if l.entities]

        pairs = []
        for line in tqdm(oracc_lines, desc="Circle 2", disable=not verbose):
            matches = self.matcher.find_circle2_matches(line, max_per_line)
            for match in matches:
                result = self.swapper.swap_entities(match)
                if result:
                    pairs.append(result)

        self.stats['circle2_count'] = len(pairs)
        if verbose:
            print(f"  Generated {len(pairs)} Circle 2 pairs")

        return pairs

    def run_circle2_parallel(
        self,
        max_per_line: int = 5,
        num_workers: Optional[int] = None,
        verbose: bool = True
    ) -> List[AugmentedPair]:
        """
        Run Circle 2 augmentation with parallel processing.

        Args:
            max_per_line: Maximum matches per template line
            num_workers: Number of workers (default: auto-detect)
            verbose: Print progress

        Returns:
            List of augmented pairs
        """
        if num_workers is None:
            num_workers = get_optimal_workers()

        if verbose:
            print(f"\nRunning Circle 2 (ORACC → ETCSL) with {num_workers} workers...")

        oracc_lines = [l for l in self.graph.get_oracc_lines() if l.entities]

        def process_line(line):
            """Process a single ORACC line."""
            results = []
            matches = self.matcher.find_circle2_matches(line, max_per_line)
            for match in matches:
                result = self.swapper.swap_entities(match)
                if result:
                    results.append(result)
            return results

        pairs = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = list(tqdm(
                executor.map(process_line, oracc_lines),
                total=len(oracc_lines),
                desc="Circle 2 (parallel)",
                disable=not verbose
            ))
            for batch in futures:
                pairs.extend(batch)

        self.stats['circle2_count'] = len(pairs)
        if verbose:
            print(f"  Generated {len(pairs)} Circle 2 pairs")

        return pairs

    def run_full_pipeline(
        self,
        max_per_line: int = 5,
        verbose: bool = True,
        parallel: bool = False,
        num_workers: Optional[int] = None
    ) -> List[Dict]:
        """
        Run full augmentation pipeline and return training records.

        Returns records in training format with control tokens.

        Args:
            max_per_line: Maximum matches per source line
            verbose: Print progress
            parallel: Use parallel processing (default: False)
            num_workers: Number of workers for parallel mode

        Returns:
            List of training records with source/target
        """
        if self.graph is None:
            self.initialize(verbose)

        # Run both circles (parallel or sequential)
        if parallel:
            circle1_pairs = self.run_circle1_parallel(max_per_line, num_workers, verbose)
            circle2_pairs = self.run_circle2_parallel(max_per_line, num_workers, verbose)
        else:
            circle1_pairs = self.run_circle1(max_per_line, verbose)
            circle2_pairs = self.run_circle2(max_per_line, verbose)

        self.augmented_pairs = circle1_pairs + circle2_pairs

        # Convert to training format
        records = []

        for pair in self.augmented_pairs:
            # Determine control token based on confidence
            if pair.skeleton_similarity >= self.SILVER_THRESHOLD:
                control_token = ControlTokens.SILVER
                self.stats['silver_count'] += 1
            else:
                control_token = ControlTokens.AUG
                self.stats['aug_count'] += 1

            if pair.is_compound_phrase:
                self.stats['flagged_count'] += 1

            record = {
                "source": {
                    "text_normalized": f"{control_token} {pair.source_text}".strip(),
                    "original": pair.original_source,
                },
                "target": {
                    "text": pair.target_text,
                    "original": pair.original_target,
                },
                "quality": {
                    "synthetic": True,
                    "method": pair.match_type,
                },
                "metadata": {
                    "source_line_id": pair.source_line_id,
                    "template_line_id": pair.template_line_id,
                    "match_type": pair.match_type,
                    "skeleton_similarity": pair.skeleton_similarity,
                    "confidence": pair.confidence,
                    "is_compound": pair.is_compound_phrase,
                    "substitutions": [
                        {"from": s.source_label, "to": s.target_label, "type": s.entity_type}
                        for s in pair.substitutions
                    ],
                },
            }
            records.append(record)

        if verbose:
            print(f"\nTotal augmented records: {len(records)}")
            print(f"  Silver (high conf): {self.stats['silver_count']}")
            print(f"  Aug (standard): {self.stats['aug_count']}")
            print(f"  Flagged for review: {self.stats['flagged_count']}")

        return records

    def export_jsonl(self, output_path: Path, records: List[Dict]):
        """Export records to JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"Exported {len(records)} records to {output_path}")

    def export_parquet(
        self,
        output_dir: Path,
        records: List[Dict],
        prefix: str = "graph_augmented",
        chunk_size: int = 10000,
    ):
        """
        Export records to chunked parquet files for GitHub-friendly storage.

        Args:
            output_dir: Directory for parquet chunks
            records: List of training records
            prefix: Filename prefix for chunks
            chunk_size: Records per chunk (default: 10000)
        """
        # Flatten records for parquet (nested dicts don't work well)
        flat_records = []
        for record in records:
            source = record.get("source", {})
            target = record.get("target", {})
            metadata = record.get("metadata", {})

            flat_record = {
                "source_text": source.get("text_normalized", ""),
                "source_original": source.get("original", ""),
                "target_text": target.get("text", ""),
                "target_original": target.get("original", ""),
                "source_line_id": metadata.get("source_line_id", ""),
                "template_line_id": metadata.get("template_line_id", ""),
                "match_type": metadata.get("match_type", ""),
                "skeleton_similarity": metadata.get("skeleton_similarity", 0.0),
                "confidence": metadata.get("confidence", 0.0),
                "is_compound": metadata.get("is_compound", False),
                "substitutions_json": json.dumps(metadata.get("substitutions", [])),
            }
            flat_records.append(flat_record)

        writer = ChunkedParquetWriter(
            output_dir=output_dir,
            prefix=prefix,
            chunk_size=chunk_size,
            compression="snappy",
            generator="graph_augmentor.py",
        )
        writer.add_records(flat_records)
        metadata = writer.finalize()

        print(f"Exported {len(records)} records to {output_dir}/")

    def get_statistics(self) -> dict:
        """Get comprehensive statistics."""
        stats = {**self.stats}

        if self.linker:
            stats['linker'] = self.linker.get_statistics()
        if self.matcher:
            stats['matcher'] = self.matcher.get_statistics()
        if self.swapper:
            stats['swapper'] = self.swapper.get_statistics()

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Graph-Enhanced Data Augmentation for Sumerian NMT"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Paths.TRAINING_DATA / "finetune" / "train_graph_augmented.jsonl",
        help="Output JSONL file"
    )
    parser.add_argument(
        "--max-per-line",
        type=int,
        default=5,
        help="Maximum matches per source line"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum entity frequency for substitution"
    )
    parser.add_argument(
        "--circle1-only",
        action="store_true",
        help="Only run Circle 1 (ETCSL ↔ ETCSL)"
    )
    parser.add_argument(
        "--circle2-only",
        action="store_true",
        help="Only run Circle 2 (ORACC → ETCSL)"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use parallel processing (recommended for 52+ vCPU systems)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker threads for parallel mode (default: auto-detect)"
    )
    parser.add_argument(
        "--output-format",
        choices=["jsonl", "parquet", "both"],
        default="both",
        help="Output format: jsonl (legacy), parquet (GitHub-friendly), or both (default)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Records per parquet chunk (default: 10000, ~5-30MB per chunk)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Graph-Enhanced Sumerian NMT Data Augmentation")
    print("=" * 70)

    if args.parallel:
        workers = args.workers or get_optimal_workers()
        print(f"Parallel mode: ENABLED ({workers} workers)")

    # Initialize augmentor
    augmentor = GraphAugmentor(min_frequency=args.min_frequency)
    augmentor.initialize()

    # Run pipeline
    if args.circle1_only:
        if args.parallel:
            pairs = augmentor.run_circle1_parallel(args.max_per_line, args.workers)
        else:
            pairs = augmentor.run_circle1(args.max_per_line)
        records = []
        for pair in pairs:
            token = ControlTokens.SILVER if pair.skeleton_similarity >= 0.95 else ControlTokens.AUG
            records.append({
                "source": {"text_normalized": f"{token} {pair.source_text}".strip()},
                "target": {"text": pair.target_text},
            })
    elif args.circle2_only:
        if args.parallel:
            pairs = augmentor.run_circle2_parallel(args.max_per_line, args.workers)
        else:
            pairs = augmentor.run_circle2(args.max_per_line)
        records = []
        for pair in pairs:
            token = ControlTokens.SILVER if pair.skeleton_similarity >= 0.95 else ControlTokens.AUG
            records.append({
                "source": {"text_normalized": f"{token} {pair.source_text}".strip()},
                "target": {"text": pair.target_text},
            })
    else:
        records = augmentor.run_full_pipeline(
            args.max_per_line,
            parallel=args.parallel,
            num_workers=args.workers
        )

    # Export based on output format
    print("\n" + "=" * 70)
    print("Exporting Output")
    print("=" * 70)

    parquet_dir = args.output.parent / "graph_augmented_parquet"

    if args.output_format in ("jsonl", "both"):
        augmentor.export_jsonl(args.output, records)

    if args.output_format in ("parquet", "both"):
        augmentor.export_parquet(
            output_dir=parquet_dir,
            records=records,
            prefix="graph_augmented",
            chunk_size=args.chunk_size,
        )

    # Print statistics
    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)
    stats = augmentor.get_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.2%}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
