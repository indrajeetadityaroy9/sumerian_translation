"""
Quality Gate for Augmented Data

Generates audit CSV for human review of augmented pairs.
Flags problematic substitutions for manual verification.

Flagging Criteria:
- Confidence < 0.3 (low skeleton similarity)
- Complex substitutions (3+ entities)
- Compound phrase swaps ("King X", "The shepherd X")
- Skeleton similarity 85-90% (edge cases)
- Pattern matches in target ("royal", "temple")

Usage:
    python processors/quality_gate.py --input train_graph_augmented.jsonl
    python processors/quality_gate.py --export-flagged-only
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from config import Paths


@dataclass
class AuditEntry:
    """Entry for quality audit."""
    needs_review: bool
    review_reasons: List[str]
    confidence: float
    skeleton_similarity: float
    source_text: str
    target_text: str
    original_source: str
    original_target: str
    substitutions: List[Dict]
    is_compound_phrase: bool
    source_line_id: str
    template_line_id: str
    match_type: str


class QualityGate:
    """
    Quality assurance for augmented data.

    Generates audit CSV for human review of flagged pairs.
    """

    # Patterns that may need review
    SENSITIVE_PATTERNS = [
        r'\broyal\b',
        r'\bking\b',
        r'\bqueen\b',
        r'\btemple\b',
        r'\bgod\b',
        r'\bgoddess\b',
        r'\bshepherd\b',
        r'\bpriest\b',
    ]

    # Thresholds
    LOW_CONFIDENCE = 0.3
    EDGE_CASE_MIN = 0.85
    EDGE_CASE_MAX = 0.90
    COMPLEX_SUBSTITUTION_COUNT = 3

    def __init__(self):
        """Initialize quality gate."""
        self.entries: List[AuditEntry] = []
        self.sensitive_re = [
            re.compile(p, re.IGNORECASE) for p in self.SENSITIVE_PATTERNS
        ]

        # Statistics
        self.stats = {
            'total': 0,
            'flagged': 0,
            'low_confidence': 0,
            'edge_case_skeleton': 0,
            'complex_substitution': 0,
            'compound_phrase': 0,
            'sensitive_pattern': 0,
        }

    def _check_sensitive_patterns(self, text: str) -> List[str]:
        """Check for sensitive patterns in text."""
        matches = []
        for pattern in self.sensitive_re:
            if pattern.search(text):
                matches.append(pattern.pattern)
        return matches

    def add_augmented_record(self, record: Dict) -> AuditEntry:
        """
        Add an augmented record and evaluate for review.

        Args:
            record: Augmented record from graph_augmentor

        Returns:
            AuditEntry with review status
        """
        self.stats['total'] += 1

        metadata = record.get('metadata', {})
        source = record.get('source', {})
        target = record.get('target', {})

        confidence = metadata.get('confidence', 1.0)
        skeleton_sim = metadata.get('skeleton_similarity', 1.0)
        substitutions = metadata.get('substitutions', [])
        is_compound = metadata.get('is_compound', False)

        # Determine if review is needed
        review_reasons = []

        # 1. Low confidence
        if confidence < self.LOW_CONFIDENCE:
            review_reasons.append("low_confidence")
            self.stats['low_confidence'] += 1

        # 2. Edge case skeleton similarity
        if self.EDGE_CASE_MIN <= skeleton_sim < self.EDGE_CASE_MAX:
            review_reasons.append("edge_case_skeleton")
            self.stats['edge_case_skeleton'] += 1

        # 3. Complex substitutions
        if len(substitutions) >= self.COMPLEX_SUBSTITUTION_COUNT:
            review_reasons.append("complex_substitution")
            self.stats['complex_substitution'] += 1

        # 4. Compound phrases
        if is_compound:
            review_reasons.append("compound_phrase")
            self.stats['compound_phrase'] += 1

        # 5. Sensitive patterns in target
        target_text = target.get('text', '')
        sensitive = self._check_sensitive_patterns(target_text)
        if sensitive:
            review_reasons.append(f"sensitive_pattern:{','.join(sensitive)}")
            self.stats['sensitive_pattern'] += 1

        needs_review = len(review_reasons) > 0
        if needs_review:
            self.stats['flagged'] += 1

        entry = AuditEntry(
            needs_review=needs_review,
            review_reasons=review_reasons,
            confidence=confidence,
            skeleton_similarity=skeleton_sim,
            source_text=source.get('text_normalized', ''),
            target_text=target_text,
            original_source=source.get('original', ''),
            original_target=target.get('original', ''),
            substitutions=substitutions,
            is_compound_phrase=is_compound,
            source_line_id=metadata.get('source_line_id', ''),
            template_line_id=metadata.get('template_line_id', ''),
            match_type=metadata.get('match_type', ''),
        )

        self.entries.append(entry)
        return entry

    def load_from_jsonl(self, path: Path):
        """Load augmented records from JSONL file."""
        with open(path, encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                self.add_augmented_record(record)

    def export_audit_csv(self, output_path: Path) -> Path:
        """Export full audit CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'needs_review',
                'review_reasons',
                'confidence',
                'skeleton_similarity',
                'source_text',
                'target_text',
                'original_source',
                'original_target',
                'substitutions',
                'is_compound',
                'source_line_id',
                'template_line_id',
                'match_type',
            ])

            # Data
            for entry in self.entries:
                writer.writerow([
                    entry.needs_review,
                    ';'.join(entry.review_reasons),
                    f"{entry.confidence:.3f}",
                    f"{entry.skeleton_similarity:.3f}",
                    entry.source_text[:200],  # Truncate for readability
                    entry.target_text[:200],
                    entry.original_source[:200],
                    entry.original_target[:200],
                    json.dumps(entry.substitutions),
                    entry.is_compound_phrase,
                    entry.source_line_id,
                    entry.template_line_id,
                    entry.match_type,
                ])

        return output_path

    def export_flagged_csv(self, output_path: Path) -> Path:
        """Export only flagged entries for review."""
        flagged = [e for e in self.entries if e.needs_review]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            writer.writerow([
                'review_reasons',
                'confidence',
                'skeleton_similarity',
                'target_text',
                'original_target',
                'substitutions',
                'source_line_id',
            ])

            for entry in flagged:
                writer.writerow([
                    ';'.join(entry.review_reasons),
                    f"{entry.confidence:.3f}",
                    f"{entry.skeleton_similarity:.3f}",
                    entry.target_text[:200],
                    entry.original_target[:200],
                    json.dumps(entry.substitutions),
                    entry.source_line_id,
                ])

        return output_path

    def get_statistics(self) -> Dict:
        """Get quality gate statistics."""
        total = max(1, self.stats['total'])
        return {
            **self.stats,
            'flag_rate': self.stats['flagged'] / total,
        }

    def print_summary(self):
        """Print quality summary."""
        stats = self.get_statistics()

        print("\nQuality Gate Summary")
        print("-" * 40)
        print(f"Total entries: {stats['total']}")
        print(f"Flagged for review: {stats['flagged']} ({stats['flag_rate']:.1%})")
        print(f"\nFlag reasons:")
        print(f"  Low confidence: {stats['low_confidence']}")
        print(f"  Edge case skeleton: {stats['edge_case_skeleton']}")
        print(f"  Complex substitution: {stats['complex_substitution']}")
        print(f"  Compound phrase: {stats['compound_phrase']}")
        print(f"  Sensitive pattern: {stats['sensitive_pattern']}")


def main():
    parser = argparse.ArgumentParser(
        description="Quality Gate for Augmented Sumerian NMT Data"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Paths.TRAINING_DATA / "finetune" / "train_graph_augmented.jsonl",
        help="Input JSONL file with augmented data"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Paths.TRAINING_DATA / "finetune" / "audit",
        help="Output directory for audit CSVs"
    )
    parser.add_argument(
        "--flagged-only",
        action="store_true",
        help="Only export flagged entries"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Quality Gate for Augmented Data")
    print("=" * 60)

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("Run graph_augmentor.py first to generate augmented data.")
        return

    # Load and analyze
    gate = QualityGate()
    print(f"\nLoading {args.input}...")
    gate.load_from_jsonl(args.input)

    # Print summary
    gate.print_summary()

    # Export CSVs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.flagged_only:
        output = args.output_dir / "flagged_for_review.csv"
        gate.export_flagged_csv(output)
        print(f"\nExported flagged entries to: {output}")
    else:
        full_output = args.output_dir / "augmentation_audit.csv"
        flagged_output = args.output_dir / "flagged_for_review.csv"

        gate.export_audit_csv(full_output)
        gate.export_flagged_csv(flagged_output)

        print(f"\nExported audit files:")
        print(f"  Full audit: {full_output}")
        print(f"  Flagged only: {flagged_output}")


if __name__ == "__main__":
    main()
