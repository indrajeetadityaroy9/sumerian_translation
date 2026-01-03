#!/usr/bin/env python3
"""
Model Comparison Script for Sumerian Translation

Compares evaluation results between:
- Legacy mT5 (Encoder-Decoder)
- Llama-3 SFT (Decoder-Only)
- Llama-3 DPO (Aligned)

Generates a unified comparison report for thesis.

Usage:
    python -m evaluation.compare_models
    python -m evaluation.compare_models --mt5-results legacy_results.json --llm-results llm_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

from config import Paths, EvalTargets


@dataclass
class ModelResult:
    """Container for model evaluation results."""
    name: str
    model_type: str  # "encoder_decoder" or "decoder_only"
    bleu: float
    chrf: float
    bertscore_f1: Optional[float] = None
    ne_accuracy: Optional[float] = None
    num_samples: int = 0
    avg_pred_len: float = 0.0
    notes: str = ""


def load_results_from_json(path: Path) -> Dict:
    """Load evaluation results from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_results_from_report(path: Path) -> Dict:
    """Parse evaluation results from markdown report."""
    results = {}

    with open(path, 'r') as f:
        content = f.read()

    # Parse BLEU
    import re
    bleu_match = re.search(r'\| BLEU \| (\d+\.?\d*) \|', content)
    if bleu_match:
        results['bleu'] = float(bleu_match.group(1))

    chrf_match = re.search(r'\| chrF\+?\+? \| (\d+\.?\d*) \|', content)
    if chrf_match:
        results['chrf'] = float(chrf_match.group(1))

    bert_match = re.search(r'\| BERTScore.*? \| (\d+\.?\d*) \|', content)
    if bert_match:
        results['bertscore_f1'] = float(bert_match.group(1))

    return results


def create_comparison_table(results: List[ModelResult]) -> str:
    """Create markdown comparison table."""
    lines = [
        "| Model | Type | BLEU | chrF | BERTScore | NE Acc |",
        "|-------|------|------|------|-----------|--------|",
    ]

    for r in results:
        bert = f"{r.bertscore_f1:.2f}" if r.bertscore_f1 else "N/A"
        ne = f"{r.ne_accuracy:.1%}" if r.ne_accuracy else "N/A"
        lines.append(
            f"| {r.name} | {r.model_type} | {r.bleu:.2f} | {r.chrf:.2f} | {bert} | {ne} |"
        )

    return "\n".join(lines)


def analyze_improvements(baseline: ModelResult, improved: ModelResult) -> str:
    """Analyze improvements between two models."""
    lines = []

    bleu_delta = improved.bleu - baseline.bleu
    chrf_delta = improved.chrf - baseline.chrf

    lines.append(f"**BLEU Improvement:** {bleu_delta:+.2f} ({baseline.bleu:.2f} → {improved.bleu:.2f})")
    lines.append(f"**chrF Improvement:** {chrf_delta:+.2f} ({baseline.chrf:.2f} → {improved.chrf:.2f})")

    if baseline.bertscore_f1 and improved.bertscore_f1:
        bert_delta = improved.bertscore_f1 - baseline.bertscore_f1
        lines.append(f"**BERTScore Improvement:** {bert_delta:+.2f}")

    if baseline.ne_accuracy is not None and improved.ne_accuracy is not None:
        ne_delta = improved.ne_accuracy - baseline.ne_accuracy
        lines.append(f"**NE Accuracy Improvement:** {ne_delta:+.1%}")

    return "\n".join(lines)


def generate_comparison_report(
    results: List[ModelResult],
    output_path: Path,
) -> None:
    """Generate comprehensive comparison report."""

    with open(output_path, 'w') as f:
        f.write("# Model Comparison Report: Sumerian-English Translation\n\n")
        f.write("Comparative analysis of translation models for low-resource Sumerian NMT.\n\n")

        f.write("## Summary Table\n\n")
        f.write(create_comparison_table(results))
        f.write("\n\n")

        f.write("## Target Metrics\n\n")
        f.write("Based on low-resource NMT literature:\n\n")
        f.write(f"- **BLEU Target:** {EvalTargets.BLEU_MIN}-{EvalTargets.BLEU_MAX}\n")
        f.write(f"- **chrF Target:** {EvalTargets.CHRF_MIN}-{EvalTargets.CHRF_MAX}\n")
        f.write(f"- **BERTScore F1 Target:** >{EvalTargets.BERTSCORE_F1_MIN}\n")
        f.write(f"- **NE Accuracy Target:** >{EvalTargets.NE_ACCURACY_TARGET:.0%}\n\n")

        # Pairwise improvements
        if len(results) >= 2:
            f.write("## Improvement Analysis\n\n")
            baseline = results[0]
            for i, improved in enumerate(results[1:], 1):
                f.write(f"### {baseline.name} → {improved.name}\n\n")
                f.write(analyze_improvements(baseline, improved))
                f.write("\n\n")

        f.write("## Model Details\n\n")
        for r in results:
            f.write(f"### {r.name}\n\n")
            f.write(f"- **Type:** {r.model_type}\n")
            f.write(f"- **Samples Evaluated:** {r.num_samples}\n")
            f.write(f"- **Avg Prediction Length:** {r.avg_pred_len:.1f} words\n")
            if r.notes:
                f.write(f"- **Notes:** {r.notes}\n")
            f.write("\n")

        f.write("## Conclusion\n\n")
        f.write("_[Add thesis conclusions here based on results]_\n")

    print(f"Comparison report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Sumerian translation model results"
    )
    parser.add_argument(
        "--mt5-results",
        type=Path,
        default=None,
        help="Path to mT5 results (JSON or markdown)"
    )
    parser.add_argument(
        "--llm-results",
        type=Path,
        default=None,
        help="Path to LLM results (JSON or markdown)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_comparison.md"),
        help="Output report path"
    )

    args = parser.parse_args()

    # Placeholder results for demonstration
    # Replace with actual loaded results
    results = []

    # mT5 Baseline (from previous training)
    results.append(ModelResult(
        name="mT5-Large (Baseline)",
        model_type="Encoder-Decoder",
        bleu=0.14,  # From academic_report.md
        chrf=7.28,
        bertscore_f1=None,
        ne_accuracy=0.0,
        num_samples=661,
        avg_pred_len=37.6,
        notes="Severe repetition issues observed"
    ))

    # Placeholder for LLM results (to be filled after training)
    results.append(ModelResult(
        name="Llama-3 SFT",
        model_type="Decoder-Only",
        bleu=0.0,  # TBD
        chrf=0.0,  # TBD
        bertscore_f1=None,
        ne_accuracy=None,
        num_samples=0,
        avg_pred_len=0.0,
        notes="Pending training"
    ))

    # Load actual results if provided
    if args.mt5_results and args.mt5_results.exists():
        if args.mt5_results.suffix == '.json':
            mt5_data = load_results_from_json(args.mt5_results)
        else:
            mt5_data = load_results_from_report(args.mt5_results)
        results[0].bleu = mt5_data.get('bleu', results[0].bleu)
        results[0].chrf = mt5_data.get('chrf', results[0].chrf)

    if args.llm_results and args.llm_results.exists():
        if args.llm_results.suffix == '.json':
            llm_data = load_results_from_json(args.llm_results)
        else:
            llm_data = load_results_from_report(args.llm_results)
        results[1].bleu = llm_data.get('bleu', results[1].bleu)
        results[1].chrf = llm_data.get('chrf', results[1].chrf)

    generate_comparison_report(results, args.output)


if __name__ == "__main__":
    main()
