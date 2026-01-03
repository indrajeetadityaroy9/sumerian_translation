#!/usr/bin/env python3
"""
Ablation Study Runner for Graph-Enhanced Sumerian NMT

Runs three experiments to prove the hypothesis that graph-based
entity substitution improves Named Entity translation accuracy.

Experiments:
1. Baseline: ETCSL only (5.8k pairs)
   - "Standard NMT fails on rare entities."

2. Naive Graph: ETCSL + Exact ORACC matches (high skeleton similarity)
   - "Volume helps general fluency, but not rare entities."

3. Smart Graph: ETCSL + Entity substitution pairs
   - "Structural augmentation significantly improves Entity Accuracy."

Key Metric: Named Entity Accuracy on test set
Expected Results:
    | Metric       | Baseline | Naive Graph | Smart Graph |
    |--------------|----------|-------------|-------------|
    | BLEU         | 18-22    | 20-24       | 22-26       |
    | chrF++       | 32-36    | 34-38       | 36-42       |
    | NE Accuracy  | ~0%      | ~20%        | >60%        |

Usage:
    python run_ablation.py --generate-datasets
    python run_ablation.py --run-all
    python run_ablation.py --evaluate-only
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import Paths, ControlTokens, TrainingDefaults


@dataclass
class ExperimentConfig:
    """Configuration for an ablation experiment."""
    name: str
    train_file: Path
    description: str
    control_token: str
    augmentation_type: Optional[str]  # None, "exact_match", "entity_substitution"


EXPERIMENTS = {
    'baseline': ExperimentConfig(
        name='baseline',
        train_file=Paths.ABLATION_BASELINE,
        description='ETCSL only (5.8k pairs)',
        control_token=ControlTokens.GOLD,
        augmentation_type=None,
    ),
    'naive_graph': ExperimentConfig(
        name='naive_graph',
        train_file=Paths.ABLATION_NAIVE,
        description='ETCSL + Exact ORACC matches (skeleton >= 95%)',
        control_token=ControlTokens.SILVER,
        augmentation_type='exact_match',
    ),
    'smart_graph': ExperimentConfig(
        name='smart_graph',
        train_file=Paths.ABLATION_SMART,
        description='ETCSL + Entity substitution (skeleton >= 85%)',
        control_token=ControlTokens.AUG,
        augmentation_type='entity_substitution',
    ),
}


def load_gold_data() -> List[Dict]:
    """Load original ETCSL gold data."""
    gold_path = Paths.TRAIN_FILE_V2 if Paths.TRAIN_FILE_V2.exists() else Paths.TRAIN_FILE

    if not gold_path.exists():
        raise FileNotFoundError(f"Gold training data not found at {gold_path}")

    records = []
    with open(gold_path, encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            records.append(record)

    return records


def inject_control_token(record: Dict, token: str) -> Dict:
    """Inject control token into source text."""
    source = record.get('source', {})
    text = source.get('text_normalized', '')

    # Don't double-inject
    if not text.startswith('<'):
        source['text_normalized'] = f"{token} {text}".strip()

    return record


def generate_baseline_dataset() -> Path:
    """Generate baseline dataset (ETCSL only with <gold> token)."""
    print("\nGenerating Baseline dataset...")

    gold_data = load_gold_data()
    output = Paths.ABLATION_BASELINE

    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w', encoding='utf-8') as f:
        for record in gold_data:
            record = inject_control_token(record, ControlTokens.GOLD)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"  Created {output} with {len(gold_data)} records")
    return output


def generate_naive_graph_dataset() -> Path:
    """
    Generate naive graph dataset (ETCSL + exact matches only).

    Exact matches are those with skeleton similarity >= 95%.
    NO entity substitution is performed.
    """
    print("\nGenerating Naive Graph dataset...")

    from processors.entity_linker import EntityLinker
    from processors.graph_engine import EntityGraph, LineMatcher

    # Load gold data
    gold_data = load_gold_data()

    # Build graph and find high-similarity matches
    linker = EntityLinker()
    graph = EntityGraph.from_etcsl(Paths.ETCSL_PARQUET)

    # Add ORACC for matching
    if Paths.ORACC_LITERARY_PARQUET.exists():
        oracc_lit = EntityGraph.from_oracc(Paths.ORACC_LITERARY_PARQUET, linker)
        graph.merge(oracc_lit)

    matcher = LineMatcher(graph)

    # Find only exact matches (skeleton >= 95%)
    # These are near-duplicates where we can propagate translations
    exact_pairs = []
    for line in graph.get_etcsl_lines():
        if not line.has_translation or not line.entities:
            continue

        matches = matcher.find_circle1_matches(line, max_matches=3)
        for match in matches:
            if match.skeleton_similarity >= 0.95:
                # For exact match, we DON'T substitute - we just propagate
                exact_pairs.append({
                    "source": {
                        "text_normalized": f"{ControlTokens.SILVER} {match.template_line.source_text}".strip(),
                    },
                    "target": {
                        "text": line.target_text,  # Propagate translation
                    },
                    "metadata": {
                        "type": "exact_match",
                        "skeleton_similarity": match.skeleton_similarity,
                    }
                })

    # Combine gold + exact matches
    output = Paths.ABLATION_NAIVE
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w', encoding='utf-8') as f:
        # Gold data first
        for record in gold_data:
            record = inject_control_token(record, ControlTokens.GOLD)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        # Exact matches
        for record in exact_pairs:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"  Created {output}")
    print(f"    Gold records: {len(gold_data)}")
    print(f"    Exact match records: {len(exact_pairs)}")
    print(f"    Total: {len(gold_data) + len(exact_pairs)}")

    return output


def generate_smart_graph_dataset() -> Path:
    """
    Generate smart graph dataset (ETCSL + entity substitution).

    Uses the full graph augmentor pipeline.
    """
    print("\nGenerating Smart Graph dataset...")

    from processors.graph_augmentor import GraphAugmentor

    # Load gold data
    gold_data = load_gold_data()

    # Run augmentor
    augmentor = GraphAugmentor()
    augmentor.initialize(verbose=True)
    aug_records = augmentor.run_full_pipeline(max_per_line=5, verbose=True)

    # Combine gold + augmented
    output = Paths.ABLATION_SMART
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w', encoding='utf-8') as f:
        # Gold data first
        for record in gold_data:
            record = inject_control_token(record, ControlTokens.GOLD)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        # Augmented records
        for record in aug_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"  Created {output}")
    print(f"    Gold records: {len(gold_data)}")
    print(f"    Augmented records: {len(aug_records)}")
    print(f"    Total: {len(gold_data) + len(aug_records)}")

    return output


def generate_all_datasets():
    """Generate all ablation study datasets."""
    print("=" * 70)
    print("Generating Ablation Study Datasets")
    print("=" * 70)

    generate_baseline_dataset()
    generate_naive_graph_dataset()
    generate_smart_graph_dataset()

    print("\n" + "=" * 70)
    print("All datasets generated successfully!")
    print("=" * 70)

    # Print summary
    print("\nDataset Summary:")
    print("-" * 50)
    for exp_name, exp_config in EXPERIMENTS.items():
        if exp_config.train_file.exists():
            with open(exp_config.train_file) as f:
                count = sum(1 for _ in f)
            print(f"  {exp_name}: {count} records")
        else:
            print(f"  {exp_name}: NOT FOUND")


def run_training(experiment: str, epochs: int = 20, batch_size: int = 64):
    """Run training for a specific experiment."""
    config = EXPERIMENTS[experiment]

    if not config.train_file.exists():
        print(f"ERROR: Training file not found: {config.train_file}")
        return None

    output_dir = Paths.MODELS / f"ablation_{experiment}"

    cmd = [
        sys.executable, "train.py",
        "--train-file", str(config.train_file),
        "--output-dir", str(output_dir),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]

    print(f"\nRunning training for {experiment}...")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"ERROR: Training failed for {experiment}")
        return None

    return output_dir


def run_evaluation(experiment: str) -> Optional[Dict]:
    """Run evaluation for a specific experiment."""
    model_dir = Paths.MODELS / f"ablation_{experiment}"

    if not model_dir.exists():
        print(f"ERROR: Model not found for {experiment}: {model_dir}")
        return None

    # Run academic evaluation
    output_report = Paths.TRAINING_DATA / "finetune" / "audit" / f"eval_{experiment}.md"

    cmd = [
        sys.executable, "evaluate_academic.py",
        "--model", str(model_dir),
        "--test-file", str(Paths.VALID_FILE),
        "--output", str(output_report),
    ]

    print(f"\nEvaluating {experiment}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Evaluation failed for {experiment}")
        print(result.stderr)
        return None

    # Parse results from output
    results = {}
    for line in result.stdout.split('\n'):
        if 'BLEU=' in line:
            import re
            bleu_match = re.search(r'BLEU=(\d+\.?\d*)', line)
            chrf_match = re.search(r'chrF\+\+=(\d+\.?\d*)', line)
            if bleu_match:
                results['BLEU'] = float(bleu_match.group(1))
            if chrf_match:
                results['chrF++'] = float(chrf_match.group(1))

    return results


def run_all_experiments(epochs: int = 20, batch_size: int = 64):
    """Run all ablation experiments."""
    print("=" * 70)
    print("Running All Ablation Experiments")
    print("=" * 70)

    results = {}

    for exp_name in EXPERIMENTS:
        print(f"\n{'='*70}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*70}")

        # Train
        run_training(exp_name, epochs, batch_size)

        # Evaluate
        eval_results = run_evaluation(exp_name)
        if eval_results:
            results[exp_name] = eval_results

    # Print comparison table
    print_comparison_table(results)


def print_comparison_table(results: Dict):
    """Print comparison table of experiment results."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS")
    print("=" * 70)

    print("\n| Experiment   | BLEU   | chrF++ | Description                          |")
    print("|--------------|--------|--------|--------------------------------------|")

    for exp_name, config in EXPERIMENTS.items():
        if exp_name in results:
            bleu = results[exp_name].get('BLEU', 0)
            chrf = results[exp_name].get('chrF++', 0)
            print(f"| {exp_name:12} | {bleu:6.2f} | {chrf:6.2f} | {config.description[:36]} |")
        else:
            print(f"| {exp_name:12} |   N/A  |   N/A  | {config.description[:36]} |")

    print("\n" + "=" * 70)


def evaluate_all():
    """Evaluate all trained models."""
    print("=" * 70)
    print("Evaluating All Models")
    print("=" * 70)

    results = {}

    for exp_name in EXPERIMENTS:
        eval_results = run_evaluation(exp_name)
        if eval_results:
            results[exp_name] = eval_results

    print_comparison_table(results)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation Study Runner for Graph-Enhanced Sumerian NMT"
    )
    parser.add_argument(
        "--generate-datasets",
        action="store_true",
        help="Generate all ablation study datasets"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all experiments (train + evaluate)"
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate existing trained models"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS.keys()),
        help="Run single experiment"
    )

    args = parser.parse_args()

    if args.generate_datasets:
        generate_all_datasets()
    elif args.run_all:
        generate_all_datasets()
        run_all_experiments(args.epochs, args.batch_size)
    elif args.evaluate_only:
        evaluate_all()
    elif args.experiment:
        if not EXPERIMENTS[args.experiment].train_file.exists():
            generate_all_datasets()
        run_training(args.experiment, args.epochs, args.batch_size)
        run_evaluation(args.experiment)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
