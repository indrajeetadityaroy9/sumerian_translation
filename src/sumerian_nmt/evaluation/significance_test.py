"""
Statistical Significance Testing for Translation Evaluation

Implements paired bootstrap resampling for statistical significance:
- Compares BLEU/chrF scores between model pairs
- Computes p-values and confidence intervals
- Follows WMT statistical testing guidelines

Usage:
    from sumerian_nmt.evaluation.significance_test import bootstrap_significance

    p_value = bootstrap_significance(
        model_a_scores, model_b_scores,
        n_bootstrap=1000, seed=42
    )
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def bootstrap_significance(
    scores_a: List[float],
    scores_b: List[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute bootstrap significance test between two models.

    Uses paired bootstrap resampling following WMT guidelines.

    Args:
        scores_a: Per-sample scores for model A
        scores_b: Per-sample scores for model B
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed for reproducibility

    Returns:
        Dictionary with p_value, mean_diff, ci_low, ci_high
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have same length")

    np.random.seed(seed)

    n_samples = len(scores_a)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    # Observed difference
    observed_diff = np.mean(scores_a) - np.mean(scores_b)

    # Bootstrap
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        diff = np.mean(scores_a[indices]) - np.mean(scores_b[indices])
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Two-tailed p-value
    p_value = 2 * min(
        np.mean(bootstrap_diffs >= 0),
        np.mean(bootstrap_diffs <= 0)
    )

    # 95% confidence interval
    ci_low = np.percentile(bootstrap_diffs, 2.5)
    ci_high = np.percentile(bootstrap_diffs, 97.5)

    return {
        'p_value': p_value,
        'mean_diff': observed_diff,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'significant_at_05': p_value < 0.05,
        'n_samples': n_samples,
        'n_bootstrap': n_bootstrap,
    }


def compare_metrics(
    results_a: Dict[str, float],
    results_b: Dict[str, float],
    sample_scores_a: Optional[List[float]] = None,
    sample_scores_b: Optional[List[float]] = None,
) -> Dict[str, Dict]:
    """
    Compare aggregate metrics with significance testing.

    Args:
        results_a: Model A aggregate results (bleu, chrf, etc.)
        results_b: Model B aggregate results
        sample_scores_a: Per-sample scores for model A (optional)
        sample_scores_b: Per-sample scores for model B (optional)

    Returns:
        Comparison dictionary with significance tests
    """
    comparison = {}

    for metric in ['bleu', 'chrf', 'ne_accuracy']:
        if metric in results_a and metric in results_b:
            comparison[metric] = {
                'model_a': results_a[metric],
                'model_b': results_b[metric],
                'diff': results_a[metric] - results_b[metric],
            }

            # Add significance test if sample scores available
            if sample_scores_a and sample_scores_b:
                sig = bootstrap_significance(sample_scores_a, sample_scores_b)
                comparison[metric].update(sig)

    return comparison


def main():
    """Test significance testing."""
    print("Statistical Significance Testing")
    print("=" * 50)

    # Example with synthetic data
    np.random.seed(42)
    scores_a = np.random.normal(0.25, 0.05, 100).tolist()
    scores_b = np.random.normal(0.22, 0.05, 100).tolist()

    result = bootstrap_significance(scores_a, scores_b)

    print(f"\nModel A mean: {np.mean(scores_a):.4f}")
    print(f"Model B mean: {np.mean(scores_b):.4f}")
    print(f"Mean difference: {result['mean_diff']:.4f}")
    print(f"95% CI: [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
    print(f"p-value: {result['p_value']:.4f}")
    print(f"Significant at p<0.05: {result['significant_at_05']}")


if __name__ == "__main__":
    main()
