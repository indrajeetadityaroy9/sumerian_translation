"""
Evaluation metrics for translation quality.

Provides BLEU and chrF metric computation with consistent
interface across training scripts.
"""

from typing import Callable, Dict, Any, Tuple, Optional

import numpy as np


def load_metrics() -> Tuple[Any, Any]:
    """
    Load BLEU and chrF metrics from evaluate library.

    Returns:
        Tuple of (bleu_metric, chrf_metric)
    """
    import evaluate

    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    return bleu, chrf


def compute_bleu_chrf(
    predictions: list,
    references: list,
    bleu_metric: Optional[Any] = None,
    chrf_metric: Optional[Any] = None
) -> Dict[str, float]:
    """
    Compute BLEU and chrF scores for predictions.

    Args:
        predictions: List of predicted strings
        references: List of reference strings (or list of lists)
        bleu_metric: Pre-loaded BLEU metric (loads if None)
        chrf_metric: Pre-loaded chrF metric (loads if None)

    Returns:
        Dictionary with 'bleu' and 'chrf' scores
    """
    if bleu_metric is None or chrf_metric is None:
        bleu_metric, chrf_metric = load_metrics()

    # Ensure references are wrapped in lists (sacrebleu format)
    if references and not isinstance(references[0], list):
        references = [[ref] for ref in references]

    predictions = [pred.strip() for pred in predictions]

    result_bleu = bleu_metric.compute(
        predictions=predictions,
        references=references
    )
    result_chrf = chrf_metric.compute(
        predictions=predictions,
        references=references
    )

    return {
        "bleu": result_bleu["score"],
        "chrf": result_chrf["score"]
    }


def create_compute_metrics_fn(
    tokenizer,
    bleu_metric: Optional[Any] = None,
    chrf_metric: Optional[Any] = None
) -> Callable:
    """
    Create a compute_metrics function for HuggingFace Trainer.

    Args:
        tokenizer: Tokenizer for decoding predictions
        bleu_metric: Pre-loaded BLEU metric
        chrf_metric: Pre-loaded chrF metric

    Returns:
        Callable that takes (eval_preds) and returns metrics dict
    """
    if bleu_metric is None or chrf_metric is None:
        bleu_metric, chrf_metric = load_metrics()

    def compute_metrics(eval_preds) -> Dict[str, float]:
        preds, labels = eval_preds

        # Handle tuple predictions (generation with logits)
        if isinstance(preds, tuple):
            preds = preds[0]

        # Sanitize predictions (clip to valid vocab range)
        vocab_size = tokenizer.vocab_size
        preds = np.where(
            (preds >= 0) & (preds < vocab_size),
            preds,
            tokenizer.pad_token_id
        )

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Clean up
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        # Compute metrics
        result_bleu = bleu_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )
        result_chrf = chrf_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
        )

        return {
            "bleu": result_bleu["score"],
            "chrf": result_chrf["score"]
        }

    return compute_metrics
