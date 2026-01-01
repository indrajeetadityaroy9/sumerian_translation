"""
Common utilities for Sumerian NMT training pipeline.

Provides shared functionality across all training scripts:
- io: JSONL file operations
- hardware: GPU detection and setup
- metrics: BLEU/chrF evaluation
- training: Training argument builders
- quality: Data quality filters
"""

from .io import load_jsonl, save_jsonl, iter_jsonl, load_text_corpus
from .hardware import get_hardware_info, setup_device, print_hardware_summary, is_main_process
from .metrics import compute_bleu_chrf, load_metrics, create_compute_metrics_fn
from .training import (
    create_training_args,
    create_seq2seq_training_args,
    create_mlm_training_args,
    apply_compile,
    setup_precision,
)
from .quality import is_mostly_broken, is_valid_length, filter_duplicates, is_valid_pair

__all__ = [
    # io
    "load_jsonl",
    "save_jsonl",
    "iter_jsonl",
    "load_text_corpus",
    # hardware
    "get_hardware_info",
    "setup_device",
    "print_hardware_summary",
    "is_main_process",
    # metrics
    "compute_bleu_chrf",
    "load_metrics",
    "create_compute_metrics_fn",
    # training
    "create_training_args",
    "create_seq2seq_training_args",
    "create_mlm_training_args",
    "apply_compile",
    "setup_precision",
    # quality
    "is_mostly_broken",
    "is_valid_length",
    "filter_duplicates",
    "is_valid_pair",
]
