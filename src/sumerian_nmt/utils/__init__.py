"""
Common utilities for Sumerian NMT training pipeline.

Provides shared functionality across all training scripts:
- io: JSONL and Parquet file operations
- hardware: GPU detection and setup
- metrics: BLEU/chrF evaluation
- quality: Data quality filters
- text: Text cleaning and normalization
"""

from .io import load_jsonl, save_jsonl, iter_jsonl, load_text_corpus
from .hardware import get_hardware_info, setup_device, print_hardware_summary, is_main_process
from .metrics import compute_bleu_chrf, load_metrics, create_compute_metrics_fn
from .quality import is_mostly_broken, is_valid_length, filter_duplicates, is_valid_pair
from .text import clean_source_text

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
    # quality
    "is_mostly_broken",
    "is_valid_length",
    "filter_duplicates",
    "is_valid_pair",
    # text
    "clean_source_text",
]
