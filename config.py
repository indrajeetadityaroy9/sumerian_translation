"""
Centralized configuration for Sumerian NMT training pipeline.

Provides unified paths, model configurations, and training defaults
used across all training and processing scripts.
"""

import os
from pathlib import Path


class Paths:
    """Centralized path configuration."""

    # Root directory (where this file is located)
    ROOT = Path(__file__).parent

    # Model checkpoints
    MODELS = ROOT / "models"
    NMT_CHECKPOINT = MODELS / "sumerian_mt5_final"

    # Data directories
    OUTPUT = ROOT / "output"
    TRAINING_DATA = ROOT / "output_training_v2_clean"

    # Training data files
    TRAIN_FILE = TRAINING_DATA / "finetune" / "train_augmented.jsonl"
    TRAIN_FILE_V2 = TRAINING_DATA / "finetune" / "train_augmented_v2.jsonl"
    VALID_FILE = TRAINING_DATA / "finetune" / "valid.jsonl"
    MONOLINGUAL_CORPUS = TRAINING_DATA / "pretrain" / "corpus_monolingual.txt"

    # Tokenizer
    TOKENIZER = TRAINING_DATA / "tokenizer_hf"
    TOKENIZER_SP = TRAINING_DATA / "tokenizer"  # SentencePiece format

    # Extracted corpora
    ETCSL_CORPUS = OUTPUT / "parallel_corpus.jsonl"
    ORACC_SYNTHETIC = OUTPUT / "oracc_synthetic_combined.jsonl"

    # Raw data
    ORACC_DATA = ROOT / "data" / "oracc"
    ETCSL_DATA = ROOT / "ota_20" / "etcsl"

    # Consolidated Parquet data (preferred format)
    CONSOLIDATED_DIR = ROOT / "data" / "consolidated"
    ETCSL_PARQUET = CONSOLIDATED_DIR / "etcsl_gold.parquet"
    ORACC_LITERARY_PARQUET = CONSOLIDATED_DIR / "oracc_literary.parquet"
    ORACC_ROYAL_PARQUET = CONSOLIDATED_DIR / "oracc_royal.parquet"
    GLOSSARY_PARQUET = CONSOLIDATED_DIR / "glossary_sux.parquet"

    # Archives
    ARCHIVES_DIR = ROOT / "archives"

    @classmethod
    def ensure_dirs(cls):
        """Create all output directories if they don't exist."""
        cls.MODELS.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT.mkdir(parents=True, exist_ok=True)
        (cls.TRAINING_DATA / "finetune").mkdir(parents=True, exist_ok=True)
        (cls.TRAINING_DATA / "pretrain").mkdir(parents=True, exist_ok=True)
        cls.CONSOLIDATED_DIR.mkdir(parents=True, exist_ok=True)
        cls.ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)


class ModelConfigs:
    """Model architecture configurations."""

    # mT5 model variants
    MT5_MODELS = {
        "small": "google/mt5-small",    # 300M params
        "base": "google/mt5-base",      # 580M params
        "large": "google/mt5-large",    # 1.2B params
    }


class TrainingDefaults:
    """Default training hyperparameters for mT5 fine-tuning."""

    # mT5 fine-tuning
    MT5 = {
        "batch_size": 64,
        "epochs": 20,
        "learning_rate": 3e-5,
        "max_length": 128,
        "warmup_ratio": 0.06,
        "task_prefix": "translate Sumerian to English: ",
    }


class EvalTargets:
    """Expected evaluation metric ranges."""

    # Low-resource NMT baseline targets
    BLEU_MIN = 15.0
    BLEU_MAX = 25.0

    # chrF is more reliable for morphological languages
    CHRF_MIN = 30.0
    CHRF_MAX = 40.0

    # Word alignment (using ORACC glosses as reference)
    ALIGNMENT_F1 = 0.6


def get_train_file() -> Path:
    """Get the appropriate training file (v2 if exists, else v1)."""
    if Paths.TRAIN_FILE_V2.exists():
        return Paths.TRAIN_FILE_V2
    return Paths.TRAIN_FILE


def get_model_checkpoint() -> Path:
    """Get the best available model checkpoint."""
    # Prefer mT5 models
    mt5_paths = [
        Paths.MODELS / "sumerian_mt5_final",
        Paths.MODELS / "sumerian_mt5_continued",
        Paths.MODELS / "sumerian_mt5",
    ]
    for path in mt5_paths:
        if path.exists():
            return path

    # Fall back to NMT checkpoint
    if Paths.NMT_CHECKPOINT.exists():
        return Paths.NMT_CHECKPOINT

    # Return default (may not exist)
    return Paths.NMT_CHECKPOINT
