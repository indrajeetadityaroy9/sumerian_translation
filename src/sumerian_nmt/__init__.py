"""
Sumerian NMT: Neural Machine Translation for Ancient Sumerian

A low-resource NMT system using graph-based entity substitution augmentation
and Llama-3 fine-tuning with LoRA.

Key modules:
- graph_augmentation: Novel two-circle entity substitution (key contribution)
- data_ingestion: ETCSL/ORACC corpus extraction
- data_processing: Training data preparation
- evaluation: BLEU/chrF/Named Entity metrics
- utils: Shared utilities
"""

__version__ = "1.0.0"

from .config import Paths, LLMConfig, ControlTokens, EvalTargets

__all__ = [
    "__version__",
    "Paths",
    "LLMConfig",
    "ControlTokens",
    "EvalTargets",
]
