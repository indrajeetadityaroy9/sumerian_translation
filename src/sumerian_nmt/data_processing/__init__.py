"""
Data Processing: Training data preparation for LLM fine-tuning.

Components:
- splitter: Composition-based train/valid splitting (prevents data leakage)
- alpaca_formatter: Convert to Alpaca format for SFT
- dpo_generator: Create preference pairs for DPO alignment
- normalization: ETCSL/ORACC character normalization
"""

__all__ = []
