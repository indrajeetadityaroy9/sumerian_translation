"""
Evaluation: LLM evaluation with WMT-standard metrics.

Components:
- llm_evaluator: Main evaluation entry point (BLEU, chrF++, NE accuracy)
- vllm_inference: High-throughput inference with vLLM
- model_comparison: Cross-checkpoint metric comparison
- error_analysis: Systematic error categorization
- significance_test: Statistical significance testing (bootstrap)
"""

__all__ = []
