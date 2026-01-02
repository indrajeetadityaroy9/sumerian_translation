"""
Training utilities and argument builders.

Provides consistent training configuration across scripts.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch


def setup_precision(hw_info: Dict[str, Any]) -> Dict[str, bool]:
    """
    Determine precision settings based on hardware.

    Args:
        hw_info: Output from get_hardware_info()

    Returns:
        Dictionary with 'bf16' and 'fp16' boolean flags
    """
    # BF16 is preferred on Ampere+ GPUs (compute capability >= 8)
    use_bf16 = hw_info.get('bf16_supported', False) and hw_info.get('cuda_available', False)

    # FP16 as fallback for older GPUs
    use_fp16 = hw_info.get('cuda_available', False) and not use_bf16

    return {
        "bf16": use_bf16,
        "fp16": use_fp16
    }


def apply_compile(
    model,
    enabled: bool = True,
    mode: str = "reduce-overhead",
    verbose: bool = True
):
    """
    Apply torch.compile() to model if available.

    Args:
        model: PyTorch model
        enabled: Whether to apply compilation
        mode: Compilation mode (reduce-overhead, max-autotune, default)
        verbose: Print status

    Returns:
        Compiled model (or original if compilation fails/disabled)
    """
    if not enabled:
        if verbose:
            print("  torch.compile(): disabled")
        return model

    if not hasattr(torch, 'compile'):
        if verbose:
            print("  torch.compile(): not available (PyTorch < 2.0)")
        return model

    try:
        compiled = torch.compile(model, mode=mode)
        if verbose:
            print(f"  torch.compile(): enabled ({mode} mode)")
        return compiled
    except Exception as e:
        if verbose:
            print(f"  torch.compile() failed: {e}")
        return model


def create_seq2seq_training_args(
    output_dir: Union[str, Path],
    batch_size: int = 64,
    epochs: int = 20,
    learning_rate: float = 3e-5,
    max_length: int = 128,
    hw_info: Optional[Dict[str, Any]] = None,
    gradient_accumulation: int = 1,
    label_smoothing: float = 0.0,
    warmup_ratio: float = 0.06,
    **kwargs
):
    """
    Create Seq2SeqTrainingArguments with sensible defaults.

    Args:
        output_dir: Directory for checkpoints
        batch_size: Per-device batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        max_length: Maximum sequence length for generation
        hw_info: Hardware info for precision settings
        gradient_accumulation: Gradient accumulation steps
        label_smoothing: Label smoothing factor
        warmup_ratio: Warmup proportion
        **kwargs: Additional arguments passed to TrainingArguments

    Returns:
        Seq2SeqTrainingArguments instance
    """
    from transformers import Seq2SeqTrainingArguments

    # Determine precision
    precision = {"bf16": False, "fp16": False}
    if hw_info:
        precision = setup_precision(hw_info)

    return Seq2SeqTrainingArguments(
        output_dir=str(output_dir),

        # Batch settings
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,

        # Generation
        predict_with_generate=True,
        generation_max_length=max_length,
        generation_num_beams=4,

        # Training
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        label_smoothing_factor=label_smoothing,

        # Precision
        bf16=precision["bf16"],
        fp16=precision["fp16"],

        # Data loading
        dataloader_num_workers=0,  # Avoid DDP issues
        dataloader_pin_memory=True,
        dataloader_drop_last=True,

        # Evaluation and saving
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        metric_for_best_model="bleu",
        load_best_model_at_end=True,
        greater_is_better=True,

        # Logging
        logging_steps=25,
        report_to="none",

        # DDP
        ddp_find_unused_parameters=False,
        dispatch_batches=False,

        **kwargs
    )


# Alias for backward compatibility
create_training_args = create_seq2seq_training_args
