#!/usr/bin/env python3
"""
Unified NMT Training Script - mT5-based Sumerian-English Translation

Consolidates all mT5 training approaches into a single, clean interface.
Uses HuggingFace Trainer with automatic DDP support via torchrun.

Key features:
- mT5 multilingual model (already knows 101+ languages)
- BF16 precision (optimized for H100)
- Automatic multi-GPU via torchrun
- Label smoothing for better generation
- Early stopping on validation BLEU
- Checkpoint resumption

Usage:
    # Basic training
    python3.11 train_nmt.py

    # With options
    python3.11 train_nmt.py --model google/mt5-base
    python3.11 train_nmt.py --resume models/checkpoint
    python3.11 train_nmt.py --label-smoothing 0.1
    python3.11 train_nmt.py --early-stopping 5

    # Multi-GPU (automatic via torchrun)
    torchrun --nproc_per_node=2 train_nmt.py --model google/mt5-base
"""

import argparse
import os
import pickle
import warnings
from pathlib import Path

# Prevent tokenizer parallelism deadlock with multiple workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# Import from common utilities
from common.io import load_jsonl
from common.hardware import get_hardware_info, print_hardware_summary, is_main_process
from common.metrics import load_metrics, create_compute_metrics_fn
from common.training import apply_compile
from config import Paths, TrainingDefaults


class SimpleMT5Dataset(torch.utils.data.Dataset):
    """Pure PyTorch dataset to bypass HuggingFace Arrow DDP bugs."""

    def __init__(self, data_dict):
        self.input_ids = data_dict['input_ids']
        self.attention_mask = data_dict['attention_mask']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
        }


def parse_args():
    """Parse command line arguments."""
    defaults = TrainingDefaults.MT5

    parser = argparse.ArgumentParser(
        description="mT5 NMT Training for Sumerian-English Translation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model selection
    parser.add_argument(
        "--model", type=str, default="google/mt5-small",
        help="mT5 variant: google/mt5-small (300M), google/mt5-base (580M), google/mt5-large (1.2B)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint directory"
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size", type=int, default=defaults["batch_size"],
        help="Per-device batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=defaults["epochs"],
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=defaults["learning_rate"],
        help="Learning rate"
    )
    parser.add_argument(
        "--max-length", type=int, default=defaults["max_length"],
        help="Maximum sequence length"
    )

    # Advanced training options
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0,
        help="Label smoothing factor (0.0-0.2)"
    )
    parser.add_argument(
        "--early-stopping", type=int, default=0,
        help="Early stopping patience (0 to disable)"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=defaults["warmup_ratio"],
        help="Warmup proportion of training"
    )

    # Optimization flags
    parser.add_argument(
        "--no-compile", action="store_true",
        help="Disable torch.compile()"
    )
    parser.add_argument(
        "--no-gradient-checkpointing", action="store_true",
        help="Disable gradient checkpointing"
    )

    # Data and output
    parser.add_argument(
        "--train-file", type=str, default=None,
        help="Training data file (JSONL)"
    )
    parser.add_argument(
        "--valid-file", type=str, default=None,
        help="Validation data file (JSONL)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(Paths.NMT_CHECKPOINT),
        help="Output directory for checkpoints"
    )

    return parser.parse_args()


def load_data(args, tokenizer, hw_info):
    """Load and prepare training data."""
    task_prefix = TrainingDefaults.MT5["task_prefix"]
    max_length = args.max_length

    # Determine data paths
    train_path = Path(args.train_file) if args.train_file else Paths.TRAIN_FILE_V2
    valid_path = Path(args.valid_file) if args.valid_file else Paths.VALID_FILE

    if not train_path.exists():
        train_path = Paths.TRAIN_FILE
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    # Check for pre-processed pickle data (DDP-safe)
    train_pickle = train_path.parent / "train_data.pkl"
    valid_pickle = valid_path.parent / "valid_data.pkl"

    if train_pickle.exists() and valid_pickle.exists():
        if is_main_process():
            print(f"  Loading pre-tokenized data (DDP-safe)...")
        with open(train_pickle, 'rb') as f:
            train_pkl = pickle.load(f)
        with open(valid_pickle, 'rb') as f:
            valid_pkl = pickle.load(f)

        train_ds = SimpleMT5Dataset(train_pkl)
        valid_ds = SimpleMT5Dataset(valid_pkl)

        # Load raw data for sample predictions
        raw_valid = load_jsonl(valid_path)
        valid_data = [{
            "source_text": item["source"]["text_normalized"],
            "target_text": item["target"]["text"],
        } for item in raw_valid]

    else:
        # Tokenize on-the-fly
        if is_main_process():
            print(f"  Tokenizing data on-the-fly...")

        raw_train = load_jsonl(train_path)
        raw_valid = load_jsonl(valid_path)

        def preprocess(raw_data):
            return [{
                "source_text": item["source"]["text_normalized"],
                "target_text": item["target"]["text"],
            } for item in raw_data]

        def tokenize(examples):
            inputs = [task_prefix + text for text in examples["source_text"]]
            model_inputs = tokenizer(
                inputs, max_length=max_length, truncation=True, padding=False
            )
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    examples["target_text"], max_length=max_length,
                    truncation=True, padding=False
                )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        num_proc = min(os.cpu_count() or 1, 8)
        train_data = preprocess(raw_train)
        valid_data = preprocess(raw_valid)

        train_ds = Dataset.from_list(train_data).map(
            tokenize, batched=True, num_proc=num_proc,
            remove_columns=["source_text", "target_text"]
        )
        valid_ds = Dataset.from_list(valid_data).map(
            tokenize, batched=True, num_proc=num_proc,
            remove_columns=["source_text", "target_text"]
        )

    if is_main_process():
        print(f"  Training examples: {len(train_ds):,}")
        print(f"  Validation examples: {len(valid_ds):,}")

    return train_ds, valid_ds, valid_data


def main():
    args = parse_args()

    # Header
    if is_main_process():
        print("=" * 70)
        print("Sumerian-English NMT Training (mT5)")
        print("=" * 70)

    # Hardware detection
    hw = get_hardware_info()
    if is_main_process():
        print_hardware_summary(hw)

    # Precision settings
    use_bf16 = hw['bf16_supported'] and hw['cuda_available']

    # 1. Load model and tokenizer
    if is_main_process():
        print(f"\n[1/5] Loading model...")
        print(f"  Model: {args.model}")

    model_path = args.resume if args.resume else args.model
    tokenizer = AutoTokenizer.from_pretrained(
        args.resume if args.resume else args.model
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Enable gradient checkpointing
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if is_main_process():
            print("  Gradient checkpointing: enabled")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(f"  Parameters: {total:,} total, {trainable:,} trainable")

    # 2. Apply torch.compile()
    if is_main_process():
        print(f"\n[2/5] Optimization...")
    model = apply_compile(model, enabled=not args.no_compile, verbose=is_main_process())

    # 3. Load data
    if is_main_process():
        print(f"\n[3/5] Loading data...")
    train_ds, valid_ds, valid_data = load_data(args, tokenizer, hw)

    # 4. Setup metrics
    if is_main_process():
        print(f"\n[4/5] Setting up metrics...")
    bleu_metric, chrf_metric = load_metrics()
    compute_metrics = create_compute_metrics_fn(tokenizer, bleu_metric, chrf_metric)

    # 5. Training setup
    if is_main_process():
        print(f"\n[5/5] Configuring training...")

    effective_batch = args.batch_size * max(1, hw['gpu_count']) * args.grad_accum
    if is_main_process():
        print(f"  Epochs: {args.epochs}")
        print(f"  Per-device batch: {args.batch_size}")
        print(f"  Effective batch: {effective_batch}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Precision: {'BF16' if use_bf16 else 'FP32'}")
        if args.label_smoothing > 0:
            print(f"  Label smoothing: {args.label_smoothing}")
        if args.early_stopping > 0:
            print(f"  Early stopping: patience={args.early_stopping}")
        print(f"  Output: {args.output_dir}")

    # Create training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,

        # Batch settings
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        # Generation
        predict_with_generate=True,
        generation_max_length=args.max_length,
        generation_num_beams=4,

        # Training
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        label_smoothing_factor=args.label_smoothing,

        # Precision (mT5 needs BF16, not FP16)
        bf16=use_bf16,
        fp16=False,

        # Data loading
        dataloader_num_workers=0,
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
    )

    # Callbacks
    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping))

    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
        ),
        compute_metrics=compute_metrics,
        callbacks=callbacks if callbacks else None,
    )

    # Train
    if is_main_process():
        print("\n" + "-" * 70)
        print("Starting training...")
        print("-" * 70)

    if args.resume and Path(args.resume).exists():
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Summary
    if is_main_process():
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Model saved to: {args.output_dir}")

        # Sample predictions
        print("\n" + "=" * 70)
        print("SAMPLE PREDICTIONS")
        print("=" * 70)

        model.eval()
        task_prefix = TrainingDefaults.MT5["task_prefix"]
        sample_inputs = [
            task_prefix + valid_data[i]["source_text"]
            for i in range(min(3, len(valid_data)))
        ]
        inputs = tokenizer(
            sample_inputs, return_tensors="pt", padding=True,
            truncation=True, max_length=args.max_length
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=args.max_length, num_beams=4)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for i, pred in enumerate(predictions):
            print(f"\nExample {i+1}:")
            print(f"  Sumerian: {valid_data[i]['source_text']}")
            print(f"  Predicted: {pred}")
            print(f"  Reference: {valid_data[i]['target_text']}")


if __name__ == "__main__":
    main()
