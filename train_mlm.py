#!/usr/bin/env python3
"""
Phase 1: MLM Pre-training for Sumerian Encoder

Trains a RoBERTa model on the Sumerian monolingual corpus.
This teaches the model Sumerian grammar (ergativity, cases) before
it ever sees English.

Usage:
    # Single GPU
    python3.11 train_mlm.py

    # Multi-GPU (recommended for 2x H100)
    torchrun --nproc_per_node=2 train_mlm.py

    # With custom model size
    python3.11 train_mlm.py --model-size base

Prerequisites:
    1. Run convert_tokenizer.py first to create HF tokenizer
    2. Ensure output_training_v2_clean/pretrain/corpus_monolingual.txt exists
"""

import argparse
import os
import warnings
from pathlib import Path

# Prevent tokenizer parallelism deadlock with multiple data workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast,
)
from datasets import disable_caching

# Import from common utilities
from common.hardware import get_hardware_info, print_hardware_summary, is_main_process
from common.training import apply_compile, setup_precision
from config import Paths, ModelConfigs, TrainingDefaults

# Disable HuggingFace datasets caching globally (required for DDP)
disable_caching()


class SimpleMLMDataset(torch.utils.data.Dataset):
    """Simple torch Dataset for MLM - more reliable for DDP than HF datasets."""

    def __init__(self, encodings):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.special_tokens_mask = encodings.get('special_tokens_mask', None)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx].tolist(),
            'attention_mask': self.attention_mask[idx].tolist(),
        }
        if self.special_tokens_mask is not None:
            item['special_tokens_mask'] = self.special_tokens_mask[idx].tolist()
        return item


def load_text_dataset(file_path: Path, tokenizer, max_length: int = 128):
    """Load and tokenize text file - returns a simple torch Dataset for DDP compatibility."""
    lines = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                lines.append(text)

    if is_main_process():
        print(f"  Loaded {len(lines):,} lines from {file_path}")
        print("  Tokenizing...")

    encodings = tokenizer(
        lines,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_special_tokens_mask=True,
        return_tensors="pt",
    )

    dataset = SimpleMLMDataset(encodings)
    if is_main_process():
        print(f"  Created dataset with {len(dataset):,} examples")

    return dataset


def parse_args():
    """Parse command line arguments."""
    defaults = TrainingDefaults.MLM

    parser = argparse.ArgumentParser(
        description="MLM Pre-training for Sumerian Encoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-size", choices=list(ModelConfigs.SIZES.keys()), default="tiny",
        help="Model size configuration"
    )
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
    parser.add_argument(
        "--no-compile", action="store_true",
        help="Disable torch.compile()"
    )
    parser.add_argument(
        "--no-flash-attn", action="store_true",
        help="Disable Flash Attention 2"
    )
    parser.add_argument(
        "--no-gradient-checkpointing", action="store_true",
        help="Disable gradient checkpointing"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(Paths.MLM_CHECKPOINT),
        help="Output directory for checkpoints"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if is_main_process():
        print("=" * 70)
        print("Phase 1: Sumerian MLM Pre-training")
        print("=" * 70)

    # Detect hardware
    hw = get_hardware_info()
    if is_main_process():
        print_hardware_summary(hw)

    # Check prerequisites
    if not Paths.TOKENIZER.exists():
        print(f"\nERROR: Tokenizer not found at {Paths.TOKENIZER}")
        print("Run 'python convert_tokenizer.py' first.")
        return

    if not Paths.MONOLINGUAL_CORPUS.exists():
        print(f"\nERROR: Training file not found at {Paths.MONOLINGUAL_CORPUS}")
        return

    # 1. Load Sumerian Tokenizer
    if is_main_process():
        print(f"\n[1/5] Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(Paths.TOKENIZER)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"
    if tokenizer.mask_token is None:
        tokenizer.mask_token = "<mask>"

    if is_main_process():
        print(f"  Vocab size: {tokenizer.vocab_size}")

    # 2. Define Model Architecture
    model_cfg = ModelConfigs.SIZES[args.model_size]
    if is_main_process():
        print(f"\n[2/5] Initializing RoBERTa-{args.model_size} architecture...")

    # Enable Flash Attention 2 if available
    attn_implementation = None
    if hw['flash_attn_available'] and not args.no_flash_attn:
        attn_implementation = "flash_attention_2"
        if is_main_process():
            print("  Using Flash Attention 2")

    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size + 10,
        max_position_embeddings=514,
        num_attention_heads=model_cfg["num_attention_heads"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        hidden_size=model_cfg["hidden_size"],
        intermediate_size=model_cfg["intermediate_size"],
        type_vocab_size=1,
        pad_token_id=tokenizer.pad_token_id,
    )

    if attn_implementation:
        model = RobertaForMaskedLM._from_config(config, attn_implementation=attn_implementation)
    else:
        model = RobertaForMaskedLM(config=config)

    # Enable gradient checkpointing
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if is_main_process():
            print("  Gradient checkpointing: enabled")

    num_params = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(f"  Parameters: {num_params:,} (~{num_params/1e6:.1f}M)")

    # 3. Apply torch.compile()
    if is_main_process():
        print(f"\n[3/5] Optimization...")
    model = apply_compile(model, enabled=not args.no_compile, verbose=is_main_process())

    # 4. Prepare Data
    if is_main_process():
        print(f"\n[4/5] Loading training data...")
    dataset = load_text_dataset(Paths.MONOLINGUAL_CORPUS, tokenizer, args.max_length)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        pad_to_multiple_of=8,
    )

    # 5. Training Arguments
    if is_main_process():
        print(f"\n[5/5] Configuring training...")

    precision = setup_precision(hw)
    effective_batch = args.batch_size * max(1, hw['gpu_count']) * args.grad_accum

    if is_main_process():
        print(f"  Epochs: {args.epochs}")
        print(f"  Per-device batch: {args.batch_size}")
        print(f"  Effective batch: {effective_batch}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Precision: {'BF16' if precision['bf16'] else 'FP16' if precision['fp16'] else 'FP32'}")
        print(f"  Output: {args.output_dir}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,

        # Precision
        bf16=precision['bf16'],
        fp16=precision['fp16'],

        # Optimizer
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",

        # Data loading
        dataloader_num_workers=0,
        dataloader_pin_memory=True,

        # Checkpointing
        save_steps=500,
        save_total_limit=3,

        # Logging
        logging_steps=50,
        report_to="none",

        # DDP
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=args.grad_accum,
    )

    # Train
    if is_main_process():
        print("\n" + "-" * 70)
        print("Starting training...")
        print("-" * 70)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()

    # Save
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if is_main_process():
        print("\n" + "=" * 70)
        print("PRE-TRAINING COMPLETE")
        print("=" * 70)
        print(f"Encoder saved to: {args.output_dir}")
        print("\nNext step: Run 'python train_nmt.py' for translation fine-tuning")


if __name__ == "__main__":
    main()
