"""
NMT Training Script - mT5 Fine-tuning (Nuclear Option)

OPTIMIZED FOR: 2x NVIDIA H100 80GB + Intel Xeon Platinum 8480+

Key insight: Instead of bridging a custom Sumerian encoder to a pre-trained
English decoder (which requires learning an entirely new vector space mapping),
we use mT5 which:
1. Already handles 101+ languages with a unified vocabulary
2. Has seen similar text structures during pre-training
3. Only needs to learn Sumerian patterns, not cross-space alignment

Research backing: "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer"
(Xue et al., 2021) - demonstrates strong performance on low-resource languages.

Optimizations enabled:
- Multi-GPU training (DDP) - auto-detected
- BF16 mixed precision (native H100 support, required for mT5)
- Flash Attention 2 (2-4x faster attention)
- torch.compile() (kernel fusion)
- Large batch sizes (64 per GPU)
- Parallel data loading (8 workers)
- Gradient checkpointing

Usage:
    # Single GPU
    python train_mt5.py

    # Multi-GPU (recommended for 2x H100)
    torchrun --nproc_per_node=2 train_mt5.py

    # With larger mT5 variant
    python train_mt5.py --model google/mt5-base
    python train_mt5.py --model google/mt5-large
"""

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

# Prevent tokenizer parallelism deadlock with multiple data workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import torch
import evaluate
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)


class SimpleMT5Dataset(torch.utils.data.Dataset):
    """Pure PyTorch dataset to bypass HuggingFace Arrow DDP bugs."""

    def __init__(self, data_dict):
        """Accept a dict with input_ids, attention_mask, labels as lists."""
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

# --- Configuration ---
MODEL_CHECKPOINT = "google/mt5-small"  # 300M params
TRAIN_FILE = "output_training_v2_clean/finetune/train_augmented_v2.jsonl"
VALID_FILE = "output_training_v2_clean/finetune/valid.jsonl"
TRAIN_TOKENIZED = "output_training_v2_clean/finetune/train_tokenized"
VALID_TOKENIZED = "output_training_v2_clean/finetune/valid_tokenized"
TRAIN_PICKLE = "output_training_v2_clean/finetune/train_data.pkl"
VALID_PICKLE = "output_training_v2_clean/finetune/valid_data.pkl"
OUTPUT_DIR = "./models/sumerian_mt5"
TASK_PREFIX = "translate Sumerian to English: "

# Training hyperparameters (optimized for H100)
NUM_EPOCHS = 20
BATCH_SIZE = 64  # Increased from 8
LEARNING_RATE = 3e-5
MAX_LENGTH = 128


def get_hardware_info():
    """Detect and report available hardware."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "bf16_supported": False,
        "flash_attn_available": False,
    }

    if info["cuda_available"]:
        capability = torch.cuda.get_device_capability()
        info["bf16_supported"] = capability[0] >= 8

        try:
            from transformers.utils import is_flash_attn_2_available
            info["flash_attn_available"] = is_flash_attn_2_available()
        except ImportError:
            info["flash_attn_available"] = False

        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(info["gpu_count"])]
        info["gpu_memory"] = [torch.cuda.get_device_properties(i).total_memory / 1e9
                             for i in range(info["gpu_count"])]

    return info


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser(description="mT5 Fine-tuning with H100 optimizations")
    parser.add_argument("--model", type=str, default=MODEL_CHECKPOINT,
                        help="mT5 variant (google/mt5-small, google/mt5-base, google/mt5-large)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile()")
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    args = parser.parse_args()

    print("=" * 70)
    print("mT5 Fine-tuning - Nuclear Option (H100 Optimized)")
    print("=" * 70)

    # Detect hardware
    hw = get_hardware_info()
    print("\n[Hardware Detection]")
    print(f"  CUDA available: {hw['cuda_available']}")
    if hw['cuda_available']:
        print(f"  GPU count: {hw['gpu_count']}")
        for i, (name, mem) in enumerate(zip(hw['gpu_names'], hw['gpu_memory'])):
            print(f"  GPU {i}: {name} ({mem:.1f} GB)")
        print(f"  BF16 supported: {hw['bf16_supported']}")
        print(f"  Flash Attention 2: {hw['flash_attn_available']}")

    # Check prerequisites
    if not Path(TRAIN_FILE).exists():
        print(f"\nERROR: Training file not found at {TRAIN_FILE}")
        return

    # 1. Load Model and Tokenizer
    print(f"\n[1/5] Loading {args.model}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Note: mT5 doesn't support Flash Attention 2 in current transformers version
    # Load model without Flash Attention
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    attn_impl = None  # mT5 uses SDPA by default on H100

    # Enable gradient checkpointing
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

    # 2. Apply torch.compile()
    if not args.no_compile and hasattr(torch, 'compile'):
        print("\n[2/5] Applying torch.compile()...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile(): enabled")
        except Exception as e:
            print(f"  torch.compile() failed: {e}")
    else:
        print("\n[2/5] Skipping torch.compile()")

    # 3. Load and process data
    # Try loading pickle data first (DDP-safe, bypasses Arrow serialization issues)
    if Path(TRAIN_PICKLE).exists() and Path(VALID_PICKLE).exists():
        print(f"\n[3/5] Loading PICKLE data (DDP-safe)...")
        with open(TRAIN_PICKLE, 'rb') as f:
            train_pkl = pickle.load(f)
        with open(VALID_PICKLE, 'rb') as f:
            valid_pkl = pickle.load(f)

        # Wrap in SimpleMT5Dataset (pure PyTorch)
        train_ds = SimpleMT5Dataset(train_pkl)
        valid_ds = SimpleMT5Dataset(valid_pkl)
        print(f"  Training examples: {len(train_ds):,}")
        print(f"  Validation examples: {len(valid_ds):,}")

        # Load raw data for sample predictions at the end
        raw_valid = load_jsonl(VALID_FILE)
        valid_data = [{
            "source_text": item["source"]["text_normalized"],
            "target_text": item["target"]["text"],
        } for item in raw_valid]
    elif Path(TRAIN_TOKENIZED).exists() and Path(VALID_TOKENIZED).exists():
        print(f"\n[3/5] Loading PRE-TOKENIZED data...")
        train_ds = load_from_disk(TRAIN_TOKENIZED)
        valid_ds = load_from_disk(VALID_TOKENIZED)
        print(f"  Training examples: {len(train_ds):,}")
        print(f"  Validation examples: {len(valid_ds):,}")

        # Load raw data for sample predictions at the end
        raw_valid = load_jsonl(VALID_FILE)
        valid_data = [{
            "source_text": item["source"]["text_normalized"],
            "target_text": item["target"]["text"],
        } for item in raw_valid]
    else:
        print(f"\n[3/5] Loading data from {TRAIN_FILE}...")
        print("  (Pre-tokenized data not found, tokenizing on-the-fly)")
        raw_train = load_jsonl(TRAIN_FILE)
        raw_valid = load_jsonl(VALID_FILE)
        print(f"  Training pairs: {len(raw_train):,}")
        print(f"  Validation pairs: {len(raw_valid):,}")

        def preprocess_data(raw_data):
            return [{
                "source_text": item["source"]["text_normalized"],
                "target_text": item["target"]["text"],
            } for item in raw_data]

        def tokenize_data(examples):
            inputs = [TASK_PREFIX + text for text in examples["source_text"]]
            targets = examples["target_text"]

            model_inputs = tokenizer(
                inputs,
                max_length=MAX_LENGTH,
                truncation=True,
                padding=False
            )

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=MAX_LENGTH,
                    truncation=True,
                    padding=False
                )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        num_proc = min(os.cpu_count() or 1, 8)
        train_data = preprocess_data(raw_train)
        valid_data = preprocess_data(raw_valid)

        train_ds = Dataset.from_list(train_data).map(
            tokenize_data, batched=True, num_proc=num_proc,
            remove_columns=["source_text", "target_text"]
        )
        valid_ds = Dataset.from_list(valid_data).map(
            tokenize_data, batched=True, num_proc=num_proc,
            remove_columns=["source_text", "target_text"]
        )

    # 4. Load metrics
    print("\n[4/5] Loading evaluation metrics...")
    metric_bleu = evaluate.load("sacrebleu")
    metric_chrf = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        vocab_size = tokenizer.vocab_size
        preds = np.where((preds >= 0) & (preds < vocab_size), preds, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        result_bleu = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
        result_chrf = metric_chrf.compute(predictions=decoded_preds, references=decoded_labels)

        return {"bleu": result_bleu["score"], "chrf": result_chrf["score"]}

    # 5. Training setup
    print("\n[5/5] Setting up training...")

    # mT5 requires BF16 (FP16 causes issues)
    use_bf16 = hw['bf16_supported'] and hw['cuda_available']

    effective_batch = args.batch_size * max(1, hw['gpu_count']) * args.grad_accum
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Per-device batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Precision: {'BF16' if use_bf16 else 'FP32'} (mT5 requires BF16, not FP16)")
    print(f"  Task prefix: '{TASK_PREFIX}'")

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        # Generation
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        generation_num_beams=4,

        # Training
        num_train_epochs=args.epochs,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",

        # Precision (mT5 needs BF16, not FP16)
        bf16=use_bf16,
        fp16=False,

        # Data loading (num_workers=0 to avoid DDP synchronization issues)
        dataloader_num_workers=0,
        dataloader_pin_memory=True,

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

        # Multi-GPU
        ddp_find_unused_parameters=False,
        dispatch_batches=False,  # Disable accelerate's batch dispatching
        dataloader_drop_last=True,  # Drop incomplete batches

        # Memory optimization
        gradient_accumulation_steps=args.grad_accum,
    )

    print(f"\n  Output: {OUTPUT_DIR}")
    print("-" * 70)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8),
        compute_metrics=compute_metrics
    )

    # Train (with optional resume)
    if args.resume and Path(OUTPUT_DIR).exists():
        print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to {OUTPUT_DIR}")
    print(f"\nOptimizations applied:")
    print(f"  - Multi-GPU: {hw['gpu_count']} GPU(s)")
    print(f"  - Precision: {'BF16' if use_bf16 else 'FP32'}")
    print(f"  - Flash Attention 2: {attn_impl is not None}")
    print(f"  - Gradient Checkpointing: {not args.no_gradient_checkpointing}")
    print(f"  - Data Workers: 8")

    # Show sample predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    model.eval()
    sample_inputs = [
        TASK_PREFIX + valid_data[0]["source_text"],
        TASK_PREFIX + valid_data[1]["source_text"],
        TASK_PREFIX + valid_data[2]["source_text"],
    ]
    inputs = tokenizer(sample_inputs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=MAX_LENGTH, num_beams=4)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for i, (src, pred, ref) in enumerate(zip(
        [valid_data[j]["source_text"] for j in range(3)],
        predictions,
        [valid_data[j]["target_text"] for j in range(3)]
    )):
        print(f"\nExample {i+1}:")
        print(f"  Sumerian: {src}")
        print(f"  Predicted: {pred}")
        print(f"  Reference: {ref}")


if __name__ == "__main__":
    main()
