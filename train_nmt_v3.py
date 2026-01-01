"""
NMT Training Script v3 - DistilGPT-2 Decoder

OPTIMIZED FOR: 2x NVIDIA H100 80GB + Intel Xeon Platinum 8480+

Key change: Replace BERT decoder with DistilGPT-2
- GPT-2 is a *generative* model trained to produce fluent text
- BERT is an *encoder* being forced to generate (not its strength)
- DistilGPT-2 already knows how to generate English sentences

Research backing: "Warm-Starting Seq2Seq" (Rothe et al., 2020)

Optimizations enabled:
- Multi-GPU training (DDP) - auto-detected
- BF16 mixed precision (native H100 support)
- Flash Attention 2 (2-4x faster attention)
- torch.compile() (kernel fusion)
- Large batch sizes (128 per GPU)
- Parallel data loading (8 workers)
- Gradient checkpointing

Usage:
    # Single GPU
    python train_nmt_v3.py

    # Multi-GPU (recommended for 2x H100)
    torchrun --nproc_per_node=2 train_nmt_v3.py

    # With larger GPT-2
    python train_nmt_v3.py --decoder-model gpt2-medium
"""

import argparse
import json
import os
import warnings
from pathlib import Path

# Prevent tokenizer parallelism deadlock with multiple data workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import torch
import evaluate
from datasets import Dataset
from transformers import (
    EncoderDecoderModel,
    PreTrainedTokenizerFast,
    GPT2TokenizerFast,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# --- Configuration ---
ENCODER_PATH = "./models/sumerian_tiny_mlm"
DECODER_PRETRAIN = "distilgpt2"
TRAIN_FILE = "output_training_v2_clean/finetune/train_augmented_v2.jsonl"
VALID_FILE = "output_training_v2_clean/finetune/valid.jsonl"
OUTPUT_DIR = "./models/sumerian_nmt_v3_gpt2"

# Training hyperparameters (optimized for H100)
NUM_EPOCHS = 50
BATCH_SIZE = 128  # Increased from 16
LEARNING_RATE = 5e-5
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
    parser = argparse.ArgumentParser(description="NMT v3 GPT-2 Decoder with H100 optimizations")
    parser.add_argument("--decoder-model", type=str, default=DECODER_PRETRAIN,
                        help="GPT-2 variant to use (distilgpt2, gpt2, gpt2-medium, gpt2-large)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Per-device batch size")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile()")
    parser.add_argument("--no-gradient-checkpointing", action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    args = parser.parse_args()

    print("=" * 70)
    print("NMT v3 - DistilGPT-2 Decoder Training (H100 Optimized)")
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
    if not Path(ENCODER_PATH).exists():
        print(f"\nERROR: Encoder not found at {ENCODER_PATH}")
        return

    if not Path(TRAIN_FILE).exists():
        print(f"\nERROR: Training file not found at {TRAIN_FILE}")
        return

    # 1. Load Tokenizers
    print("\n[1/6] Loading tokenizers...")
    tokenizer_src = PreTrainedTokenizerFast.from_pretrained(ENCODER_PATH)
    tokenizer_src.pad_token = "<pad>"

    tokenizer_tgt = AutoTokenizer.from_pretrained(args.decoder_model)
    # GPT-2 specific: set pad token to eos token
    tokenizer_tgt.pad_token = tokenizer_tgt.eos_token

    print(f"  Source (Sumerian): {tokenizer_src.vocab_size} tokens")
    print(f"  Target (English/GPT-2): {tokenizer_tgt.vocab_size} tokens")

    # 2. Initialize Model
    print(f"\n[2/6] Loading pre-trained encoder-decoder...")
    print(f"  Encoder: {ENCODER_PATH}")
    print(f"  Decoder: {args.decoder_model}")

    # Use Flash Attention if available
    attn_impl = "flash_attention_2" if hw['flash_attn_available'] else None

    if attn_impl:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            ENCODER_PATH,
            args.decoder_model,
            encoder_attn_implementation=attn_impl,
            decoder_attn_implementation=attn_impl
        )
        print("  Using Flash Attention 2")
    else:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            ENCODER_PATH,
            args.decoder_model
        )

    # Configure Model for GPT-2 decoder
    model.config.decoder_start_token_id = tokenizer_tgt.bos_token_id or tokenizer_tgt.eos_token_id
    model.config.pad_token_id = tokenizer_tgt.pad_token_id
    model.config.eos_token_id = tokenizer_tgt.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = MAX_LENGTH

    # Enable cross-attention in GPT-2
    model.config.decoder.add_cross_attention = True

    # Enable gradient checkpointing
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

    # 3. Apply torch.compile()
    if not args.no_compile and hasattr(torch, 'compile'):
        print("\n[3/6] Applying torch.compile()...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile(): enabled")
        except Exception as e:
            print(f"  torch.compile() failed: {e}")
    else:
        print("\n[3/6] Skipping torch.compile()")

    # 4. Load and process data
    print(f"\n[4/6] Loading data from {TRAIN_FILE}...")
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
        model_inputs = tokenizer_src(
            examples["source_text"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer_tgt(
            examples["target_text"],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length"
        )
        labels["input_ids"] = [
            [(l if l != tokenizer_tgt.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
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

    # 5. Load metrics
    print("\n[5/6] Loading evaluation metrics...")
    metric_bleu = evaluate.load("sacrebleu")
    metric_chrf = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        # Sanitize predictions
        vocab_size = tokenizer_tgt.vocab_size
        preds = np.where((preds >= 0) & (preds < vocab_size), preds, tokenizer_tgt.pad_token_id)

        decoded_preds = tokenizer_tgt.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer_tgt.pad_token_id)
        decoded_labels = tokenizer_tgt.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        result_bleu = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
        result_chrf = metric_chrf.compute(predictions=decoded_preds, references=decoded_labels)

        return {"bleu": result_bleu["score"], "chrf": result_chrf["score"]}

    # 6. Training setup
    print("\n[6/6] Setting up training...")

    use_bf16 = hw['bf16_supported'] and hw['cuda_available']
    use_fp16 = hw['cuda_available'] and not use_bf16

    effective_batch = args.batch_size * max(1, hw['gpu_count']) * args.grad_accum
    print(f"  Epochs: {args.epochs}")
    print(f"  Per-device batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Precision: {'BF16' if use_bf16 else 'FP16' if use_fp16 else 'FP32'}")

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

        # Precision
        bf16=use_bf16,
        fp16=use_fp16,

        # Data loading
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,

        # Evaluation and saving
        eval_strategy="epoch",
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

        # Memory optimization
        gradient_accumulation_steps=args.grad_accum,
    )

    print(f"\n  Output: {OUTPUT_DIR}")
    print("-" * 70)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer_src,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer_src, model=model, pad_to_multiple_of=8),
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer_src.save_pretrained(f"{OUTPUT_DIR}/tokenizer_src")
    tokenizer_tgt.save_pretrained(f"{OUTPUT_DIR}/tokenizer_tgt")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to {OUTPUT_DIR}")
    print(f"\nOptimizations applied:")
    print(f"  - Multi-GPU: {hw['gpu_count']} GPU(s)")
    print(f"  - Precision: {'BF16' if use_bf16 else 'FP16' if use_fp16 else 'FP32'}")
    print(f"  - Flash Attention 2: {attn_impl is not None}")
    print(f"  - Gradient Checkpointing: {not args.no_gradient_checkpointing}")
    print(f"  - Data Workers: 8")


if __name__ == "__main__":
    main()
