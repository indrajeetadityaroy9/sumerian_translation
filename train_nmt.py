"""
Phase 2: Seq2Seq Fine-tuning for Sumerian-English Translation

Initializes the model with the Pre-trained Sumerian Encoder (from Phase 1)
and adds a fresh English Decoder (using bert-base-uncased tokenizer).

OPTIMIZED FOR: 2x NVIDIA H100 80GB + Intel Xeon Platinum 8480+

Optimizations enabled:
- Multi-GPU training (DDP) - auto-detected
- BF16 mixed precision (native H100 support)
- Flash Attention 2 (2-4x faster attention)
- torch.compile() (kernel fusion)
- Large batch sizes (256 per GPU)
- Parallel data loading (8 workers)
- Gradient checkpointing (for larger models)

Architecture:
- Encoder: Pre-trained RoBERTa (Sumerian)
- Decoder: Fresh BERT (English) with matched/larger dimensions

Features:
- Real-time BLEU score monitoring at each epoch (using sacrebleu)
- Best model saved by BLEU score (not just loss)
- Proper -100 masking for padding tokens in labels

Usage:
    # Single GPU
    python train_nmt.py

    # Multi-GPU (recommended for 2x H100)
    torchrun --nproc_per_node=2 train_nmt.py

    # With larger decoder
    python train_nmt.py --decoder-size base

Prerequisites:
    1. Run train_mlm.py first to create the pre-trained encoder
    2. Ensure fine-tuning data exists in output_training_v2_clean/finetune/
    3. Install: pip install evaluate sacrebleu
"""

import argparse
import json
import os
import warnings
from pathlib import Path

# Prevent tokenizer parallelism deadlock with multiple data workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import torch
import evaluate
from datasets import Dataset
from transformers import (
    EncoderDecoderModel,
    PreTrainedTokenizerFast,
    BertTokenizerFast,
    BertConfig,
    BertLMHeadModel,
    AutoModel,
    AutoConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# --- Configuration ---
ENCODER_PATH = Path("./models/sumerian_tiny_mlm")  # Result from Phase 1
TRAIN_FILE = Path("output_training_v2_clean/finetune/train_augmented.jsonl")
VALID_FILE = Path("output_training_v2_clean/finetune/valid.jsonl")
OUTPUT_DIR = Path("./models/sumerian_nmt_final")

# Decoder size configurations
DECODER_CONFIGS = {
    "tiny": {  # Match original tiny encoder
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
    },
    "small": {  # Larger decoder
        "hidden_size": 512,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "intermediate_size": 2048,
    },
    "base": {  # BERT-base sized decoder
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
    },
}

# Training hyperparameters (optimized for H100)
NUM_EPOCHS = 20
BATCH_SIZE = 256  # Increased from 32
LEARNING_RATE = 5e-5
MAX_LENGTH = 128


def load_jsonl(path: Path) -> list:
    """Load JSONL file."""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


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


def main():
    parser = argparse.ArgumentParser(description="NMT Fine-tuning with H100 optimizations")
    parser.add_argument("--decoder-size", choices=list(DECODER_CONFIGS.keys()), default="tiny",
                        help="Decoder size configuration")
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
    print("Phase 2: Sumerian-English Translation Fine-tuning (H100 Optimized)")
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
    if not ENCODER_PATH.exists():
        print(f"\nERROR: Pre-trained encoder not found at {ENCODER_PATH}")
        print("Run 'python train_mlm.py' first.")
        return

    if not TRAIN_FILE.exists():
        print(f"\nERROR: Training file not found at {TRAIN_FILE}")
        return

    if not VALID_FILE.exists():
        print(f"\nERROR: Validation file not found at {VALID_FILE}")
        return

    # 1. Load Tokenizers
    print("\n[1/7] Loading tokenizers...")
    tokenizer_src = PreTrainedTokenizerFast.from_pretrained(ENCODER_PATH)
    if tokenizer_src.pad_token is None:
        tokenizer_src.pad_token = "<pad>"
    print(f"  Source (Sumerian): {tokenizer_src.vocab_size} tokens")

    tokenizer_tgt = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer_tgt.bos_token = "[CLS]"
    tokenizer_tgt.eos_token = "[SEP]"
    print(f"  Target (English): {tokenizer_tgt.vocab_size} tokens")

    # Load metrics
    print("  Loading BLEU metric (sacrebleu)...")
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    def compute_metrics(eval_preds):
        """Compute BLEU and chrF scores for translation quality evaluation."""
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

        result_bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result_chrf = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)

        return {"bleu": result_bleu["score"], "chrf": result_chrf["score"]}

    # 2. Load pre-trained encoder
    print(f"\n[2/7] Loading pre-trained encoder from {ENCODER_PATH}...")

    # Get encoder config to check dimensions
    encoder_config = AutoConfig.from_pretrained(str(ENCODER_PATH))
    encoder_hidden_size = encoder_config.hidden_size
    print(f"  Encoder hidden size: {encoder_hidden_size}")

    # Load encoder with Flash Attention if available
    attn_impl = "flash_attention_2" if hw['flash_attn_available'] else None
    if attn_impl:
        encoder = AutoModel.from_pretrained(
            str(ENCODER_PATH),
            attn_implementation=attn_impl
        )
        print("  Using Flash Attention 2 for encoder")
    else:
        encoder = AutoModel.from_pretrained(str(ENCODER_PATH))

    # 3. Create decoder
    decoder_cfg = DECODER_CONFIGS[args.decoder_size]
    print(f"\n[3/7] Creating {args.decoder_size} decoder...")

    decoder_config = BertConfig(
        vocab_size=tokenizer_tgt.vocab_size,
        hidden_size=decoder_cfg["hidden_size"],
        num_hidden_layers=decoder_cfg["num_hidden_layers"],
        num_attention_heads=decoder_cfg["num_attention_heads"],
        intermediate_size=decoder_cfg["intermediate_size"],
        is_decoder=True,
        add_cross_attention=True,
    )

    # Handle encoder-decoder dimension mismatch with projection layer
    if encoder_hidden_size != decoder_cfg["hidden_size"]:
        print(f"  Adding projection layer: {encoder_hidden_size} -> {decoder_cfg['hidden_size']}")
        decoder_config.encoder_hidden_size = encoder_hidden_size

    decoder = BertLMHeadModel(decoder_config)

    # 4. Combine into EncoderDecoderModel
    print("\n[4/7] Building Encoder-Decoder model...")
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

    # Configure generation parameters
    model.config.decoder_start_token_id = tokenizer_tgt.cls_token_id
    model.config.pad_token_id = tokenizer_tgt.pad_token_id
    model.config.eos_token_id = tokenizer_tgt.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = MAX_LENGTH

    # Enable gradient checkpointing
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f"  Total parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
    print(f"  Encoder: {encoder_params:,} (pre-trained)")
    print(f"  Decoder: {decoder_params:,} (random init)")

    # 5. Apply torch.compile()
    if not args.no_compile and hasattr(torch, 'compile'):
        print("\n[5/7] Applying torch.compile() optimization...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile(): enabled")
        except Exception as e:
            print(f"  torch.compile() failed: {e}")
    else:
        print("\n[5/7] Skipping torch.compile()")

    # 6. Load and process data
    print("\n[6/7] Loading and tokenizing data...")
    raw_train = load_jsonl(TRAIN_FILE)
    raw_valid = load_jsonl(VALID_FILE)
    print(f"  Training pairs: {len(raw_train):,}")
    print(f"  Validation pairs: {len(raw_valid):,}")

    def process_data(examples):
        """Tokenize source and target texts (batched format)."""
        inputs = [src["text_normalized"] for src in examples["source"]]
        targets = [tgt["text"] for tgt in examples["target"]]

        model_inputs = tokenizer_src(
            inputs,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length"
        )

        labels = tokenizer_tgt(
            targets,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length"
        )

        model_inputs["labels"] = [
            [(l if l != tokenizer_tgt.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        return model_inputs

    sample_keys = set(raw_train[0].keys()) if raw_train else set()
    remove_cols = list(sample_keys)
    num_proc = min(os.cpu_count() or 1, 8)

    train_ds = Dataset.from_list(raw_train).map(
        process_data,
        batched=True,
        batch_size=100,
        num_proc=num_proc,
        remove_columns=remove_cols,
        desc="Tokenizing train"
    )

    valid_ds = Dataset.from_list(raw_valid).map(
        process_data,
        batched=True,
        batch_size=100,
        num_proc=num_proc,
        remove_columns=remove_cols,
        desc="Tokenizing valid"
    )

    print(f"  Tokenized train: {len(train_ds):,} examples")
    print(f"  Tokenized valid: {len(valid_ds):,} examples")

    # 7. Training setup
    print("\n[7/7] Setting up training...")

    use_bf16 = hw['bf16_supported'] and hw['cuda_available']
    use_fp16 = hw['cuda_available'] and not use_bf16

    effective_batch = args.batch_size * max(1, hw['gpu_count']) * args.grad_accum
    print(f"  Epochs: {args.epochs}")
    print(f"  Per-device batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Precision: {'BF16' if use_bf16 else 'FP16' if use_fp16 else 'FP32'}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,

        # Generation settings
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        generation_num_beams=4,

        # Training settings
        num_train_epochs=args.epochs,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",

        # Precision (BF16 preferred for H100)
        bf16=use_bf16,
        fp16=use_fp16,

        # Data loading optimization
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,

        # Evaluation and saving
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,

        # Logging
        logging_steps=25,
        logging_dir=str(OUTPUT_DIR / "logs"),
        report_to="none",

        # Multi-GPU
        ddp_find_unused_parameters=False,

        # Memory optimization
        gradient_accumulation_steps=args.grad_accum,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer_src,
        model=model,
        padding=True,
        pad_to_multiple_of=8,  # Optimize for Tensor Cores and avoid recompilation
    )

    print(f"\n  Output directory: {OUTPUT_DIR}")
    print("-" * 70)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer_src,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save final model
    print("\n" + "-" * 70)
    print("Saving model and tokenizers...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer_src.save_pretrained(str(OUTPUT_DIR / "tokenizer_src"))
    tokenizer_tgt.save_pretrained(str(OUTPUT_DIR / "tokenizer_tgt"))

    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"Translation model saved to: {OUTPUT_DIR}")
    print(f"\nOptimizations applied:")
    print(f"  - Multi-GPU: {hw['gpu_count']} GPU(s)")
    print(f"  - Precision: {'BF16' if use_bf16 else 'FP16' if use_fp16 else 'FP32'}")
    print(f"  - Flash Attention 2: {attn_impl is not None}")
    print(f"  - Gradient Checkpointing: {not args.no_gradient_checkpointing}")
    print(f"  - Data Workers: 8")


if __name__ == "__main__":
    main()
