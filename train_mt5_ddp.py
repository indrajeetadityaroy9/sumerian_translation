"""
mT5 DDP Training Script - Pure PyTorch Implementation

This script bypasses the HuggingFace Trainer/accelerate integration issues
by using pure PyTorch DDP. Tested and verified working on 2x H100 80GB.

Usage:
    torchrun --nproc_per_node=2 train_mt5_ddp.py

Features:
- Pure PyTorch DDP (works reliably)
- BF16 mixed precision
- Gradient checkpointing
- BLEU/chrF evaluation
- Checkpointing with best model saving
"""
import os
import pickle
import json
import math
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast
from tqdm import tqdm
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

# Configuration
MODEL_CHECKPOINT = "google/mt5-small"
TRAIN_PICKLE = "output_training_v2_clean/finetune/train_data.pkl"
VALID_PICKLE = "output_training_v2_clean/finetune/valid_data.pkl"
VALID_JSONL = "output_training_v2_clean/finetune/valid.jsonl"
OUTPUT_DIR = "./models/sumerian_mt5_final"
TASK_PREFIX = "translate Sumerian to English: "

# Training hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 128  # Increased to saturate H100s
LEARNING_RATE = 3e-5
MAX_LENGTH = 128
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01


class SimpleMT5Dataset(Dataset):
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


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup then linear decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_model(model, tokenizer, valid_loader, device, rank, metric_bleu, metric_chrf):
    """Run evaluation and compute BLEU/chrF scores."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in valid_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Generate predictions
            generated = model.module.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=MAX_LENGTH,
                num_beams=4,
            )

            # Decode
            preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
            labels = batch['labels'].cpu().numpy()
            labels[labels == -100] = tokenizer.pad_token_id
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_preds.extend([p.strip() for p in preds])
            all_labels.extend([[l.strip()] for l in labels])

    # Gather predictions from all ranks
    if dist.is_initialized():
        world_size = dist.get_world_size()
        gathered_preds = [None] * world_size
        gathered_labels = [None] * world_size
        dist.all_gather_object(gathered_preds, all_preds)
        dist.all_gather_object(gathered_labels, all_labels)

        # Flatten
        all_preds = [p for preds in gathered_preds for p in preds]
        all_labels = [l for labels in gathered_labels for l in labels]

    # Compute metrics (only on rank 0)
    if rank == 0:
        bleu = metric_bleu.compute(predictions=all_preds, references=all_labels)['score']
        chrf = metric_chrf.compute(predictions=all_preds, references=all_labels)['score']
        return bleu, chrf
    return None, None


def main():
    # Initialize DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print("=" * 70)
        print("mT5 Fine-tuning - Pure PyTorch DDP")
        print("=" * 70)
        print(f"\nWorld size: {world_size}")
        print(f"Epochs: {NUM_EPOCHS}")
        print(f"Batch size per GPU: {BATCH_SIZE}")
        print(f"Effective batch size: {BATCH_SIZE * world_size}")
        print(f"Learning rate: {LEARNING_RATE}")

    # Load tokenizer and model
    if rank == 0:
        print(f"\n[1/5] Loading {MODEL_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")
        print(f"  Gradient checkpointing: enabled")

    # Load data
    if rank == 0:
        print(f"\n[2/5] Loading data from pickle files...")

    with open(TRAIN_PICKLE, "rb") as f:
        train_data = pickle.load(f)
    with open(VALID_PICKLE, "rb") as f:
        valid_data = pickle.load(f)

    train_dataset = SimpleMT5Dataset(train_data)
    valid_dataset = SimpleMT5Dataset(valid_data)

    if rank == 0:
        print(f"  Training examples: {len(train_dataset):,}")
        print(f"  Validation examples: {len(valid_dataset):,}")

    # Create data loaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model.module, label_pad_token_id=-100, pad_to_multiple_of=8)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collator, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler, collate_fn=collator, num_workers=0)

    # Optimizer and scheduler
    if rank == 0:
        print(f"\n[3/5] Setting up optimizer...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    num_training_steps = len(train_loader) * NUM_EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    if rank == 0:
        print(f"  Training steps: {num_training_steps:,}")
        print(f"  Warmup steps: {num_warmup_steps:,}")

    # Load metrics
    if rank == 0:
        print(f"\n[4/5] Loading evaluation metrics...")
    metric_bleu = evaluate.load("sacrebleu")
    metric_chrf = evaluate.load("chrf")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Training loop
    if rank == 0:
        print(f"\n[5/5] Starting training...")
        print(f"  Output: {OUTPUT_DIR}")
        print("-" * 70)

    best_bleu = 0.0
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        model.train()

        total_loss = 0.0
        num_batches = 0

        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        else:
            pbar = train_loader

        for batch in pbar:
            batch = {k: v.to(local_rank) for k, v in batch.items()}

            # Forward pass with BF16
            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

            if rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        # Epoch summary
        avg_loss = total_loss / num_batches
        if rank == 0:
            print(f"\nEpoch {epoch+1} - Avg Train Loss: {avg_loss:.4f}")

        # Evaluation
        if rank == 0:
            print("Running evaluation...")

        bleu, chrf = evaluate_model(model, tokenizer, valid_loader, local_rank, rank, metric_bleu, metric_chrf)

        if rank == 0:
            print(f"  BLEU: {bleu:.2f}, chrF: {chrf:.2f}")

            # Save best model
            if bleu > best_bleu:
                best_bleu = bleu
                print(f"  New best BLEU! Saving model...")
                model.module.save_pretrained(OUTPUT_DIR)
                tokenizer.save_pretrained(OUTPUT_DIR)

        dist.barrier()

    # Final summary
    if rank == 0:
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best BLEU: {best_bleu:.2f}")
        print(f"Model saved to: {OUTPUT_DIR}")

        # Show sample predictions
        print("\n" + "=" * 70)
        print("SAMPLE PREDICTIONS")
        print("=" * 70)

        raw_valid = load_jsonl(VALID_JSONL)
        valid_data_raw = [{
            "source_text": item["source"]["text_normalized"],
            "target_text": item["target"]["text"],
        } for item in raw_valid]

        model.eval()
        sample_inputs = [TASK_PREFIX + valid_data_raw[i]["source_text"] for i in range(3)]
        inputs = tokenizer(sample_inputs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(local_rank) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.module.generate(**inputs, max_length=MAX_LENGTH, num_beams=4)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for i in range(3):
            print(f"\nExample {i+1}:")
            print(f"  Sumerian: {valid_data_raw[i]['source_text']}")
            print(f"  Predicted: {predictions[i]}")
            print(f"  Reference: {valid_data_raw[i]['target_text']}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
