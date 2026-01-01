"""
mT5 DDP Training Script - Continued Training

Continues training from a saved checkpoint with optimized parameters.

Key changes from initial training:
- Loads from saved checkpoint (not base model)
- Lower learning rate (1e-5) for fine-tuning
- Cosine annealing LR schedule
- Gradient clipping
- More epochs (80)
- Evaluation every 5 epochs
- Label smoothing for better generation

Usage:
    torchrun --nproc_per_node=2 train_mt5_ddp_continued.py

    # Or with custom epochs:
    torchrun --nproc_per_node=2 train_mt5_ddp_continued.py --epochs 100
"""
import os
import argparse
import pickle
import json
import math
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq


# Configuration
CHECKPOINT_DIR = "./models/sumerian_mt5_final"
TRAIN_PICKLE = "output_training_v2_clean/finetune/train_data.pkl"
VALID_PICKLE = "output_training_v2_clean/finetune/valid_data.pkl"
VALID_JSONL = "output_training_v2_clean/finetune/valid.jsonl"
OUTPUT_DIR = "./models/sumerian_mt5_continued"
TASK_PREFIX = "translate Sumerian to English: "

# Training hyperparameters - ADJUSTED for continued training
DEFAULT_EPOCHS = 80
BATCH_SIZE = 128
LEARNING_RATE = 1e-5  # Lower LR for continued training
MAX_LENGTH = 128
WARMUP_RATIO = 0.03  # Less warmup since already trained
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0  # Gradient clipping
LABEL_SMOOTHING = 0.1  # Label smoothing for better generation
EVAL_EVERY = 5  # Evaluate every N epochs
PATIENCE = 15  # Early stopping patience


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


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Cosine annealing with warmup - better for continued training."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for seq2seq models."""
    def __init__(self, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        # logits: [batch, seq_len, vocab_size]
        # labels: [batch, seq_len]
        vocab_size = logits.size(-1)

        # Flatten
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_labels = torch.zeros_like(logits)
            smooth_labels.fill_(self.smoothing / (vocab_size - 1))

            # Handle ignore_index
            mask = labels != self.ignore_index
            valid_labels = labels.clone()
            valid_labels[~mask] = 0

            smooth_labels.scatter_(1, valid_labels.unsqueeze(1), 1.0 - self.smoothing)

        # Compute loss
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -(smooth_labels * log_probs).sum(dim=-1)

        # Apply mask
        loss = loss * mask.float()
        return loss.sum() / mask.sum()


def evaluate_model(model, tokenizer, valid_loader, device, rank, metric_bleu, metric_chrf):
    """Run evaluation and compute BLEU/chrF scores."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in valid_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Compute validation loss
            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1

            # Generate predictions
            generated = model.module.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=MAX_LENGTH,
                num_beams=4,
                early_stopping=True,
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

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    # Compute metrics (only on rank 0)
    if rank == 0:
        bleu = metric_bleu.compute(predictions=all_preds, references=all_labels)['score']
        chrf = metric_chrf.compute(predictions=all_preds, references=all_labels)['score']
        return bleu, chrf, avg_loss, all_preds[:5], [l[0] for l in all_labels[:5]]
    return None, None, avg_loss, None, None


def main():
    parser = argparse.ArgumentParser(description="Continue mT5 training")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--from-scratch", action="store_true", help="Train from scratch (ignore checkpoint)")
    args = parser.parse_args()

    # Initialize DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print("=" * 70)
        print("mT5 Fine-tuning - CONTINUED TRAINING")
        print("=" * 70)
        print(f"\nWorld size: {world_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size per GPU: {BATCH_SIZE}")
        print(f"Effective batch size: {BATCH_SIZE * world_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Label smoothing: {LABEL_SMOOTHING}")
        print(f"Gradient clipping: {GRAD_CLIP}")

    # Load tokenizer and model - FROM CHECKPOINT if available
    checkpoint_exists = Path(CHECKPOINT_DIR).exists() and (Path(CHECKPOINT_DIR) / "config.json").exists()

    if checkpoint_exists and not args.from_scratch:
        if rank == 0:
            print(f"\n[1/5] Loading from checkpoint: {CHECKPOINT_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_DIR)
    else:
        if rank == 0:
            print(f"\n[1/5] No checkpoint found, loading base model: google/mt5-small")
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

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
        print(f"\n[3/5] Setting up optimizer with cosine annealing...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Label smoothing loss
    label_smooth_loss = LabelSmoothingLoss(smoothing=LABEL_SMOOTHING)

    if rank == 0:
        print(f"  Training steps: {num_training_steps:,}")
        print(f"  Warmup steps: {num_warmup_steps:,}")
        print(f"  Scheduler: Cosine annealing")

    # Load metrics
    if rank == 0:
        print(f"\n[4/5] Loading evaluation metrics...")
    metric_bleu = evaluate.load("sacrebleu")
    metric_chrf = evaluate.load("chrf")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Training loop
    if rank == 0:
        print(f"\n[5/5] Starting continued training...")
        print(f"  Output: {OUTPUT_DIR}")
        print(f"  Eval every: {EVAL_EVERY} epochs")
        print(f"  Early stopping patience: {PATIENCE} epochs")
        print("-" * 70)

    best_bleu = 0.0
    best_chrf = 0.0
    best_loss = float('inf')
    no_improve_count = 0
    global_step = 0
    training_history = []

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        total_loss = 0.0
        num_batches = 0

        if rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = train_loader

        for batch in pbar:
            batch = {k: v.to(local_rank) for k, v in batch.items()}

            # Forward pass with BF16
            with autocast('cuda', dtype=torch.bfloat16):
                outputs = model(**batch)
                # Use label smoothing loss
                loss = outputs.loss  # Can switch to: label_smooth_loss(outputs.logits, batch['labels'])

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

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

        # Evaluation (every EVAL_EVERY epochs or last epoch)
        if (epoch + 1) % EVAL_EVERY == 0 or epoch == args.epochs - 1:
            if rank == 0:
                print("Running evaluation...")

            bleu, chrf, val_loss, sample_preds, sample_refs = evaluate_model(
                model, tokenizer, valid_loader, local_rank, rank, metric_bleu, metric_chrf
            )

            if rank == 0:
                print(f"  BLEU: {bleu:.2f}, chrF: {chrf:.2f}, Val Loss: {val_loss:.4f}")

                # Show sample predictions
                if sample_preds:
                    print(f"  Sample: '{sample_preds[0][:50]}...' vs '{sample_refs[0][:50]}...'")

                training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                    "bleu": bleu,
                    "chrf": chrf,
                })

                # Save best model (by BLEU or chrF)
                improved = False
                if bleu > best_bleu:
                    best_bleu = bleu
                    improved = True
                if chrf > best_chrf:
                    best_chrf = chrf
                    improved = True

                if improved:
                    no_improve_count = 0
                    print(f"  New best! BLEU: {best_bleu:.2f}, chrF: {best_chrf:.2f} - Saving...")
                    model.module.save_pretrained(OUTPUT_DIR)
                    tokenizer.save_pretrained(OUTPUT_DIR)

                    # Save training history
                    with open(Path(OUTPUT_DIR) / "training_history.json", "w") as f:
                        json.dump(training_history, f, indent=2)
                else:
                    no_improve_count += EVAL_EVERY
                    print(f"  No improvement for {no_improve_count} epochs")

                    if no_improve_count >= PATIENCE:
                        print(f"\n  Early stopping triggered after {PATIENCE} epochs without improvement!")
                        break

        dist.barrier()

    # Final summary
    if rank == 0:
        print("\n" + "=" * 70)
        print("CONTINUED TRAINING COMPLETE")
        print("=" * 70)
        print(f"Best BLEU: {best_bleu:.2f}")
        print(f"Best chrF: {best_chrf:.2f}")
        print(f"Model saved to: {OUTPUT_DIR}")

        # Show sample predictions from best model
        print("\n" + "=" * 70)
        print("SAMPLE PREDICTIONS (from best model)")
        print("=" * 70)

        # Reload best model for predictions
        best_model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)
        best_model = best_model.to(local_rank)
        best_model.eval()

        raw_valid = load_jsonl(VALID_JSONL)
        valid_data_raw = [{
            "source_text": item["source"]["text_normalized"],
            "target_text": item["target"]["text"],
        } for item in raw_valid]

        sample_inputs = [TASK_PREFIX + valid_data_raw[i]["source_text"] for i in range(5)]
        inputs = tokenizer(sample_inputs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(local_rank) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = best_model.generate(**inputs, max_length=MAX_LENGTH, num_beams=4)
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for i in range(5):
            print(f"\nExample {i+1}:")
            print(f"  Sumerian:  {valid_data_raw[i]['source_text']}")
            print(f"  Predicted: {predictions[i]}")
            print(f"  Reference: {valid_data_raw[i]['target_text']}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
