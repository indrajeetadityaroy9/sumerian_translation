"""Minimal DDP test without HuggingFace Trainer."""
import os
import pickle
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def main():
    # Initialize DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    print(f"[Rank {rank}/{world_size}] Initialized on GPU {local_rank}")

    # Load tokenizer and model
    print(f"[Rank {rank}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Load data from pickle
    print(f"[Rank {rank}] Loading data...")
    with open("output_training_v2_clean/finetune/train_data.pkl", "rb") as f:
        train_data = pickle.load(f)

    dataset = SimpleMT5Dataset(train_data)
    print(f"[Rank {rank}] Dataset size: {len(dataset)}")

    # Create DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # Create DataLoader
    collator = DataCollatorForSeq2Seq(tokenizer, model=model.module, label_pad_token_id=-100, pad_to_multiple_of=8)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler, collate_fn=collator, num_workers=0)

    print(f"[Rank {rank}] DataLoader batches: {len(dataloader)}")

    # Train for a few steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    model.train()

    print(f"[Rank {rank}] Starting training...")
    for step, batch in enumerate(dataloader):
        if step >= 10:
            break

        batch = {k: v.to(local_rank) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if rank == 0:
            print(f"  Step {step+1}: loss = {loss.item():.4f}")

    print(f"[Rank {rank}] Training complete!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
