"""Minimal MLM stress test for H100."""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import (
    RobertaConfig, RobertaForMaskedLM, PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
from datasets import Dataset

print('='*60)
print('H100 MLM Stress Test')
print('='*60)

print('\n[1] Loading tokenizer...')
tokenizer = PreTrainedTokenizerFast.from_pretrained('output_training_v2_clean/tokenizer_hf')
tokenizer.pad_token = '<pad>'
tokenizer.mask_token = '<mask>'
print(f'    Vocab size: {tokenizer.vocab_size}')

print('\n[2] Loading corpus...')
lines = []
with open('output_training_v2_clean/pretrain/corpus_monolingual.txt') as f:
    for line in f:
        if line.strip():
            lines.append(line.strip())
print(f'    Loaded {len(lines):,} lines')

print('\n[3] Tokenizing...')
def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, max_length=128, 
                     padding='max_length', return_special_tokens_mask=True)

dataset = Dataset.from_dict({'text': lines})
dataset = dataset.map(tokenize, batched=True, num_proc=1, remove_columns=['text'])
print(f'    Dataset: {len(dataset)} examples')

print('\n[4] Creating model...')
config = RobertaConfig(
    vocab_size=tokenizer.vocab_size + 10,
    max_position_embeddings=514,
    num_hidden_layers=12, hidden_size=768, 
    num_attention_heads=12, intermediate_size=3072,
    type_vocab_size=1, pad_token_id=tokenizer.pad_token_id
)
model = RobertaForMaskedLM(config)
model.gradient_checkpointing_enable()
print(f'    Parameters: {sum(p.numel() for p in model.parameters()):,}')

print('\n[5] Setting up training...')
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15, pad_to_multiple_of=8
)

args = TrainingArguments(
    output_dir='./models/test_mlm',
    per_device_train_batch_size=128,
    num_train_epochs=1,
    bf16=True,
    dataloader_num_workers=0,
    logging_steps=10,
    save_strategy='no',
    report_to='none',
    max_steps=100,
)
print(f'    Batch size: 128')
print(f'    Precision: BF16')
print(f'    Max steps: 100')

trainer = Trainer(model=model, args=args, data_collator=collator, train_dataset=dataset)

print('\n[6] Training...')
print('-'*60)
trainer.train()
print('-'*60)
print('\n✅ TRAINING SUCCESSFUL!')
