# Reproducibility Guide

This document provides step-by-step instructions to reproduce the results from our paper on Graph-Based Entity Substitution for Low-Resource Neural Machine Translation.

## Prerequisites

- Python >= 3.10
- CUDA >= 11.8 (for GPU training)
- 2x NVIDIA H100 80GB (recommended) or 1x A100 80GB (minimum)
- 450GB RAM (for parallel processing)

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/your-repo/sumerian-translation.git
cd sumerian-translation

# 2. Setup environment
./scripts/00_setup.sh
source venv/bin/activate

# 3. Extract data from archives
cd data && unzip -o "*.zip" && cd ..

# 4. Run full pipeline
./scripts/run_all.sh
```

## Step-by-Step Instructions

### Step 0: Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package in editable mode
pip install -e .

# Install Axolotl for training (separate)
pip install axolotl peft bitsandbytes

# Optional: Flash Attention for faster inference
pip install ninja packaging
pip install flash-attn --no-build-isolation
```

### Step 1: Corpus Extraction

Extract parallel corpus from ETCSL XML:

```bash
sumerian-extract --output-dir output
```

Output: `output/parallel_corpus.jsonl`

### Step 2: Data Split

Create train/valid split by composition (prevents data leakage):

```bash
sumerian-prepare --etcsl-path output/parallel_corpus.jsonl \
    --output-dir output_training_v2_clean
```

### Step 3: Graph-Based Entity Substitution (Novel Contribution)

Run the Two-Circle augmentation approach:

```bash
sumerian-augment --parallel --output-format both
```

This runs:
- Circle 1: ETCSL ↔ ETCSL (same entities across compositions)
- Circle 2: ORACC → ETCSL (glossary-linked monolingual texts)

### Step 4: Consolidate for LLM Training

Convert to Alpaca format for SFT:

```bash
sumerian-consolidate --output-format both
```

### Step 5: Generate DPO Pairs

Create preference pairs for alignment:

```bash
sumerian-dpo --parallel --output-format both
```

### Step 6: Train SFT Model

```bash
accelerate launch --num_processes 2 --multi_gpu \
    -m axolotl.cli.train configs/llama3_sft.yaml
```

### Step 7: Train DPO Model

```bash
accelerate launch --num_processes 2 --multi_gpu \
    -m axolotl.cli.train configs/llama3_dpo.yaml
```

### Step 8: Evaluate

```bash
# High-throughput with vLLM
sumerian-evaluate --model models_llm/sumerian_llama3_sft --use-vllm

# With Named Entity evaluation
sumerian-evaluate --with-ne-eval
```

## Expected Results

| Model | BLEU | chrF++ | NE Accuracy |
|-------|------|--------|-------------|
| Baseline (no augmentation) | ~12 | ~28 | ~45% |
| + Graph Augmentation | ~18 | ~35 | ~65% |
| + DPO Alignment | ~21 | ~38 | ~68% |

## Key Files

- `src/sumerian_nmt/graph_augmentation/` - Novel Two-Circle approach implementation
- `configs/llama3_sft.yaml` - SFT training configuration
- `configs/llama3_dpo.yaml` - DPO training configuration

## Citation

```bibtex
@inproceedings{author2024sumerian,
  title={Graph-Based Entity Substitution for Low-Resource Neural Machine Translation},
  author={Author},
  booktitle={Proceedings of ...},
  year={2024}
}
```
