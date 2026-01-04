# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Low-resource Neural Machine Translation system for Sumerian-to-English translation using **~46,000 training samples** from ETCSL (Electronic Text Corpus of Sumerian Literature).

**Architecture**: Llama-3.1-8B-Instruct with LoRA fine-tuning via Axolotl.

**Requirements**: Python ≥3.10, CUDA-capable GPU (A100/H100 recommended for BF16)

**Target Hardware**: 2x NVIDIA H100 80GB SXM5, 52 vCPUs, 450 GiB RAM

## Package Structure

The codebase uses a `src/` layout with the `sumerian_nmt` namespace:

```
src/sumerian_nmt/
├── config.py                  # Central configuration (Paths, LLMConfig, ControlTokens)
├── graph_augmentation/        # NOVEL CONTRIBUTION - Two-Circle Entity Substitution
│   ├── entity_graph.py        # Bipartite line-entity graph
│   ├── structural_matcher.py  # Skeleton similarity matching (85% threshold)
│   ├── substitution.py        # Type-safe entity substitution
│   ├── constraints.py         # DN↔DN, RN↔RN, GN↔GN type constraints
│   └── pipeline.py            # Main augmentation pipeline
├── data_ingestion/            # ETCSL/ORACC corpus extraction
├── data_processing/           # Training data preparation
├── evaluation/                # LLM evaluation with WMT metrics
└── utils/                     # Common utilities (I/O, metrics, text cleaning)
```

## Common Commands

### LLM Training (Axolotl)

```bash
# Phase 1: Supervised Fine-Tuning (SFT) - Dual H100 DDP
accelerate launch --num_processes 2 --multi_gpu \
    -m axolotl.cli.train configs/llama3_sft.yaml

# Phase 2: DPO Alignment (after SFT) - Dual H100 DDP
accelerate launch --num_processes 2 --multi_gpu \
    -m axolotl.cli.train configs/llama3_dpo.yaml

# Single GPU (fallback)
accelerate launch -m axolotl.cli.train configs/llama3_sft.yaml
```

### Evaluation

```bash
# High-throughput evaluation with vLLM (dual H100, ~50x faster)
sumerian-evaluate --model models_llm/sumerian_llama3_sft --use-vllm

# Standard HuggingFace evaluation (single GPU)
sumerian-evaluate --model models_llm/sumerian_llama3_sft

# Quick test (limited samples)
sumerian-evaluate --max-samples 50

# With Named Entity evaluation
sumerian-evaluate --with-ne-eval
```

### Data Pipeline

The pipeline transforms ETCSL XML → training-ready Alpaca JSON:

```
etcsl_extractor/ota_20/*.xml
    ↓ (sumerian-extract)
output/parallel_corpus.jsonl
    ↓ (sumerian-prepare)
output_training_v2_clean/finetune/{train,valid}.jsonl
    ↓ (sumerian-augment)
output_training_v2_clean/finetune/train_graph_augmented.jsonl
    ↓ (sumerian-consolidate)
data/final_llm_ready/sft_{train,test}.json (Alpaca format)
    ↓ (sumerian-dpo)
data/final_llm_ready/dpo_pairs.json
```

```bash
# Step 1: Extract ETCSL corpus from XML
sumerian-extract --output-dir output

# Step 2: Prepare train/valid split (composition-based)
sumerian-prepare --etcsl-path output/parallel_corpus.jsonl --output-dir output_training_v2_clean

# Step 3: Run graph-based entity substitution augmentation (NOVEL)
sumerian-augment --parallel --output-format both

# Step 4: Consolidate for LLM fine-tuning (Alpaca format)
sumerian-consolidate --output-format both

# Step 5: Create DPO preference pairs
sumerian-dpo --parallel --output-format both
```

### CLI Entry Points

After `pip install -e .`, these commands are available:

| Command | Module | Purpose |
|---------|--------|---------|
| `sumerian-extract` | `data_ingestion.extractor` | Extract ETCSL corpus from XML |
| `sumerian-prepare` | `data_processing.splitter` | Create train/valid split |
| `sumerian-augment` | `graph_augmentation.pipeline` | Run entity substitution |
| `sumerian-consolidate` | `data_processing.alpaca_formatter` | Convert to Alpaca format |
| `sumerian-dpo` | `data_processing.dpo_generator` | Create DPO pairs |
| `sumerian-evaluate` | `evaluation.llm_evaluator` | Evaluate model |
| `sumerian-compare` | `evaluation.model_comparison` | Compare models |

### Automation Scripts

```bash
# Run full pipeline (setup + data processing)
./scripts/run_all.sh

# Individual steps
./scripts/00_setup.sh         # Create venv, install dependencies
./scripts/01_extract_corpus.sh # Extract and prepare corpus
./scripts/03_augment.sh       # Run graph augmentation
```

### Parquet Output for Reproducibility

Data pipeline scripts support chunked parquet output for GitHub-friendly storage (<50MB per file):

```bash
# Output format options: json, parquet, or both (default)
sumerian-dpo --output-format parquet --chunk-size 10000
```

Parquet utilities in `sumerian_nmt.utils.io`:
- `ChunkedParquetWriter`: Write chunked parquet
- `load_chunked_parquet`: Load all chunks into DataFrame
- `load_chunked_parquet_lazy`: Iterate chunks without loading all
- `zip_parquet_dataset` / `unzip_parquet_dataset`: Archive management
- `parquet_to_json`: Convert to JSON for Axolotl

## Architecture

### Two-Phase Training Pipeline

1. **SFT (Supervised Fine-Tuning)**: Train on 45,915 Sumerian→English pairs in Alpaca format
2. **DPO (Direct Preference Optimization)**: Align with 5,017 preference pairs (human translation vs. gloss concatenation)

### Data Augmentation: Two-Circle Approach (Novel Contribution)

The `sumerian_nmt.graph_augmentation/` module implements graph-based entity substitution:
- **Circle 1**: ETCSL ↔ ETCSL (swap entities between compositions in same corpus)
- **Circle 2**: ORACC → ETCSL (link monolingual ORACC texts via glossary, substitute into ETCSL templates)

**Key classes**:
- `StructuralMatcher`: Skeleton similarity matching (85% threshold prevents "Bag-of-Entities" fallacy)
- `EntitySubstitutor`: Type-safe entity substitution with word boundary safety
- `TypeConstraints`: Enforces DN↔DN, RN↔RN, GN↔GN constraints

Control tokens track data provenance: `<gold>` (original), `<silver>` (high-confidence match), `<aug>` (entity substitution)

### Centralized Utilities

**Control Tokens** (`sumerian_nmt.config`):
```python
from sumerian_nmt.config import ControlTokens

ControlTokens.GOLD      # "<gold>" - Original ETCSL data
ControlTokens.SILVER    # "<silver>" - High-confidence matches (skeleton >= 95%)
ControlTokens.AUG       # "<aug>" - Entity substitution augmented
ControlTokens.GLOSS     # "<gloss>" - Glossary-based augmented
ControlTokens.PATTERN   # Compiled regex for cleaning
```

**Text Cleaning** (`sumerian_nmt.utils.text`):
```python
from sumerian_nmt.utils.text import clean_source_text

# Remove control tokens and normalize whitespace
clean_text = clean_source_text("<gold> lugal-e e2-gal-la-na")
# Returns: "lugal-e e2-gal-la-na"
```

**Schema Validation** (`sumerian_nmt.utils.validation`):
```python
from sumerian_nmt.utils.validation import (
    validate_etcsl_record,      # Validate ETCSL extractor output
    validate_augmented_record,  # Validate graph augmentor output
    validate_training_record,   # Validate Alpaca-format records
    validate_dpo_record,        # Validate DPO preference pairs
)
```

### Data Formats

**SFT (Alpaca format)** - `data/final_llm_ready/sft_train.json`:
```json
{"instruction": "Translate this Sumerian text into English:", "input": "lugal-e e2-gal-la-na ba-gen", "output": "The king went to his palace."}
```

**DPO pairs** - `data/final_llm_ready/dpo_pairs.json`:
```json
{"instruction": "...", "input": "...", "chosen": "<fluent translation>", "rejected": "<gloss concatenation>"}
```

## Target Metrics

- **BLEU**: 15-25, **chrF++**: 30-40
- **Named Entity Accuracy**: >60% (key differentiator from mT5 baseline)

## Critical Constraints

- **Data split uses `composition_id`** to prevent leakage between train/test
- **Prompt format must match training**: Use `LLMConfig.format_prompt()` for Llama-3 Instruct format
- **Named Entity evaluation** requires `data/archive/valid.jsonl` (preserves `named_entities` field)
- **Hardware**: BF16 requires compute capability >= 8.0 (A100/H100); use `--use-4bit` for lower VRAM

## Setup

```bash
# Install core dependencies
pip install -e .

# Install with dev tools (linting)
pip install -e ".[dev]"

# Install Axolotl for LLM training
pip install axolotl peft bitsandbytes

# Optional: vLLM for high-throughput evaluation
pip install -e ".[vllm]"

# Optional: Flash Attention (requires compatible GPU)
pip install ninja packaging
pip install flash-attn --no-build-isolation

# Extract data archives (required before training)
cd data && unzip -o "*.zip" && cd ..
```

## Linting

```bash
# Format code
black .
isort .

# Lint
ruff check .
mypy .
```

Line length is 100 characters (configured in pyproject.toml).
