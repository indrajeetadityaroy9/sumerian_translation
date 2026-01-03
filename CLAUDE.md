# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Low-resource Neural Machine Translation system for Sumerian-to-English translation using **~46,000 training samples** from ETCSL (Electronic Text Corpus of Sumerian Literature).

**Architecture**: Llama-3.1-8B-Instruct with LoRA fine-tuning via Axolotl.

**Requirements**: Python ≥3.10, CUDA-capable GPU (A100/H100 recommended for BF16)

**Target Hardware**: 2x NVIDIA H100 80GB SXM5, 52 vCPUs, 450 GiB RAM

## Common Commands

### LLM Training (Axolotl)

```bash
# Phase 1: Supervised Fine-Tuning (SFT) - Dual H100 DDP
accelerate launch --num_processes 2 --multi_gpu \
    -m axolotl.cli.train configs_llm/llama3_sft.yaml

# Phase 2: DPO Alignment (after SFT) - Dual H100 DDP
accelerate launch --num_processes 2 --multi_gpu \
    -m axolotl.cli.train configs_llm/llama3_dpo.yaml

# Single GPU (fallback)
accelerate launch -m axolotl.cli.train configs_llm/llama3_sft.yaml

# Multi-GPU with DeepSpeed ZeRO-3 (for larger models)
accelerate launch --config_file configs_llm/deepspeed_zero3.json \
    -m axolotl.cli.train configs_llm/llama3_sft.yaml
```

### Evaluation

```bash
# High-throughput evaluation with vLLM (dual H100, ~50x faster)
python -m evaluation.evaluate_llm --model models_llm/sumerian_llama3_sft --use-vllm

# Standard HuggingFace evaluation (single GPU)
python -m evaluation.evaluate_llm --model models_llm/sumerian_llama3_sft

# Quick test (limited samples)
python -m evaluation.evaluate_llm --max-samples 50

# With 4-bit quantization (low VRAM)
python -m evaluation.evaluate_llm --model models_llm/sumerian_llama3_sft --use-4bit
```

### Data Pipeline

```bash
# Step 1: Extract ETCSL corpus from XML (source: etcsl_extractor/ota_20/)
python3 -m etcsl_extractor.main --output-dir output

# Step 2: Prepare train/valid split (composition-based)
python3 processors/prepare_training_data.py \
    --etcsl-path output/parallel_corpus.jsonl \
    --output-dir output_training_v2_clean

# Step 3: Run graph-based entity substitution augmentation (parallel + parquet)
python3 processors/graph_augmentor.py --parallel --output-format both

# Step 4: Consolidate for LLM fine-tuning (Alpaca format + parquet)
python3 processors/consolidate_for_llm.py --output-format both

# Step 5: Create DPO preference pairs (parallel + parquet)
python3 processors/create_dpo_pairs.py --parallel --output-format both
```

### Parquet Output for Reproducibility

Data pipeline scripts support chunked parquet output for GitHub-friendly storage (<50MB per file):

```bash
# Output format options: json, parquet, or both (default)
python3 processors/create_dpo_pairs.py --output-format parquet --chunk-size 10000

# Parquet utilities in common/io.py:
from common.io import (
    ChunkedParquetWriter,      # Write chunked parquet
    load_chunked_parquet,       # Load all chunks into DataFrame
    load_chunked_parquet_lazy,  # Iterate chunks without loading all
    zip_parquet_dataset,        # Zip for GitHub storage
    unzip_parquet_dataset,      # Extract zipped dataset
    parquet_to_json,            # Convert to JSON for Axolotl
)
```

## Architecture

### Two-Phase Training Pipeline

1. **SFT (Supervised Fine-Tuning)**: Train on 45,915 Sumerian→English pairs in Alpaca format
2. **DPO (Direct Preference Optimization)**: Align with 5,017 preference pairs (human translation vs. gloss concatenation)

### Data Augmentation: Two-Circle Approach

The `processors/graph_engine/` implements graph-based entity substitution:
- **Circle 1**: ETCSL ↔ ETCSL (swap entities between compositions in same corpus)
- **Circle 2**: ORACC → ETCSL (link monolingual ORACC texts via glossary, substitute into ETCSL templates)

**Entity types**: DN (Divine Name), RN (Royal Name), GN (Geographic Name) — swaps are type-safe (DN↔DN only)

Control tokens track data provenance: `<gold>` (original), `<silver>` (high-confidence match), `<aug>` (entity substitution)

### Centralized Utilities

**Control Tokens** (`config.py`):
```python
from config import ControlTokens

ControlTokens.GOLD      # "<gold>" - Original ETCSL data
ControlTokens.SILVER    # "<silver>" - High-confidence matches (skeleton >= 95%)
ControlTokens.AUG       # "<aug>" - Entity substitution augmented
ControlTokens.GLOSS     # "<gloss>" - Glossary-based augmented
ControlTokens.PATTERN   # Compiled regex for cleaning
ControlTokens.VARIANTS  # Extended patterns for validation
```

**Text Cleaning** (`common/text.py`):
```python
from common.text import clean_source_text

# Remove control tokens and normalize whitespace
clean_text = clean_source_text("<gold> lugal-e e2-gal-la-na")
# Returns: "lugal-e e2-gal-la-na"
```

**Schema Validation** (`common/validation.py`):
```python
from common.validation import (
    validate_etcsl_record,      # Validate ETCSL extractor output
    validate_augmented_record,  # Validate graph augmentor output
    validate_training_record,   # Validate Alpaca-format records
    validate_dpo_record,        # Validate DPO preference pairs
    validate_batch,             # Validate batch with statistics
)
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `config.py` | Central config: `Paths`, `LLMConfig`, `EvalTargets`, `ControlTokens` |
| `common/io.py` | JSONL and chunked Parquet I/O utilities |
| `common/text.py` | Text cleaning (`clean_source_text`) and normalization |
| `common/validation.py` | Schema validation for pipeline boundaries |
| `common/metrics.py` | BLEU/chrF computation |
| `common/hardware.py` | GPU detection, BF16/Flash Attention capability |
| `processors/graph_engine/` | Entity graph, skeleton matching, substitution safety |
| `evaluation/evaluate_llm.py` | LLM inference and metric computation |

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

# Install with dev tools (linting, testing)
pip install -e ".[dev]"

# Install Axolotl for LLM training (separate)
pip install axolotl peft bitsandbytes

# Flash Attention (requires ninja, compatible GPU)
pip install ninja packaging
pip install flash-attn --no-build-isolation

# Extract data archives (required before training)
cd data && unzip -o "*.zip" && cd ..
```

## Linting & Testing

```bash
# Format code
black .
isort .

# Lint
ruff check .
mypy .
```

## Recent Refactoring (2026-01)

The codebase underwent significant cleanup and consolidation:

### Bug Fixes
- **Quality metadata interface**: `graph_augmentor.py` now outputs `quality` field at top level (was nested in `metadata`)
- **Composition ID tracking**: `consolidate_for_llm.py` now checks `metadata.template_line_id` for augmented records
- **Stop token consistency**: HuggingFace evaluation now uses Llama-3 `<|eot_id|>` stop token (matching vLLM)

### Code Consolidation
- **Control tokens**: Centralized in `config.py:ControlTokens` (previously duplicated in 4 files)
- **Text cleaning**: Centralized in `common/text.py:clean_source_text()` (previously duplicated)
- **Schema validation**: New `common/validation.py` for pipeline boundary checks

### Dead Code Removed
- `legacy_mt5/` directory (superseded by LLM approach)
- `common/training.py` (only used by legacy)
- `processors/gloss_augmentor.py` (superseded by graph_engine)
- `processors/quality_gate.py` (standalone, unused)
