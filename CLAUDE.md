# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Low-resource Neural Machine Translation system for Sumerian-to-English translation using **~46,000 training samples** from ETCSL (Electronic Text Corpus of Sumerian Literature).

**Architecture**: Llama-3.1-8B-Instruct with LoRA fine-tuning via Axolotl. Legacy mT5 code archived in `legacy_mt5/`.

## Common Commands

### LLM Training (Axolotl)

```bash
# Phase 1: Supervised Fine-Tuning (SFT)
accelerate launch -m axolotl.cli.train configs_llm/llama3_sft.yaml

# Phase 2: DPO Alignment (after SFT)
accelerate launch -m axolotl.cli.train configs_llm/llama3_dpo.yaml

# Multi-GPU with DeepSpeed ZeRO-3
accelerate launch --config_file configs_llm/deepspeed_zero3.json \
    -m axolotl.cli.train configs_llm/llama3_sft.yaml
```

### Evaluation

```bash
# Evaluate fine-tuned model
python -m evaluation.evaluate_llm --model models_llm/sumerian_llama3_sft

# Quick test (limited samples)
python -m evaluation.evaluate_llm --max-samples 50

# With 4-bit quantization (low VRAM)
python -m evaluation.evaluate_llm --model models_llm/sumerian_llama3_sft --use-4bit
```

### Data Pipeline

```bash
# Step 1: Extract ETCSL corpus from XML
python3 -m etcsl_extractor.main --output-dir output

# Step 2: Prepare train/valid split (composition-based)
python3 processors/prepare_training_data.py \
    --etcsl-path output/parallel_corpus.jsonl \
    --output-dir output_training_v2_clean

# Step 3: Run graph-based entity substitution augmentation
python3 processors/graph_augmentor.py

# Step 4: Consolidate for LLM fine-tuning (Alpaca format)
python3 processors/consolidate_for_llm.py

# Step 5: Create DPO preference pairs
python3 processors/create_dpo_pairs.py
```

## Architecture

### Two-Phase Training Pipeline

1. **SFT (Supervised Fine-Tuning)**: Train on 45,915 Sumerian→English pairs in Alpaca format
2. **DPO (Direct Preference Optimization)**: Align with 5,017 preference pairs (human translation vs. gloss concatenation)

### Data Augmentation: Two-Circle Approach

The `processors/graph_engine/` implements graph-based entity substitution:
- **Circle 1**: ETCSL ↔ ETCSL (swap entities between compositions in same corpus)
- **Circle 2**: ORACC → ETCSL (link monolingual ORACC texts via glossary, substitute into ETCSL templates)

Control tokens track data provenance: `<gold>` (original), `<silver>` (high-confidence match), `<aug>` (entity substitution)

### Key Modules

| Module | Purpose |
|--------|---------|
| `config.py` | Central config: `Paths`, `LLMConfig`, `EvalTargets` |
| `common/metrics.py` | BLEU/chrF computation (shared across architectures) |
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

## Dependencies

```bash
pip install axolotl peft bitsandbytes flash-attn sacrebleu bert-score evaluate
```
