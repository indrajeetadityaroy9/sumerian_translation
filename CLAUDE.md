# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Sumerian-English neural machine translation (NMT)** research project using a 3-phase training pipeline:
1. **MLM Pre-training** on monolingual Sumerian
2. **Alignment Pre-training** on synthetic parallel data (ORACC Silver)
3. **Seq2Seq Fine-tuning** on human-translated parallel corpus (ETCSL Gold)

**Optimized for:** 2× NVIDIA H100 80GB + Intel Xeon Platinum 8480+ (52 cores)

## Training Data

| Corpus | Pairs | Unique Translations | Quality | Source |
|--------|-------|---------------------|---------|--------|
| ETCSL Gold | 5,826 | 5,252 | Human-translated | Oxford Text Archive |
| ORACC Silver | 47,927 | 32,962 | Synthetic (word glosses) | Munich ORACC Server |
| **Combined** | **53,753** | **~38,000** | Mixed | - |

**Key improvement:** Previous pipeline had only 14 unique translations (mode collapse). Now fixed with proper ETCSL extraction and ORACC synthetic augmentation.

## Commands

### Full Training Pipeline

```bash
# 1. Tokenizer conversion (required first)
python convert_tokenizer.py

# 2. Phase 1: MLM Pre-training
python train_mlm.py                    # Default: tiny model
python train_mlm.py --model-size base  # Larger model for H100

# 3. Phase 2: NMT Fine-tuning
python train_nmt.py
python train_nmt.py --decoder-size base  # Larger decoder

# Multi-GPU training (recommended for 2x H100)
torchrun --nproc_per_node=2 train_mlm.py --model-size base
torchrun --nproc_per_node=2 train_nmt.py

# Alternative training approaches
python train_nmt_v2.py   # Pre-trained English decoder (frozen)
python train_nmt_v3.py   # DistilGPT-2 decoder
python train_mt5.py      # mT5 multilingual (300M params)
python train_mt5.py --model google/mt5-large  # Larger mT5

# Inference
python translate.py "lugal-e e2-gal-la-na ba-gen"
python translate.py --interactive
python translate.py --file input.txt --batch-size 32
```

### Data Extraction

```bash
# Extract ETCSL parallel corpus (Gold Standard - 5,826 pairs)
python -m etcsl_extractor.main --output-dir output

# Extract ORACC synthetic parallel corpus (Silver Standard)
python processors/oracc_gloss_extractor.py data/oracc/epsd2-literary -o output/oracc_literary_synthetic.jsonl
python processors/oracc_gloss_extractor.py data/oracc/epsd2-royal -o output/oracc_royal_synthetic.jsonl

# Combine ORACC corpora
cat output/oracc_literary_synthetic.jsonl output/oracc_royal_synthetic.jsonl > output/oracc_synthetic_combined.jsonl

# Prepare ML training data (merges ETCSL Gold + ORACC Silver)
python processors/prepare_training_data.py
```

## Hardware Optimizations

All training scripts auto-detect and enable:

| Optimization | Description | Speedup |
|--------------|-------------|---------|
| Multi-GPU (DDP) | Auto-detected via `torchrun` | 2× |
| BF16 precision | Native H100 support | 2× memory efficiency |
| Flash Attention 2 | Optimized attention kernels | 2-4× attention speed |
| torch.compile() | Kernel fusion | 10-30% |
| Gradient checkpointing | Trade compute for memory | 3-4× larger models |
| Parallel data loading | 8 workers, prefetch | 10-20% |

**Batch sizes** (per GPU, optimized for H100 80GB):
- `train_mlm.py`: 512 (was 64)
- `train_nmt.py`: 256 (was 32)
- `train_mt5.py`: 64 (was 8)

**Model size options** (`--model-size` / `--decoder-size`):
- `tiny`: ~4M params (original)
- `small`: ~30M params
- `base`: ~125M params (recommended for H100)
- `large`: ~355M params

## Architecture

### 3-Phase Training

1. **Phase 1 (MLM)**: Train RoBERTa on 500K words of monolingual Sumerian from ORACC. Learns Sumerian grammar/morphology.

2. **Phase 2 (Alignment)**: Pre-train Seq2Seq on 47,927 ORACC Silver pairs (synthetic English from word glosses). Learns Sumerian→English alignment patterns.

3. **Phase 3 (Fine-tuning)**: Fine-tune on 5,826 ETCSL Gold pairs (human translations). Refines fluency and translation quality.

### Data Flow

```
ETCSL XML ──→ etcsl_extractor/ ──→ parallel_corpus.jsonl (Gold: 5,826 pairs)
                                              ↓
ORACC JSON ─→ oracc_gloss_extractor.py ─→ oracc_synthetic.jsonl (Silver: 47,927 pairs)
                                              ↓
                               prepare_training_data.py
                                              ↓
                   train.jsonl + valid.jsonl + corpus_monolingual.txt
                                              ↓
                   train_mlm.py → train_nmt.py (Silver→Gold) → translate.py
```

### Key Directories

- `etcsl_extractor/` - ETCSL parsing: XML → JSONL parallel corpus
- `processors/` - Data prep: normalization, augmentation, ORACC extraction
- `models/` - Trained model checkpoints
- `output/` - Extracted corpora (parallel_corpus.jsonl, oracc_synthetic_combined.jsonl)
- `output_training_v2_clean/` - Prepared training data (train/valid splits)
- `data/oracc/` - Raw ORACC corpus data (epsd2-literary, epsd2-royal, etcsl)

### Critical Modules

- `etcsl_extractor/config.py` - Determinative mappings, special characters, POS tags
- `processors/oracc_gloss_extractor.py` - ORACC Silver corpus: Sumerian lines + synthetic English from glosses
- `processors/normalization_bridge.py` - ETCSL↔ORACC text normalization (ĝ vs ŋ)
- `processors/gloss_augmentor.py` - Lexical substitution for data augmentation

## Domain Knowledge

### Normalization

ETCSL and ORACC use different conventions that must be unified:
- ETCSL: `ĝ`, subscripts like `₂`
- ORACC: `ŋ`, different subscript handling

Use `normalization_bridge.py` for consistent tokenization.

### Data Quality

- Administrative texts (epsd2-admin-ur3) are formulaic receipts - use sparingly
- Literary texts (epsd2-literary) match ETCSL style - prioritize these
- Quality flags track damage/supplied/unclear markers in source texts

### Evaluation Targets

- BLEU: 15-25 (low-resource baseline)
- chrF: 30-40
- Word alignment F1: 0.6+

## Configuration

Paths and hyperparameters in script headers. CLI flags for runtime customization:

```bash
# Common flags across training scripts
--batch-size N          # Per-device batch size
--epochs N              # Training epochs
--model-size SIZE       # tiny/small/base/large
--no-compile            # Disable torch.compile()
--no-gradient-checkpointing  # Disable gradient checkpointing
--no-flash-attn         # Disable Flash Attention 2
```

## Dependencies

```
transformers datasets torch numpy evaluate sacrebleu lxml sentencepiece flash-attn
```
