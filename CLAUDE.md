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

## Commands

### Full Training Pipeline

```bash
# 1. Tokenizer conversion (required first)
python3.11 convert_tokenizer.py

# 2. Phase 1: MLM Pre-training (Sumerian language model)
python3.11 train_mlm.py                    # Default: tiny model
python3.11 train_mlm.py --model-size base  # Larger model for H100

# 3. Phase 2/3: NMT Fine-tuning (mT5-based, recommended)
python3.11 train_nmt.py                              # Default: mt5-small
python3.11 train_nmt.py --model google/mt5-base      # Larger model
python3.11 train_nmt.py --resume models/checkpoint   # Continue training
python3.11 train_nmt.py --label-smoothing 0.1        # With label smoothing
python3.11 train_nmt.py --early-stopping 5           # With early stopping

# Multi-GPU training (recommended for 2x H100)
torchrun --nproc_per_node=2 train_mlm.py --model-size base
torchrun --nproc_per_node=2 train_nmt.py --model google/mt5-base

# Inference
python3.11 translate.py "lugal-e e2-gal-la-na ba-gen"
python3.11 translate.py --interactive
python3.11 translate.py --file input.txt --batch-size 32
```

### Data Extraction

```bash
# Extract ETCSL parallel corpus (Gold Standard - 5,826 pairs)
python3.11 -m etcsl_extractor.main --output-dir output

# Extract ORACC synthetic parallel corpus (Silver Standard)
python3.11 processors/oracc_core.py data/oracc/epsd2-literary -o output/oracc_literary.jsonl --mode parallel
python3.11 processors/oracc_core.py data/oracc/epsd2-royal -o output/oracc_royal.jsonl --mode parallel

# Extract ORACC monolingual corpus (for MLM pre-training)
python3.11 processors/oracc_core.py data/oracc/epsd2-literary -o output/oracc_monolingual.txt --mode monolingual

# Combine ORACC corpora
cat output/oracc_literary.jsonl output/oracc_royal.jsonl > output/oracc_synthetic_combined.jsonl

# Prepare ML training data (merges ETCSL Gold + ORACC Silver)
python3.11 processors/prepare_training_data.py
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

**Batch sizes** (per GPU, optimized for H100 80GB):
- `train_mlm.py`: 512
- `train_nmt.py`: 64 (mT5)

**Model size options** (`--model-size`):
- `tiny`: ~4M params
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
ORACC JSON ─→ processors/oracc_core.py ─→ oracc_synthetic.jsonl (Silver: 47,927 pairs)
                                              ↓
                               prepare_training_data.py
                                              ↓
                   train.jsonl + valid.jsonl + corpus_monolingual.txt
                                              ↓
                   train_mlm.py → train_nmt.py → translate.py
```

### Directory Structure

```
sumerian_translation/
├── common/                          # Shared utilities
│   ├── io.py                        # load_jsonl, save_jsonl, iter_jsonl
│   ├── hardware.py                  # get_hardware_info, setup_precision
│   ├── metrics.py                   # compute_bleu_chrf
│   ├── training.py                  # create_training_args, apply_compile
│   └── quality.py                   # is_mostly_broken, is_valid_length
├── config.py                        # Centralized paths & defaults
├── train_mlm.py                     # Phase 1: MLM pre-training
├── train_nmt.py                     # Phase 2/3: mT5 NMT fine-tuning
├── translate.py                     # Inference CLI
├── test_inference.py                # Validation test script
├── convert_tokenizer.py             # SentencePiece → HuggingFace
├── processors/
│   ├── normalization_bridge.py      # ETCSL↔ORACC normalization
│   ├── oracc_core.py                # Unified ORACC parser
│   └── prepare_training_data.py     # Master data pipeline
├── etcsl_extractor/                 # ETCSL corpus extraction
│   ├── config.py                    # Determinatives, special chars
│   ├── main.py                      # CLI entry point
│   ├── parsers/                     # XML parsing
│   ├── exporters/                   # JSONL export
│   └── processors/                  # Alignment
├── models/                          # Trained checkpoints
├── output/                          # Extracted corpora
└── output_training_v2_clean/        # ML training data
```

### Critical Modules

- `config.py` - Centralized paths (Paths class) and hyperparameters (TrainingDefaults)
- `common/hardware.py` - Hardware detection (BF16, Flash Attention, GPU count)
- `processors/normalization_bridge.py` - ETCSL↔ORACC text normalization (ĝ vs ŋ)
- `processors/oracc_core.py` - Unified ORACC parser for monolingual/parallel extraction
- `etcsl_extractor/config.py` - Determinative mappings, special characters, POS tags

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

All configuration is centralized in `config.py`:

```python
from config import Paths, TrainingDefaults, ModelConfigs

# Paths
Paths.TRAIN_FILE      # Training data
Paths.VALID_FILE      # Validation data
Paths.NMT_CHECKPOINT  # Model output directory

# Hyperparameters
TrainingDefaults.MT5["batch_size"]      # 64
TrainingDefaults.MT5["learning_rate"]   # 3e-5
TrainingDefaults.MT5["task_prefix"]     # "translate Sumerian to English: "
```

CLI flags for runtime customization:

```bash
# Common flags across training scripts
--batch-size N          # Per-device batch size
--epochs N              # Training epochs
--lr RATE               # Learning rate
--no-compile            # Disable torch.compile()
--no-gradient-checkpointing  # Disable gradient checkpointing
--no-flash-attn         # Disable Flash Attention 2

# NMT-specific
--model MODEL           # mT5 variant (google/mt5-small, google/mt5-base, google/mt5-large)
--resume PATH           # Continue from checkpoint
--label-smoothing F     # Label smoothing factor (0.0-0.2)
--early-stopping N      # Early stopping patience
```

## Model Checkpoints

Trained models saved to `models/`:
- `sumerian_tiny_mlm/` - Phase 1 RoBERTa encoder
- `sumerian_mt5_final/` - mT5 Seq2Seq (recommended)

To continue training from checkpoint:
```bash
python3.11 train_nmt.py --resume models/sumerian_mt5_final
```

## Dependencies

```
transformers datasets torch numpy evaluate sacrebleu lxml sentencepiece flash-attn
```
