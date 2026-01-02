# Sumerian NMT Execution Runbook

Complete execution order from fresh environment to trained SOTA model on H100 hardware.

## Step 0: Environment & Hardware Setup

**Goal:** Configure the Python environment and verify H100 acceleration.

### 1. Install System Dependencies
```bash
apt-get update && apt-get install -y git curl unzip
```

### 2. Install Python Libraries
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets evaluate sacrebleu bert_score pyarrow pandas sentencepiece protobuf tqdm
pip install ninja packaging  # Required for Flash Attention build
pip install flash-attn --no-build-isolation
```

### 3. Verify Hardware
```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, BF16: {torch.cuda.is_bf16_supported()}, GPUs: {torch.cuda.device_count()}')"
```
**Success Criteria:** `CUDA: True`, `BF16: True`, `GPUs: 2`

---

## Step 1: Data Acquisition (Raw Sources)

**Goal:** Download the raw legacy data.

### 1. Download ORACC (JSON)
```bash
mkdir -p data/oracc
cd data/oracc
curl -o epsd2-literary.zip http://oracc.museum.upenn.edu/json/epsd2-literary.zip
curl -o epsd2-royal.zip http://oracc.museum.upenn.edu/json/epsd2-royal.zip
unzip -q epsd2-literary.zip -d epsd2-literary
unzip -q epsd2-royal.zip -d epsd2-royal
cd ../..
```

### 2. Download ETCSL (XML)
Place ETCSL XML files in:
- `data/oracc/etcsl/transliterations/`
- `data/oracc/etcsl/translations/`

Or use Oxford Text Archive `ota_20` directory if available.

---

## Step 2: ETCSL Extraction (The "Gold" Corpus)

**Goal:** Parse the complex XML into clean JSONL.

```bash
python3 -m etcsl_extractor.main --output-dir output
```

**Output:** `output/parallel_corpus.jsonl` (~5,800 pairs)

---

## Step 3: Consolidation (The Parquet Transformation)

**Goal:** Convert all raw data to Parquet and archive the raw files.

### 1. Run Consolidation Script
```bash
python3 processors/consolidate_to_parquet.py \
    --oracc-dir data/oracc \
    --etcsl-jsonl output/parallel_corpus.jsonl \
    --output-dir data/consolidated \
    --include-glossary
```

**Output:** `data/consolidated/*.parquet` (Total ~11MB)

### 2. Verify Data Integrity (Optional)
```bash
python3 -c "import pandas as pd; print(pd.read_parquet('data/consolidated/etcsl_gold.parquet').iloc[0])"
```

---

## Step 4: Training Prep & Augmentation

**Goal:** Split data and create training inputs.

```bash
python3 processors/prepare_training_data.py \
    --etcsl-path output/parallel_corpus.jsonl \
    --output-dir output_training_v2_clean
```

**Output:**
- `output_training_v2_clean/finetune/train.jsonl`
- `output_training_v2_clean/finetune/valid.jsonl`

---

## Step 5: Model Training (The H100 Run)

**Goal:** Fine-tune mT5-Large using DDP across both GPUs.

```bash
USE_TF=0 torchrun --nproc_per_node=2 train.py \
    --model google/mt5-large \
    --batch-size 16 \
    --epochs 20 \
    --early-stopping 5
```

**Notes:**
- `mt5-large` is optimal for 2x H100s
- If OOM (unlikely on 80GB), reduce batch size to 8
- Watch for "Crossover Point" where Train Loss < Eval Loss (usually Epoch 10-15)

**Output:** `models/sumerian_mt5_final/`

---

## Step 6: Cleanup Raw Data (Optional)

**Goal:** Save disk space after successful consolidation.

```bash
# Only run if archives/raw_sources_2024.tar.gz backup exists
rm -rf data/oracc
rm -rf ota_20
```

---

## Step 7: Evaluation & Inference

### 1. Academic Evaluation (BLEU/chrF/BERTScore)
```bash
python3 evaluate_academic.py
```

**Output:** `academic_report.md`

### 2. Interactive Demo
```bash
python3 translate.py --interactive
```

**Example:**
- Input: `lugal-e e2 mu-du3`
- Output: `The king built the house.`

---

## Quick Reference

| Step | Command | Output |
|------|---------|--------|
| 2 | `python3 -m etcsl_extractor.main` | parallel_corpus.jsonl |
| 3 | `python3 processors/consolidate_to_parquet.py` | *.parquet files |
| 4 | `python3 processors/prepare_training_data.py` | train/valid.jsonl |
| 5 | `torchrun train.py` | trained model |
| 7 | `python3 evaluate_academic.py` | academic_report.md |

---

## Expected Results

| Metric | Target Range |
|--------|--------------|
| BLEU | 15-25 |
| chrF++ | 30-40 |
| BERTScore F1 | 0.6+ |

---

## Troubleshooting

### Flash Attention Build Fails
```bash
pip uninstall flash-attn
pip install flash-attn --no-build-isolation
```

### CUDA Out of Memory
Reduce batch size: `--batch-size 8`

### DDP Hangs
Set environment variable: `NCCL_DEBUG=INFO`
