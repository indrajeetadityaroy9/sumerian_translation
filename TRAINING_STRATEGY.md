# Sumerian-English Translation Training Strategy

## Overview

This document outlines the recommended approach for training a Sumerian-English translation model using the combined ETCSL and ORACC datasets.

## Datasets Available

| Dataset | Texts | Words | Type | Use |
|---------|-------|-------|------|-----|
| **ETCSL** (processed) | 394 | 160K | Parallel corpus | Fine-tuning |
| **epsd2-literary** | 1,022 | 257K | Monolingual + glosses | Pre-training |
| **epsd2-admin-ur3** | 80,181 | 3.6M | Administrative | Embeddings only |
| **dcclt** | 4,980 | 255K | Lexical lists | Sign learning |

**Additional recommended downloads:**
- `epsd2-royal.zip` - Royal inscriptions (stylistically close to ETCSL)
- `rinap.zip` - Neo-Assyrian royal inscriptions

## Critical Technical Requirements

### 1. Normalization Bridge

The datasets use different character conventions:

| Corpus | ŋ/ĝ | š | Subscripts | Determinatives |
|--------|-----|---|------------|----------------|
| ETCSL | ĝ | š | ₂ | `{{DET_DIVINE}}` |
| ORACC | ŋ | š | ₂ | `{d}` |

**Solution**: Use `normalization_bridge.py` to unify:

```python
from normalization_bridge import normalize_oracc, normalize_etcsl

# Both produce: "lugal-ĝu"
normalize_oracc("lugal-ŋu₁₀", keep_subscripts=False)
normalize_etcsl("lugal-ĝu")
```

### 2. Domain Filtering

**Problem**: 80K Ur III administrative texts are formulaic receipts:
```
1 sheep received, Ur-Namma, year 2
```

This biases models toward list patterns, not literary sentences.

**Solution**: Prioritize literary domains for pre-training:

| Priority | Corpus | Reason |
|----------|--------|--------|
| 1 | epsd2-literary | Direct match to ETCSL style |
| 2 | epsd2-royal | Royal praise poetry |
| 3 | rinap | Royal inscriptions |
| 4 | epsd2-admin-ur3 | Use for NE only |

### 3. Data Augmentation via Gloss Substitution

**Problem**: Only 5.6K ETCSL parallel pairs - too few for NMT.

**Solution**: Use ORACC glossary for lexical substitution:

```python
# Original ETCSL pair
src: "lugal-e e2-gal-la-na ba-gen"
tgt: "The king went to his palace"

# Substitute using ORACC gloss-sux.json
# lugal (king) -> nun (prince)
# e2-gal (palace) -> e2-an-na (temple)

src_aug: "nun-e e2-an-na-na ba-gen"
tgt_aug: "The prince went to his temple"
```

This can expand 5.6K pairs to ~20K synthetic pairs.

## Training Pipeline

### Phase 0: Tokenizer Training

Train BPE/WordPiece on combined normalized vocabulary:

```bash
# Combine all Sumerian text
cat etcsl_text.txt oracc_literary.txt oracc_royal.txt > all_sumerian.txt

# Train tokenizer (target: 8K-15K vocab)
sentencepiece_train --input=all_sumerian.txt \
    --model_prefix=sumerian_bpe \
    --vocab_size=12000 \
    --character_coverage=0.9995
```

### Phase 1: Pre-training (MLM)

Train Masked Language Model on ORACC (filtered):

| Config | Value |
|--------|-------|
| Architecture | RoBERTa-base |
| Data | epsd2-literary + epsd2-royal (~500K words) |
| Masking | 15% tokens |
| Epochs | 40 |
| Batch size | 32 |

**Objective**: Learn that `lugal` is a noun, `ba-gen` is a verb chain.

### Phase 2: Alignment Initialization

Use ORACC glossary to initialize cross-lingual embeddings:

```python
# From gloss-sux.json
alignments = {
    "lugal": "king",
    "nin": "lady",
    "dingir": "god",
    "e2": "house",
    # ... 3,600+ pairs
}

# Initialize encoder-decoder attention bias
for sux, eng in alignments.items():
    model.set_alignment_prior(sux, eng, weight=0.5)
```

### Phase 3: Fine-tuning (Seq2Seq)

Train translation on ETCSL parallel corpus:

| Config | Value |
|--------|-------|
| Architecture | Encoder (pre-trained) + Decoder (random init) |
| Data | ETCSL parallel_corpus.jsonl |
| Augmentation | Gloss substitution (4x expansion) |
| Epochs | 100 |
| Learning rate | 1e-5 (encoder), 1e-4 (decoder) |

## File Locations

```
/home/ubuntu/sumerian_improvement/
├── output/                          # ETCSL processed
│   ├── parallel_corpus.jsonl        # Translation pairs
│   ├── linguistic_annotations.jsonl # Token annotations
│   └── vocabulary.json              # Lemma inventory
│
├── oracc_data/                      # ORACC raw
│   ├── epsd2-literary/              # Literary texts
│   ├── dcclt/                       # Lexical texts
│   ├── epsd2-admin-ur3/             # Admin texts
│   └── normalization_bridge.py      # Normalization utils
│
└── TRAINING_STRATEGY.md             # This document
```

## Evaluation Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| BLEU | 15-25 | Low-resource baseline |
| chrF | 30-40 | Better for morphological languages |
| Word alignment F1 | 0.6+ | Using ORACC glosses as gold |

## Expected Challenges

1. **Morphological complexity**: Sumerian is agglutinative
2. **Word order**: SOV vs English SVO
3. **Implicit subjects**: Often dropped in Sumerian
4. **Poetic register**: ETCSL is literary, not colloquial

## Next Steps

1. [ ] Download `epsd2-royal.zip` for additional pre-training data
2. [ ] Create unified tokenizer training corpus
3. [ ] Implement gloss substitution augmentation
4. [ ] Set up MLM pre-training pipeline
5. [ ] Establish baseline with ETCSL-only fine-tuning
