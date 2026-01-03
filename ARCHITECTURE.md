# Technical Overview: Sumerian-to-English Neural Machine Translation

## Project Summary

This project implements a **low-resource Neural Machine Translation (NMT)** system for translating ancient **Sumerian cuneiform texts** into modern **English**. It uses a two-phase LLM fine-tuning approach (SFT + DPO) with Llama-3.1-8B-Instruct, trained on ~46,000 samples from the Electronic Text Corpus of Sumerian Literature (ETCSL).

---

## Main Components

### 1. Corpus Extraction (`etcsl_extractor/`)
Parses ETCSL XML files to extract parallel Sumerian-English sentence pairs.

### 2. Data Augmentation (`processors/graph_engine/`)
Graph-based entity substitution that expands training data by swapping named entities (gods, kings, places) between structurally similar sentences.

### 3. Training Pipeline (`configs_llm/`)
Two-phase LLM fine-tuning via Axolotl:
- **SFT**: Supervised fine-tuning on translation pairs
- **DPO**: Direct Preference Optimization for fluency alignment

### 4. Evaluation Suite (`evaluation/`)
Metric computation (BLEU, chrF) and cross-architecture comparison tools.

---

## Theoretical Concepts

### Low-Resource NMT
Sumerian has limited parallel data (~6,000 original pairs). The project addresses this through:
- **Data Augmentation**: Graph-based entity substitution multiplies training examples
- **Transfer Learning**: Fine-tuning a pre-trained LLM (Llama-3) rather than training from scratch
- **LoRA**: Parameter-efficient fine-tuning reduces memory requirements

### Two-Circle Augmentation
```
Circle 1: ETCSL ↔ ETCSL
  - Swap entities between different compositions in the same corpus
  - Example: "Enlil built the temple" → "Enki built the temple"

Circle 2: ORACC → ETCSL
  - Link monolingual ORACC texts via glossary
  - Use ETCSL translations as templates for ORACC source lines
```

### Skeleton Similarity
Prevents "Bag-of-Entities" fallacy where lines with same entity pattern but different grammar are incorrectly matched:
```
"Enlil built the temple" (skeleton: "ENTITY built the temple")
"Enlil destroyed the temple" (skeleton: "ENTITY destroyed the temple")
→ Different skeletons, rejected match (Levenshtein < 85%)
```

### Direct Preference Optimization (DPO)
Trains the model to prefer fluent human translations over mechanical gloss concatenations:
```
Chosen:   "The king went to his palace" (fluent)
Rejected: "king palace go" (word-by-word gloss)
```

---

## File Tree with Functionality

```
sumerian_translation/
│
├── config.py                           # Central configuration hub
│   └── Classes: Paths, LLMConfig, LegacyMT5Config, ControlTokens, EvalTargets
│   └── Functions: get_train_file(), get_model_checkpoint(), get_llm_checkpoint()
│   └── Purpose: Single source of truth for paths, hyperparameters, and targets
│
├── CLAUDE.md                           # Claude Code guidance document
│
├── common/                             # Shared utilities across all modules
│   ├── __init__.py                     # Package exports
│   ├── hardware.py                     # GPU/hardware detection
│   │   └── get_hardware_info(): Detect CUDA, BF16, Flash Attention capability
│   │   └── setup_device(): Configure device for training (DDP-aware)
│   │   └── is_main_process(): Check if rank 0 in distributed training
│   ├── io.py                           # JSONL file I/O utilities
│   │   └── load_jsonl(): Load entire file into memory
│   │   └── iter_jsonl(): Memory-efficient streaming iterator
│   │   └── save_jsonl(): Write records to JSONL format
│   ├── metrics.py                      # Translation quality metrics
│   │   └── load_metrics(): Load SacreBLEU and chrF metrics
│   │   └── compute_bleu_chrf(): Compute scores for predictions vs references
│   │   └── create_compute_metrics_fn(): HuggingFace Trainer-compatible callback
│   └── quality.py                      # Data quality filters
│       └── is_mostly_broken(): Detect illegible lines (>50% broken markers)
│       └── is_valid_length(): Check word count bounds
│       └── filter_duplicates(): Remove overly frequent patterns
│       └── is_valid_pair(): Validate source-target length ratios
│
├── configs_llm/                        # Axolotl training configurations
│   ├── llama3_sft.yaml                 # Supervised Fine-Tuning config
│   │   └── Defines: base_model, LoRA params (r=64, alpha=128)
│   │   └── Defines: dataset path, sequence_len=512, epochs=3
│   │   └── Defines: sample_packing=true, flash_attention=true
│   ├── llama3_dpo.yaml                 # DPO alignment config
│   │   └── Starts from: SFT checkpoint
│   │   └── Defines: rl=dpo, dpo_beta=0.1 (KL penalty)
│   │   └── Defines: Lower learning_rate=5e-5, epochs=1
│   └── deepspeed_zero3.json            # Multi-GPU DeepSpeed ZeRO-3 config
│       └── Enables: Memory-efficient distributed training
│
├── data/
│   ├── final_llm_ready/                # Production training data
│   │   ├── sft_train.json              # 45,915 SFT examples (Alpaca format)
│   │   ├── sft_test.json               # 640 test examples (composition-split)
│   │   └── dpo_pairs.json              # 5,017 DPO preference pairs
│   ├── archive/                        # Intermediate datasets with metadata
│   │   └── valid.jsonl                 # Validation set with named_entities field
│   └── raw_source/                     # Original Parquet source files
│       ├── etcsl_gold.parquet          # ETCSL parallel corpus
│       ├── glossary_sux.parquet        # Sumerian glossary for entity linking
│       ├── oracc_literary.parquet      # ORACC literary texts (monolingual)
│       └── oracc_royal.parquet         # ORACC royal inscriptions (monolingual)
│
├── etcsl_extractor/                    # ETCSL corpus extraction pipeline
│   ├── __init__.py
│   ├── config.py                       # ETCSL-specific paths and settings
│   ├── main.py                         # CLI entry point
│   │   └── Orchestrates full extraction pipeline
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── preprocessor.py             # XML preprocessing utilities
│   │   ├── transliteration_parser.py   # Parse Sumerian transliteration XML
│   │   │   └── Extracts: tokens, lemmas, entity types (DN/RN/GN), positions
│   │   └── translation_parser.py       # Parse English translation XML
│   │       └── Extracts: paragraphs, named_entities, corresp references
│   ├── processors/
│   │   ├── __init__.py
│   │   └── parallel_aligner.py         # Align source lines with translations
│   │       └── LineAligner: LUT-based range resolution for non-sequential IDs
│   │       └── ParallelAligner: Creates aligned pairs using corresp attributes
│   ├── exporters/
│   │   ├── __init__.py
│   │   ├── parallel_corpus_exporter.py # Export to JSONL format
│   │   └── text_generator.py           # Reconstructs normalized text from tokens
│   └── utils/
│       ├── __init__.py
│       └── xml_utils.py                # XML parsing helpers
│
├── processors/                         # Data processing and augmentation
│   ├── prepare_training_data.py        # Create train/valid split
│   │   └── Splits by composition_id to prevent data leakage
│   │   └── Outputs: train.jsonl, valid.jsonl
│   │
│   ├── entity_linker.py                # Glossary-based entity recognition
│   │   └── EntityLinker: Maps Sumerian lemmas → entity types (DN/RN/GN)
│   │   └── Fuzzy matching chain: exact → strip determinatives → normalize unicode
│   │   └── Manual aliases: Handle philological mismatches (inanna → Inanak)
│   │
│   ├── graph_engine/                   # Graph-based augmentation engine
│   │   ├── __init__.py                 # Package exports
│   │   ├── builder.py                  # Bipartite graph construction
│   │   │   └── EntityGraph: NetworkX graph connecting lines ↔ entities
│   │   │   └── LineNode: Represents a corpus line with entities and translations
│   │   │   └── from_etcsl(): Build graph from ETCSL parquet
│   │   │   └── from_oracc(): Build graph from ORACC using glossary linking
│   │   ├── matcher.py                  # Line matching with skeleton similarity
│   │   │   └── LineMatcher: Two-circle matching (ETCSL↔ETCSL, ORACC→ETCSL)
│   │   │   └── get_skeleton(): Extract non-entity tokens for comparison
│   │   │   └── compute_skeleton_similarity(): Levenshtein ratio (≥85% threshold)
│   │   │   └── CRITICAL: Prevents bag-of-entities fallacy
│   │   ├── swapper.py                  # Entity substitution with safety
│   │   │   └── EntitySwapper: Performs type-safe substitutions
│   │   │   └── Word boundary regex: Prevents substring matches ("Ur" in "Ur-Namma")
│   │   │   └── Compound phrase detection: Flags "King Enlil", "The god Utu"
│   │   └── safety.py                   # Type constraint enforcement
│   │       └── SafetyChecker: DN↔DN, RN↔RN, GN↔GN only
│   │       └── Blacklist: Context-sensitive words (an=sky, ki=earth)
│   │       └── Minimum frequency threshold for rare entities
│   │
│   ├── graph_augmentor.py              # Main augmentation pipeline
│   │   └── GraphAugmentor: Orchestrates full pipeline
│   │   └── run_circle1(): ETCSL ↔ ETCSL augmentation
│   │   └── run_circle2(): ORACC → ETCSL augmentation
│   │   └── Control tokens: <gold>, <silver>, <aug> for data provenance
│   │
│   ├── consolidate_for_llm.py          # Convert to Alpaca format
│   │   └── Creates: sft_train.json, sft_test.json
│   │   └── Diverse prompts: 5 instruction variants to prevent overfitting
│   │   └── Quality tier detection: gold/augmented/synthetic
│   │
│   ├── create_dpo_pairs.py             # Generate preference pairs
│   │   └── Chosen: Human translation (fluent)
│   │   └── Rejected: Concatenated glosses (word-by-word literal)
│   │   └── is_meaningfully_different(): Jaccard distance check
│   │
│   ├── consolidate_to_parquet.py       # Convert intermediate formats to Parquet
│   ├── gloss_augmentor.py              # Glossary-based augmentation (alternative)
│   ├── normalization_bridge.py         # Text normalization utilities
│   ├── oracc_core.py                   # ORACC-specific parsing utilities
│   └── quality_gate.py                 # Audit CSV generation for review
│
├── evaluation/                         # Model evaluation suite
│   ├── __init__.py
│   ├── evaluate_llm.py                 # LLM evaluation script
│   │   └── load_model_and_tokenizer(): Load fine-tuned model
│   │   └── load_peft_model(): Load LoRA adapter on base model
│   │   └── generate_translation(): Inference with Llama-3 format
│   │   └── extract_response(): Parse output, remove special tokens
│   │   └── evaluate_model(): Run on test set, compute BLEU/chrF
│   │   └── generate_report(): Create markdown evaluation report
│   └── compare_models.py               # Cross-architecture comparison
│       └── ModelResult: Container for evaluation metrics
│       └── create_comparison_table(): Markdown table generation
│       └── analyze_improvements(): Delta calculations between models
│       └── Purpose: Valid mT5 vs Llama-3 thesis comparison
│
├── legacy_mt5/                         # Archived mT5 code (for thesis comparison)
│   ├── common/
│   │   ├── data_loader.py              # mT5 data loading utilities
│   │   └── training.py                 # mT5 training loop
│   ├── train.py                        # mT5 fine-tuning script
│   ├── translate.py                    # mT5 interactive inference
│   ├── evaluate_academic.py            # mT5 evaluation with metrics
│   ├── test_inference.py               # Quick inference tests
│   └── run_ablation.py                 # Ablation study runner
│
├── models_llm/                         # LLM checkpoints (created during training)
│   ├── sumerian_llama3_sft/            # SFT checkpoint (Phase 1)
│   └── sumerian_llama3_dpo/            # DPO checkpoint (Phase 2)
│
└── output_training_v2_clean/           # Intermediate training outputs
    └── finetune/
        ├── train_augmented.jsonl       # Augmented training data
        ├── train_substitution.jsonl    # Entity substitution augmented
        ├── valid.jsonl                 # Validation set
        └── audit/                      # Quality audit CSVs
```

---

## Key Mechanisms

### 1. ETCSL Extraction Pipeline
```
XML Files → TransliterationParser → tokens with entity annotations
         → TranslationParser     → paragraphs with corresp refs
         → ParallelAligner       → aligned (source, target) pairs
         → JSONL export          → parallel_corpus.jsonl
```

### 2. Graph-Based Augmentation
```
1. Build bipartite graph: Lines ←→ Entities (NetworkX)
2. Index lines by entity pattern: "DN-GN-DN" → [line_ids]
3. For each source line with translation:
   a. Find candidate lines with same pattern
   b. Check skeleton similarity ≥ 85% (Levenshtein)
   c. Validate entity type safety (DN↔DN only)
   d. Perform substitution with word boundary regex
   e. Flag compound phrases for human review
4. Inject control token: <silver> (≥95% skeleton) or <aug>
```

### 3. Two-Phase LLM Training
```
Phase 1: SFT (Supervised Fine-Tuning)
  - Input: Alpaca-format translation pairs
  - LoRA: r=64, alpha=128, all attention + MLP layers
  - 3 epochs, lr=2e-4, batch=128 effective

Phase 2: DPO (Direct Preference Optimization)
  - Input: (instruction, input, chosen, rejected) pairs
  - Starts from SFT checkpoint
  - 1 epoch, lr=5e-5, beta=0.1 (KL penalty)
```

### 4. Prompt Format (Llama-3 Instruct)
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert Sumerologist...<|eot_id|><|start_header_id|>user<|end_header_id|>

Translate this Sumerian text into English:

lugal-e e2-gal-la-na ba-gen<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

---

## Target Metrics

| Metric | Target Range | Purpose |
|--------|--------------|---------|
| BLEU | 15-25 | Standard MT quality |
| chrF++ | 30-40 | Character-level similarity |
| BERTScore F1 | >0.60 | Semantic similarity |
| Named Entity Accuracy | >60% | Key differentiator from mT5 |
