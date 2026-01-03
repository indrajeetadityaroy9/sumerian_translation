# Sumerian Translation System - Code Audit Report

**Audit Date**: 2026-01-03
**Scope**: Primary Llama-3 pipeline (excluding `legacy_mt5/`)
**Goal**: Documentation/audit of correctness, robustness, and integration issues

---

## Executive Summary

This audit reviewed the complete Sumerian-to-English neural machine translation pipeline, covering data extraction, augmentation, training configuration, and evaluation.

**Critical Finding**: A prompt format mismatch between training and evaluation will cause severely degraded metrics when evaluating trained models.

| Severity | Count | Description |
|----------|-------|-------------|
| P0 (Critical) | 2 | Prompt format mismatch, DPO format inconsistency |
| P1 (High) | 4 | Missing NE evaluation, data format issues |
| P2 (Medium) | 4 | Hardware assumptions, robustness gaps |
| P3 (Low) | 3 | Missing features, edge cases |

---

## P0: Critical Issues

### P0-1: Training/Evaluation Prompt Format Mismatch

| Field | Value |
|-------|-------|
| **File:Line** | `configs_llm/llama3_sft.yaml:33-35`, `evaluation/evaluate_llm.py:154-160` |
| **Category** | Interface Mismatch |
| **Impact** | Model trained on Alpaca format receives Llama-3 Instruct format at evaluation, causing severely degraded BLEU/chrF |

**Details:**

The SFT training config uses:
```yaml
datasets:
  - path: data/final_llm_ready/sft_train.json
    type: alpaca  # <-- Produces "### Instruction:\n..." format
```

The evaluation code uses:
```python
def format_prompt(instruction: str, input_text: str) -> str:
    return LLMConfig.format_prompt(instruction, input_text)  # <-- Llama-3 Instruct format
```

`LLMConfig.format_prompt()` produces:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>
{instruction}
{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

**Recommended Fix**: Update `llama3_sft.yaml` to use Llama-3 native chat template:
```yaml
chat_template: llama3
datasets:
  - path: data/final_llm_ready/sft_train.json
    type: chat_template
```

---

### P0-2: DPO Training Format Inconsistency

| Field | Value |
|-------|-------|
| **File:Line** | `configs_llm/llama3_dpo.yaml:29-33` |
| **Category** | Interface Mismatch |
| **Impact** | DPO uses third format (`chatml.intel`), creating three-way format mismatch |

**Details:**

```yaml
datasets:
  - path: data/final_llm_ready/dpo_pairs.json
    type: chatml.intel  # <-- ChatML format
```

This creates:
- SFT: Alpaca format
- DPO: ChatML format
- Evaluation: Llama-3 Instruct format

All three are different, causing cumulative degradation.

**Recommended Fix**: Align all formats to Llama-3 Instruct (`chat_template: llama3`).

---

## P1: High Priority Issues

### P1-1: Missing Named Entity Evaluation

| Field | Value |
|-------|-------|
| **File:Line** | `evaluation/evaluate_llm.py` (entire file) |
| **Category** | Missing Functionality |
| **Impact** | Cannot measure NE accuracy (target >60%), key differentiator vs mT5 baseline |

**Details:**

`evaluate_llm.py` only computes BLEU and chrF. The CLAUDE.md documents NE accuracy as the "killer metric" but:
- No NE extraction from predictions
- No NE comparison logic
- `valid.jsonl` has `named_entities` field but it's unused

**Recommended Fix**: Add Named Entity evaluation:
1. Extract entities from predictions using regex/NER
2. Load `data/archive/valid.jsonl` with `named_entities` field
3. Compute precision/recall/F1 for entity types

---

### P1-2: DPO Output Contains Extra Meta Field

| Field | Value |
|-------|-------|
| **File:Line** | `processors/create_dpo_pairs.py:155-165` |
| **Category** | Data Format |
| **Impact** | Extra `meta` field may cause parsing issues with Axolotl |

**Details:**

```python
dpo_pairs.append({
    "instruction": INSTRUCTION,
    "input": clean_src,
    "chosen": chosen,
    "rejected": rejected,
    "meta": {  # <-- Extra field
        "source_id": ...,
        "chosen_len": ...,
        "rejected_len": ...
    }
})
```

Axolotl's `chatml.intel` format expects only `instruction`, `input`, `chosen`, `rejected`.

**Recommended Fix**: Remove `meta` field from DPO output or verify Axolotl ignores extra fields.

---

### P1-3: No Validation During SFT Training

| Field | Value |
|-------|-------|
| **File:Line** | `configs_llm/llama3_sft.yaml:38` |
| **Category** | Training Configuration |
| **Impact** | No early stopping signal, potential overfitting undetected |

**Details:**

```yaml
val_set_size: 0.0  # Using separate test file
```

While a separate test file exists (`sft_test.json`), no evaluation runs during training. This prevents:
- Early stopping based on validation loss
- Learning curve monitoring
- Hyperparameter tuning feedback

**Recommended Fix**: Add validation dataset path or set `val_set_size: 0.05`.

---

### P1-4: Special Tokens Configuration Gap

| Field | Value |
|-------|-------|
| **File:Line** | `configs_llm/llama3_sft.yaml:10-11`, `evaluation/evaluate_llm.py:61-62` |
| **Category** | Interface Mismatch |
| **Impact** | Pad token configuration may differ between training and inference |

**Details:**

Training config:
```yaml
special_tokens:
  pad_token: "<|end_of_text|>"
```

Evaluation code:
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

The training explicitly sets pad token, but evaluation fallback may use different token.

**Recommended Fix**: Explicitly set pad_token in evaluation to match training config.

---

## P2: Medium Priority Issues

### P2-1: Hardware Assumptions Not Validated

| Field | Value |
|-------|-------|
| **File:Line** | `configs_llm/llama3_sft.yaml:47`, `common/hardware.py:37-39` |
| **Category** | Hidden Assumption |
| **Impact** | Training will fail or silently degrade on non-H100 GPUs |

**Details:**

Config assumes:
- H100 80GB GPU (micro_batch_size=32)
- 52 vCPUs (dataloader_num_workers=16)
- BF16 precision (bf16=true)

No fallback configuration or documentation for:
- A100 (80GB but different performance)
- Consumer GPUs (24-48GB)
- FP16 fallback implications

**Recommended Fix**: Document hardware requirements and provide alternative configs.

---

### P2-2: Entity Linker Manual Aliases Limited

| Field | Value |
|-------|-------|
| **File:Line** | `processors/entity_linker.py:73-111` |
| **Category** | Hidden Assumption |
| **Impact** | Entities not in alias list may fail to link |

**Details:**

Manual aliases cover ~25 common forms but:
- Many royal names missing
- Regional geographic names missing
- Variant spellings not exhaustive

Failed lookups are logged but not reviewed.

**Recommended Fix**:
1. Generate alias list from corpus statistics
2. Log failed lookups to file for review
3. Consider expanding aliases based on actual failures

---

### P2-3: Skeleton Similarity Threshold Unvalidated

| Field | Value |
|-------|-------|
| **File:Line** | `processors/graph_engine/matcher.py:57` |
| **Category** | Implementation Deviation |
| **Impact** | 85% threshold may be too permissive or restrictive |

**Details:**

```python
SKELETON_SIMILARITY_THRESHOLD = 0.85
```

This threshold was chosen without documented empirical validation:
- Too low (80%): May include semantically different augmentations
- Too high (90%): May exclude valid augmentations
- Edge cases (85-90%): Currently included but flagged

**Recommended Fix**: Run ablation study on skeleton threshold (80%, 85%, 90%) and measure downstream BLEU.

---

### P2-4: Gloss Extraction Minimum Length

| Field | Value |
|-------|-------|
| **File:Line** | `processors/create_dpo_pairs.py:64` |
| **Category** | Implementation Deviation |
| **Impact** | DPO pairs with short glosses excluded |

**Details:**

```python
if len(glosses) < 3:  # Too short to be meaningful
    return None
```

Short but meaningful phrases may be excluded. No analysis of:
- How many pairs are excluded
- Whether 3 is optimal threshold
- Impact on DPO training

**Recommended Fix**: Log excluded pair count and validate threshold.

---

## P3: Low Priority Issues

### P3-1: Control Token Stripping Regex

| Field | Value |
|-------|-------|
| **File:Line** | `processors/consolidate_for_llm.py:46` |
| **Category** | Robustness |
| **Impact** | Non-standard token formats may not be stripped |

**Details:**

```python
CONTROL_TOKEN_PATTERN = re.compile(r'<(?:gold|aug|silver|gloss)>\s*')
```

Pattern assumes:
- Exact token names
- Optional trailing whitespace only
- No nesting

Edge cases not handled:
- `<GOLD>` (uppercase)
- `< gold>` (leading space)
- Multiple consecutive tokens

**Recommended Fix**: Add validation that no control tokens remain in final output.

---

### P3-2: Compound Phrase Detection Incomplete

| Field | Value |
|-------|-------|
| **File:Line** | `processors/graph_engine/swapper.py:68-74` |
| **Category** | Robustness |
| **Impact** | Some compound phrases may not be flagged for review |

**Details:**

Current patterns:
```python
COMPOUND_PATTERNS = [
    r'(king|queen|lord|lady|god|goddess)\s+',
    r'(the\s+\w+)\s+',
    r'(shepherd|ruler|priest|priestess)\s+',
    r"'s\s*$",
    r'\s+of\s+\w+$',
]
```

Missing patterns:
- `"son of X"`, `"daughter of X"`
- `"temple of X"`, `"city of X"`
- `"O X"` (vocative)
- `"divine X"`

**Recommended Fix**: Expand compound patterns based on corpus analysis.

---

### P3-3: Single Reference Evaluation

| Field | Value |
|-------|-------|
| **File:Line** | `evaluation/evaluate_llm.py:280-282` |
| **Category** | Missing Functionality |
| **Impact** | Lower BLEU scores than possible with multiple references |

**Details:**

```python
# References wrapped as single-element lists
references.append(reference)  # Single reference per example
```

Low-resource translation benefits from multiple reference translations, but:
- Only one reference used per example
- No support for loading multiple references
- BLEU penalizes valid paraphrases

**Recommended Fix**: Support multiple references if available in test data.

---

## Verified Correct Implementations

### Composition-Based Data Split
`processors/prepare_training_data.py:56-70` correctly:
- Groups alignments by composition_id
- Shuffles at composition level
- Maintains 90/10 split ratio
- Uses deterministic seed (42)

### Word Boundary Entity Substitution
`processors/graph_engine/swapper.py:178-179` correctly:
```python
pattern = r'\b' + re.escape(old_label) + r'\b'
```
Prevents substring matches (e.g., "Ur" in "Ur-Namma").

### Type-Safe Entity Swapping
`processors/graph_engine/safety.py:79-128` correctly:
- Enforces DN↔DN, RN↔RN, GN↔GN
- Prevents self-swaps
- Applies frequency threshold
- Maintains blacklist ('an', 'ki')

### Control Token Lifecycle
Tokens are:
- Added: `graph_augmentor.py` (during augmentation)
- Used: `quality_gate.py` (for audit)
- Stripped: `consolidate_for_llm.py` (before training)

### BLEU/chrF Computation
`common/metrics.py:28-67` correctly:
- Uses sacrebleu library
- Wraps references in lists
- Strips predictions before scoring

---

## Integration Contract Verification

### Data Flow Schemas

| Stage | Schema Check | Status |
|-------|--------------|--------|
| parallel_corpus.jsonl → prepare_training_data.py | ✓ `composition_id`, `source`, `target` fields present | OK |
| train.jsonl → graph_augmentor.py | ✓ `tokens` field with entity annotations | OK |
| train_substitution.jsonl → consolidate_for_llm.py | ✓ Control tokens stripped correctly | OK |
| sft_train.json → Axolotl | ⚠️ Format mismatch (Alpaca vs Llama-3) | **FAIL** |
| dpo_pairs.json → Axolotl | ⚠️ Extra `meta` field | **WARN** |

### Prompt Format Alignment

| Component | Format | Status |
|-----------|--------|--------|
| config.py `format_prompt()` | Llama-3 Instruct | Reference |
| llama3_sft.yaml | Alpaca (`type: alpaca`) | **MISMATCH** |
| llama3_dpo.yaml | ChatML (`type: chatml.intel`) | **MISMATCH** |
| evaluate_llm.py | Llama-3 Instruct (via config.py) | OK |

---

## Remediation Priority

### Immediate (Before Next Training Run)
1. **P0-1**: Fix SFT config prompt format
2. **P0-2**: Fix DPO config prompt format
3. **P1-4**: Align pad token configuration

### Short-Term (Within 1 Week)
4. **P1-1**: Implement Named Entity evaluation
5. **P1-2**: Remove meta field from DPO output or test compatibility
6. **P1-3**: Add validation during training

### Medium-Term (Within 1 Month)
7. **P2-1**: Document hardware requirements
8. **P2-3**: Validate skeleton threshold empirically
9. **P2-2**: Expand entity linker aliases

### Long-Term (Future Enhancements)
10. **P3-1**: Add control token validation
11. **P3-2**: Expand compound phrase patterns
12. **P3-3**: Support multiple references

---

## Files Reviewed

| Stage | Files |
|-------|-------|
| Core Config | `config.py`, `pyproject.toml` |
| ETCSL Extraction | `etcsl_extractor/parsers/transliteration_parser.py` |
| Data Prep | `processors/prepare_training_data.py`, `processors/normalization_bridge.py` |
| Graph Engine | `processors/graph_engine/builder.py`, `matcher.py`, `swapper.py`, `safety.py` |
| Consolidation | `processors/consolidate_for_llm.py`, `processors/create_dpo_pairs.py` |
| Training Config | `configs_llm/llama3_sft.yaml`, `llama3_dpo.yaml`, `deepspeed_zero3.json` |
| Evaluation | `evaluation/evaluate_llm.py`, `common/metrics.py`, `common/hardware.py` |

---

## Appendix: Key Code Paths

### Data Flow
```
ETCSL XML → parallel_corpus.jsonl → train/valid split → graph augmentation
→ consolidate_for_llm → sft_train.json → Axolotl SFT → DPO → Evaluation
```

### Entity Substitution Flow
```
EntityLinker (glossary lookup) → EntityGraph (bipartite graph)
→ LineMatcher (skeleton similarity) → EntitySwapper (word boundary safe)
→ QualityGate (audit CSV) → AugmentedPair
```

### Prompt Format (Expected)
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert Sumerologist...<|eot_id|><|start_header_id|>user<|end_header_id|>
Translate this Sumerian text into English:
lugal-e e2-gal-la-na ba-gen<|eot_id|><|start_header_id|>assistant<|end_header_id|>
The king went to his palace.
```
