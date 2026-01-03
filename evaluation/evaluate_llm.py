#!/usr/bin/env python3
"""
LLM Evaluation Suite for Sumerian-English Translation

Evaluates fine-tuned Llama-3 models on translation quality metrics.
Uses the same BLEU/chrF metrics as mT5 for valid comparison.

Includes Named Entity evaluation (P1-1 fix) - the key differentiator metric.

Supports high-throughput inference with vLLM for 10-100x speedup on dual H100.

Usage:
    python -m evaluation.evaluate_llm
    python -m evaluation.evaluate_llm --model models_llm/sumerian_llama3_sft
    python -m evaluation.evaluate_llm --test-file data/final_llm_ready/sft_test.json
    python -m evaluation.evaluate_llm --with-ne-eval  # Enable Named Entity evaluation
    python -m evaluation.evaluate_llm --use-vllm      # High-throughput with vLLM (dual H100)
"""

import argparse
import json
import re
import torch
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm

# Reuse existing metric logic for valid comparison
from common.metrics import load_metrics, compute_bleu_chrf
from common.hardware import get_hardware_info
from config import Paths, LLMConfig, get_llm_checkpoint


# =============================================================================
# NAMED ENTITY EVALUATION (P1-1 fix)
# =============================================================================

# Known entity patterns for extraction from English translations
# These are common Sumerian proper nouns that appear in ETCSL translations
# Entity lists derived from ETCSL corpus translations
# Source: Manual extraction from ETCSL parallel corpus
DIVINE_NAMES = {
    # Major deities
    'Enlil', 'Enki', 'Inanna', 'Nanna', 'Utu', 'An', 'Ninlil', 'Ninhursag',
    'Ninurta', 'Nergal', 'Ereshkigal', 'Dumuzi', 'Ningal', 'Suen', 'Ishkur',
    'Nammu', 'Nanshe', 'Ningirsu', 'Nisaba', 'Ashgi', 'Shara', 'Lulal',
    'Hendursag', 'Nuska', 'Gibil', 'Nidaba', 'Gatumdu', 'Bau', 'Asarluhi',
    # Additional deities from ETCSL
    'Adad', 'Kaal', 'Meslam', 'Dagan', 'Shul-pa-e', 'Geshtinanna',
    'Ninshubur', 'Ninazu', 'Numushda', 'Ninkasi', 'Enbilulu', 'Enkimdu',
}

ROYAL_NAMES = {
    # Ur III dynasty
    'Shulgi', 'Ur-Namma', 'Ur-Nammu', 'Urnamma', 'Amar-Sin', 'Shu-Sin', 'Ibbi-Sin',
    # Isin dynasty
    'Isin-Dagan', 'Ishme-Dagan', 'Iddin-Dagan', 'Lipit-Ishtar',
    # Lagash rulers
    'Gudea', 'Ur-Ningirsu', 'Ur-Bau',
    # Legendary/mythological kings
    'Gilgamesh', 'Enmerkar', 'Lugalbanda', 'Etana',
    # Akkadian kings
    'Sargon', 'Naram-Sin',
    # Other rulers
    'Rim-Sin', 'Hammurabi', 'Shulgi-simti', 'Arad-Nanna',
}

GEOGRAPHIC_NAMES = {
    # Major cities (modern names)
    'Nippur', 'Ur', 'Uruk', 'Eridu', 'Lagash', 'Girsu', 'Umma', 'Akkad',
    'Sumer', 'Babylon', 'Kish', 'Sippar', 'Larsa', 'Isin', 'Shuruppak',
    'Adab', 'Zabalam', 'Eshnunna', 'Der', 'Mari', 'Assur', 'Dilmun',
    # Foreign regions
    'Magan', 'Meluhha', 'Elam', 'Gutium', 'Aratta', 'Anshan',
    # Sumerian city names
    'Ekur', 'Eanna', 'Abzu', 'Eridug', 'Nibru', 'Unug', 'Urim',
    # Additional locations from ETCSL
    'Girtab', 'Bad-tibira', 'Kutha', 'Kulaba', 'Keš', 'Zimbir',
}

# Compile entity patterns for efficient matching
ALL_ENTITIES = DIVINE_NAMES | ROYAL_NAMES | GEOGRAPHIC_NAMES
ENTITY_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(e) for e in sorted(ALL_ENTITIES, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)


def extract_entities_from_text(text: str) -> Dict[str, Set[str]]:
    """
    Extract named entities from English translation text.

    Args:
        text: English translation text

    Returns:
        Dictionary with entity types as keys and sets of found entities
    """
    found = {
        'DN': set(),  # Divine Names
        'RN': set(),  # Royal Names
        'GN': set(),  # Geographic Names
    }

    # Find all matches
    for match in ENTITY_PATTERN.finditer(text):
        entity = match.group(1)
        entity_normalized = entity.title()

        # Classify by type
        if entity_normalized in DIVINE_NAMES or entity.lower() in {e.lower() for e in DIVINE_NAMES}:
            found['DN'].add(entity_normalized)
        elif entity_normalized in ROYAL_NAMES or entity.lower() in {e.lower() for e in ROYAL_NAMES}:
            found['RN'].add(entity_normalized)
        elif entity_normalized in GEOGRAPHIC_NAMES or entity.lower() in {e.lower() for e in GEOGRAPHIC_NAMES}:
            found['GN'].add(entity_normalized)

    return found


def load_validation_with_entities(valid_path: Path) -> List[Dict]:
    """
    Load validation data that includes named_entities annotations.

    Args:
        valid_path: Path to valid.jsonl with named_entities field

    Returns:
        List of validation records with entity annotations
    """
    records = []
    with open(valid_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records.append(record)
    return records


def compute_ne_metrics(
    predictions: List[str],
    references: List[str],
    reference_entities: Optional[List[Dict]] = None
) -> Dict[str, float]:
    """
    Compute Named Entity evaluation metrics.

    Computes precision, recall, and F1 for entity extraction:
    - Precision: Of entities in prediction, how many are in reference?
    - Recall: Of entities in reference, how many are in prediction?
    - F1: Harmonic mean of precision and recall

    Args:
        predictions: List of predicted translations
        references: List of reference translations
        reference_entities: Optional list of annotated entities from validation data

    Returns:
        Dictionary with NE metrics by type and overall
    """
    total_pred = defaultdict(int)
    total_ref = defaultdict(int)
    total_correct = defaultdict(int)

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # Extract entities from prediction
        pred_entities = extract_entities_from_text(pred)

        # Use annotated entities if available, otherwise extract from reference
        if reference_entities and i < len(reference_entities):
            ref_ent_data = reference_entities[i]
            # Handle different formats of entity annotations
            if isinstance(ref_ent_data, dict):
                ref_entities = {
                    'DN': set(ref_ent_data.get('DN', [])),
                    'RN': set(ref_ent_data.get('RN', [])),
                    'GN': set(ref_ent_data.get('GN', [])),
                }
            else:
                ref_entities = extract_entities_from_text(ref)
        else:
            ref_entities = extract_entities_from_text(ref)

        # Count matches by type
        for entity_type in ['DN', 'RN', 'GN']:
            pred_set = {e.lower() for e in pred_entities[entity_type]}
            ref_set = {e.lower() for e in ref_entities[entity_type]}

            total_pred[entity_type] += len(pred_set)
            total_ref[entity_type] += len(ref_set)
            total_correct[entity_type] += len(pred_set & ref_set)

    # Compute metrics
    metrics = {}

    for entity_type in ['DN', 'RN', 'GN']:
        precision = total_correct[entity_type] / max(1, total_pred[entity_type])
        recall = total_correct[entity_type] / max(1, total_ref[entity_type])
        f1 = 2 * precision * recall / max(1e-10, precision + recall)

        metrics[f'ne_{entity_type.lower()}_precision'] = precision
        metrics[f'ne_{entity_type.lower()}_recall'] = recall
        metrics[f'ne_{entity_type.lower()}_f1'] = f1

    # Overall metrics
    total_pred_all = sum(total_pred.values())
    total_ref_all = sum(total_ref.values())
    total_correct_all = sum(total_correct.values())

    overall_precision = total_correct_all / max(1, total_pred_all)
    overall_recall = total_correct_all / max(1, total_ref_all)
    overall_f1 = 2 * overall_precision * overall_recall / max(1e-10, overall_precision + overall_recall)

    metrics['ne_precision'] = overall_precision
    metrics['ne_recall'] = overall_recall
    metrics['ne_f1'] = overall_f1
    # Note: ne_accuracy is actually F1-score, kept as alias for EvalTargets compatibility
    # The name is misleading but preserved for backward compatibility
    metrics['ne_accuracy'] = overall_f1

    # Raw counts for debugging
    metrics['ne_pred_count'] = total_pred_all
    metrics['ne_ref_count'] = total_ref_all
    metrics['ne_correct_count'] = total_correct_all

    return metrics


# =============================================================================
# MULTI-REFERENCE SUPPORT (P3-3 fix)
# =============================================================================

def compute_bleu_chrf_multi_ref(
    predictions: List[str],
    references: List[List[str]],
    bleu_metric,
    chrf_metric
) -> Dict[str, float]:
    """
    Compute BLEU and chrF with multiple references per example.

    Args:
        predictions: List of predictions
        references: List of lists of references (multiple refs per example)
        bleu_metric: Pre-loaded BLEU metric
        chrf_metric: Pre-loaded chrF metric

    Returns:
        Dictionary with BLEU and chrF scores
    """
    predictions = [pred.strip() for pred in predictions]

    result_bleu = bleu_metric.compute(
        predictions=predictions,
        references=references
    )
    result_chrf = chrf_metric.compute(
        predictions=predictions,
        references=references
    )

    return {
        "bleu": result_bleu["score"],
        "chrf": result_chrf["score"]
    }


def load_test_data(test_file: Path) -> List[Dict]:
    """Load test data from JSON file."""
    with open(test_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_model_and_tokenizer(
    model_path: Path,
    use_4bit: bool = False,
    device_map: str = "auto"
) -> Tuple:
    """
    Load fine-tuned LLM model and tokenizer.

    Args:
        model_path: Path to fine-tuned model (or adapter)
        use_4bit: Use 4-bit quantization for inference
        device_map: Device mapping strategy

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    hw = get_hardware_info()
    print(f"Loading model from {model_path}...")

    if hw['cuda_available']:
        print(f"  GPU: {hw['gpu_names'][0] if hw['gpu_names'] else 'Unknown'}")
        print(f"  BF16: {hw['bf16_supported']}")

    # Tokenizer with explicit pad token to match training config (P1-4 fix)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Use same pad token as training config: "<|end_of_text|>"
    # This matches configs_llm/llama3_sft.yaml special_tokens.pad_token
    if tokenizer.pad_token is None:
        # For Llama-3, use end_of_text as pad token (same as training)
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = "<|end_of_text|>"
    tokenizer.padding_side = "left"  # For generation, pad on left

    # Quantization config
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if hw['bf16_supported'] else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        dtype = torch.bfloat16 if hw['bf16_supported'] else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    model.eval()
    print(f"  Model loaded: {model.config._name_or_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer


def load_peft_model(
    base_model: str,
    adapter_path: Path,
    use_4bit: bool = True
) -> Tuple:
    """
    Load a PEFT/LoRA adapter on top of base model.

    Args:
        base_model: HuggingFace model ID for base
        adapter_path: Path to LoRA adapter
        use_4bit: Use 4-bit quantization

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    hw = get_hardware_info()
    print(f"Loading base model: {base_model}")
    print(f"Loading adapter from: {adapter_path}")

    # Tokenizer with explicit pad token to match training config (P1-4 fix)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = "<|end_of_text|>"
    tokenizer.padding_side = "left"

    # Base model with quantization
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if hw['bf16_supported'] else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        dtype = torch.bfloat16 if hw['bf16_supported'] else torch.float16
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    # Load adapter
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    print(f"  Adapter loaded successfully")

    return model, tokenizer


def format_prompt(instruction: str, input_text: str) -> str:
    """
    Format input for Llama-3 Instruct model.

    Uses the same format as training for consistency.
    """
    return LLMConfig.format_prompt(instruction, input_text)


def extract_response(full_output: str, prompt: str) -> str:
    """
    Extract the model's response from the full generated output.

    Removes the prompt and any special tokens.
    """
    # Remove the prompt
    if prompt in full_output:
        response = full_output[len(prompt):]
    else:
        # Try to find assistant header
        match = re.search(r'<\|start_header_id\|>assistant<\|end_header_id\|>\s*', full_output)
        if match:
            response = full_output[match.end():]
        else:
            response = full_output

    # Clean up special tokens
    response = re.sub(r'<\|eot_id\|>.*', '', response, flags=re.DOTALL)
    response = re.sub(r'<\|.*?\|>', '', response)

    return response.strip()


def generate_translation(
    model,
    tokenizer,
    input_text: str,
    instruction: str = "Translate this Sumerian text into English:",
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    """
    Generate a translation using the LLM.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        input_text: Sumerian text to translate
        instruction: The instruction prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling probability

    Returns:
        Translated English text
    """
    prompt = format_prompt(instruction, input_text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=LLMConfig.MAX_SEQ_LEN,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Build stop token list (Llama-3 uses <|eot_id|> for end-of-turn)
    stop_token_ids = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id != tokenizer.unk_token_id:  # Only add if token exists
        stop_token_ids.append(eot_id)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=stop_token_ids,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    response = extract_response(full_output, prompt)

    return response


def evaluate_model(
    model,
    tokenizer,
    test_data: List[Dict],
    batch_size: int = 1,  # LLMs typically generate one at a time
    max_samples: Optional[int] = None,
    verbose: bool = True,
    compute_ne_eval: bool = False,
    reference_entities: Optional[List[Dict]] = None,
    use_multi_ref: bool = False,
) -> Dict:
    """
    Evaluate model on test set.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        test_data: List of test examples with 'input' and 'output' keys
        batch_size: Batch size (typically 1 for generation)
        max_samples: Limit number of samples (for quick testing)
        verbose: Print progress
        compute_ne_eval: Whether to compute Named Entity metrics (P1-1)
        reference_entities: Optional annotated entities for NE evaluation
        use_multi_ref: P3-3 fix - Use multiple references if available

    Returns:
        Dictionary with metrics and predictions

    P3-3 Multi-Reference Support:
        If use_multi_ref=True, test data can include a 'references' field
        containing a list of alternative translations. Example:

        {
            "input": "lugal-e e2-gal-la-na ba-gen",
            "output": "The king went to his palace.",
            "references": [
                "The king went to his palace.",
                "The king proceeded to his palace.",
                "The king journeyed to the palace."
            ]
        }

        BLEU/chrF will use all references for scoring, which reduces
        penalty for valid paraphrases.
    """
    if max_samples:
        test_data = test_data[:max_samples]
        if reference_entities:
            reference_entities = reference_entities[:max_samples]

    predictions = []
    references = []  # Single references for NE eval
    references_multi = []  # Multi-references for BLEU/chrF

    iterator = tqdm(test_data, desc="Generating") if verbose else test_data

    for example in iterator:
        input_text = example['input']
        reference = example['output']
        instruction = example.get('instruction', "Translate this Sumerian text into English:")

        prediction = generate_translation(
            model, tokenizer, input_text, instruction
        )

        predictions.append(prediction)
        references.append(reference)

        # P3-3 fix: Collect multiple references if available
        if use_multi_ref and 'references' in example:
            refs = example['references']
            if isinstance(refs, list) and len(refs) > 0:
                references_multi.append(refs)
            else:
                references_multi.append([reference])
        else:
            references_multi.append([reference])

    # Compute metrics using same functions as mT5
    bleu_metric, chrf_metric = load_metrics()

    # P3-3 fix: Use multi-reference scoring if available
    if use_multi_ref and any(len(refs) > 1 for refs in references_multi):
        metrics = compute_bleu_chrf_multi_ref(
            predictions, references_multi, bleu_metric, chrf_metric
        )
        metrics['multi_ref'] = True
        metrics['avg_refs_per_example'] = sum(len(r) for r in references_multi) / len(references_multi)
    else:
        metrics = compute_bleu_chrf(predictions, references, bleu_metric, chrf_metric)
        metrics['multi_ref'] = False

    # Add prediction stats
    metrics['num_samples'] = len(predictions)
    metrics['avg_pred_len'] = sum(len(p.split()) for p in predictions) / len(predictions)
    metrics['avg_ref_len'] = sum(len(r.split()) for r in references) / len(references)

    # Compute Named Entity metrics if requested (P1-1 fix)
    if compute_ne_eval:
        ne_metrics = compute_ne_metrics(predictions, references, reference_entities)
        metrics.update(ne_metrics)

    return {
        'metrics': metrics,
        'predictions': predictions,
        'references': references,
        'inputs': [ex['input'] for ex in test_data],
    }


def evaluate_with_vllm(
    model_path: Path,
    test_data: List[Dict],
    batch_size: int = 64,
    max_samples: Optional[int] = None,
    tensor_parallel_size: int = 2,
    compute_ne_eval: bool = False,
    reference_entities: Optional[List[Dict]] = None,
    use_multi_ref: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate using vLLM for high-throughput inference.

    Provides 10-100x throughput improvement over sequential HuggingFace generation
    by leveraging vLLM's continuous batching and tensor parallelism.

    Args:
        model_path: Path to fine-tuned model
        test_data: List of test examples
        batch_size: Batch size for vLLM (default: 64 for H100)
        max_samples: Optional limit on samples
        tensor_parallel_size: Number of GPUs (default: 2 for dual H100)
        compute_ne_eval: Whether to compute Named Entity metrics
        reference_entities: Optional annotated entities for NE evaluation
        use_multi_ref: Use multiple references if available
        verbose: Print progress

    Returns:
        Dictionary with metrics and predictions
    """
    try:
        from evaluation.vllm_inference import VLLMInference
    except ImportError as e:
        print(f"ERROR: vLLM not available: {e}")
        print("Install with: pip install vllm>=0.4.0")
        print("Or: pip install -e '.[vllm]'")
        return None

    if max_samples:
        test_data = test_data[:max_samples]
        if reference_entities:
            reference_entities = reference_entities[:max_samples]

    # Initialize vLLM engine
    print(f"\nInitializing vLLM with tensor_parallel_size={tensor_parallel_size}...")
    engine = VLLMInference(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
    )

    # Prepare inputs
    inputs = [ex['input'] for ex in test_data]
    references = [ex['output'] for ex in test_data]
    instructions = [
        ex.get('instruction', "Translate this Sumerian text into English:")
        for ex in test_data
    ]

    # Handle multi-reference mode
    references_multi = []
    for ex in test_data:
        if use_multi_ref and 'references' in ex:
            refs = ex['references']
            if isinstance(refs, list) and len(refs) > 0:
                references_multi.append(refs)
            else:
                references_multi.append([ex['output']])
        else:
            references_multi.append([ex['output']])

    # Generate in batches with vLLM
    print(f"\nGenerating {len(inputs)} translations with vLLM (batch_size={batch_size})...")
    predictions = []

    for i in tqdm(range(0, len(inputs), batch_size), desc="vLLM Batches", disable=not verbose):
        batch_inputs = inputs[i : i + batch_size]
        batch_instructions = instructions[i : i + batch_size]

        # Format prompts
        prompts = [
            LLMConfig.format_prompt(inst, inp)
            for inst, inp in zip(batch_instructions, batch_inputs)
        ]

        # Generate with vLLM
        batch_preds = engine.generate_batch(prompts)
        predictions.extend(batch_preds)

    # Compute metrics
    bleu_metric, chrf_metric = load_metrics()

    if use_multi_ref and any(len(refs) > 1 for refs in references_multi):
        metrics = compute_bleu_chrf_multi_ref(
            predictions, references_multi, bleu_metric, chrf_metric
        )
        metrics['multi_ref'] = True
        metrics['avg_refs_per_example'] = sum(len(r) for r in references_multi) / len(references_multi)
    else:
        metrics = compute_bleu_chrf(predictions, references, bleu_metric, chrf_metric)
        metrics['multi_ref'] = False

    metrics['num_samples'] = len(predictions)
    metrics['avg_pred_len'] = sum(len(p.split()) for p in predictions) / len(predictions)
    metrics['avg_ref_len'] = sum(len(r.split()) for r in references) / len(references)
    metrics['inference_mode'] = 'vllm'
    metrics['tensor_parallel_size'] = tensor_parallel_size

    # Compute NE metrics if requested
    if compute_ne_eval:
        ne_metrics = compute_ne_metrics(predictions, references, reference_entities)
        metrics.update(ne_metrics)

    return {
        'metrics': metrics,
        'predictions': predictions,
        'references': references,
        'inputs': inputs,
    }


def generate_report(
    results: Dict,
    output_path: Path,
    model_name: str = "Llama-3 SFT",
) -> None:
    """Generate markdown evaluation report."""
    metrics = results['metrics']
    predictions = results['predictions']
    references = results['references']
    inputs = results['inputs']

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# LLM Evaluation Report\n\n")
        f.write(f"**Model:** `{model_name}`\n\n")
        f.write(f"**Test Set Size:** {metrics['num_samples']} examples\n\n")

        f.write("## Translation Quality Metrics\n\n")
        f.write("| Metric | Score |\n")
        f.write("|--------|-------|\n")
        f.write(f"| BLEU | {metrics['bleu']:.2f} |\n")
        f.write(f"| chrF | {metrics['chrf']:.2f} |\n")
        f.write(f"| Avg Prediction Length | {metrics['avg_pred_len']:.1f} words |\n")
        f.write(f"| Avg Reference Length | {metrics['avg_ref_len']:.1f} words |\n")

        # Named Entity metrics if available (P1-1 fix)
        if 'ne_accuracy' in metrics:
            f.write("\n## Named Entity Metrics\n\n")
            f.write("| Metric | Score |\n")
            f.write("|--------|-------|\n")
            f.write(f"| **NE Accuracy (F1)** | **{metrics['ne_accuracy']:.2%}** |\n")
            f.write(f"| NE Precision | {metrics['ne_precision']:.2%} |\n")
            f.write(f"| NE Recall | {metrics['ne_recall']:.2%} |\n")
            f.write(f"| Entities in Predictions | {metrics['ne_pred_count']} |\n")
            f.write(f"| Entities in References | {metrics['ne_ref_count']} |\n")
            f.write(f"| Correct Matches | {metrics['ne_correct_count']} |\n")

            f.write("\n### By Entity Type\n\n")
            f.write("| Type | Precision | Recall | F1 |\n")
            f.write("|------|-----------|--------|----|\n")
            for etype in ['dn', 'rn', 'gn']:
                f.write(f"| {etype.upper()} | {metrics.get(f'ne_{etype}_precision', 0):.2%} | "
                       f"{metrics.get(f'ne_{etype}_recall', 0):.2%} | "
                       f"{metrics.get(f'ne_{etype}_f1', 0):.2%} |\n")

        f.write("\n## Sample Predictions\n\n")

        # Show first 10 examples
        for i in range(min(10, len(predictions))):
            f.write(f"### Example {i+1}\n\n")
            f.write(f"**Sumerian:** `{inputs[i][:100]}{'...' if len(inputs[i]) > 100 else ''}`\n\n")
            f.write(f"**Prediction:** {predictions[i]}\n\n")
            f.write(f"**Reference:** {references[i]}\n\n")
            f.write("---\n\n")

    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM on Sumerian-English Translation"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to fine-tuned model or adapter"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=LLMConfig.BASE_MODEL,
        help="Base model for PEFT adapters"
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Paths.SFT_TEST,
        help="Path to test JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("llm_evaluation_report.md"),
        help="Output report path"
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of test samples"
    )
    parser.add_argument(
        "--use-peft",
        action="store_true",
        help="Load as PEFT adapter"
    )
    parser.add_argument(
        "--with-ne-eval",
        action="store_true",
        help="Enable Named Entity evaluation (computes NE precision/recall/F1)"
    )
    parser.add_argument(
        "--valid-with-entities",
        type=Path,
        default=Paths.VALID_WITH_NE,
        help="Path to validation file with named_entities annotations"
    )
    parser.add_argument(
        "--multi-ref",
        action="store_true",
        help="P3-3 fix: Use multiple references if available in test data (expects 'references' field)"
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM for high-throughput inference (10-100x faster, requires dual H100)"
    )
    parser.add_argument(
        "--vllm-batch-size",
        type=int,
        default=64,
        help="Batch size for vLLM inference (default: 64)"
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=2,
        help="Tensor parallel size for vLLM (default: 2 for dual H100)"
    )

    args = parser.parse_args()

    # Resolve model path
    if args.model is None:
        args.model = get_llm_checkpoint()

    if not args.model.exists():
        print(f"ERROR: Model not found at {args.model}")
        print("Train a model first or specify a valid path.")
        return

    if not args.test_file.exists():
        print(f"ERROR: Test file not found at {args.test_file}")
        return

    # Load model (skip for vLLM mode - it handles model loading internally)
    model, tokenizer = None, None
    if not args.use_vllm:
        if args.use_peft:
            model, tokenizer = load_peft_model(
                args.base_model, args.model, args.use_4bit
            )
        else:
            model, tokenizer = load_model_and_tokenizer(
                args.model, args.use_4bit
            )

    # Load test data
    print(f"\nLoading test data from {args.test_file}...")
    test_data = load_test_data(args.test_file)
    print(f"  {len(test_data)} examples loaded")

    # Load reference entities for NE evaluation if requested (P1-1 fix)
    reference_entities = None
    if args.with_ne_eval:
        print(f"\nNamed Entity evaluation enabled")
        if args.valid_with_entities.exists():
            print(f"  Loading entity annotations from {args.valid_with_entities}...")
            valid_with_ne = load_validation_with_entities(args.valid_with_entities)
            # Extract named_entities field from each record
            reference_entities = [
                record.get('named_entities', {}) for record in valid_with_ne
            ]
            print(f"  {len(reference_entities)} entity annotations loaded")
        else:
            print(f"  Warning: {args.valid_with_entities} not found, extracting entities from references")

    # P3-3 fix: Check for multi-reference mode
    if args.multi_ref:
        print("\nMulti-reference mode: ENABLED")
        print("  (Will use 'references' field if available in test data)")

    # Evaluate using vLLM or HuggingFace
    if args.use_vllm:
        print("\n" + "=" * 60)
        print("Using vLLM for high-throughput inference")
        print("=" * 60)
        results = evaluate_with_vllm(
            model_path=args.model,
            test_data=test_data,
            batch_size=args.vllm_batch_size,
            max_samples=args.max_samples,
            tensor_parallel_size=args.tensor_parallel,
            compute_ne_eval=args.with_ne_eval,
            reference_entities=reference_entities,
            use_multi_ref=args.multi_ref,
        )
        if results is None:
            print("\n" + "=" * 60)
            print("WARNING: vLLM evaluation failed!")
            print("Falling back to HuggingFace (results may differ due to stop tokens)")
            print("=" * 60 + "\n")
            args.use_vllm = False

    if not args.use_vllm:
        # Standard HuggingFace evaluation
        print("\nRunning evaluation with HuggingFace...")
        results = evaluate_model(
            model, tokenizer, test_data,
            max_samples=args.max_samples,
            compute_ne_eval=args.with_ne_eval,
            reference_entities=reference_entities,
            use_multi_ref=args.multi_ref,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    if results['metrics'].get('inference_mode') == 'vllm':
        print(f"Inference: vLLM (tensor_parallel={results['metrics']['tensor_parallel_size']})")
    print(f"BLEU:  {results['metrics']['bleu']:.2f}")
    print(f"chrF:  {results['metrics']['chrf']:.2f}")

    # P3-3 fix: Show multi-ref info if used
    if results['metrics'].get('multi_ref'):
        print(f"  (Multi-reference scoring used, avg {results['metrics']['avg_refs_per_example']:.1f} refs/example)")

    # Print NE metrics if computed (P1-1 fix)
    if 'ne_accuracy' in results['metrics']:
        print(f"\n--- Named Entity Metrics ---")
        print(f"NE Accuracy (F1): {results['metrics']['ne_accuracy']:.2%}")
        print(f"NE Precision:     {results['metrics']['ne_precision']:.2%}")
        print(f"NE Recall:        {results['metrics']['ne_recall']:.2%}")
        print(f"Entities Found:   {results['metrics']['ne_pred_count']} pred / {results['metrics']['ne_ref_count']} ref")

    # Generate report
    generate_report(results, args.output, str(args.model))


if __name__ == "__main__":
    main()
