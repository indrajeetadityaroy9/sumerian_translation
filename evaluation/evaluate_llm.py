#!/usr/bin/env python3
"""
LLM Evaluation Suite for Sumerian-English Translation

Evaluates fine-tuned Llama-3 models on translation quality metrics.
Uses the same BLEU/chrF metrics as mT5 for valid comparison.

Usage:
    python -m evaluation.evaluate_llm
    python -m evaluation.evaluate_llm --model models_llm/sumerian_llama3_sft
    python -m evaluation.evaluate_llm --test-file data/final_llm_ready/sft_test.json
"""

import argparse
import json
import re
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Reuse existing metric logic for valid comparison
from common.metrics import load_metrics, compute_bleu_chrf
from common.hardware import get_hardware_info
from config import Paths, LLMConfig, get_llm_checkpoint


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

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
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

    Returns:
        Dictionary with metrics and predictions
    """
    if max_samples:
        test_data = test_data[:max_samples]

    predictions = []
    references = []

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

    # Compute metrics using same functions as mT5
    bleu_metric, chrf_metric = load_metrics()
    metrics = compute_bleu_chrf(predictions, references, bleu_metric, chrf_metric)

    # Add prediction stats
    metrics['num_samples'] = len(predictions)
    metrics['avg_pred_len'] = sum(len(p.split()) for p in predictions) / len(predictions)
    metrics['avg_ref_len'] = sum(len(r.split()) for r in references) / len(references)

    return {
        'metrics': metrics,
        'predictions': predictions,
        'references': references,
        'inputs': [ex['input'] for ex in test_data],
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

        f.write("## Quantitative Metrics\n\n")
        f.write("| Metric | Score |\n")
        f.write("|--------|-------|\n")
        f.write(f"| BLEU | {metrics['bleu']:.2f} |\n")
        f.write(f"| chrF | {metrics['chrf']:.2f} |\n")
        f.write(f"| Avg Prediction Length | {metrics['avg_pred_len']:.1f} words |\n")
        f.write(f"| Avg Reference Length | {metrics['avg_ref_len']:.1f} words |\n")

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

    # Load model
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

    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate_model(
        model, tokenizer, test_data,
        max_samples=args.max_samples
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"BLEU:  {results['metrics']['bleu']:.2f}")
    print(f"chrF:  {results['metrics']['chrf']:.2f}")

    # Generate report
    generate_report(results, args.output, str(args.model))


if __name__ == "__main__":
    main()
