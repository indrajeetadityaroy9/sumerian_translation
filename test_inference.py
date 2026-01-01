"""
Test inference on the trained Sumerian-English translation model.

Runs inference on multiple examples for robust verification.
"""
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Configuration
MODEL_PATH = "./models/sumerian_mt5_continued"
VALID_JSONL = "output_training_v2_clean/finetune/valid.jsonl"
TASK_PREFIX = "translate Sumerian to English: "
MAX_LENGTH = 128
NUM_EXAMPLES = 30  # Test on 30 examples

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def main():
    print("=" * 70)
    print("Sumerian-English Translation Model - Inference Test")
    print("=" * 70)

    # Load model and tokenizer
    print(f"\nLoading model from {MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully!")

    # Load validation data
    print(f"\nLoading validation data from {VALID_JSONL}...")
    raw_valid = load_jsonl(VALID_JSONL)

    # Prepare examples
    examples = [{
        "source": item["source"]["text_normalized"],
        "reference": item["target"]["text"],
    } for item in raw_valid[:NUM_EXAMPLES]]

    print(f"Testing on {len(examples)} examples\n")
    print("-" * 70)

    # Track metrics
    exact_matches = 0
    partial_matches = 0

    for i, example in enumerate(examples):
        source = example["source"]
        reference = example["reference"]

        # Prepare input
        input_text = TASK_PREFIX + source
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                num_beams=4,
                early_stopping=True,
            )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check match quality
        pred_lower = prediction.lower().strip()
        ref_lower = reference.lower().strip()

        if pred_lower == ref_lower:
            match_status = "✓ EXACT"
            exact_matches += 1
        elif any(word in pred_lower for word in ref_lower.split() if len(word) > 3):
            match_status = "~ PARTIAL"
            partial_matches += 1
        else:
            match_status = "✗ MISS"

        print(f"\nExample {i+1}: {match_status}")
        print(f"  Sumerian:  {source}")
        print(f"  Predicted: {prediction}")
        print(f"  Reference: {reference}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total examples: {len(examples)}")
    print(f"Exact matches:   {exact_matches} ({100*exact_matches/len(examples):.1f}%)")
    print(f"Partial matches: {partial_matches} ({100*partial_matches/len(examples):.1f}%)")
    print(f"Misses:          {len(examples) - exact_matches - partial_matches} ({100*(len(examples) - exact_matches - partial_matches)/len(examples):.1f}%)")

    # Test with some custom inputs
    print("\n" + "=" * 70)
    print("CUSTOM INFERENCE TESTS")
    print("=" * 70)

    custom_inputs = [
        "lugal e2 gal",
        "dingir an ki",
        "nin munus gal",
        "dumu lugal",
        "kur gal",
    ]

    for custom in custom_inputs:
        input_text = TASK_PREFIX + custom
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                num_beams=4,
                early_stopping=True,
            )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n  Input:  {custom}")
        print(f"  Output: {prediction}")

if __name__ == "__main__":
    main()
