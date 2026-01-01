"""
LLM-Based Paraphrase Augmentation for Sumerian-English Translation

Uses a local LLM (Flan-T5 or similar) to generate diverse paraphrases
of the 14 original English translations.

Usage:
    python augment_with_paraphrasing.py --num-paraphrases 30
"""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

# Configuration
INPUT_FILE = "output_training_v2_clean/finetune/train_augmented_v2.jsonl"
OUTPUT_FILE = "output_training_v2_clean/finetune/train_paraphrase_augmented.jsonl"

# Paraphrase prompts for diversity
PARAPHRASE_PROMPTS = [
    "paraphrase: {}",
    "rewrite this sentence: {}",
    "say this differently: {}",
    "rephrase: {}",
]

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def generate_paraphrases(model, tokenizer, sentence, num_paraphrases=10, device='cuda'):
    """Generate diverse paraphrases using beam search with different seeds."""
    paraphrases = set()
    paraphrases.add(sentence)  # Include original

    for prompt_template in PARAPHRASE_PROMPTS:
        input_text = prompt_template.format(sentence)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate multiple outputs with different parameters
        for temp in [0.7, 0.9, 1.1]:
            for top_p in [0.85, 0.95]:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=64,
                        num_return_sequences=min(5, num_paraphrases),
                        do_sample=True,
                        temperature=temp,
                        top_p=top_p,
                        num_beams=1,
                    )

                for output in outputs:
                    paraphrase = tokenizer.decode(output, skip_special_tokens=True)
                    # Filter out bad paraphrases
                    if (len(paraphrase) > 5 and
                        paraphrase.lower() != sentence.lower() and
                        not paraphrase.startswith('<') and
                        len(paraphrase.split()) >= 3):
                        paraphrases.add(paraphrase)

                if len(paraphrases) >= num_paraphrases:
                    return list(paraphrases)[:num_paraphrases]

    return list(paraphrases)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-paraphrases", type=int, default=30,
                        help="Number of paraphrases per original sentence")
    parser.add_argument("--model", type=str, default="google/flan-t5-base",
                        help="Paraphrase model to use")
    args = parser.parse_args()

    print("=" * 70)
    print("LLM Paraphrase Augmentation")
    print("=" * 70)

    # Load original data
    print(f"\nLoading data from {INPUT_FILE}...")
    train_data = load_jsonl(INPUT_FILE)

    # Group by target (English translation)
    target_to_sources = defaultdict(list)
    for item in train_data:
        target = item["target"]["text"]
        source = item["source"]["text_normalized"]
        target_to_sources[target].append(source)

    print(f"Found {len(target_to_sources)} unique English translations")
    print(f"Total examples: {len(train_data)}")

    # Load paraphrase model
    print(f"\nLoading paraphrase model: {args.model}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Generate paraphrases for each unique target
    print(f"\nGenerating {args.num_paraphrases} paraphrases per translation...")
    target_paraphrases = {}

    for target in tqdm(target_to_sources.keys(), desc="Generating paraphrases"):
        paraphrases = generate_paraphrases(
            model, tokenizer, target,
            num_paraphrases=args.num_paraphrases,
            device=device
        )
        target_paraphrases[target] = paraphrases

    # Show examples
    print("\n" + "=" * 70)
    print("PARAPHRASE EXAMPLES")
    print("=" * 70)
    for target, paraphrases in list(target_paraphrases.items())[:3]:
        print(f"\nOriginal: {target}")
        print(f"Paraphrases ({len(paraphrases)}):")
        for p in paraphrases[:5]:
            print(f"  - {p}")

    # Create augmented dataset
    print("\n" + "=" * 70)
    print("CREATING AUGMENTED DATASET")
    print("=" * 70)

    augmented_data = []

    for item in tqdm(train_data, desc="Augmenting"):
        original_target = item["target"]["text"]
        source = item["source"]["text_normalized"]

        # Add original
        augmented_data.append(item)

        # Add paraphrased versions
        paraphrases = target_paraphrases.get(original_target, [original_target])
        for paraphrase in paraphrases:
            if paraphrase != original_target:
                new_item = {
                    "source": item["source"].copy(),
                    "target": {
                        "text": paraphrase,
                        "original": original_target,
                        "augmentation": "paraphrase"
                    }
                }
                augmented_data.append(new_item)

    # Shuffle
    random.shuffle(augmented_data)

    # Save
    save_jsonl(augmented_data, OUTPUT_FILE)

    # Statistics
    new_targets = set(item["target"]["text"] for item in augmented_data)

    print(f"\n" + "=" * 70)
    print("AUGMENTATION COMPLETE")
    print("=" * 70)
    print(f"Original examples: {len(train_data)}")
    print(f"Augmented examples: {len(augmented_data)}")
    print(f"Expansion factor: {len(augmented_data)/len(train_data):.1f}x")
    print(f"Original unique translations: {len(target_to_sources)}")
    print(f"New unique translations: {len(new_targets)}")
    print(f"Diversity improvement: {len(new_targets)/len(target_to_sources):.1f}x")
    print(f"\nSaved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
