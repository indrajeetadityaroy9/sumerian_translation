"""
Compare inference between the initial and continued training models.
Also analyze the training data distribution.
"""
import json
from collections import Counter
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

VALID_JSONL = "output_training_v2_clean/finetune/valid.jsonl"
TRAIN_JSONL = "output_training_v2_clean/finetune/train.jsonl"
TASK_PREFIX = "translate Sumerian to English: "

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def main():
    print("=" * 70)
    print("Training Data Analysis")
    print("=" * 70)

    # Analyze training data distribution
    train_data = load_jsonl(TRAIN_JSONL)
    targets = [item["target"]["text"] for item in train_data]

    target_counts = Counter(targets)
    print(f"\nTotal training examples: {len(targets)}")
    print(f"Unique target translations: {len(target_counts)}")

    print("\nTop 15 most common target translations:")
    for i, (target, count) in enumerate(target_counts.most_common(15), 1):
        print(f"  {i:2}. ({count:4} times) {target}")

    # Check distribution
    print(f"\nTarget distribution analysis:")
    single_occurrence = sum(1 for c in target_counts.values() if c == 1)
    print(f"  Targets appearing only once: {single_occurrence}")
    print(f"  Targets appearing 2-10 times: {sum(1 for c in target_counts.values() if 2 <= c <= 10)}")
    print(f"  Targets appearing >10 times: {sum(1 for c in target_counts.values() if c > 10)}")

    # Validation data
    print("\n" + "=" * 70)
    print("Validation Data Analysis")
    print("=" * 70)

    valid_data = load_jsonl(VALID_JSONL)
    valid_targets = [item["target"]["text"] for item in valid_data]
    valid_target_counts = Counter(valid_targets)

    print(f"\nTotal validation examples: {len(valid_targets)}")
    print(f"Unique target translations: {len(valid_target_counts)}")

    print("\nTop 10 most common validation targets:")
    for i, (target, count) in enumerate(valid_target_counts.most_common(10), 1):
        print(f"  {i:2}. ({count:4} times) {target}")

    # Model comparison
    print("\n" + "=" * 70)
    print("Model Comparison: Initial vs Continued Training")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {
        "Initial (20 epochs)": "./models/sumerian_mt5_final",
        "Continued (30 epochs)": "./models/sumerian_mt5_continued",
    }

    # Test examples from validation set
    test_indices = [0, 1, 2, 7, 15, 20]  # Mix of different examples
    examples = [{
        "source": valid_data[i]["source"]["text_normalized"],
        "reference": valid_data[i]["target"]["text"],
    } for i in test_indices]

    for model_name, model_path in models.items():
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model = model.to(device)
        model.eval()

        for i, example in enumerate(examples):
            input_text = TASK_PREFIX + example["source"]
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=128, num_beams=4)

            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

            match = "✓" if prediction.lower().strip() == example["reference"].lower().strip() else "✗"
            print(f"\n  Example {i+1}: {match}")
            print(f"    Source:     {example['source']}")
            print(f"    Prediction: {prediction}")
            print(f"    Reference:  {example['reference']}")

if __name__ == "__main__":
    main()
