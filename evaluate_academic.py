"""
Academic Evaluation Suite for Sumerian-English NMT

Generates quantitative metrics (BLEU, chrF, BERTScore) and qualitative
error analysis for research publication.

Usage:
    python3 evaluate_academic.py
    python3 evaluate_academic.py --model models/sumerian_mt5_final
    python3 evaluate_academic.py --test-file output_training_v2_clean/finetune/valid.jsonl
"""

import argparse
import json
import torch
import evaluate
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from config import Paths, TrainingDefaults


def load_data(path: Path) -> list:
    """Load JSONL test data."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def evaluate_model(
    model_path: str,
    test_file: str,
    output_report: str = "academic_report.md",
    batch_size: int = 16,
):
    """
    Run full academic evaluation suite.

    Args:
        model_path: Path to trained mT5 model
        test_file: Path to test JSONL file
        output_report: Path to output markdown report
        batch_size: Batch size for inference
    """
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use BF16 on compatible hardware
    if device == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    print(f"Loading test data from {test_file}...")
    data = load_data(Path(test_file))

    # Load metrics
    print("Loading evaluation metrics...")
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    # BERTScore is optional (requires bert-score package)
    try:
        bertscore = evaluate.load("bertscore")
        use_bertscore = True
    except Exception as e:
        print(f"BERTScore unavailable: {e}")
        use_bertscore = False

    sources = []
    references = []
    predictions = []
    task_prefix = TrainingDefaults.MT5["task_prefix"]

    print("Running inference...")
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i + batch_size]

        batch_sources = [entry["source"]["text_normalized"] for entry in batch]
        batch_refs = [entry["target"]["text"] for entry in batch]

        # Prepare inputs with task prefix
        input_texts = [task_prefix + src for src in batch_sources]

        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )

        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        sources.extend(batch_sources)
        references.extend(batch_refs)
        predictions.extend(batch_preds)

    print("Calculating metrics...")

    # BLEU score
    b_score = bleu.compute(predictions=predictions, references=[[r] for r in references])

    # chrF++ score
    c_score = chrf.compute(predictions=predictions, references=[[r] for r in references])

    results = {
        "BLEU": b_score['score'],
        "chrF++": c_score['score'],
    }

    # BERTScore (semantic similarity)
    bert_scores = None
    if use_bertscore:
        print("Computing BERTScore (this may take a while)...")
        bs_score = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            batch_size=batch_size
        )
        mean_bert = sum(bs_score['f1']) / len(bs_score['f1'])
        results["BERTScore_F1"] = mean_bert
        bert_scores = bs_score['f1']

    # Create DataFrame for analysis
    df = pd.DataFrame({
        "Source": sources,
        "Reference": references,
        "Prediction": predictions,
    })

    if bert_scores:
        df["BERTScore"] = bert_scores
        # Get worst predictions for error analysis
        worst_preds = df.sort_values(by="BERTScore").head(20)
    else:
        # Use string distance as fallback
        from difflib import SequenceMatcher
        df["Similarity"] = [
            SequenceMatcher(None, r, p).ratio()
            for r, p in zip(references, predictions)
        ]
        worst_preds = df.sort_values(by="Similarity").head(20)

    # Generate Markdown Report
    print(f"Generating report: {output_report}")
    with open(output_report, "w") as f:
        f.write("# Academic Evaluation Report\n\n")
        f.write(f"**Model:** `{model_path}`\n\n")
        f.write(f"**Test Set:** `{test_file}` ({len(data)} examples)\n\n")

        f.write("## 1. Quantitative Metrics\n\n")
        f.write("| Metric | Score |\n")
        f.write("|--------|-------|\n")
        for k, v in results.items():
            f.write(f"| {k} | {v:.2f} |\n")

        f.write("\n## 2. Score Distribution\n\n")
        f.write(f"- **Mean prediction length:** {sum(len(p.split()) for p in predictions) / len(predictions):.1f} words\n")
        f.write(f"- **Mean reference length:** {sum(len(r.split()) for r in references) / len(references):.1f} words\n")

        f.write("\n## 3. Qualitative Error Analysis\n\n")
        f.write("Lowest-scoring predictions for manual review:\n\n")

        for idx, row in worst_preds.iterrows():
            f.write(f"### Example {idx + 1}\n\n")
            f.write(f"**Sumerian:** `{row['Source']}`\n\n")
            f.write(f"**Reference:** {row['Reference']}\n\n")
            f.write(f"**Prediction:** {row['Prediction']}\n\n")
            if bert_scores:
                f.write(f"**BERTScore:** {row['BERTScore']:.4f}\n\n")
            else:
                f.write(f"**Similarity:** {row['Similarity']:.4f}\n\n")
            f.write("---\n\n")

        f.write("## 4. Sample Good Predictions\n\n")
        if bert_scores:
            best_preds = df.sort_values(by="BERTScore", ascending=False).head(10)
        else:
            best_preds = df.sort_values(by="Similarity", ascending=False).head(10)

        for idx, row in best_preds.iterrows():
            f.write(f"- **Sumerian:** `{row['Source'][:50]}...`\n")
            f.write(f"  - **Pred:** {row['Prediction']}\n")
            f.write(f"  - **Ref:** {row['Reference']}\n\n")

    print(f"\nEvaluation complete!")
    print(f"Results: BLEU={results['BLEU']:.2f}, chrF++={results['chrF++']:.2f}")
    if 'BERTScore_F1' in results:
        print(f"         BERTScore={results['BERTScore_F1']:.2f}")
    print(f"Report saved to: {output_report}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Academic Evaluation Suite for Sumerian-English NMT"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(Paths.NMT_CHECKPOINT),
        help="Path to trained model"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=str(Paths.VALID_FILE),
        help="Path to test JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="academic_report.md",
        help="Output report path"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size"
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        test_file=args.test_file,
        output_report=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
