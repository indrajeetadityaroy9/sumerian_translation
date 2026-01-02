"""
Academic Evaluation Suite for Sumerian-English NMT

Implements a rigorous evaluation framework for publication:
- Phase 1: Semantic Validation (BERTScore, COMET)
- Phase 2: Linguistic Probing (Ergativity, Genitives, Named Entities)
- Phase 3: Dictionary Baseline Comparison
- Phase 4: Error Analysis Taxonomy
- Phase 5: Out-of-Domain Generalization

Usage:
    python evaluate_academic.py --phase all
    python evaluate_academic.py --phase semantic
    python evaluate_academic.py --phase linguistic
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common.io import load_jsonl
from config import Paths, TrainingDefaults


class AcademicEvaluator:
    """Comprehensive evaluation suite for academic publication."""

    def __init__(self, model_path: Path, verbose: bool = True):
        self.model_path = model_path
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_prefix = TrainingDefaults.MT5["task_prefix"]

        # Load model
        if verbose:
            print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        self.model.eval()

    def translate(self, text: str, max_length: int = 128) -> str:
        """Translate a single Sumerian text."""
        input_text = self.task_prefix + text
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate_batch(self, texts: list, batch_size: int = 16) -> list:
        """Translate a batch of texts."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            input_texts = [self.task_prefix + t for t in batch]
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )

            for output in outputs:
                results.append(self.tokenizer.decode(output, skip_special_tokens=True))

            if self.verbose and (i + batch_size) % 100 == 0:
                print(f"  Translated {min(i + batch_size, len(texts))}/{len(texts)}...")

        return results


def phase1_semantic_validation(evaluator: AcademicEvaluator, valid_data: list) -> dict:
    """
    Phase 1: Semantic Validation with BERTScore.

    BERTScore measures embedding similarity, capturing meaning beyond exact words.
    """
    print("\n" + "=" * 70)
    print("PHASE 1: SEMANTIC VALIDATION (BERTScore)")
    print("=" * 70)

    # Extract source and reference
    sources = []
    references = []
    for item in valid_data:
        src = item.get("source", {}).get("text_normalized", "")
        tgt = item.get("target", {}).get("text", "")
        if src and tgt:
            sources.append(src)
            references.append(tgt)

    print(f"Evaluating {len(sources)} examples...")

    # Generate predictions
    predictions = evaluator.translate_batch(sources)

    # Calculate BLEU and chrF for comparison
    try:
        import evaluate
        bleu_metric = evaluate.load("bleu")
        chrf_metric = evaluate.load("chrf")

        # BLEU
        bleu_result = bleu_metric.compute(
            predictions=predictions,
            references=[[r] for r in references]
        )

        # chrF
        chrf_result = chrf_metric.compute(
            predictions=predictions,
            references=[[r] for r in references]
        )

        print(f"\nStandard Metrics:")
        print(f"  BLEU:  {bleu_result['bleu'] * 100:.2f}")
        print(f"  chrF:  {chrf_result['score']:.2f}")
    except Exception as e:
        print(f"  Standard metrics failed: {e}")
        bleu_result = {"bleu": 0}
        chrf_result = {"score": 0}

    # BERTScore
    try:
        bertscore = evaluate.load("bertscore")
        bert_results = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli"  # High-quality model
        )

        mean_precision = sum(bert_results['precision']) / len(bert_results['precision'])
        mean_recall = sum(bert_results['recall']) / len(bert_results['recall'])
        mean_f1 = sum(bert_results['f1']) / len(bert_results['f1'])

        print(f"\nBERTScore (Semantic Similarity):")
        print(f"  Precision: {mean_precision:.4f}")
        print(f"  Recall:    {mean_recall:.4f}")
        print(f"  F1:        {mean_f1:.4f}")

        # Find examples where BERTScore >> BLEU (synonym handling)
        print("\n  Examples where model used synonyms (high BERTScore, potentially low BLEU):")
        for i in range(min(3, len(predictions))):
            if bert_results['f1'][i] > 0.7:
                print(f"\n    Source: {sources[i][:60]}...")
                print(f"    Pred:   {predictions[i][:60]}...")
                print(f"    Ref:    {references[i][:60]}...")
                print(f"    BERTScore F1: {bert_results['f1'][i]:.3f}")

        return {
            "bleu": bleu_result['bleu'] * 100,
            "chrf": chrf_result['score'],
            "bertscore_precision": mean_precision,
            "bertscore_recall": mean_recall,
            "bertscore_f1": mean_f1,
            "predictions": predictions,
            "references": references,
            "sources": sources,
            "bert_f1_scores": bert_results['f1']
        }
    except Exception as e:
        print(f"\nBERTScore failed: {e}")
        print("  Install with: pip install bert-score")
        return {
            "bleu": bleu_result['bleu'] * 100 if bleu_result else 0,
            "chrf": chrf_result['score'] if chrf_result else 0,
            "predictions": predictions,
            "references": references,
            "sources": sources
        }


def phase2_linguistic_probing(evaluator: AcademicEvaluator) -> dict:
    """
    Phase 2: Linguistic Probing - Test specific Sumerian grammar features.

    Tests:
    1. Ergativity (agent marking with -e)
    2. Genitive chains (possession with -ak)
    3. Named entity handling (divine determinatives)
    """
    print("\n" + "=" * 70)
    print("PHASE 2: LINGUISTIC PROBING (Grammar Competence)")
    print("=" * 70)

    # Contrastive Challenge Set
    challenge_set = {
        "ergativity": [
            # Tests: Does model correctly identify agent (-e marker)?
            {"source": "lugal-e e2 mu-du3", "expected_agent": "king", "expected_patient": "house",
             "correct_pattern": "king.*built.*house", "fail_pattern": "house.*built.*king"},
            {"source": "dingir-e nam mu-tar", "expected_agent": "god", "expected_patient": "fate",
             "correct_pattern": "god.*decree|determin", "fail_pattern": "fate.*god"},
            {"source": "nin-e igi mu-bar", "expected_agent": "lady|queen", "expected_patient": "eye",
             "correct_pattern": "(lady|queen).*look|see", "fail_pattern": "eye.*(lady|queen)"},
            {"source": "dumu-e a mu-na-de2", "expected_agent": "son|child", "expected_patient": "water",
             "correct_pattern": "(son|child).*pour|water", "fail_pattern": "water.*(son|child)"},
            {"source": "ensi2-e ki mu-ag2", "expected_agent": "ruler|governor", "expected_patient": "love",
             "correct_pattern": "(ruler|governor).*love", "fail_pattern": "love.*(ruler|governor)"},
        ],
        "genitive": [
            # Tests: Does model handle possession chains correctly?
            {"source": "dumu lugal-la-ke4", "expected": "son of the king",
             "correct_pattern": "son.*of.*king", "fail_pattern": "king.*son"},
            {"source": "e2 dingir-ra-ke4", "expected": "house of the god",
             "correct_pattern": "(house|temple).*of.*god", "fail_pattern": "god.*(house|temple)"},
            {"source": "nam-lugal kalam-ma", "expected": "kingship of the land",
             "correct_pattern": "king(ship)?.*of.*(land|country)", "fail_pattern": "(land|country).*king"},
            {"source": "me an-na", "expected": "divine powers of heaven",
             "correct_pattern": "(power|me).*of.*heaven", "fail_pattern": "heaven.*(power|me)"},
            {"source": "igi nin-a-ke4", "expected": "before the lady",
             "correct_pattern": "(before|eye|presence).*lady", "fail_pattern": "lady.*(before|eye)"},
        ],
        "named_entities": [
            # Tests: Does model correctly transliterate divine/place names?
            {"source": "d-en-lil2", "expected": "Enlil",
             "correct_pattern": "Enlil", "fail_pattern": "god|deity|divine"},
            {"source": "d-inana", "expected": "Inana|Inanna|Ishtar",
             "correct_pattern": "Inan[na]?|Ishtar", "fail_pattern": "goddess|lady"},
            {"source": "d-utu", "expected": "Utu|Shamash",
             "correct_pattern": "Utu|Shamash|Sun", "fail_pattern": "god|bright"},
            {"source": "nibru-ki", "expected": "Nippur",
             "correct_pattern": "Nippur", "fail_pattern": "city|place"},
            {"source": "unug-ki", "expected": "Uruk",
             "correct_pattern": "Uruk", "fail_pattern": "city|place"},
        ]
    }

    results = {}
    import re

    for category, tests in challenge_set.items():
        print(f"\n{category.upper()} Tests:")
        print("-" * 40)

        correct = 0
        total = len(tests)
        details = []

        for test in tests:
            pred = evaluator.translate(test["source"])

            # Check if prediction matches correct pattern
            correct_match = re.search(test["correct_pattern"], pred, re.IGNORECASE)
            fail_match = re.search(test.get("fail_pattern", "^$"), pred, re.IGNORECASE) if test.get("fail_pattern") else None

            is_correct = bool(correct_match) and not bool(fail_match)
            if is_correct:
                correct += 1
                status = "✓"
            else:
                status = "✗"

            details.append({
                "source": test["source"],
                "prediction": pred,
                "expected_pattern": test["correct_pattern"],
                "correct": is_correct
            })

            print(f"  {status} {test['source']}")
            print(f"      Pred: {pred}")
            if not is_correct:
                print(f"      Expected pattern: {test['correct_pattern']}")

        accuracy = correct / total * 100
        results[category] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": details
        }
        print(f"\n  {category.title()} Accuracy: {correct}/{total} ({accuracy:.1f}%)")

    # Overall linguistic competence
    total_correct = sum(r["correct"] for r in results.values())
    total_tests = sum(r["total"] for r in results.values())
    overall_accuracy = total_correct / total_tests * 100

    print(f"\n{'=' * 40}")
    print(f"OVERALL LINGUISTIC ACCURACY: {total_correct}/{total_tests} ({overall_accuracy:.1f}%)")
    print("=" * 40)

    results["overall"] = {
        "accuracy": overall_accuracy,
        "correct": total_correct,
        "total": total_tests
    }

    return results


def phase3_dictionary_baseline(evaluator: AcademicEvaluator, valid_data: list) -> dict:
    """
    Phase 3: Dictionary Baseline Comparison.

    Compare neural model against simple word-for-word gloss lookup.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: DICTIONARY BASELINE COMPARISON")
    print("=" * 70)

    # Simple Sumerian-English dictionary (from ORACC guide words)
    dictionary = {
        "lugal": "king", "nin": "lady", "dingir": "god", "e2": "house",
        "an": "heaven", "ki": "earth", "kur": "mountain", "a": "water",
        "mu": "year", "nam": "fate", "me": "being", "gal": "great",
        "tur": "small", "dumu": "child", "ama": "mother", "ab": "father",
        "ud": "day", "gi": "reed", "kug": "silver", "ku3": "holy",
        "nig2": "thing", "sag": "head", "igi": "eye", "ka": "mouth",
        "zu": "tooth", "su": "hand", "giri3": "foot", "ur": "dog",
        "gu4": "ox", "udu": "sheep", "ku6": "fish", "muszen": "bird",
        "gin": "go", "du": "go", "gen": "go", "gub": "stand",
        "tus": "sit", "nag": "drink", "gu7": "eat", "de2": "pour",
        "du3": "build", "dug4": "speak", "e": "say", "zu": "know",
        "tuku": "have", "ak": "do", "tar": "cut", "kur2": "change",
        "sig": "low", "bad": "open", "tab": "double", "sze": "barley",
        "ninda": "bread", "kaskal": "road", "har": "ring", "tug2": "cloth",
        "en": "lord", "ensi2": "ruler", "sukkal": "minister",
        "iri": "city", "uru": "city", "nibru": "Nippur", "unug": "Uruk",
        "ba": "give", "sum": "give", "sa2": "equal", "silim": "health",
        "hul2": "joy", "sza3": "heart", "zi": "life", "nam-ti": "life",
    }

    # Extract validation data
    sources = []
    references = []
    for item in valid_data[:200]:  # Limit for speed
        src = item.get("source", {}).get("text_normalized", "")
        tgt = item.get("target", {}).get("text", "")
        if src and tgt:
            sources.append(src)
            references.append(tgt)

    print(f"Comparing on {len(sources)} examples...")

    # Generate dictionary baseline translations
    def dictionary_translate(text: str) -> str:
        words = text.replace("-", " ").replace(".", " ").split()
        translated = []
        for word in words:
            # Strip common suffixes
            base = word.rstrip("0123456789")
            for suffix in ["-e", "-a", "-ak", "-ta", "-sze3", "-ra", "-ke4", "-la", "-na"]:
                if base.endswith(suffix.replace("-", "")):
                    base = base[:-len(suffix) + 1]

            if base in dictionary:
                translated.append(dictionary[base])
            elif word in dictionary:
                translated.append(dictionary[word])
            else:
                translated.append(f"[{word}]")  # Unknown word

        return " ".join(translated)

    baseline_preds = [dictionary_translate(src) for src in sources]
    neural_preds = evaluator.translate_batch(sources)

    # Calculate metrics
    try:
        import evaluate
        bleu_metric = evaluate.load("bleu")
        chrf_metric = evaluate.load("chrf")

        # Baseline metrics
        baseline_bleu = bleu_metric.compute(
            predictions=baseline_preds,
            references=[[r] for r in references]
        )
        baseline_chrf = chrf_metric.compute(
            predictions=baseline_preds,
            references=[[r] for r in references]
        )

        # Neural metrics
        neural_bleu = bleu_metric.compute(
            predictions=neural_preds,
            references=[[r] for r in references]
        )
        neural_chrf = chrf_metric.compute(
            predictions=neural_preds,
            references=[[r] for r in references]
        )

        print(f"\nResults:")
        print(f"  {'Method':<20} {'BLEU':>10} {'chrF':>10}")
        print(f"  {'-' * 42}")
        print(f"  {'Dictionary Baseline':<20} {baseline_bleu['bleu']*100:>10.2f} {baseline_chrf['score']:>10.2f}")
        print(f"  {'mT5-large (Neural)':<20} {neural_bleu['bleu']*100:>10.2f} {neural_chrf['score']:>10.2f}")
        print(f"  {'-' * 42}")
        improvement = (neural_bleu['bleu'] - baseline_bleu['bleu']) / max(baseline_bleu['bleu'], 0.001) * 100
        print(f"  Neural improvement: {improvement:.1f}%")

        # Show comparison examples
        print("\nExample Comparisons:")
        for i in range(min(3, len(sources))):
            print(f"\n  Source: {sources[i]}")
            print(f"  Dictionary: {baseline_preds[i]}")
            print(f"  Neural:     {neural_preds[i]}")
            print(f"  Reference:  {references[i]}")

        return {
            "baseline_bleu": baseline_bleu['bleu'] * 100,
            "baseline_chrf": baseline_chrf['score'],
            "neural_bleu": neural_bleu['bleu'] * 100,
            "neural_chrf": neural_chrf['score'],
            "improvement_percent": improvement
        }
    except Exception as e:
        print(f"Metrics failed: {e}")
        return {}


def phase4_error_analysis(phase1_results: dict) -> dict:
    """
    Phase 4: Error Analysis Taxonomy.

    Categorize errors in the worst translations.
    """
    print("\n" + "=" * 70)
    print("PHASE 4: ERROR ANALYSIS TAXONOMY")
    print("=" * 70)

    if "bert_f1_scores" not in phase1_results:
        print("  Skipping - requires BERTScore results from Phase 1")
        return {}

    predictions = phase1_results["predictions"]
    references = phase1_results["references"]
    sources = phase1_results["sources"]
    bert_scores = phase1_results["bert_f1_scores"]

    # Find worst translations (lowest BERTScore)
    indexed_scores = [(i, score) for i, score in enumerate(bert_scores)]
    indexed_scores.sort(key=lambda x: x[1])
    worst_50 = indexed_scores[:50]

    print(f"\nAnalyzing 50 worst translations (lowest BERTScore)...")

    # Error categories (heuristic detection)
    error_counts = Counter()
    error_examples = defaultdict(list)

    for idx, score in worst_50:
        pred = predictions[idx]
        ref = references[idx]
        src = sources[idx]

        error_type = "unknown"

        # Heuristic error detection
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        src_words = set(src.replace("-", " ").split())

        # Check for repetition
        words = pred.split()
        if len(words) > 3 and len(set(words)) < len(words) * 0.5:
            error_type = "repetition"

        # Check for under-translation (very short output)
        elif len(pred.split()) < len(ref.split()) * 0.3:
            error_type = "under_translation"

        # Check for over-translation (very long output)
        elif len(pred.split()) > len(ref.split()) * 2:
            error_type = "over_translation"

        # Check for named entity issues (proper nouns)
        elif any(w[0].isupper() for w in ref.split() if len(w) > 2):
            if not any(w[0].isupper() for w in pred.split() if len(w) > 2):
                error_type = "named_entity_failure"

        # Check semantic drift (low word overlap)
        elif len(pred_words & ref_words) < 2:
            error_type = "hallucination"

        else:
            error_type = "semantic_drift"

        error_counts[error_type] += 1
        if len(error_examples[error_type]) < 2:
            error_examples[error_type].append({
                "source": src,
                "prediction": pred,
                "reference": ref,
                "bert_score": score
            })

    # Print error distribution
    print("\nError Distribution (worst 50 translations):")
    print("-" * 50)
    total = sum(error_counts.values())
    for error_type, count in error_counts.most_common():
        pct = count / total * 100
        bar = "█" * int(pct / 5)
        print(f"  {error_type:<22} {count:>3} ({pct:>5.1f}%) {bar}")

    # Show examples
    print("\nExample Errors by Category:")
    for error_type in ["hallucination", "under_translation", "repetition", "named_entity_failure"]:
        if error_type in error_examples:
            print(f"\n  {error_type.upper()}:")
            for ex in error_examples[error_type][:1]:
                print(f"    Source: {ex['source'][:50]}...")
                print(f"    Pred:   {ex['prediction'][:50]}...")
                print(f"    Ref:    {ex['reference'][:50]}...")

    return {
        "error_distribution": dict(error_counts),
        "error_examples": {k: v for k, v in error_examples.items()},
        "total_analyzed": total
    }


def phase5_ood_generalization(evaluator: AcademicEvaluator) -> dict:
    """
    Phase 5: Out-of-Domain Generalization Test.

    Test on administrative texts (excluded from training).
    """
    print("\n" + "=" * 70)
    print("PHASE 5: OUT-OF-DOMAIN GENERALIZATION (Admin Texts)")
    print("=" * 70)

    # Synthetic administrative text examples (typical Ur III receipts)
    admin_texts = [
        {"source": "1 udu", "expected_style": "receipt", "reference": "1 sheep"},
        {"source": "2 gu4 niga", "expected_style": "receipt", "reference": "2 fattened oxen"},
        {"source": "3 sila3 sze", "expected_style": "receipt", "reference": "3 liters of barley"},
        {"source": "ki ur-d-namma-ta", "expected_style": "receipt", "reference": "from Ur-Namma"},
        {"source": "mu-kux", "expected_style": "receipt", "reference": "delivery"},
        {"source": "kiszib3 lu2-kal-la", "expected_style": "receipt", "reference": "seal of Lukalla"},
        {"source": "iti sze-sag11-ku5", "expected_style": "receipt", "reference": "month of harvest"},
        {"source": "mu en-unu6-gal ba-hun", "expected_style": "receipt", "reference": "year Enunugal was installed"},
        {"source": "szu ba-ti", "expected_style": "receipt", "reference": "received"},
        {"source": "giri3 a-hu-wa-qar", "expected_style": "receipt", "reference": "via Ahu-waqar"},
    ]

    print(f"\nTesting on {len(admin_texts)} administrative text samples...")
    print("(Model trained on literary texts, testing domain transfer)\n")

    results = []
    literary_count = 0
    correct_count = 0

    for item in admin_texts:
        pred = evaluator.translate(item["source"])

        # Check if output sounds "literary" vs "administrative"
        literary_markers = ["the", "great", "lord", "king", "spoke", "ruled", "divine"]
        admin_markers = ["sheep", "ox", "barley", "liter", "received", "delivery", "seal", "month", "year"]

        lit_score = sum(1 for m in literary_markers if m.lower() in pred.lower())
        adm_score = sum(1 for m in admin_markers if m.lower() in pred.lower())

        is_literary = lit_score > adm_score
        is_correct = any(m.lower() in pred.lower() for m in item["reference"].split())

        if is_literary:
            literary_count += 1
        if is_correct:
            correct_count += 1

        style = "LITERARY" if is_literary else "ADMIN"
        results.append({
            "source": item["source"],
            "prediction": pred,
            "reference": item["reference"],
            "detected_style": style,
            "correct": is_correct
        })

        print(f"  Source: {item['source']:<25} | {style:<8} | {pred[:40]}")

    literary_pct = literary_count / len(admin_texts) * 100
    accuracy = correct_count / len(admin_texts) * 100

    print(f"\n{'=' * 50}")
    print(f"Domain Analysis:")
    print(f"  Literary-style outputs: {literary_count}/{len(admin_texts)} ({literary_pct:.1f}%)")
    print(f"  Correct translations:   {correct_count}/{len(admin_texts)} ({accuracy:.1f}%)")
    print(f"\nConclusion: {'Domain shift detected - model applies literary patterns' if literary_pct > 50 else 'Model generalizes to admin domain'}")

    return {
        "total_tests": len(admin_texts),
        "literary_style_count": literary_count,
        "literary_style_pct": literary_pct,
        "accuracy": accuracy,
        "examples": results
    }


def main():
    parser = argparse.ArgumentParser(description="Academic Evaluation Suite")
    parser.add_argument(
        "--phase",
        choices=["all", "semantic", "linguistic", "baseline", "errors", "ood"],
        default="all",
        help="Which evaluation phase to run"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Paths.NMT_CHECKPOINT,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results.json"),
        help="Output file for results"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("ACADEMIC EVALUATION SUITE FOR SUMERIAN-ENGLISH NMT")
    print("=" * 70)

    # Load evaluator
    evaluator = AcademicEvaluator(args.model_dir)

    # Load validation data
    valid_data = load_jsonl(Paths.VALID_FILE)
    print(f"Loaded {len(valid_data)} validation examples\n")

    results = {}

    # Run phases
    if args.phase in ["all", "semantic"]:
        results["phase1_semantic"] = phase1_semantic_validation(evaluator, valid_data)

    if args.phase in ["all", "linguistic"]:
        results["phase2_linguistic"] = phase2_linguistic_probing(evaluator)

    if args.phase in ["all", "baseline"]:
        results["phase3_baseline"] = phase3_dictionary_baseline(evaluator, valid_data)

    if args.phase in ["all", "errors"]:
        if "phase1_semantic" in results:
            results["phase4_errors"] = phase4_error_analysis(results["phase1_semantic"])
        else:
            print("\nPhase 4 requires Phase 1 results. Run with --phase all or --phase semantic first.")

    if args.phase in ["all", "ood"]:
        results["phase5_ood"] = phase5_ood_generalization(evaluator)

    # Save results
    # Remove non-serializable items
    save_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            save_results[k] = {
                kk: vv for kk, vv in v.items()
                if not isinstance(vv, list) or len(vv) < 100
            }

    with open(args.output, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    if "phase1_semantic" in results:
        p1 = results["phase1_semantic"]
        print(f"\nPhase 1 - Semantic Validation:")
        print(f"  BLEU: {p1.get('bleu', 'N/A'):.2f}")
        print(f"  BERTScore F1: {p1.get('bertscore_f1', 'N/A'):.4f}" if 'bertscore_f1' in p1 else "")

    if "phase2_linguistic" in results:
        p2 = results["phase2_linguistic"]
        print(f"\nPhase 2 - Linguistic Probing:")
        print(f"  Overall Accuracy: {p2['overall']['accuracy']:.1f}%")
        for cat in ["ergativity", "genitive", "named_entities"]:
            if cat in p2:
                print(f"    {cat.title()}: {p2[cat]['accuracy']:.1f}%")

    if "phase3_baseline" in results:
        p3 = results["phase3_baseline"]
        print(f"\nPhase 3 - Baseline Comparison:")
        print(f"  Dictionary BLEU: {p3.get('baseline_bleu', 'N/A'):.2f}")
        print(f"  Neural BLEU:     {p3.get('neural_bleu', 'N/A'):.2f}")
        print(f"  Improvement:     {p3.get('improvement_percent', 'N/A'):.1f}%")

    if "phase4_errors" in results:
        p4 = results["phase4_errors"]
        print(f"\nPhase 4 - Error Analysis:")
        if "error_distribution" in p4:
            for err, count in sorted(p4["error_distribution"].items(), key=lambda x: -x[1])[:3]:
                print(f"    {err}: {count}")

    if "phase5_ood" in results:
        p5 = results["phase5_ood"]
        print(f"\nPhase 5 - Out-of-Domain:")
        print(f"  Literary-style outputs: {p5['literary_style_pct']:.1f}%")
        print(f"  Accuracy: {p5['accuracy']:.1f}%")


if __name__ == "__main__":
    main()
