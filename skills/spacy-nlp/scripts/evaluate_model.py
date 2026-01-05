#!/usr/bin/env python3
"""
evaluate_model.py

Evaluate a trained spaCy text classification model with detailed metrics.

Usage:
    python evaluate_model.py --model ./output/model-best --test test.spacy
    python evaluate_model.py --model ./output/model-best --test test.spacy --output report.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import spacy
    from spacy.training import Corpus
except ImportError:
    print("Error: spaCy not installed. Run: pip install spacy")
    sys.exit(1)


def evaluate_model(nlp, test_path: str) -> dict:
    """Run spaCy's built-in evaluation."""
    corpus = Corpus(test_path)
    examples = list(corpus(nlp))
    
    if not examples:
        raise ValueError(f"No examples found in {test_path}")
    
    return nlp.evaluate(examples)


def confusion_matrix(nlp, test_path: str) -> dict:
    """Generate confusion matrix from predictions."""
    corpus = Corpus(test_path)
    
    matrix = defaultdict(lambda: defaultdict(int))
    total = 0
    correct = 0
    
    for example in corpus(nlp):
        doc = nlp(example.text)
        
        # Get true label (highest score in reference)
        true_cats = example.reference.cats
        true_label = max(true_cats, key=true_cats.get)
        
        # Get predicted label
        pred_label = max(doc.cats, key=doc.cats.get)
        
        matrix[true_label][pred_label] += 1
        total += 1
        if true_label == pred_label:
            correct += 1
    
    return {
        "matrix": {k: dict(v) for k, v in matrix.items()},
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0
    }


def per_class_metrics(confusion: dict) -> dict:
    """Calculate precision, recall, F1 per class from confusion matrix."""
    matrix = confusion["matrix"]
    categories = sorted(set(matrix.keys()))
    
    metrics = {}
    for category in categories:
        # True positives: diagonal
        tp = matrix.get(category, {}).get(category, 0)
        
        # False positives: column sum minus TP
        fp = sum(matrix.get(other, {}).get(category, 0) 
                 for other in categories if other != category)
        
        # False negatives: row sum minus TP
        fn = sum(matrix.get(category, {}).get(other, 0) 
                 for other in categories if other != category)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[category] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn
        }
    
    return metrics


def find_errors(nlp, test_path: str, max_errors: int = 20) -> list:
    """Find misclassified examples."""
    corpus = Corpus(test_path)
    errors = []
    
    for example in corpus(nlp):
        doc = nlp(example.text)
        
        true_cats = example.reference.cats
        true_label = max(true_cats, key=true_cats.get)
        
        pred_label = max(doc.cats, key=doc.cats.get)
        confidence = doc.cats[pred_label]
        
        if true_label != pred_label:
            errors.append({
                "text": example.text[:200],  # Truncate long texts
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": confidence,
                "scores": {k: round(v, 3) for k, v in doc.cats.items()}
            })
    
    # Sort by confidence (most confident errors first)
    errors.sort(key=lambda x: -x["confidence"])
    return errors[:max_errors]


def print_report(scores: dict, confusion: dict, class_metrics: dict, errors: list):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("TEXT CLASSIFICATION EVALUATION REPORT")
    print("=" * 60)
    
    # Overall metrics
    print("\n📊 OVERALL METRICS")
    print("-" * 40)
    print(f"  Accuracy:         {confusion['accuracy']:.1%}")
    print(f"  Macro Precision:  {scores.get('cats_macro_p', 0):.3f}")
    print(f"  Macro Recall:     {scores.get('cats_macro_r', 0):.3f}")
    print(f"  Macro F1:         {scores.get('cats_macro_f', 0):.3f}")
    print(f"  cats_score:       {scores.get('cats_score', 0):.3f}")
    
    # Per-class metrics
    print("\n📈 PER-CLASS METRICS")
    print("-" * 40)
    print(f"{'Category':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)
    
    for category in sorted(class_metrics.keys()):
        m = class_metrics[category]
        print(f"{category:<20} {m['precision']:>10.3f} {m['recall']:>10.3f} "
              f"{m['f1']:>10.3f} {m['support']:>10}")
    
    # Confusion matrix
    print("\n🔀 CONFUSION MATRIX")
    print("-" * 40)
    categories = sorted(confusion["matrix"].keys())
    
    # Header
    print(f"{'True \\ Pred':<15}", end="")
    for cat in categories:
        print(f"{cat[:10]:>12}", end="")
    print()
    
    # Rows
    for true_cat in categories:
        print(f"{true_cat:<15}", end="")
        for pred_cat in categories:
            count = confusion["matrix"].get(true_cat, {}).get(pred_cat, 0)
            print(f"{count:>12}", end="")
        print()
    
    # Errors
    if errors:
        print("\n❌ TOP MISCLASSIFICATIONS")
        print("-" * 40)
        for i, err in enumerate(errors[:5], 1):
            print(f"\n{i}. True: {err['true_label']} → Predicted: {err['predicted_label']} "
                  f"({err['confidence']:.1%})")
            print(f"   Text: {err['text'][:100]}...")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate spaCy text classification model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic evaluation
    python evaluate_model.py --model ./output/model-best --test test.spacy

    # Save report to JSON
    python evaluate_model.py --model ./output/model-best --test test.spacy --output report.json

    # Show more error examples
    python evaluate_model.py --model ./output/model-best --test test.spacy --max-errors 50
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--test", "-t",
        required=True,
        help="Path to test data (.spacy file)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file for detailed report"
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Maximum number of errors to analyze (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    model_path = Path(args.model)
    if not model_path.exists():
        parser.error(f"Model not found: {args.model}")
    
    test_path = Path(args.test)
    if not test_path.exists():
        parser.error(f"Test data not found: {args.test}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    nlp = spacy.load(args.model)
    
    # Check for textcat component
    if "textcat" not in nlp.pipe_names and "textcat_multilabel" not in nlp.pipe_names:
        parser.error("Model does not have a text classification component")
    
    # Run evaluation
    print(f"Evaluating on {args.test}...")
    
    scores = evaluate_model(nlp, args.test)
    confusion = confusion_matrix(nlp, args.test)
    class_metrics = per_class_metrics(confusion)
    errors = find_errors(nlp, args.test, args.max_errors)
    
    # Print report
    print_report(scores, confusion, class_metrics, errors)
    
    # Save JSON report
    if args.output:
        report = {
            "model": str(args.model),
            "test_data": str(args.test),
            "overall": {
                "accuracy": confusion["accuracy"],
                "macro_precision": scores.get("cats_macro_p", 0),
                "macro_recall": scores.get("cats_macro_r", 0),
                "macro_f1": scores.get("cats_macro_f", 0),
                "cats_score": scores.get("cats_score", 0),
                "total_examples": confusion["total"],
                "correct": confusion["correct"]
            },
            "per_class": class_metrics,
            "confusion_matrix": confusion["matrix"],
            "errors": errors
        }
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Detailed report saved to {args.output}")


if __name__ == "__main__":
    main()
