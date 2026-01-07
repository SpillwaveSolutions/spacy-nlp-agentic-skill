#!/usr/bin/env python3
"""
prepare_training_data.py

Convert labeled training data to spaCy DocBin format for text classification.

Usage:
    python prepare_training_data.py --input data.json --output-train train.spacy --output-dev dev.spacy

Input format (JSON):
    Single-label: [{"text": "...", "label": "Category"}, ...]
    Multi-label:  [{"text": "...", "labels": ["Cat1", "Cat2"]}, ...]
"""

import argparse
import json
import random
import sys
from pathlib import Path

try:
    import spacy
    from spacy.tokens import DocBin
except ImportError:
    print("Error: spaCy not installed. Run: pip install spacy")
    sys.exit(1)


def load_data(input_path: str) -> list:
    """Load training data from JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects")
    
    if len(data) == 0:
        raise ValueError("Input JSON is empty")
    
    return data


def detect_categories(data: list) -> tuple:
    """Detect categories and classification type from data."""
    categories = set()
    is_multilabel = False
    
    for item in data:
        if "labels" in item:
            is_multilabel = True
            categories.update(item["labels"])
        elif "label" in item:
            categories.add(item["label"])
        else:
            raise ValueError(f"Item missing 'label' or 'labels': {item}")
    
    return sorted(categories), is_multilabel


def create_docbin(data: list, nlp, categories: list, is_multilabel: bool) -> DocBin:
    """Convert data list to spaCy DocBin format."""
    db = DocBin()
    
    for item in data:
        text = item.get("text", "").strip()
        if not text:
            continue
        
        doc = nlp.make_doc(text)
        
        # Initialize all categories to 0
        doc.cats = {cat: 0.0 for cat in categories}
        
        # Set positive labels
        if is_multilabel:
            for label in item.get("labels", []):
                if label in doc.cats:
                    doc.cats[label] = 1.0
        else:
            label = item.get("label")
            if label in doc.cats:
                doc.cats[label] = 1.0
        
        db.add(doc)
    
    return db


def split_data(data: list, train_ratio: float, seed: int = 42) -> tuple:
    """Split data into train and dev sets."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    split_point = int(len(shuffled) * train_ratio)
    return shuffled[:split_point], shuffled[split_point:]


def print_stats(data: list, categories: list, is_multilabel: bool, name: str):
    """Print dataset statistics."""
    print(f"\n{name} Statistics:")
    print(f"  Total examples: {len(data)}")
    
    # Count per category
    counts = {cat: 0 for cat in categories}
    for item in data:
        if is_multilabel:
            for label in item.get("labels", []):
                counts[label] = counts.get(label, 0) + 1
        else:
            label = item.get("label")
            counts[label] = counts.get(label, 0) + 1
    
    print("  Category distribution:")
    for cat in sorted(counts.keys()):
        pct = counts[cat] / len(data) * 100
        print(f"    {cat}: {counts[cat]} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert labeled data to spaCy DocBin format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with train/dev split
    python prepare_training_data.py --input data.json

    # Custom output paths
    python prepare_training_data.py --input data.json \\
        --output-train train.spacy --output-dev dev.spacy

    # Custom split ratio (90% train, 10% dev)
    python prepare_training_data.py --input data.json --split 0.9

    # Single output (no split)
    python prepare_training_data.py --input data.json \\
        --output-train all_data.spacy --split 1.0
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSON file with labeled data"
    )
    parser.add_argument(
        "--output-train", "-t",
        default="train.spacy",
        help="Output path for training data (default: train.spacy)"
    )
    parser.add_argument(
        "--output-dev", "-d",
        default="dev.spacy",
        help="Output path for dev data (default: dev.spacy)"
    )
    parser.add_argument(
        "--split", "-s",
        type=float,
        default=0.8,
        help="Train/dev split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--lang", "-l",
        default="en",
        help="Language code (default: en)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.0 < args.split <= 1.0:
        parser.error("Split ratio must be between 0 and 1")
    
    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input file not found: {args.input}")
    
    # Load data
    print(f"Loading data from {args.input}...")
    data = load_data(args.input)
    print(f"Loaded {len(data)} examples")
    
    # Detect categories
    categories, is_multilabel = detect_categories(data)
    print(f"Detected {len(categories)} categories: {', '.join(categories)}")
    print(f"Classification type: {'multi-label' if is_multilabel else 'single-label'}")
    
    # Create blank NLP for tokenization
    nlp = spacy.blank(args.lang)
    
    # Split data
    if args.split < 1.0:
        train_data, dev_data = split_data(data, args.split, args.seed)
        print_stats(train_data, categories, is_multilabel, "Training")
        print_stats(dev_data, categories, is_multilabel, "Dev")
    else:
        train_data = data
        dev_data = []
        print_stats(train_data, categories, is_multilabel, "All Data")
    
    # Create and save DocBins
    print(f"\nCreating {args.output_train}...")
    train_db = create_docbin(train_data, nlp, categories, is_multilabel)
    train_db.to_disk(args.output_train)
    print(f"  Saved {len(train_data)} examples")
    
    if dev_data:
        print(f"Creating {args.output_dev}...")
        dev_db = create_docbin(dev_data, nlp, categories, is_multilabel)
        dev_db.to_disk(args.output_dev)
        print(f"  Saved {len(dev_data)} examples")
    
    print("\n✓ Data preparation complete!")
    print(f"\nNext steps:")
    print(f"  1. Generate config: python generate_config.py --categories \"{','.join(categories)}\"")
    print(f"  2. Train model: python -m spacy train config.cfg --output ./output")


if __name__ == "__main__":
    main()
