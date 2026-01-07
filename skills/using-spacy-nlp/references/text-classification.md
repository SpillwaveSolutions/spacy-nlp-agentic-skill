# Text Classification Reference

Training custom text classifiers with spaCy 3.x TextCategorizer.

## Contents

- [Overview](#overview)
- [Single vs Multi-Label](#single-vs-multi-label)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Model Architectures](#model-architectures)
- [Using Trained Models](#using-trained-models)
- [Best Practices](#best-practices)

---

## Overview

spaCy's `TextCategorizer` assigns categories to text documents. Two variants:

| Component | Use Case | Output |
|-----------|----------|--------|
| `textcat` | Single-label (mutually exclusive) | One category per doc |
| `textcat_multilabel` | Multi-label | Multiple categories per doc |

### When to Use Which

**Single-label (`textcat`):**
- Sentiment: positive/negative
- Document type: report/email/memo
- Language detection: en/es/fr

**Multi-label (`textcat_multilabel`):**
- Topics: sports AND politics
- Tags: urgent AND customer AND billing
- Attributes: formal AND technical AND long

---

## Single vs Multi-Label

### Single-Label Example

```python
# Categories are mutually exclusive
doc.cats = {
    "Business": 1.0,
    "Technology": 0.0,
    "Programming": 0.0,
    "DevOps": 0.0
}
# Sum = 1.0
```

### Multi-Label Example

```python
# Document can have multiple labels
doc.cats = {
    "urgent": 1.0,
    "customer": 1.0,
    "billing": 0.0,
    "technical": 1.0
}
# Sum can be > 1.0
```

---

## Data Preparation

### Input Format

JSON array of text-label pairs:

```json
[
    {"text": "Quarterly revenue exceeded expectations", "label": "Business"},
    {"text": "Fixed null pointer exception", "label": "Programming"},
    {"text": "Deploy to Kubernetes cluster", "label": "DevOps"}
]
```

For multi-label:

```json
[
    {"text": "Customer billing issue urgent", "labels": ["customer", "billing", "urgent"]},
    {"text": "Technical documentation update", "labels": ["technical"]}
]
```

### Convert to DocBin

Use `scripts/prepare_training_data.py` or manually:

```python
import spacy
from spacy.tokens import DocBin
import json
import random

def create_docbin(input_file, output_file, nlp, categories, split=None):
    """Convert JSON training data to spaCy DocBin format."""
    
    with open(input_file) as f:
        data = json.load(f)
    
    # Shuffle for training
    random.shuffle(data)
    
    db = DocBin()
    for item in data:
        doc = nlp.make_doc(item["text"])
        
        # Set category scores
        doc.cats = {cat: 0.0 for cat in categories}
        
        if "label" in item:  # Single-label
            doc.cats[item["label"]] = 1.0
        elif "labels" in item:  # Multi-label
            for label in item["labels"]:
                doc.cats[label] = 1.0
        
        db.add(doc)
    
    db.to_disk(output_file)
    print(f"Created {output_file} with {len(data)} examples")

# Usage
nlp = spacy.blank("en")
categories = ["Business", "Technology", "Programming", "DevOps"]

create_docbin("data.json", "train.spacy", nlp, categories)
```

### Train/Dev Split

Standard split: 80% train, 20% dev

```python
import random

data = load_data()
random.shuffle(data)
split_point = int(len(data) * 0.8)

train_data = data[:split_point]
dev_data = data[split_point:]

create_docbin(train_data, "train.spacy", nlp, categories)
create_docbin(dev_data, "dev.spacy", nlp, categories)
```

### Data Quality Requirements

| Factor | Recommendation |
|--------|----------------|
| Minimum examples per class | 50+ (100+ preferred) |
| Class balance | Within 1:10 ratio |
| Text length | Similar to production data |
| Label quality | Manual review recommended |

---

## Configuration

### Generate Base Config

```bash
# Single-label
python -m spacy init config config.cfg --pipeline textcat --lang en

# Multi-label
python -m spacy init config config.cfg --pipeline textcat_multilabel --lang en
```

### Key Configuration Sections

```ini
[paths]
train = "train.spacy"
dev = "dev.spacy"

[nlp]
lang = "en"
pipeline = ["textcat"]
batch_size = 128

[components.textcat]
factory = "textcat"
threshold = 0.5

[components.textcat.model]
@architectures = "spacy.TextCatEnsemble.v2"

[training]
max_steps = 20000
eval_frequency = 200
patience = 1600
dropout = 0.1

[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
initial_rate = 0.00005
warmup_steps = 250
total_steps = 20000
```

### Configure Labels

Labels are auto-detected from training data. To verify:

```bash
python -m spacy debug data config.cfg
```

---

## Training

### Basic Training

```bash
# Fill config with defaults
python -m spacy init fill-config base_config.cfg config.cfg

# Validate data before training (catches issues early)
python -m spacy debug data config.cfg

# Train
python -m spacy train config.cfg --output ./output
```

### GPU Training

```bash
python -m spacy train config.cfg --output ./output --gpu-id 0
```

### Resume Training

```bash
python -m spacy train config.cfg --output ./output --resume
```

### Training Output

```
output/
├── model-best/      # Best model by dev score
├── model-last/      # Latest checkpoint
└── training.log     # Training metrics
```

### Monitor Training

Watch for:
- `cats_score` increasing
- Loss decreasing
- No overfitting (dev score tracking train)

```bash
# View training progress
tail -f output/training.log
```

---

## Model Architectures

### TextCatEnsemble (Default)

Combines bag-of-words with neural features. Good balance of speed and accuracy.

```ini
[components.textcat.model]
@architectures = "spacy.TextCatEnsemble.v2"
```

### TextCatBOW

Pure bag-of-words. Fastest, good for quick experiments.

```ini
[components.textcat.model]
@architectures = "spacy.TextCatBOW.v3"
exclusive_classes = true  # For single-label
ngram_size = 1
no_output_layer = false
```

### TextCatCNN

Convolutional neural network. Better for longer texts.

```ini
[components.textcat.model]
@architectures = "spacy.TextCatCNN.v2"
```

### Transformer-Based

Highest accuracy, requires GPU.

```ini
[components.textcat.model]
@architectures = "spacy-transformers.TextCatTransformer.v1"

[components.textcat.model.transformer]
@architectures = "spacy-transformers.TransformerModel.v3"
name = "roberta-base"
```

### Architecture Selection

| Architecture | Speed | Accuracy | GPU Required |
|--------------|-------|----------|--------------|
| TextCatBOW | Fastest | Good | No |
| TextCatEnsemble | Fast | Better | No |
| TextCatCNN | Medium | Better | No (helps) |
| Transformer | Slow | Best | Yes |

---

## Using Trained Models

### Load and Predict

```python
import spacy

nlp = spacy.load("./output/model-best")

def classify(text):
    doc = nlp(text)
    predicted = max(doc.cats, key=doc.cats.get)
    confidence = doc.cats[predicted]
    return predicted, confidence, doc.cats

# Single prediction
category, conf, scores = classify("Deploy to production")
print(f"{category}: {conf:.1%}")

# Batch prediction
texts = ["Revenue report", "Bug fix", "CI/CD setup"]
for doc in nlp.pipe(texts):
    predicted = max(doc.cats, key=doc.cats.get)
    print(f"{doc.text}: {predicted}")
```

### Threshold Adjustment

Default threshold is 0.5. Adjust for precision/recall tradeoff:

```python
def classify_with_threshold(text, threshold=0.7):
    doc = nlp(text)
    predicted = max(doc.cats, key=doc.cats.get)
    confidence = doc.cats[predicted]
    
    if confidence < threshold:
        return "uncertain", confidence, doc.cats
    return predicted, confidence, doc.cats
```

### Add to Existing Pipeline

```python
# Load base model
nlp = spacy.load("en_core_web_sm")

# Add trained textcat
textcat = spacy.load("./output/model-best").get_pipe("textcat")
nlp.add_pipe("textcat", source=textcat)

# Now have NER + classification
doc = nlp("Apple released quarterly earnings")
print(f"Entities: {[(e.text, e.label_) for e in doc.ents]}")
print(f"Category: {max(doc.cats, key=doc.cats.get)}")
```

---

## Best Practices

### Data Quality

1. **Balance classes** - Undersample majority or oversample minority
2. **Clean text** - Consistent preprocessing
3. **Review labels** - Manual spot-check for accuracy
4. **Representative samples** - Match production data distribution

### Training

1. **Start simple** - TextCatBOW for baseline
2. **Track experiments** - Log configs and results
3. **Early stopping** - Use patience parameter
4. **Validation** - Always evaluate on held-out dev set

### Hyperparameters to Tune

| Parameter | Range | Effect |
|-----------|-------|--------|
| `dropout` | 0.1-0.5 | Regularization |
| `learn_rate` | 1e-5 to 1e-3 | Training speed |
| `batch_size` | 16-256 | Memory/speed |
| `max_steps` | 5000-50000 | Training duration |

### Debugging Poor Performance

```bash
# Check data distribution
python -m spacy debug data config.cfg

# Check for label imbalance
# Check for too few examples
# Check for data quality issues
```

### Production Checklist

- [ ] Evaluated on held-out test set
- [ ] Tested on real production examples
- [ ] Packaged model for distribution
- [ ] Memory usage acceptable
- [ ] Latency meets requirements
- [ ] Error handling implemented
