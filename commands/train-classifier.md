---
name: train-classifier
description: Train a custom text classification model
parameters:
  - name: data
    description: Path to training data (JSON)
    required: true
  - name: categories
    description: Comma-separated categories
    required: false
  - name: output
    description: Output directory
    required: false
    default: ./output
  - name: epochs
    description: Training epochs
    required: false
    default: 10
  - name: gpu
    description: Use GPU
    required: false
    default: false
skills:
  - spacy-nlp
---

# Train Classifier

Train a TextCategorizer model for document classification.

## Usage

```
/train-classifier --data training.json --output ./model
/train-classifier --data data.json --categories "Tech,Business,Legal"
/train-classifier --data data.json --epochs 20 --gpu
```

## Data Format

```json
[
  {"text": "Revenue exceeded expectations", "label": "Business"},
  {"text": "Fixed null pointer exception", "label": "Programming"},
  {"text": "K8s manifest updated", "label": "DevOps"}
]
```

## Training Workflow

1. **Prepare data** → Convert JSON to DocBin
2. **Generate config** → Create training config
3. **Validate** → `spacy debug data`
4. **Train** → `spacy train`
5. **Evaluate** → Check metrics

## Output

```
Epoch 10/10: Loss=0.45  F1=0.90
Best model: ./output/model-best
```
