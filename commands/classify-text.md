---
name: classify-text
description: Classify text into categories using a trained model
parameters:
  - name: text
    description: Text to classify
    required: false
  - name: file
    description: File to classify
    required: false
  - name: model
    description: Path to trained model
    required: true
  - name: threshold
    description: Confidence threshold (0.0-1.0)
    required: false
    default: 0.5
skills:
  - spacy-nlp
---

# Classify Text

Classify documents using spaCy TextCategorizer.

## Usage

```
/classify-text --text "K8s deployment failed" --model ./classifier
/classify-text --file tickets.txt --model ./support_model
/classify-text --text "Revenue up 20%" --model ./model --threshold 0.7
```

## What It Does

1. Loads classification model
2. Processes text through pipeline
3. Returns category with confidence
4. Filters by threshold

## Output

```
Classification: DevOps (92.3%)

All Scores:
  DevOps:      92.3%  ████████████████████
  Programming:  5.1%  █
  Business:     2.6%
```

## No Model?

Train one first:
```
/train-classifier --data labeled_data.json
```
