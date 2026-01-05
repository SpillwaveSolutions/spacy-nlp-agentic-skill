---
name: spacy-setup
description: Install and configure spaCy with optimal model selection
parameters:
  - name: model
    description: Model size (sm, md, lg, trf)
    required: false
    default: sm
  - name: gpu
    description: Enable GPU support
    required: false
    default: false
skills:
  - spacy-nlp
---

# spaCy Setup

Install and configure spaCy for your NLP project.

## Usage

```
/spacy-setup                     # Install with small model
/spacy-setup --model lg          # Install large model
/spacy-setup --model trf --gpu   # Transformer with GPU
```

## What It Does

1. Verifies Python environment (3.8+)
2. Installs spaCy and dependencies
3. Downloads selected model
4. Configures GPU if requested
5. Verifies installation

## Model Selection

| Model | Size | Speed | Use Case |
|-------|------|-------|----------|
| `sm` | 12 MB | Fastest | Prototyping |
| `md` | 40 MB | Fast | General use + vectors |
| `lg` | 560 MB | Fast | Similarity tasks |
| `trf` | 438 MB | Slow | Max accuracy (GPU) |
