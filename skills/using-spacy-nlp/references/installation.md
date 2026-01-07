# Installation Reference

Detailed installation options for spaCy 3.x across environments.

## Contents

- [Standard Installation](#standard-installation)
- [Conda Installation](#conda-installation)
- [GPU Support](#gpu-support)
- [Transformer Models](#transformer-models)
- [Model Selection Guide](#model-selection-guide)
- [Verification](#verification)
- [Virtual Environments](#virtual-environments)

---

## Standard Installation

### Prerequisites

```bash
pip install -U pip setuptools wheel
```

### Basic Install

```bash
pip install -U spacy
```

### With Extras

```bash
# Transformer support
pip install -U spacy[transformers]

# CUDA 12.x GPU
pip install -U spacy[cuda12x]

# Apple Silicon GPU
pip install -U spacy[apple]

# Lemmatization lookup tables
pip install -U spacy[lookups]

# Multiple extras
pip install -U spacy[transformers,cuda12x]
```

---

## Conda Installation

```bash
# Create environment
conda create -n spacy_env python=3.10
conda activate spacy_env

# Install from conda-forge
conda install -c conda-forge spacy

# Download model
python -m spacy download en_core_web_sm
```

**Note:** Conda-forge may lag behind PyPI for latest versions.

---

## GPU Support

### NVIDIA CUDA

```bash
# Check CUDA version
nvidia-smi

# Install matching spaCy
pip install -U spacy[cuda11x]  # CUDA 11.x
pip install -U spacy[cuda12x]  # CUDA 12.x
```

### Apple Silicon (M1/M2/M3)

```bash
pip install -U spacy[apple]
```

### Verify GPU

```python
import spacy

# Must call before loading model
gpu_available = spacy.prefer_gpu()
print(f"GPU available: {gpu_available}")

# For specific GPU
spacy.require_gpu(0)  # GPU ID 0
```

---

## Transformer Models

Transformer models require additional dependencies:

```bash
pip install -U spacy[transformers]
python -m spacy download en_core_web_trf
```

### Memory Requirements

| Model | VRAM (GPU) | RAM (CPU) |
|-------|------------|-----------|
| `en_core_web_trf` | 4-6 GB | 8+ GB |
| `en_core_web_sm` | N/A | <1 GB |

### CPU Fallback

Transformer models work on CPU but are 15-20x slower:

```python
# Will use CPU if no GPU
nlp = spacy.load("en_core_web_trf")
```

---

## Model Selection Guide

### English Models

| Model | Size | NER F1 | Speed (CPU) | Best For |
|-------|------|--------|-------------|----------|
| `en_core_web_sm` | 12 MB | 85.5% | 10,000 WPS | Speed-critical, prototyping |
| `en_core_web_md` | 40 MB | 85.5% | 10,000 WPS | General use, word vectors |
| `en_core_web_lg` | 560 MB | 85.5% | 10,000 WPS | Semantic similarity |
| `en_core_web_trf` | 438 MB | 89.8% | 700 WPS | Maximum accuracy |

### Download Commands

```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_trf
```

### Decision Tree

```
Need maximum accuracy?
├── Yes → en_core_web_trf (requires GPU for speed)
└── No → Need word vectors?
         ├── Yes → Need similarity comparison?
         │        ├── Yes → en_core_web_lg (685k vectors)
         │        └── No → en_core_web_md (20k vectors)
         └── No → en_core_web_sm (fastest)
```

### Other Languages

```bash
# German
python -m spacy download de_core_news_sm

# French
python -m spacy download fr_core_news_sm

# Spanish
python -m spacy download es_core_news_sm

# Chinese
python -m spacy download zh_core_web_sm

# List available models
python -m spacy info
```

Full list: https://spacy.io/models

---

## Verification

### Check Installation

```python
import spacy
print(f"spaCy version: {spacy.__version__}")
```

### Validate Models

```bash
python -m spacy validate
```

### Test Model

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple Inc. is based in Cupertino, California.")

# Check pipeline components
print(f"Pipeline: {nlp.pipe_names}")

# Check entities
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")

# Expected output:
# Pipeline: ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']
# Apple Inc.: ORG
# Cupertino: GPE
# California: GPE
```

### System Info

```bash
python -m spacy info
```

Output includes: spaCy version, Python version, platform, installed models.

---

## Virtual Environments

### venv (Recommended)

```bash
python -m venv spacy_env
source spacy_env/bin/activate  # Linux/Mac
spacy_env\Scripts\activate     # Windows

pip install -U spacy
python -m spacy download en_core_web_sm
```

### Poetry

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.9"
spacy = "^3.7"
```

```bash
poetry install
poetry run python -m spacy download en_core_web_sm
```

### Docker

```dockerfile
FROM python:3.10-slim

RUN pip install --no-cache-dir spacy
RUN python -m spacy download en_core_web_sm

WORKDIR /app
COPY . .

CMD ["python", "app.py"]
```

---

## Platform-Specific Notes

### Windows

- Use virtual environments
- For Git-tracked models: `git config core.autocrlf false`
- PowerShell may need: `Set-ExecutionPolicy RemoteSigned`

### macOS

- Apple Silicon: Use `spacy[apple]` for GPU acceleration
- Intel: Standard installation works

### Linux

- Install build dependencies: `apt install python3-dev build-essential`
- For headless servers, no additional setup needed
