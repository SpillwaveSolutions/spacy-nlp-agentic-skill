# Troubleshooting Reference

Common spaCy issues and their solutions.

## Contents

- [Model Loading Errors](#model-loading-errors)
- [Memory Issues](#memory-issues)
- [GPU Problems](#gpu-problems)
- [Version Compatibility](#version-compatibility)
- [Performance Issues](#performance-issues)
- [Training Errors](#training-errors)
- [Diagnostic Commands](#diagnostic-commands)

---

## Model Loading Errors

### E050: Can't find model

```
OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be 
a shortcut link, a Python package or a valid path to a data directory.
```

**Cause:** Model not installed or installed in different environment.

**Solutions:**

```bash
# 1. Download the model
python -m spacy download en_core_web_sm

# 2. Verify correct Python environment
which python  # Unix
where python  # Windows

# 3. Check installed models
python -m spacy info
```

**Alternative loading (avoids path issues):**

```python
# Direct import instead of spacy.load()
import en_core_web_sm
nlp = en_core_web_sm.load()
```

### E941: Can't find model meta.json

```
OSError: [E941] Can't find model directory: .../en_core_web_sm
```

**Cause:** Corrupted model installation.

**Solution:**

```bash
# Remove and reinstall
pip uninstall en_core_web_sm
python -m spacy download en_core_web_sm
```

### Model Loading Timeout in Jupyter

**Cause:** Large model + slow disk.

**Solution:**

```python
# Load once, reuse
import spacy
nlp = spacy.load("en_core_web_lg")  # Do this in first cell

# Then use nlp in subsequent cells without reloading
```

---

## Memory Issues

### Out of Memory During Processing

**Symptoms:**
- `MemoryError`
- System becomes unresponsive
- Process killed by OS

**Solutions:**

```python
# 1. Disable unused components
nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner"])

# 2. Process in chunks
def chunk_text(text, max_length=100000):
    for i in range(0, len(text), max_length):
        yield text[i:i + max_length]

for chunk in chunk_text(very_long_document):
    doc = nlp(chunk)
    process(doc)

# 3. Use generator instead of list
for doc in nlp.pipe(texts):  # Generator
    process(doc)
# NOT: docs = list(nlp.pipe(texts))  # Loads all into memory

# 4. Memory zones (spaCy 3.8+)
with nlp.memory_zone():
    for doc in nlp.pipe(batch):
        process(doc)
# Memory freed when exiting context
```

### Memory Leak in Long-Running Process

**Cause:** spaCy's string store grows indefinitely.

**Solution:**

```python
# Method 1: Memory zones
with nlp.memory_zone():
    process_batch(texts)

# Method 2: Periodic restart (for web services)
import gc

request_count = 0
for request in requests:
    process(request)
    request_count += 1
    if request_count % 10000 == 0:
        gc.collect()
```

### Training Out of Memory

**Solutions:**

```ini
# In config.cfg
[training.batcher.size]
start = 50       # Reduce from 100
stop = 500       # Reduce from 1000

[nlp]
batch_size = 64  # Reduce from 128
```

---

## GPU Problems

### GPU Not Being Used

**Symptom:** Training/inference slow despite having GPU.

**Solution:**

```python
import spacy

# MUST call before loading model
if spacy.prefer_gpu():
    print("Using GPU")
else:
    print("GPU not available, using CPU")

nlp = spacy.load("en_core_web_trf")
```

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. Reduce batch size:
```ini
[training.batcher.size]
start = 16
stop = 64
```

2. Use smaller model or `en_core_web_sm`

3. Clear CUDA cache:
```python
import torch
torch.cuda.empty_cache()
```

### CUDA Version Mismatch

**Symptom:** Cryptic CUDA errors.

**Diagnosis:**

```bash
# Check CUDA version
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"
```

**Solution:**

```bash
# Reinstall with matching CUDA
pip uninstall spacy
pip install spacy[cuda11x]  # or cuda12x
```

---

## Version Compatibility

### spaCy 2.x Models with spaCy 3.x

```
ValueError: [E927] Can't load model...
```

**Rule:** spaCy 2.x models do NOT work with spaCy 3.x. Must retrain.

**Diagnosis:**

```bash
python -m spacy validate
```

**Solution:** Download new model or retrain custom model.

### Python Version Issues

**spaCy 3.x supports:** Python 3.7+

**Check:**

```bash
python --version
```

### Dependency Conflicts

```bash
# Check for conflicts
pip check

# Common fix
pip install --upgrade spacy
pip install --upgrade thinc
```

---

## Performance Issues

### Slow Single-Document Processing

**Cause:** Pipeline overhead per call.

**Wrong:**
```python
for text in texts:
    doc = nlp(text)  # Slow - overhead each time
```

**Right:**
```python
for doc in nlp.pipe(texts, batch_size=50):
    process(doc)
```

### Slow Despite Using nlp.pipe()

**Causes and fixes:**

1. **Running unnecessary components:**
```python
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
```

2. **Batch size too small:**
```python
nlp.pipe(texts, batch_size=100)  # Increase from default
```

3. **CPU-bound with multiple cores:**
```python
nlp.pipe(texts, n_process=4)  # Use multiprocessing
```

### Transformer Model Slow on CPU

**Expected:** Transformer models are 15-20x slower on CPU vs GPU.

**Options:**
1. Use GPU: `spacy.prefer_gpu()`
2. Use non-transformer model: `en_core_web_lg`
3. Accept slower speed for higher accuracy

---

## Training Errors

### E024: No examples to train

```
ValueError: [E024] Could not find an example...
```

**Cause:** Empty or invalid training data.

**Check:**

```python
from spacy.training import Corpus
corpus = Corpus("train.spacy")
nlp = spacy.blank("en")
examples = list(corpus(nlp))
print(f"Examples: {len(examples)}")
```

### Config Validation Errors

```bash
# Validate config
python -m spacy debug config config.cfg

# Fill missing values
python -m spacy init fill-config base.cfg config.cfg
```

### Training Not Improving

**Diagnosis:**

```bash
python -m spacy debug data config.cfg
```

**Common fixes:**
1. More training data
2. Adjust learning rate
3. Check label distribution (balance classes)
4. Increase `max_steps`

---

## Diagnostic Commands

### System Information

```bash
python -m spacy info
```

### Validate Installation

```bash
python -m spacy validate
```

### Debug Training Config

```bash
python -m spacy debug config config.cfg
```

### Debug Training Data

```bash
python -m spacy debug data config.cfg
```

### Profile Performance

```python
import spacy
from spacy.util import get_words_and_spaces

nlp = spacy.load("en_core_web_sm")

# Time individual components
import time
text = "Sample text " * 1000

doc = nlp.make_doc(text)
for name in nlp.pipe_names:
    start = time.time()
    doc = nlp.get_pipe(name)(doc)
    print(f"{name}: {time.time() - start:.3f}s")
```

### Check Model Size

```python
import spacy

nlp = spacy.load("en_core_web_sm")
print(f"Vocab size: {len(nlp.vocab)}")
print(f"Pipeline: {nlp.pipe_names}")
```
