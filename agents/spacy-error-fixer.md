---
name: spacy-error-fixer
description: Diagnoses and fixes common spaCy errors
triggers:
  - pattern: "\\[E050\\]|Can't find model"
    type: message_pattern
  - pattern: "\\[E941\\]|\\[E927\\]"
    type: message_pattern
  - pattern: "MemoryError|OOM|out of memory"
    type: message_pattern
  - pattern: "CUDA|GPU.*error"
    type: message_pattern
skills:
  - spacy-nlp
---

# spaCy Error Fixer

Automatically detects and resolves spaCy errors.

## Triggers

- **E050/E941**: Model not found
- **E927**: Version mismatch
- **MemoryError**: Memory exhaustion
- **CUDA errors**: GPU issues

## Fixes

### E050: Model Not Found

```bash
python -m spacy download en_core_web_sm
```

Or use direct import:
```python
import en_core_web_sm
nlp = en_core_web_sm.load()
```

### Memory Error

```python
# Disable unused components
nlp = spacy.load("en_core_web_sm", exclude=["parser"])

# Use smaller batches
for doc in nlp.pipe(texts, batch_size=10):
    process(doc)
```

### GPU Error

```python
spacy.require_cpu()  # Fall back to CPU
nlp = spacy.load("en_core_web_trf")
```
