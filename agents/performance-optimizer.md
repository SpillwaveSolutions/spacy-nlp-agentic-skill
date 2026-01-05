---
name: performance-optimizer
description: Detects slow NLP patterns and suggests optimizations
triggers:
  - pattern: "for.*(text|doc).*nlp\\("
    type: message_pattern
  - pattern: "(slow|taking forever|speed up|optimize)"
    type: message_pattern
  - pattern: "nlp\\(.*\\).*for"
    type: message_pattern
skills:
  - spacy-nlp
---

# Performance Optimizer

Detects inefficient patterns and provides optimizations.

## Triggers

- Slow processing loops
- Performance complaints
- Inefficient code patterns

## Anti-Patterns Detected

### Model in Loop (100-1000x slower)

```python
# BAD
for text in texts:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

# GOOD
nlp = spacy.load("en_core_web_sm")
for text in texts:
    doc = nlp(text)
```

### Sequential Processing (5-10x slower)

```python
# BAD
for text in texts:
    doc = nlp(text)

# GOOD
for doc in nlp.pipe(texts, batch_size=50):
    process(doc)
```

### Unused Components (2-3x slower)

```python
# Only need NER? Disable the rest
nlp = spacy.load("en_core_web_sm",
    disable=["parser", "tagger"])
```

## Optimization Summary

| Technique | Speedup |
|-----------|---------|
| Load once | 100-1000x |
| nlp.pipe() | 5-10x |
| Disable components | 2-3x |
| Multiprocessing | 2-4x |
