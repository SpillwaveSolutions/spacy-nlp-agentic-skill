---
name: batch-nlp
description: Efficiently process multiple documents with batch optimization
parameters:
  - name: input
    description: Input directory or file
    required: true
  - name: output
    description: Output file or directory
    required: true
  - name: task
    description: Task (entities, classify, analyze)
    required: false
    default: entities
  - name: batch-size
    description: Documents per batch
    required: false
    default: 50
  - name: workers
    description: Parallel workers
    required: false
    default: 0
skills:
  - spacy-nlp
---

# Batch NLP

Efficient batch processing using nlp.pipe optimization.

## Usage

```
/batch-nlp --input docs/ --output results/ --task entities
/batch-nlp --input texts.jsonl --output out.json --batch-size 100
/batch-nlp --input corpus.txt --output analysis.json --workers 4
```

## Why Batch?

```python
# SLOW - don't do this
for text in texts:
    doc = nlp(text)

# FAST - 5-10x speedup
for doc in nlp.pipe(texts, batch_size=50):
    process(doc)
```

## Performance

| Technique | Speedup |
|-----------|---------|
| nlp.pipe() | 5-10x |
| Disable components | 2-3x |
| Multiprocessing | 2-4x |
| Combined | 10-40x |

## Progress

```
[████████████░░░░░░] 67% | 10,050/15,000
Speed: 245 docs/sec | ETA: 20s
```
