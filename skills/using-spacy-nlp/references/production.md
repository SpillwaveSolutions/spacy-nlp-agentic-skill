# Production Reference

Model evaluation, fine-tuning, and production deployment.

## Contents

- [Evaluation Metrics](#evaluation-metrics)
- [Error Analysis](#error-analysis)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Packaging Models](#packaging-models)
- [Serving Models](#serving-models)
- [Performance Optimization](#performance-optimization)
- [Monitoring](#monitoring)

---

## Evaluation Metrics

### Command-Line Evaluation

```bash
python -m spacy evaluate ./output/model-best ./dev.spacy --output metrics.json
```

### Programmatic Evaluation

```python
import spacy
from spacy.training import Corpus

def evaluate_model(model_path, test_path):
    nlp = spacy.load(model_path)
    corpus = Corpus(test_path)
    examples = list(corpus(nlp))
    
    scores = nlp.evaluate(examples)
    
    print("=== Evaluation Results ===")
    print(f"Overall Score: {scores.get('cats_score', 0):.3f}")
    print(f"Macro Precision: {scores.get('cats_macro_p', 0):.3f}")
    print(f"Macro Recall: {scores.get('cats_macro_r', 0):.3f}")
    print(f"Macro F1: {scores.get('cats_macro_f', 0):.3f}")
    
    # Per-class breakdown
    if 'cats_f_per_type' in scores:
        print("\nPer-Class F1:")
        for label, f1 in scores['cats_f_per_type'].items():
            print(f"  {label}: {f1:.3f}")
    
    return scores

scores = evaluate_model("./output/model-best", "test.spacy")
```

### Understanding Metrics

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| Precision | TP / (TP + FP) | Of predicted X, how many are correct? |
| Recall | TP / (TP + FN) | Of actual X, how many did we find? |
| F1 Score | 2 × (P × R) / (P + R) | Harmonic mean of precision and recall |
| Macro avg | Mean across classes | Treats all classes equally |
| Weighted avg | Weighted by support | Accounts for class imbalance |

### Confusion Matrix

```python
from collections import defaultdict

def confusion_matrix(nlp, test_data):
    matrix = defaultdict(lambda: defaultdict(int))
    
    for text, true_label in test_data:
        doc = nlp(text)
        predicted = max(doc.cats, key=doc.cats.get)
        matrix[true_label][predicted] += 1
    
    return matrix

# Display
for true_label, predictions in matrix.items():
    print(f"\nTrue: {true_label}")
    for pred_label, count in predictions.items():
        print(f"  Predicted {pred_label}: {count}")
```

---

## Error Analysis

### Find Misclassifications

```python
def analyze_errors(nlp, test_data):
    errors = []
    
    for text, true_label in test_data:
        doc = nlp(text)
        predicted = max(doc.cats, key=doc.cats.get)
        confidence = doc.cats[predicted]
        
        if predicted != true_label:
            errors.append({
                "text": text,
                "true": true_label,
                "predicted": predicted,
                "confidence": confidence,
                "scores": dict(doc.cats)
            })
    
    return errors

errors = analyze_errors(nlp, test_data)

# Sort by confidence (most confident errors are most interesting)
errors.sort(key=lambda x: -x["confidence"])

for err in errors[:10]:
    print(f"\nText: {err['text'][:80]}...")
    print(f"True: {err['true']}, Predicted: {err['predicted']} ({err['confidence']:.1%})")
```

### Low-Confidence Predictions

```python
def find_uncertain(nlp, texts, threshold=0.6):
    uncertain = []
    
    for doc in nlp.pipe(texts):
        top_score = max(doc.cats.values())
        if top_score < threshold:
            uncertain.append({
                "text": doc.text,
                "scores": dict(doc.cats)
            })
    
    return uncertain
```

### Common Error Patterns

Look for:
1. **Confusable classes** - Similar categories often confused
2. **Boundary cases** - Text between two categories
3. **Label noise** - Training data mislabeled
4. **Distribution shift** - Test data different from training

---

## Hyperparameter Tuning

### Key Parameters

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| `dropout` | Regularization (prevent overfit) | 0.1 - 0.5 |
| `learn_rate` | Training speed/stability | 1e-5 - 1e-3 |
| `batch_size` | Memory/speed tradeoff | 16 - 256 |
| `max_steps` | Training duration | 5000 - 50000 |
| `patience` | Early stopping | 500 - 2000 |

### Tuning Strategies

**Grid Search:**

```python
configs = [
    {"dropout": 0.1, "learn_rate": 0.0001},
    {"dropout": 0.2, "learn_rate": 0.0001},
    {"dropout": 0.1, "learn_rate": 0.00005},
    {"dropout": 0.2, "learn_rate": 0.00005},
]

for config in configs:
    # Generate config file
    # Train model
    # Evaluate and record results
    pass
```

**Learning Rate Schedule:**

```ini
[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
warmup_steps = 250
total_steps = 20000
initial_rate = 0.00005
```

**Early Stopping:**

```ini
[training]
patience = 1600  # Stop if no improvement for 1600 steps
eval_frequency = 200  # Evaluate every 200 steps
```

### Architecture Selection

| Data Size | Recommended Architecture |
|-----------|--------------------------|
| < 500 examples | TextCatBOW |
| 500-5000 | TextCatEnsemble |
| 5000+ | TextCatCNN or Transformer |

---

## Packaging Models

### Create Installable Package

```bash
python -m spacy package ./output/model-best ./packages \
    --name my_classifier \
    --version 1.0.0 \
    --meta-path meta.json
```

### Custom Metadata

```json
{
    "name": "my_classifier",
    "version": "1.0.0",
    "description": "Document classifier for internal use",
    "author": "Your Name",
    "email": "you@example.com",
    "license": "MIT"
}
```

### Install Package

```bash
pip install ./packages/en_my_classifier-1.0.0/
```

### Use Installed Package

```python
import spacy
nlp = spacy.load("en_my_classifier")
```

### Distribution

```bash
# Build wheel
cd packages/en_my_classifier-1.0.0
python setup.py bdist_wheel

# Distribute wheel
# pip install en_my_classifier-1.0.0-py3-none-any.whl
```

---

## Serving Models

### FastAPI Server

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy

app = FastAPI(title="Text Classifier API")
nlp = spacy.load("en_my_classifier")

class TextInput(BaseModel):
    text: str

class ClassificationResult(BaseModel):
    category: str
    confidence: float
    scores: dict

@app.post("/classify", response_model=ClassificationResult)
async def classify(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    
    with nlp.memory_zone():
        doc = nlp(input.text)
        category = max(doc.cats, key=doc.cats.get)
        return ClassificationResult(
            category=category,
            confidence=doc.cats[category],
            scores=doc.cats
        )

@app.post("/classify/batch")
async def classify_batch(texts: list[str]):
    results = []
    with nlp.memory_zone():
        for doc in nlp.pipe(texts):
            category = max(doc.cats, key=doc.cats.get)
            results.append({
                "text": doc.text,
                "category": category,
                "confidence": doc.cats[category]
            })
    return results

# Run: uvicorn server:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and app
COPY ./model ./model
COPY app.py .

# Pre-load model on startup
ENV MODEL_PATH=/app/model

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: text-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: text-classifier
  template:
    spec:
      containers:
      - name: classifier
        image: your-registry/text-classifier:1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
```

---

## Performance Optimization

### Inference Optimization

```python
# 1. Disable unused components
nlp = spacy.load("en_my_classifier", exclude=["ner", "parser"])

# 2. Use batch processing
results = list(nlp.pipe(texts, batch_size=100))

# 3. Use memory zones (prevents leaks)
with nlp.memory_zone():
    for doc in nlp.pipe(texts):
        process(doc)
```

### Benchmark

```python
import time

def benchmark(nlp, texts, iterations=5):
    times = []
    for _ in range(iterations):
        start = time.time()
        list(nlp.pipe(texts))
        times.append(time.time() - start)
    
    avg = sum(times) / len(times)
    docs_per_sec = len(texts) / avg
    print(f"Average: {avg:.3f}s ({docs_per_sec:.0f} docs/sec)")

benchmark(nlp, test_texts)
```

### Optimization Checklist

| Technique | Impact | When to Use |
|-----------|--------|-------------|
| Disable components | 2-3x | Always in production |
| Batch processing | 5-10x | Multiple documents |
| Memory zones | Prevents leaks | Long-running services |
| Multiprocessing | 2-4x | CPU-bound, many cores |
| GPU | 2-5x | Transformer models |

---

## Monitoring

### Logging Predictions

```python
import logging
import json

logger = logging.getLogger("classifier")

def classify_with_logging(text):
    doc = nlp(text)
    category = max(doc.cats, key=doc.cats.get)
    confidence = doc.cats[category]
    
    logger.info(json.dumps({
        "text_length": len(text),
        "category": category,
        "confidence": confidence,
        "all_scores": doc.cats
    }))
    
    return category, confidence
```

### Metrics to Track

| Metric | Why |
|--------|-----|
| Request latency | SLA compliance |
| Prediction confidence | Model drift detection |
| Category distribution | Usage patterns |
| Error rate | System health |
| Memory usage | Resource planning |

### Drift Detection

```python
from collections import Counter
import datetime

class DriftDetector:
    def __init__(self, window_size=1000):
        self.predictions = []
        self.window_size = window_size
    
    def record(self, category, confidence):
        self.predictions.append({
            "category": category,
            "confidence": confidence,
            "timestamp": datetime.datetime.now()
        })
        
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
    
    def get_distribution(self):
        categories = [p["category"] for p in self.predictions]
        return Counter(categories)
    
    def avg_confidence(self):
        return sum(p["confidence"] for p in self.predictions) / len(self.predictions)
```

### Alerting Thresholds

| Condition | Alert |
|-----------|-------|
| Avg confidence < 0.5 | Model may need retraining |
| Single category > 80% | Possible data issue |
| Latency > 100ms | Performance degradation |
| Error rate > 1% | System issue |
