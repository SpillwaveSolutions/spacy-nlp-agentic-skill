---
name: nlp-workflow-assistant
description: Proactively assists with NLP tasks and workflows
triggers:
  - pattern: "classify.*(text|document|email)"
    type: message_pattern
  - pattern: "extract.*(entities|keywords|names)"
    type: message_pattern
  - pattern: "(NLP|natural language|text processing)"
    type: message_pattern
  - pattern: "spacy|spaCy"
    type: keyword
skills:
  - spacy-nlp
---

# NLP Workflow Assistant

Suggests optimal NLP workflows based on task context.

## Triggers

- Classification requests
- Entity extraction needs
- NLP/text processing mentions
- spaCy references

## Workflow Suggestions

### For Classification

```
Recommended workflow:
1. /create-taxonomy --domain "Your Domain"
2. /train-classifier --data training.json
3. /classify-text --model ./output/model-best
```

### For Entity Extraction

```
/extract-entities --text "..." --types "PERSON,ORG"
```

### For Large Datasets

```
/batch-nlp --input docs/ --output results/ --workers 4
```

## Model Recommendations

| Use Case | Model |
|----------|-------|
| Speed critical | en_core_web_sm |
| Similarity | en_core_web_lg |
| Max accuracy | en_core_web_trf |
