---
name: analyze-text
description: Full NLP analysis - tokens, POS, dependencies, entities
parameters:
  - name: text
    description: Text to analyze
    required: false
  - name: file
    description: File to analyze
    required: false
  - name: components
    description: Components (all, tokens, pos, dep, ner, chunks)
    required: false
    default: all
  - name: output
    description: Format (table, json, detailed)
    required: false
    default: detailed
skills:
  - spacy-nlp
---

# Analyze Text

Comprehensive NLP analysis with full linguistic annotations.

## Usage

```
/analyze-text --text "The quick brown fox jumps"
/analyze-text --file doc.txt --components "ner,chunks"
/analyze-text --text "Apple announced products" --output json
```

## Components

- **tokens** - Words and punctuation
- **pos** - Part-of-speech tags
- **dep** - Dependency parsing
- **ner** - Named entities
- **chunks** - Noun phrases

## Output Example

```
TOKENS
Token       POS    Dep      Head
──────────  ─────  ───────  ────────
Apple       PROPN  nsubj    announced
announced   VERB   ROOT     announced
products    NOUN   dobj     announced

ENTITIES
Apple    ORG

NOUN CHUNKS
Apple, products
```
