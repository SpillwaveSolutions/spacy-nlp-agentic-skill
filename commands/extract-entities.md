---
name: extract-entities
description: Extract named entities (people, orgs, locations) from text
parameters:
  - name: text
    description: Text to analyze
    required: false
  - name: file
    description: File path to analyze
    required: false
  - name: types
    description: Entity types (PERSON, ORG, GPE, DATE, MONEY)
    required: false
  - name: output
    description: Output format (table, json, csv)
    required: false
    default: table
skills:
  - spacy-nlp
---

# Extract Entities

Extract named entities from text using spaCy NER.

## Usage

```
/extract-entities --text "Apple hired Tim Cook in California"
/extract-entities --file document.txt --types "PERSON,ORG"
/extract-entities --file report.txt --output json
```

## Entity Types

| Label | Description | Examples |
|-------|-------------|----------|
| `PERSON` | People | Tim Cook, Marie Curie |
| `ORG` | Organizations | Apple Inc., UN |
| `GPE` | Countries/cities | California, France |
| `DATE` | Dates | January 2024 |
| `MONEY` | Money | $1 billion |
| `PRODUCT` | Products | iPhone |

## Output Example

```
| Text       | Label  | Start | End |
|------------|--------|-------|-----|
| Apple      | ORG    | 0     | 5   |
| Tim Cook   | PERSON | 12    | 20  |
| California | GPE    | 24    | 34  |
```
