---
name: create-taxonomy
description: Design classification categories and subcategories
parameters:
  - name: domain
    description: Domain (e.g., "Customer Support", "Legal")
    required: true
  - name: depth
    description: Hierarchy depth (1-3)
    required: false
    default: 2
  - name: output
    description: Output file
    required: false
    default: taxonomy.json
  - name: examples
    description: Generate training examples
    required: false
    default: true
skills:
  - spacy-nlp
---

# Create Taxonomy

Design hierarchical classification categories.

## Usage

```
/create-taxonomy --domain "Customer Support"
/create-taxonomy --domain "Legal Documents" --depth 3
/create-taxonomy --domain "IT Tickets" --examples
```

## What It Does

1. Analyzes domain patterns
2. Suggests category structure
3. Creates hierarchy with keywords
4. Generates training examples

## Output Example

```json
{
  "domain": "Customer Support",
  "categories": [
    {
      "name": "Technical Issues",
      "subcategories": [
        {"name": "Installation", "keywords": ["install", "setup"]},
        {"name": "Performance", "keywords": ["slow", "crash"]}
      ]
    },
    {
      "name": "Billing",
      "subcategories": [
        {"name": "Charges", "keywords": ["charge", "payment"]},
        {"name": "Refunds", "keywords": ["refund", "cancel"]}
      ]
    }
  ]
}
```

## Design Principles

1. **Mutually exclusive** - No overlap
2. **Exhaustive** - Cover all cases
3. **Balanced depth** - 2-3 levels max

## Next Steps

```
/train-classifier --data taxonomy_examples.json
```
