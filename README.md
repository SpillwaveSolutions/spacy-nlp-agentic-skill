# spaCy NLP Agentic Skill

Industrial-strength NLP with spaCy 3.x for text processing and classification.

## Installation

```bash
cd ~/.claude/skills
git clone https://github.com/SpillwaveSolutions/spacy-nlp-agentic-skill.git
```

## Commands

| Command | Description |
|---------|-------------|
| `/spacy-setup` | Install and configure spaCy |
| `/extract-entities` | Extract named entities from text |
| `/classify-text` | Classify documents with trained models |
| `/train-classifier` | Train custom classification models |
| `/analyze-text` | Full NLP analysis (tokens, POS, deps) |
| `/batch-nlp` | Efficient batch document processing |
| `/create-taxonomy` | Design classification categories |

## Agents

| Agent | Triggers On |
|-------|-------------|
| `spacy-error-fixer` | E050, E941, memory errors |
| `nlp-workflow-assistant` | NLP task mentions |
| `performance-optimizer` | Slow code patterns |

## Quick Start

```
/spacy-setup --model md
/extract-entities --text "Apple hired Tim Cook"
/create-taxonomy --domain "Customer Support"
/train-classifier --data training.json
```

## Structure

```
spacy-nlp-agentic-skill/
├── .claude-plugin/
│   └── marketplace.json
├── skills/
│   └── spacy-nlp/
│       ├── SKILL.md
│       ├── references/
│       ├── scripts/
│       └── assets/
├── commands/
│   ├── spacy-setup.md
│   ├── extract-entities.md
│   ├── classify-text.md
│   ├── train-classifier.md
│   ├── analyze-text.md
│   ├── batch-nlp.md
│   └── create-taxonomy.md
├── agents/
│   ├── spacy-error-fixer.md
│   ├── nlp-workflow-assistant.md
│   └── performance-optimizer.md
└── README.md
```

## License

MIT License

## Author

Richard Hightower (rick@spillwave.com)
