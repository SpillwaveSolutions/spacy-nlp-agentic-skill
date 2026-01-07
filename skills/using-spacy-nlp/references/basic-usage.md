# Basic Usage Reference

Working with spaCy's core objects and pipeline components.

## Contents

- [The Doc Object](#the-doc-object)
- [Token Attributes](#token-attributes)
- [Spans and Slicing](#spans-and-slicing)
- [Named Entity Recognition](#named-entity-recognition)
- [Noun Chunks](#noun-chunks)
- [Dependency Parsing](#dependency-parsing)
- [Sentence Segmentation](#sentence-segmentation)
- [Batch Processing](#batch-processing)
- [Pipeline Optimization](#pipeline-optimization)
- [Similarity](#similarity)

---

## The Doc Object

The `Doc` is spaCy's container for processed text. It holds tokens and their annotations.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion.")

# Doc properties
print(f"Text: {doc.text}")
print(f"Tokens: {len(doc)}")
print(f"Sentences: {len(list(doc.sents))}")
print(f"Entities: {len(doc.ents)}")

# Iteration
for token in doc:
    print(token.text)

# Indexing
first_token = doc[0]
last_token = doc[-1]
```

### Creating Docs Without Processing

```python
# Make doc without running pipeline (just tokenization)
doc = nlp.make_doc("Just tokenize this text.")

# Useful for: preparing training data, manual annotation
```

---

## Token Attributes

Each token carries rich linguistic annotations:

```python
doc = nlp("The striped bats are hanging on their feet.")

for token in doc:
    print(f"{token.text:12} | "
          f"Lemma: {token.lemma_:10} | "
          f"POS: {token.pos_:6} | "
          f"Tag: {token.tag_:6} | "
          f"Dep: {token.dep_:10} | "
          f"Head: {token.head.text}")
```

### Attribute Reference

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `text` | str | Original text | "running" |
| `lemma_` | str | Base form | "run" |
| `pos_` | str | Coarse POS | "VERB" |
| `tag_` | str | Fine-grained POS | "VBG" |
| `dep_` | str | Dependency label | "ROOT" |
| `head` | Token | Syntactic parent | token[2] |
| `ent_type_` | str | Entity type | "ORG" |
| `ent_iob_` | str | Entity IOB tag | "B" |
| `is_alpha` | bool | Alphabetic | True |
| `is_stop` | bool | Stop word | False |
| `is_punct` | bool | Punctuation | False |
| `like_num` | bool | Number-like | True for "10" |
| `shape_` | str | Word shape | "Xxxx" |

### Boolean Filters

```python
# Filter tokens
words = [token for token in doc if token.is_alpha]
content = [token for token in doc if not token.is_stop and not token.is_punct]
numbers = [token for token in doc if token.like_num]
```

---

## Spans and Slicing

Spans are slices of a Doc—sequences of tokens.

```python
doc = nlp("Apple is looking at buying U.K. startup for $1 billion.")

# Slice to create span
first_three = doc[0:3]  # "Apple is looking"
print(f"Span text: {first_three.text}")
print(f"Span root: {first_three.root.text}")  # Syntactic head

# Span properties
print(f"Start: {first_three.start}")
print(f"End: {first_three.end}")
print(f"Label: {first_three.label_}")  # Empty unless entity
```

### Creating Custom Spans

```python
from spacy.tokens import Span

doc = nlp("Apple is looking at buying U.K. startup.")

# Create entity span manually
company_span = Span(doc, 5, 7, label="COMPANY")  # "U.K. startup"
print(f"{company_span.text}: {company_span.label_}")

# Add to doc.ents (must be tuple)
doc.ents = list(doc.ents) + [company_span]
```

---

## Named Entity Recognition

```python
text = """
Apple Inc. was founded by Steve Jobs in Cupertino, California.
The company is now worth over $2 trillion and employs 164,000 people.
"""

doc = nlp(text)

for ent in doc.ents:
    print(f"{ent.text:20} | {ent.label_:10} | {spacy.explain(ent.label_)}")
```

### Common Entity Types

| Label | Description | Examples |
|-------|-------------|----------|
| PERSON | People | Steve Jobs, Marie Curie |
| ORG | Organizations | Apple Inc., United Nations |
| GPE | Countries, cities, states | California, Japan |
| LOC | Non-GPE locations | Mount Everest, the Pacific |
| DATE | Dates | January 2020, next week |
| TIME | Times | 3 PM, midnight |
| MONEY | Monetary values | $1 billion, €50 |
| PERCENT | Percentages | 15%, twenty percent |
| PRODUCT | Products | iPhone, Model 3 |
| EVENT | Named events | World War II, Olympics |

### Entity Spans

```python
# Get character positions
for ent in doc.ents:
    print(f"{ent.text}: chars {ent.start_char}-{ent.end_char}")

# Get token positions
for ent in doc.ents:
    print(f"{ent.text}: tokens {ent.start}-{ent.end}")
```

---

## Noun Chunks

Noun chunks are base noun phrases—flat phrases without nested structures.

```python
doc = nlp("The big brown fox jumped over the lazy dog near the old barn.")

for chunk in doc.noun_chunks:
    print(f"{chunk.text:25} | Root: {chunk.root.text:10} | Root Dep: {chunk.root.dep_}")
```

Output:
```
The big brown fox         | Root: fox        | Root Dep: nsubj
the lazy dog              | Root: dog        | Root Dep: pobj
the old barn              | Root: barn       | Root Dep: pobj
```

### Chunk Properties

```python
for chunk in doc.noun_chunks:
    print(f"Text: {chunk.text}")
    print(f"Root: {chunk.root}")       # Main noun
    print(f"Root head: {chunk.root.head}")  # Verb chunk depends on
    print(f"Root dep: {chunk.root.dep_}")   # Dependency relation
```

---

## Dependency Parsing

Understand syntactic structure through dependency relations.

```python
doc = nlp("Apple is looking at buying a startup.")

for token in doc:
    print(f"{token.text:10} --{token.dep_:10}--> {token.head.text}")
```

### Navigate the Tree

```python
# Find root of sentence
root = [token for token in doc if token.head == token][0]
print(f"Root: {root.text}")

# Get children
for child in root.children:
    print(f"Child of root: {child.text} ({child.dep_})")

# Get subtree
for token in root.subtree:
    print(token.text, end=" ")

# Left/right children
left_children = list(root.lefts)
right_children = list(root.rights)
```

### Common Dependency Labels

| Label | Description | Example |
|-------|-------------|---------|
| nsubj | Nominal subject | "Apple" in "Apple bought..." |
| dobj | Direct object | "startup" in "bought startup" |
| pobj | Object of preposition | "California" in "in California" |
| amod | Adjective modifier | "big" in "big company" |
| det | Determiner | "the" in "the company" |
| prep | Preposition | "in" in "based in" |
| ROOT | Root of sentence | Main verb |

---

## Sentence Segmentation

```python
doc = nlp("This is the first sentence. This is the second. And the third!")

for sent in doc.sents:
    print(f"[{sent.start}:{sent.end}] {sent.text}")
```

### Custom Sentence Boundaries

```python
from spacy.language import Language

@Language.component("custom_sentencizer")
def custom_sentencizer(doc):
    for token in doc[:-1]:
        if token.text == ";":
            doc[token.i + 1].is_sent_start = True
    return doc

nlp.add_pipe("custom_sentencizer", before="parser")
```

---

## Batch Processing

**Critical for production performance.**

### Wrong Way (Slow)

```python
# DON'T DO THIS
results = []
for text in texts:
    doc = nlp(text)  # Pipeline overhead each time
    results.append(doc)
```

### Right Way (Fast)

```python
# Use nlp.pipe() for batch processing
docs = list(nlp.pipe(texts, batch_size=50))

# As generator (memory efficient)
for doc in nlp.pipe(texts):
    process(doc)

# With context (preserve metadata)
text_tuples = [("Text one", {"id": 1}), ("Text two", {"id": 2})]
for doc, context in nlp.pipe(text_tuples, as_tuples=True):
    print(f"Doc {context['id']}: {len(doc)} tokens")
```

### Multiprocessing

```python
# CPU-bound workloads on multi-core machines
docs = list(nlp.pipe(texts, n_process=4))

# Note: n_process > 1 doesn't work with GPU models
```

### Memory Zones (spaCy 3.8+)

```python
# Prevent memory leaks in long-running processes
with nlp.memory_zone():
    for doc in nlp.pipe(batch):
        process(doc)
# Memory automatically freed when exiting context
```

---

## Pipeline Optimization

### Disable Unused Components

```python
# At load time
nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

# Temporarily with context manager
with nlp.select_pipes(enable=["ner"]):
    doc = nlp(text)  # Only NER runs

# Check what's enabled
print(nlp.pipe_names)
```

### Component Costs

| Component | Relative Cost | When to Disable |
|-----------|---------------|-----------------|
| tok2vec | Medium | Never (needed by others) |
| tagger | Low | If POS not needed |
| parser | High | If dependencies not needed |
| ner | Medium | If entities not needed |
| lemmatizer | Low | If lemmas not needed |
| attribute_ruler | Low | Rarely |

### Exclude vs Disable

```python
# exclude: Not loaded at all (can't re-enable)
nlp = spacy.load("en_core_web_sm", exclude=["parser"])

# disable: Loaded but not run (can re-enable)
nlp = spacy.load("en_core_web_sm", disable=["parser"])
nlp.enable_pipe("parser")  # Works
```

---

## Similarity

Requires word vectors (`en_core_web_md` or `en_core_web_lg`).

```python
nlp = spacy.load("en_core_web_md")

doc1 = nlp("I like cats")
doc2 = nlp("I love dogs")
doc3 = nlp("The stock market crashed")

print(f"cats/dogs similarity: {doc1.similarity(doc2):.3f}")   # High (~0.8)
print(f"cats/stocks similarity: {doc1.similarity(doc3):.3f}")  # Low (~0.3)
```

### Token and Span Similarity

```python
doc = nlp("The cat sat on the mat near the dog.")
cat = doc[1]
dog = doc[9]

print(f"cat/dog similarity: {cat.similarity(dog):.3f}")

# Span similarity
cat_phrase = doc[0:2]  # "The cat"
dog_phrase = doc[7:10]  # "the dog"
print(f"Phrase similarity: {cat_phrase.similarity(dog_phrase):.3f}")
```

### Word Vectors

```python
# Access vector directly
token = nlp("cat")[0]
print(f"Vector shape: {token.vector.shape}")
print(f"Has vector: {token.has_vector}")
print(f"Vector norm: {token.vector_norm}")
```
