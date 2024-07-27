# OWL Ontology Verbalizer

## Installation

```shell
TBD
```

## Usage

### Configure URIs to ignore or rename

```python
ignore = {
    'http://www.w3.org/2002/07/owl#onDatatype',
    'http://www.w3.org/2000/01/rdf-schema#seeAlso',
    'http://www.w3.org/2000/01/rdf-schema#label',
    'http://www.w3.org/2000/01/rdf-schema#comment',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
    'http://www.w3.org/2000/01/rdf-schema#isDefinedBy',
}

rephrased = {
    'http://www.w3.org/2002/07/owl#equivalentClass': 'is same as',
    'http://www.w3.org/2000/01/rdf-schema#subClassOf': 'is a type of',
    'http://www.w3.org/2002/07/owl#intersectionOf': 'all of',
    'http://www.w3.org/2002/07/owl#unionOf': 'any of',
    'http://www.w3.org/2002/07/owl#disjointWith': 'is different from',
    'http://www.w3.org/2002/07/owl#withRestrictions': 'must be'
}
```

### Configure LLM of choice

```python
model = ChatGptModelParaphrase(api_key='sk-xyz', model='gpt-4o', temperature=0.7)
```

### Perform Verbalization

```python
ontology = 'pizza.ttl'
name = 'pizza'
processor = Processor(llm=model, vocab_ignore=ignore, vocab_rephrased=rephrased, min_statements=1)
processor.process(name, ontology, output_dir='/path/to/my/output')
```