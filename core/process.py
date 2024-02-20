import os

import pandas
from rdflib import Graph

from core.nlp import ChatGptModel
from core.verbalizer import Vocabulary, Verbalizer

if __name__ == '__main__':
    graph = Graph()
    graph.parse('../data/envo.owl')

    ignore = {
        'http://www.w3.org/2000/01/rdf-schema#seeAlso',
        'http://www.w3.org/2000/01/rdf-schema#label',
        'http://www.w3.org/2000/01/rdf-schema#comment',
        'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
        'http://www.w3.org/1999/02/22-rdf-syntax-ns#nil',

        # OBO documentation related
        'http://purl.obolibrary.org/obo/IAO_0000111',
        'http://purl.obolibrary.org/obo/IAO_0000114',
        'http://purl.obolibrary.org/obo/IAO_0000116',
        'http://purl.obolibrary.org/obo/IAO_0000117',
        'http://purl.obolibrary.org/obo/IAO_0000119',
        'http://purl.obolibrary.org/obo/IAO_0000589',

        'http://www.geneontology.org/formats/oboInOwl#creation_date',
        'http://www.geneontology.org/formats/oboInOwl#hasDbXref',
        'http://www.geneontology.org/formats/oboInOwl#hasOBONamespace',
        'http://www.geneontology.org/formats/oboInOwl#id',
        'http://www.geneontology.org/formats/oboInOwl#hasAlternativeId',
        'http://www.geneontology.org/formats/oboInOwl#inSubset',
        'http://www.geneontology.org/formats/oboInOwl#created_by',
        'http://purl.org/dc/elements/1.1/contributor',
        'http://purl.org/dc/elements/1.1/creator',
        'http://purl.org/dc/elements/1.1/date'
    }

    rephrased = {
        'http://www.w3.org/2002/07/owl#equivalentClass': 'concept same as',
        'http://www.w3.org/2002/07/owl#subClassOf': 'sub concept of',
        'http://purl.obolibrary.org/obo/IAO_0000115': 'has definition'
    }

    # create a vocabulary from the ontology.
    vocab = Vocabulary(graph, ignore=ignore, rephrased=rephrased)

    # Initialize language model. TODO: Use LangChain for better interoperability
    llm = ChatGptModel(
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4-0613',
        temperature=0.5
    )
    verbalizer = Verbalizer(graph, vocabulary=vocab, language_model=llm)
    #
    # fragment, text = verbalizer.verbalize('http://purl.obolibrary.org/obo/ENVO_01001810')
    # print(fragment)
    # print(text)

    dataset = []
    classes = [result[0] for result in graph.query('select ?o ?label where { ?o a owl:Class ; rdfs:label ?label }')]
    for entry in classes[3400:3500]:
        print(f'verbalizing {entry}')
        fragment, text = verbalizer.verbalize(entry)

        print(fragment)
        print(text)
        print("****" * 20)

        dataset.append({'fragment': fragment, 'text': text})

    pandas.DataFrame(dataset).to_csv('../output/dataset_2.csv', index=False)
