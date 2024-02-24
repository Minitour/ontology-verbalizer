import os

import pandas
from rdflib import Graph, URIRef

from core.nlp import ChatGptModel
from core.verbalizer import Vocabulary, Verbalizer

from core.verbalizer import Pattern, VerbalizationNode, VerbalizationEdge


class FirstRestOwlPattern(Pattern):
    def check(self, results) -> bool:
        expected = {'http://www.w3.org/1999/02/22-rdf-syntax-ns#first',
                    'http://www.w3.org/1999/02/22-rdf-syntax-ns#rest'}
        return {relation.toPython() for (relation, obj) in results} == expected

    def normalize(self, node: 'VerbalizationNode', triple_collector):
        current = node
        while current.concept != URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#nil'):
            query = self.verbalizer.next_step_query_builder(current)
            results = graph.query(query)
            rest_node = None
            for (relation, obj) in results:
                next_node = VerbalizationNode(obj,
                                              parent_path=current.get_parent_path() + [(current.concept, relation)])
                if relation == URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#first'):
                    # first
                    edge = VerbalizationEdge(URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#collection'), next_node)
                    node.add_edge(edge)
                    edge.display = '#collection'
                else:
                    # rest
                    rest_node = next_node
                triple_collector.append((current.concept, relation, obj))

            current = rest_node

        return [(reference.relationship, reference.node.concept) for reference in node.references]


class OboRelationPattern(Pattern):
    def check(self, results) -> bool:
        expected = {
            'http://www.w3.org/2002/07/owl#onProperty',
            'http://www.w3.org/2002/07/owl#someValuesFrom',
            'http://www.w3.org/2002/07/owl#allValuesFrom'
        }
        actual = {relation.toPython() for (relation, obj) in results}

        return len(expected.intersection(actual)) == 2

    def normalize(self, node: 'VerbalizationNode',
                  triple_collector):
        query = self.verbalizer.next_step_query_builder(node)
        results = graph.query(query)

        next_node = None
        quantifier_relation = None
        property_relation = None

        for (relation, obj) in results:
            if relation.toPython() in {'http://www.w3.org/2002/07/owl#someValuesFrom',
                                       'http://www.w3.org/2002/07/owl#allValuesFrom'}:
                next_node = VerbalizationNode(concept=obj,
                                              parent_path=node.get_parent_path() + [(node.concept, relation)])
                quantifier_relation = relation
            if relation.toPython() == 'http://www.w3.org/2002/07/owl#onProperty':
                property_relation = obj

            triple_collector.append((node.concept, relation, obj))

        edge = VerbalizationEdge(
            relationship=URIRef(quantifier_relation.toPython() + property_relation.toPython()),
            node=next_node
        )
        edge.display = f'{self.vocab.get_cls_label(property_relation)} {self.vocab.get_rel_label(quantifier_relation)}'
        node.add_edge(edge)

        return [(reference.relationship, reference.node.concept) for reference in node.references]


if __name__ == '__main__':
    graph = Graph()
    graph.parse('../data/envo.owl')

    ignore = {
        'http://www.w3.org/2000/01/rdf-schema#seeAlso',
        'http://www.w3.org/2000/01/rdf-schema#label',
        'http://www.w3.org/2000/01/rdf-schema#comment',
        'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',

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
        'http://purl.org/dc/elements/1.1/date',

        'http://purl.obolibrary.org/obo/IAO_0000412',
        'http://purl.obolibrary.org/obo/RO_0002175',
        'http://purl.obolibrary.org/obo/RO_0002161',
        'http://purl.obolibrary.org/obo/ado#from_Alzheimer_Ontology',
        'http://xmlns.com/foaf/0.1/depicted_by'
    }

    rephrased = {
        'http://www.w3.org/2002/07/owl#equivalentClass': 'is same as',
        'http://www.w3.org/2000/01/rdf-schema#subClassOf': 'is a type of',
        'http://purl.obolibrary.org/obo/IAO_0000115': 'has definition'
    }

    # create a vocabulary from the ontology.
    vocab = Vocabulary(graph, ignore=ignore, rephrased=rephrased)

    # Use patterns to normalize the graph
    patterns = [FirstRestOwlPattern, OboRelationPattern]

    # Initialize language model. TODO: Use LangChain for better interoperability
    llm = ChatGptModel(
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4-0613',
        temperature=0.5
    )
    verbalizer = Verbalizer(graph, vocab, patterns, llm)

    # fragment, text = verbalizer.verbalize('http://purl.obolibrary.org/obo/ENVO_01001810')
    # print(fragment)
    # print(text)

    dataset = []
    classes = [result[0] for result in graph.query('select ?o ?label where { ?o a owl:Class ; rdfs:label ?label }')]
    for entry in classes[-10:]:
        print(f'verbalizing {entry}')
        fragment, text = verbalizer.verbalize(entry)

        print(fragment)
        print(text)
        print("****" * 20)

        dataset.append({'fragment': fragment, 'text': text})

    pandas.DataFrame(dataset).to_csv('../output/dataset_6.csv', index=False)
