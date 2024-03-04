import os

import pandas
from rdflib import Graph, URIRef, Literal

from core.nlp import ChatGptModel
from core.verbalizer import Vocabulary, Verbalizer, VerbalizerModelUsageConfig

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


class RelationPattern(Pattern):
    def check(self, results) -> bool:
        expected = {
            'http://www.w3.org/2002/07/owl#onProperty',
            'http://www.w3.org/2002/07/owl#someValuesFrom',
            'http://www.w3.org/2002/07/owl#allValuesFrom',
            'http://www.w3.org/2002/07/owl#hasValue',
            'http://www.w3.org/2002/07/owl#cardinality',
            'http://www.w3.org/2002/07/owl#minCardinality',
            'http://www.w3.org/2002/07/owl#maxCardinality'
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
        literal_value = None

        for (relation, obj) in results:

            if relation.toPython() == 'http://www.w3.org/2002/07/owl#onProperty':
                property_relation = obj

            if relation.toPython() in {'http://www.w3.org/2002/07/owl#someValuesFrom',
                                       'http://www.w3.org/2002/07/owl#allValuesFrom',
                                       'http://www.w3.org/2002/07/owl#hasValue'}:
                quantifier_relation = relation
                next_node = VerbalizationNode(concept=obj,
                                              parent_path=node.get_parent_path() + [(node.concept, relation)])

            if relation.toPython() in {'http://www.w3.org/2002/07/owl#cardinality',
                                       'http://www.w3.org/2002/07/owl#minCardinality',
                                       'http://www.w3.org/2002/07/owl#maxCardinality'}:
                quantifier_relation = relation
                literal_value = obj
                next_node = VerbalizationNode(concept='',
                                              parent_path=node.get_parent_path() + [(node.concept, relation)])
                next_node.display = ''

            triple_collector.append((node.concept, relation, obj))

        edge = VerbalizationEdge(
            relationship=URIRef(quantifier_relation.toPython() + property_relation.toPython()),
            node=next_node
        )

        quantifier_relation_label = self.vocab.get_rel_label(quantifier_relation)
        property_relation_label = self.vocab.get_cls_label(property_relation)

        if quantifier_relation.toPython().endswith('someValuesFrom'):
            edge.display = f'at least {property_relation_label} some'
        elif quantifier_relation.toPython().endswith('allValuesFrom'):
            edge.display = f'only {property_relation_label}'
        elif quantifier_relation.toPython().endswith('hasValue'):
            edge.display = f'must {property_relation_label}'
        elif quantifier_relation.toPython().lower().endswith('cardinality'):
            edge.display = self._handle_cardinality(quantifier_relation, property_relation, literal_value)
        else:
            edge.display = f'{property_relation_label} {quantifier_relation_label}'

        node.add_edge(edge)

        return [(reference.relationship, reference.node.concept) for reference in node.references]

    def _handle_cardinality(self, quantifier_relation, property_relation, obj_literal) -> str:
        property_relation_label = self.vocab.get_cls_label(property_relation)
        literal_value = obj_literal.toPython()
        if property_relation_label.startswith('has'):
            property_relation_label = property_relation_label.replace('has ', '')

        plural_s = 's' if literal_value > 1 else ''
        if quantifier_relation.endswith('cardinality') and literal_value == 0:
            return f'has no {property_relation_label}s'
        elif quantifier_relation.endswith('cardinality'):
            return f'has exactly {literal_value} {property_relation_label}{plural_s}'
        elif quantifier_relation.endswith('minCardinality'):
            return f'has at least {literal_value} {property_relation_label}{plural_s}'
        elif quantifier_relation.endswith('maxCardinality'):
            return f'has at most {literal_value} {property_relation_label}{plural_s}'


if __name__ == '__main__':
    ontology_name = 'sweet'
    use_llm = False

    graph = Graph()
    graph.parse(f'../data/{ontology_name}.owl')

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
        'http://xmlns.com/foaf/0.1/depicted_by',

        # SWEET
        'http://data.bioontology.org/metadata/prefixIRI',
        'http://purl.org/dc/terms/contributor',
        'http://purl.org/dc/terms/creator',
        'http://purl.org/dc/terms/created',
        'http://purl.org/dc/terms/source'
    }

    rephrased = {
        'http://www.w3.org/2002/07/owl#equivalentClass': 'is same as',
        'http://www.w3.org/2000/01/rdf-schema#subClassOf': 'is a type of',
        'http://purl.obolibrary.org/obo/IAO_0000115': 'has definition'
    }

    # create a vocabulary from the ontology.
    vocab = Vocabulary(graph, ignore=ignore, rephrased=rephrased)

    # Use patterns to normalize the graph
    patterns = [FirstRestOwlPattern, RelationPattern]

    # Initialize language model. TODO: Use LangChain for better interoperability
    llm = None
    if use_llm:
        llm = ChatGptModel(
            api_key=os.getenv('OPENAI_API_KEY'),
            model='gpt-3.5-turbo-0125',
            temperature=0.5
        )

    verbalizer = Verbalizer(
        graph, vocab, patterns,
        language_model=llm,
        usage_config=VerbalizerModelUsageConfig(
            min_patterns_evaluated=1,
            min_statements=2
        )
    )

    # Simple test
    # fragment, text, count = verbalizer.verbalize('http://purl.obolibrary.org/obo/ENVO_01001810')
    # print(fragment)
    # print(text)

    query = """
        SELECT ?o ?label
        WHERE {
            ?o a owl:Class ; rdfs:label ?label
            FILTER NOT EXISTS {
                ?o owl:deprecated true
            }
        }
    """
    dataset = []
    classes = [result[0] for result in graph.query(query)]
    for i, entry in enumerate(classes):
        print(f'verbalizing {entry}')
        fragment, text, count, llm_used = verbalizer.verbalize(entry)

        print(fragment)
        print(text)
        print("****" * 20)

        dataset.append({'fragment': fragment, 'text': text, 'statements': count, 'llm_used': llm_used})
        if i % 100 == 0 and llm:
            print(f'Cost so far: ${llm.cost}')

    pandas.DataFrame(dataset).to_csv(f'../output/{ontology_name}{"" if use_llm else ".raw"}.csv', index=False)
    print(f'Final cost: ${llm.cost}')
