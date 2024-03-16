import os
from tqdm import tqdm

import pandas
from rdflib import Graph, URIRef, Literal

from core.nlp import ChatGptModel, LlamaModel
from core.verbalizer import Vocabulary, Verbalizer, VerbalizerModelUsageConfig

from core.verbalizer import Pattern, VerbalizationNode, VerbalizationEdge


class OwlFirstRestPattern(Pattern):
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


class OwlRestrictionPattern(Pattern):
    def check(self, results) -> bool:
        expected = {
            'http://www.w3.org/2002/07/owl#onProperty',
            'http://www.w3.org/2002/07/owl#someValuesFrom',
            'http://www.w3.org/2002/07/owl#allValuesFrom',
            'http://www.w3.org/2002/07/owl#hasValue',
            'http://www.w3.org/2002/07/owl#cardinality',
            'http://www.w3.org/2002/07/owl#minCardinality',
            'http://www.w3.org/2002/07/owl#maxCardinality',
            'http://www.w3.org/2002/07/owl#qualifiedCardinality',
            'http://www.w3.org/2002/07/owl#minQualifiedCardinality',
            'http://www.w3.org/2002/07/owl#maxQualifiedCardinality',
            'http://www.w3.org/2002/07/owl#onClass'
        }
        actual = {relation.toPython() for (relation, obj) in results}

        return len(expected.intersection(actual)) >= 2

    def normalize(self, node: 'VerbalizationNode',
                  triple_collector):
        query = self.verbalizer.next_step_query_builder(node)
        results = graph.query(query)

        next_node = None
        quantifier_relation = None
        property_relation = None
        literal_value = None
        on_class = None

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
                                       'http://www.w3.org/2002/07/owl#maxCardinality',
                                       'http://www.w3.org/2002/07/owl#qualifiedCardinality',
                                       'http://www.w3.org/2002/07/owl#minQualifiedCardinality',
                                       'http://www.w3.org/2002/07/owl#maxQualifiedCardinality'
                                       }:
                quantifier_relation = relation
                literal_value = obj

                # initialize next_node only if it hasn't be initialized yet.
                if next_node is None:
                    next_node = VerbalizationNode(
                        concept='',
                        parent_path=node.get_parent_path() + [(node.concept, relation)]
                    )
                    next_node.display = ''

            if relation.toPython() in {
                'http://www.w3.org/2002/07/owl#onClass'
            }:
                on_class = obj
                next_node = VerbalizationNode(
                    concept=obj,
                    parent_path=node.get_parent_path() + [(node.concept, relation)]
                )
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
            edge.display = self._handle_cardinality(quantifier_relation, property_relation, literal_value, on_class)
        else:
            edge.display = f'{property_relation_label} {quantifier_relation_label}'

        node.add_edge(edge)

        return [(reference.relationship, reference.node.concept) for reference in node.references]

    def _handle_cardinality(self, quantifier_relation, property_relation, obj_literal, on_class) -> str:
        property_relation_label = self.vocab.get_cls_label(property_relation)
        literal_value = obj_literal.toPython()
        on_class_label = ' '

        relation_plural_s = 's' if literal_value > 1 and not property_relation_label.endswith('s') else ''

        if property_relation_label.startswith('has'):
            property_relation_label = property_relation_label.replace('has ', '')

        if on_class:
            on_class_label = f' {self.vocab.get_cls_label(on_class)} '

        if quantifier_relation.endswith('cardinality') and literal_value == 0:
            return f'has zero {on_class_label} {property_relation_label}s'
        elif quantifier_relation.endswith('cardinality') or quantifier_relation.endswith('qualifiedCardinality'):
            return f'has exactly {literal_value}{on_class_label}{property_relation_label}{relation_plural_s}'
        elif quantifier_relation.endswith('minCardinality') or quantifier_relation.endswith('minQualifiedCardinality'):
            return f'has at least {literal_value}{on_class_label}{property_relation_label}{relation_plural_s}'
        elif quantifier_relation.endswith('maxCardinality') or quantifier_relation.endswith('maxQualifiedCardinality'):
            return f'has at most {literal_value}{on_class_label}{property_relation_label}{relation_plural_s}'


def get_obo_relations():
    ro = Graph()
    ro.parse(f'../data/ro.owl')
    query = """
            SELECT ?o ?label
            WHERE {
                ?o a owl:ObjectProperty ; rdfs:label ?label
                FILTER NOT EXISTS {
                    ?o owl:deprecated true
                }
            }
        """
    return {result[0].toPython(): result[1].toPython() for result in ro.query(query)}


if __name__ == '__main__':
    ontology_name = 'doid'
    use_llm = True

    obo_relations = get_obo_relations()

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

    obo_relations.update(rephrased)

    # create a vocabulary from the ontology.
    vocab = Vocabulary(graph, ignore=ignore, rephrased=obo_relations)

    # Use patterns to normalize the graph
    patterns = [OwlFirstRestPattern, OwlRestrictionPattern]

    # Initialize language model. TODO: Use LangChain for better interoperability
    llm = LlamaModel(base_url='http://127.0.0.1:8080/v1', temperature=0.7)

    verbalizer = Verbalizer(
        graph, vocab, patterns,
        language_model=llm,
        usage_config=VerbalizerModelUsageConfig(
            min_patterns_evaluated=0,
            min_statements=2
        )
    )

    # Simple test
    # fragment, text, count, llm_used = verbalizer.verbalize('http://purl.obolibrary.org/obo/ENVO_00000397')
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
    print(f'Found {len(classes)} classes.')
    for i, entry in tqdm(enumerate(classes)):
        # print(f'verbalizing {entry}')
        fragment, text, count, llm_used = verbalizer.verbalize(entry)

        # print(fragment)
        # print(text)
        # print("****" * 20)

        dataset.append({
            'ontology': ontology_name,
            'root': entry,
            'fragment': fragment,
            'text': text,
            'statements': count,
            'llm_used': llm_used
        })
        if i % 100 == 0 and llm:
            print(f'Cost so far: ${llm.cost}')

    pandas.DataFrame(dataset).to_csv(f'../output/{ontology_name}{"" if use_llm else ".raw"}.csv', index=False)
    print(f'Final cost: ${llm.cost}')
