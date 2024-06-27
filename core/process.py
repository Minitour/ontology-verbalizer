import datetime
from pathlib import Path

import pandas
from rdflib import Graph, URIRef
from tqdm import tqdm

from core.nlp import LlamaModel, LanguageModel
from core.verbalizer import Pattern, VerbalizationNode, VerbalizationEdge
from core.verbalizer import Vocabulary, Verbalizer, VerbalizerModelUsageConfig


class OwlFirstRestPattern(Pattern):
    def check(self, results) -> bool:
        expected = {'http://www.w3.org/1999/02/22-rdf-syntax-ns#first',
                    'http://www.w3.org/1999/02/22-rdf-syntax-ns#rest'}
        return {relation.toPython() for (relation, obj) in results} == expected

    def normalize(self, node: 'VerbalizationNode', triple_collector):
        current = node
        while current.concept != URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#nil'):
            query = self.verbalizer.next_step_query_builder(current)
            results = self._graph.query(query)
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
        results = self._graph.query(query)

        next_node = None
        quantifier_relation = None
        property_relation = None
        literal_value = None
        on_class = None

        for (relation, obj) in results:

            if self.vocab.should_ignore(relation.toPython()):
                triple_collector.append((node.concept, relation, obj))
                continue

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


def get_graph_from_file(file_path: str) -> Graph:
    graph = Graph()
    try:
        graph.parse(file_path, format='xml')
        return graph
    except:
        pass

    try:
        graph.parse(file_path, format='n3')
        return graph
    except:
        pass
    return graph


class Processor:

    def __init__(self,
                 llm: LanguageModel = LlamaModel(base_url='http://127.0.0.1:8080/v1', temperature=0.7),
                 patterns: list = None,
                 min_patterns_evaluated=0,
                 min_statements=2,
                 vocab_ignore: set = None,
                 vocab_rephrased: dict = None,
                 extra_context: str = ""):
        self.llm = llm
        self.patterns = patterns or [OwlFirstRestPattern, OwlRestrictionPattern]
        self.verbalizer_model_usage_config = VerbalizerModelUsageConfig(
            min_patterns_evaluated,
            min_statements,
            extra_context
        )
        self.vocab_ignore = vocab_ignore
        self.vocab_rephrase = vocab_rephrased

    def process(self, name: str, file_path: str, output_dir: str = './output', chunk_size=1000):
        # current timestamp
        now = datetime.datetime.utcnow()
        timestamp = int(now.timestamp())

        # make output directory
        out = f'{output_dir}/{name}/{timestamp}'
        Path(out).mkdir(parents=True, exist_ok=True)

        graph = get_graph_from_file(file_path)

        vocab = Vocabulary(graph, self.vocab_ignore, self.vocab_rephrase)

        verbalizer = Verbalizer(
            graph,
            vocabulary=vocab,
            patterns=self.patterns,
            language_model=self.llm,
            usage_config=self.verbalizer_model_usage_config
        )

        classes = self._get_classes(graph)

        dataset = []
        partition = 0
        for entry in tqdm(classes, desc='Verbalizing Classes'):
            fragment, text, llm_text, count = verbalizer.verbalize(entry)

            dataset.append({
                'ontology': name,
                'root': entry,
                'fragment': fragment,
                'text': text,
                'llm_text': llm_text,
                'statements': count,
                'llm_used': True if llm_text else False
            })
            if len(dataset) == chunk_size:
                pandas.DataFrame(dataset).to_csv(f'{out}/file_{partition}.csv', index=False)
                partition += 1
                dataset = []

        if dataset:
            pandas.DataFrame(dataset).to_csv(f'{out}/file_{partition}.csv', index=False)

        print(f'Finished verbalizing')
        if self.llm:
            print(f'LLM usage cost: ${self.llm.cost}')

    @staticmethod
    def _get_classes(graph):
        query = """
                SELECT ?o ?label
                WHERE {
                    ?o a owl:Class .
                    OPTIONAL {
                        ?o rdfs:label ?label
                    }
                    FILTER NOT EXISTS {
                        ?o owl:deprecated true
                    }
                }
            """
        return [result[0] for result in graph.query(query) if isinstance(result[0], URIRef)]
