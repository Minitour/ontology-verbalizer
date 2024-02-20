import re
from collections import defaultdict

import pandas
from rdflib import Graph
from rdflib import RDFS, RDF, OWL
from rdflib import URIRef, Literal, BNode
from rdflib.term import Node

from core.nlp import LanguageModel


class Vocabulary:
    IGNORE_VALUE = object()

    def __init__(self, graph, ignore: set = None, rephrased: dict[str, str] = None):
        self._graph = graph
        self.relationship_labels = self._get_ontology_relationship_labels()
        self.object_labels = self._get_ontology_object_labels()
        self.rephrased = rephrased or dict()

        self._ignore_list = ignore

    def _get_ontology_relationship_labels(self) -> dict[str, str]:
        results = self._graph.query(
            """
            SELECT DISTINCT ?p ?relation
            WHERE {
                ?o1 ?p ?o2 .
                OPTIONAL {
                    ?p rdfs:label ?pLabel
                }
                BIND (
                  COALESCE(
                    ?pLabel,
                    ?p
                  ) AS ?relation
                )
            }
            """
        )

        owl_rdf_ns = [OWL, RDF, RDFS]
        ontology_relations = {}

        for result in results:
            iri, label = result
            iri_str = iri.toPython()
            label_str = label.toPython()

            is_owl_rdf = any([label_str.startswith(str(ns)) for ns in owl_rdf_ns])

            # if the label is also the URI then try to parse it.
            if label_str.startswith('http'):
                label_str_parts = label_str.split('#')
                if len(label_str_parts) == 1:
                    continue
                label_str = label_str_parts[1]
                # convert camel case to snake case
                label_str = re.sub(r'(?<!^)(?=[A-Z][a-z])', '_', label_str).lower()

            label_str = re.sub('[^0-9a-zA-Z]+', ' ', label_str)

            ontology_relations[iri_str] = label_str

        return ontology_relations

    def _get_ontology_object_labels(self) -> dict[str, str]:
        results = self._graph.query(
            """
            SELECT DISTINCT ?o1 ?o1Label
            WHERE {
                ?o1 ?p ?o2 .
                ?o1 rdfs:label ?o1Label
            }
            """
        )
        object_labels = {}
        for result in results:
            iri, label = result
            iri_str = iri.toPython()
            label_str = label.toPython()
            object_labels[iri_str] = label_str
        return object_labels

    def _util_lookup(self, dictionary, val, default):
        if isinstance(val, URIRef):
            val = val.toPython()

        if not isinstance(val, str):
            return None

        if val in self._ignore_list:
            return self.__class__.IGNORE_VALUE

        # try get from overrides first
        if result := self.rephrased.get(val):
            return result

        if default is None:
            default = val

        return dictionary.get(val, default)

    def get_rel_label(self, val, default=None) -> str:
        return self._util_lookup(self.relationship_labels, val, default)

    def get_cls_label(self, val, default=None) -> str:
        return self._util_lookup(self.object_labels, val, default)


class Verbalizer:
    prefix = 'https://zaitoun.dev/onto/'

    # These are relations that should not be verbalized because they are OWL specific.
    def __init__(self, graph: Graph, vocabulary: Vocabulary, language_model: LanguageModel = None):
        self.graph = graph
        self.vocab = vocabulary
        self.llm = language_model
        self.special_relations = {
            'http://www.w3.org/1999/02/22-rdf-syntax-ns#first',
            'http://www.w3.org/1999/02/22-rdf-syntax-ns#rest'
        }

    def verbalize(self, starting_concept, max_depth=5) -> (str, str):
        """
        Returns the Turtle Fragment and its corresponding textual description.
        """
        sentences = []
        triples = []

        if isinstance(starting_concept, str):
            starting_concept = URIRef(starting_concept)

        self._verbalize_as_text_from([(starting_concept, None)],
                                     sentences,
                                     triples,
                                     depth=max_depth,
                                     max_depth=max_depth)
        text = '\n'.join(sentences)
        onto_fragment: str = self.generate_fragment(triples)

        # convert the pseudo text into proper English.
        if self.llm:
            text = self.llm.pseudo_to_text(text)

        return onto_fragment, text

    def _verbalize_as_text_from(self,
                                current_path: list,
                                sentence_collector: list,
                                triple_collector: list, depth=3, max_depth=3):
        if depth < 0:
            return

        def indent(i: int):
            return ' ' * i * 4

        def add_group(index, start=True):
            indentation = indent(max_depth - depth)
            group = f'group_{index}'
            sentence_collector.append(f'{indentation}<{group}>' if start else f'{indentation}</{group}>')

        query = self.query_builder(current_path)
        results = self.graph.query(query)
        obj_display_1 = self._verbalize_trail(current_path)

        # add spacing
        # add_space_if_possible()
        add_group(max_depth - depth + 1, start=True)
        for result in results:
            relation, obj_2 = result
            print(current_path[-1][0], relation, obj_2)
            print(type(current_path[-1][0]), type(relation), type(obj_2))
            relation_display = self.vocab.get_rel_label(relation)

            if relation_display == Vocabulary.IGNORE_VALUE:
                continue

            if isinstance(obj_2, URIRef):
                obj_display_2 = self.vocab.get_cls_label(obj_2)
            elif isinstance(obj_2, Literal):
                obj_display_2 = obj_2.toPython()
            elif isinstance(obj_2, BNode):
                triple_collector.append((current_path[-1][0], relation, obj_2))
                self._verbalize_as_text_from(current_path[:-1] + [(current_path[-1][0], relation), (obj_2, None)],
                                             sentence_collector,
                                             triple_collector,
                                             depth - 1,
                                             max_depth)
                # add_space_if_possible()
                continue
            else:
                obj_display_2 = None

            if obj_display_2 == Vocabulary.IGNORE_VALUE:
                continue
            indentation = indent(max_depth - depth + 1)
            if relation.toPython() in self.special_relations:
                sentence_collector.append(f'{indentation}{obj_display_1}{obj_display_2}')
            else:
                sentence_collector.append(f'{indentation}{obj_display_1}-[{relation_display}]->{obj_display_2}')

            # collect triple
            triple_collector.append((current_path[-1][0], relation, obj_2))

        add_group(max_depth - depth + 1, start=False)

    def _verbalize_trail(self, trail: list[tuple[URIRef, URIRef]]) -> str:
        """
        Given a trail (A path from one node to another), generate a pseudo sentence that describes it.
        """
        expressions = []

        for i, (t, p) in enumerate(trail):
            is_last = i == len(trail) - 1

            if is_last:
                if isinstance(t, BNode):
                    exp = ''
                else:
                    exp = f'{self.vocab.get_cls_label(t)}'
            else:
                if p.toPython() in self.special_relations:
                    continue
                if isinstance(t, BNode):
                    exp = f'-[{self.vocab.get_rel_label(p)}]->'
                else:
                    exp = f'{self.vocab.get_cls_label(t)}-[{self.vocab.get_rel_label(p)}]->'
            expressions.append(exp)

        if expressions:
            return ''.join(expressions)
        return ''

    @classmethod
    def query_builder(cls, trail: list[tuple[URIRef, URIRef]]) -> str:
        """
        Given a list of RDF Node, where each nodes is somehow connected to the next node in the list,
        return a SPARQL query expression to get all the relationships of the last node.
        This is needed in order to traverse over Blank Nodes.
        """
        if isinstance(trail, Node):
            trail = [trail]

        expressions = []
        filters = []
        for i, (t, p) in enumerate(trail):
            is_last = i == len(trail) - 1
            ref = cls.get_reference_expression(t, index=i)
            if is_last:
                exp = f'{ref} ?p ?o'
            else:
                next_t, _ = trail[i + 1]
                next_ref = cls.get_reference_expression(next_t, index=i + 1)
                exp = f'{ref} <{p.toPython()}> {next_ref}'

            if isinstance(t, BNode):
                filters.append(f"filter(str(?o{i}) = '{t.toPython()}')")

            expressions.append(exp)

        query = f"SELECT ?p ?o WHERE {{ {' . '.join(expressions)} . {' '.join(filters)} }}"
        return query

    @classmethod
    def get_reference_expression(cls, t: Node, index: int):
        """
        Util function
        """
        if isinstance(t, URIRef):
            return f'<{t.toPython()}>'
        if isinstance(t, BNode):
            return f'?o{index}'

        return None

    @staticmethod
    def _starts_with_one_of(val: str, items: list):
        return any([val.startswith(str(e)) for e in items])

    def generate_fragment(self, triples: list[tuple[Node, URIRef, Node]]) -> str:
        """
        Return a string that represents the triples as an ontology fragment in turtle format.
        """

        def display_to_uri(display: str) -> URIRef:
            if display.startswith('http'):
                return URIRef(display)
            identifier = re.sub('[^a-zA-Z0-9 \n\.]', ' ', display).lower().replace(' ', '_')
            return URIRef(self.prefix + identifier)

        g = Graph()
        g.bind(prefix='', namespace=self.prefix)
        for triple in triples:
            subject, predicate, obj = triple

            subject_node = subject
            predicate_node = predicate
            object_node = obj

            if isinstance(subject, URIRef):
                subject_node = display_to_uri(self.vocab.get_cls_label(subject))

            if isinstance(obj, URIRef):
                object_node = display_to_uri(self.vocab.get_cls_label(obj))

            if not self._starts_with_one_of(predicate_node.toPython(), [OWL, RDF, RDFS]):
                predicate_node = display_to_uri(self.vocab.get_rel_label(predicate))

            g.add((subject_node, predicate_node, object_node))

            # If subject or object have labels - add them as rdfs:label
            if isinstance(subject, URIRef) and self.vocab.get_cls_label(subject, default=''):
                g.add((subject_node, RDFS.label, Literal(self.vocab.get_cls_label(subject))))

            if isinstance(obj, URIRef) and self.vocab.get_cls_label(obj, default=''):
                g.add((object_node, RDFS.label, Literal(self.vocab.get_cls_label(obj))))

        return g.serialize(format='turtle')
