import re
from typing import Type

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
        self._ignore_list = ignore or {}

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


class Pattern:

    def __init__(self, graph: Graph, verbalizer: 'Verbalizer', vocabulary: Vocabulary):
        self._graph = graph
        self.verbalizer = verbalizer
        self.vocab = vocabulary

    def check(self, results) -> bool:
        return False

    def normalize(self, node: 'VerbalizationNode',
                  triple_collector) -> list[tuple[Node, Node]]:
        return []


class VerbalizationEdge:

    def __init__(self, relationship, node: 'VerbalizationNode'):
        self.relationship = relationship
        self.node = node
        self._display = None

    @property
    def display(self):
        return self._display

    @display.setter
    def display(self, value):
        if not self._display:
            self._display = value

    def __repr__(self):
        return self.display

    def verbalize(self) -> str:
        display = self.display

        if not display:
            display = ''

        if display.startswith('#'):
            display = ''

        return f'{display} {self.node.verbalize()}'


class VerbalizationNode:
    def __init__(self, concept, parent_path=None):
        self.concept = concept
        if parent_path is None:
            self._true_path = [(concept, None)]
        else:
            self._true_path = parent_path + [(concept, None)]
        self.references: list[VerbalizationEdge] = []
        self._display = None

    def add_edge(self, edge: VerbalizationEdge):
        self.references.append(edge)

    def get_path(self):
        return list(self._true_path)

    def get_parent_path(self):
        return list(self._true_path[:-1])

    def get_next_node(self, relationship, concept):
        for reference in self.references:
            if reference.relationship == relationship and reference.node.concept == concept:
                return reference.node
        return None

    @property
    def display(self):
        return self._display

    @display.setter
    def display(self, value):
        if not self._display:
            self._display = value

    def __repr__(self):
        return self.display

    def verbalize(self) -> str:
        sentences = [edge.verbalize() for edge in self.references]

        display = self.display
        if isinstance(self.concept, BNode):
            display = ''

        return f'{display} {" and ".join(sentences)}'


class Verbalizer:
    prefix = 'https://zaitoun.dev/onto/'

    def __init__(self, graph: Graph,
                 vocabulary: Vocabulary,
                 patterns: list[Type[Pattern]] = None,
                 language_model: LanguageModel = None):
        self.graph = graph
        self.vocab = vocabulary
        self.llm = language_model

        self.patterns = [pattern(graph, self, vocabulary) for pattern in patterns or []]

    def verbalize(self, starting_concept) -> (str, str):
        """
        Returns the Turtle Fragment and its corresponding textual description.
        """
        sentences = []
        triples = []

        if isinstance(starting_concept, str):
            starting_concept = URIRef(starting_concept)

        node = VerbalizationNode(starting_concept)
        self._verbalize_as_text_from(node, self.vocab, triples)

        for ref in node.references:
            sentences.append(f'{node.display} {ref.verbalize()}.')

        text = '\n'.join(sentences)
        onto_fragment: str = self.generate_fragment(triples)

        # convert the pseudo text into proper English.
        if self.llm:
            text = self.llm.pseudo_to_text(text)

        return onto_fragment, text

    def _verbalize_as_text_from(self, node: VerbalizationNode,
                                vocab: Vocabulary,
                                triple_collector: list):
        query = self.next_step_query_builder(node)
        results = self.graph.query(query)

        results_normalized = False

        # check patterns
        for pattern in self.patterns:
            if not pattern.check(results):
                continue

            results = pattern.normalize(node, triple_collector)
            results_normalized = True
            break

        # set node display
        node.display = vocab.get_cls_label(node.concept)

        for result in results:
            relation, obj_2 = result

            relation_display = vocab.get_rel_label(relation)

            if relation_display == Vocabulary.IGNORE_VALUE:
                continue

            if not results_normalized:
                # Continue to expand the graph
                next_node = VerbalizationNode(obj_2, parent_path=node.get_parent_path() + [(node.concept, relation)])
                edge = VerbalizationEdge(relation, next_node)
                node.add_edge(edge)

                edge.display = relation_display

                # collect triple
                triple_collector.append((node.concept, relation, next_node.concept))
            else:
                # we already built this part of the graph so we just need to fetch it out.
                next_node = node.get_next_node(relation, obj_2)

            print(node.concept, relation, obj_2)

            if isinstance(obj_2, URIRef):
                obj_display_2 = vocab.get_cls_label(obj_2)
            elif isinstance(obj_2, Literal):
                obj_display_2 = obj_2.toPython()
            elif isinstance(obj_2, BNode):
                self._verbalize_as_text_from(next_node, vocab, triple_collector)
                continue
            else:
                obj_display_2 = None

            next_node.display = obj_display_2

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

    @classmethod
    def next_step_query_builder(cls, node: VerbalizationNode) -> str:
        """
        Given a list of RDF Node, where each nodes is somehow connected to the next node in the list,
        return a SPARQL query expression to get all the relationships of the last node.
        This is needed in order to traverse over Blank Nodes.
        """

        trail = node.get_path()
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
                subject_vocab_rep = self.vocab.get_cls_label(subject)
                if subject_vocab_rep == Vocabulary.IGNORE_VALUE:
                    continue
                subject_node = display_to_uri(subject_vocab_rep)

            if isinstance(obj, URIRef):
                object_vocab_rep = self.vocab.get_cls_label(obj)
                if object_vocab_rep == Vocabulary.IGNORE_VALUE:
                    continue
                object_node = display_to_uri(object_vocab_rep)

            if not self._starts_with_one_of(predicate_node.toPython(), [OWL, RDF, RDFS]):
                predicate_node = display_to_uri(self.vocab.get_rel_label(predicate))

            g.add((subject_node, predicate_node, object_node))

            # If subject or object have labels - add them as rdfs:label
            if isinstance(subject, URIRef) and self.vocab.get_cls_label(subject, default=''):
                g.add((subject_node, RDFS.label, Literal(self.vocab.get_cls_label(subject))))

            if isinstance(obj, URIRef) and self.vocab.get_cls_label(obj, default=''):
                g.add((object_node, RDFS.label, Literal(self.vocab.get_cls_label(obj))))

        return g.serialize(format='turtle')

    @staticmethod
    def _starts_with_one_of(val: str, items: list):
        return any([val.startswith(str(e)) for e in items])
