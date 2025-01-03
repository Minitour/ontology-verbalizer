import unittest

from rdflib import Graph

from verbalizer.process import Processor
from verbalizer.sampler import Sampler
from verbalizer.vocabulary import Vocabulary
from verbalizer import Verbalizer

rename_iri = {
    'http://www.w3.org/2002/07/owl#equivalentClass': 'is same as',
    'http://www.w3.org/2000/01/rdf-schema#subClassOf': 'is a type of',
    'http://www.w3.org/2002/07/owl#intersectionOf': 'all of',
    'http://www.w3.org/2002/07/owl#unionOf': 'any of',
    'http://www.w3.org/2002/07/owl#disjointWith': 'is different from',
    'http://www.w3.org/2002/07/owl#withRestrictions': 'must be'
}
ignore_iri = {
    'http://www.w3.org/2002/07/owl#onDatatype',
    'http://www.w3.org/2000/01/rdf-schema#seeAlso',
    'http://www.w3.org/2000/01/rdf-schema#label',
    'http://www.w3.org/2000/01/rdf-schema#comment',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
    'http://www.w3.org/2000/01/rdf-schema#isDefinedBy',
    'http://www.w3.org/2003/06/sw-vocab-status/ns#term_status',
    'http://www.w3.org/2000/01/rdf-schema#Class'
}


class TestVerbalization(unittest.TestCase):

    def test_verbalization(self):
        # graph
        ontology = Processor.from_file('./data/foaf.owl')

        # create vocabulary
        vocab = Vocabulary(ontology, ignore=ignore_iri, rephrased=rename_iri)

        # create verbalizer
        verbalizer = Verbalizer(vocab)

        results = Processor.verbalize_with(verbalizer, namespace='foaf')
        self.assertEqual(12, len(results))

        # Add default prefix (won't work without this)
        fragment_sample = '@prefix : <https://zaitoun.dev#> .\n' + results[0]['fragment']
        g = Graph()
        g.parse(data=fragment_sample, format="turtle")

        self.assertEqual(7, len(list(g.triples((None, None, None)))))

    def test_verbalization_with_sampler(self):
        # graph
        ontology = Processor.from_file('./data/foaf.owl')

        # create vocabulary
        vocab = Vocabulary(ontology, ignore=ignore_iri, rephrased=rename_iri)

        # create verbalizer
        verbalizer = Verbalizer(vocab)

        sampler = Sampler(sample_n=10, seed=42)
        results = Processor.verbalize_with(verbalizer, namespace='foaf', sampler=sampler)

        # although we sampled 10, only 7 were applicable.
        self.assertEqual(7, len(results))
