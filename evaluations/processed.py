import logging
import os
import unittest
from unittest.mock import patch

from rdflib import URIRef

from verbalizer.nlp import ChatGptModelParaphrase, LlamaModelParaphrase
from verbalizer.process import Processor
from verbalizer.sampler import Sampler
from verbalizer.verbalizer import VerbalizerInstanceStats, VerbalizationNode, _RE_COMBINE_WHITESPACE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomSampler(Sampler):
    def __init__(self, samples):
        super().__init__(sample_n=10)
        self.samples = samples

    def get_sample(self, items):
        return {'classes': self.samples, 'individuals': []}


ontologies = {
    'pizza': {
        'file': './data/pizza.ttl',
        'samples': [
            'http://www.co-ode.org/ontologies/pizza/2005/10/18/pizza.owl#Margherita',
            'http://www.co-ode.org/ontologies/pizza/2005/10/18/pizza.owl#InterestingPizza'
        ]
    },
    'people': {
        'file': './data/people.ttl',
        'samples': [
            'http://www.semanticweb.org/mdebe/ontologies/example#Child',
            'http://www.semanticweb.org/mdebe/ontologies/example#John_Doe'

        ]
    },
    'mondo': {
        'file': './data/mondo.owl',
        'samples': [
            'http://purl.obolibrary.org/obo/MONDO_0006755',
            'http://purl.obolibrary.org/obo/MONDO_0009162'
        ]
    },
    'fma': {
        'file': './data/fma.owl',
        'samples': [
            'http://purl.org/sig/ont/fma/fma83332',
            'http://purl.org/sig/ont/fma/fma7101'
        ]
    }
}


def _verbalization_function_patch(self, starting_concept):
    """
    Custom function that behaves almost the same the the original function. Only difference is what gets sent to the LLM.
    """
    sentences = []
    triples = []
    stats = VerbalizerInstanceStats()

    if isinstance(starting_concept, str):
        starting_concept = URIRef(starting_concept)

    node = VerbalizationNode(starting_concept)
    self._verbalize_as_text_from(node, self.vocab, triples, stats)

    for ref in node.references:
        sentences.append(_RE_COMBINE_WHITESPACE.sub(" ", f'{node.display} {ref.verbalize().strip()}.').strip())

    text = '\n'.join(sorted(sentences))
    onto_fragment: str = self.generate_fragment(triples)

    # update stats
    for triple in triples:
        subject, predicate, obj = triple
        stats.relationship_counter[predicate] += 1

        if isinstance(subject, URIRef):
            stats.concepts.add(subject)

        if isinstance(obj, URIRef):
            stats.concepts.add(obj)

    stats.statements = len(sentences)

    # convert the pseudo text into proper English.
    use_llm = bool(self.llm and self._check_llm_usage_policy(stats))
    llm_text = None
    if use_llm:
        # Note the difference here: we are passing `onto_fragment` instead of `text`
        llm_text = self.llm.pseudo_to_text(onto_fragment, extra=self.llm_config.extra_context)

    return onto_fragment, text, llm_text, stats


class EvaluateVerbalizationOfOwl(unittest.TestCase):

    @patch('verbalizer.verbalizer.Verbalizer.verbalize', _verbalization_function_patch)
    def test_evaluation(self):

        llama_model = LlamaModelParaphrase('http://localhost:11434/v1', temperature=0.1)
        openai_model = ChatGptModelParaphrase(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-4o', temperature=0.7)
        models = [openai_model, llama_model]

        for model in models:
            processor = Processor(llm=model, min_statements=1)
            for ontology_name, contents in ontologies.items():
                file = contents['file']
                sampler = CustomSampler(samples=contents['samples'])
                processor.verbalize_with(ontology_name, file, sampler=sampler)
