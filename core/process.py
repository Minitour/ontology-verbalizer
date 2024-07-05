import datetime
import logging
from pathlib import Path
from typing import Optional
from xml.sax import SAXParseException

import pandas
from rdflib import Graph, URIRef
from tqdm import tqdm

from core.nlp import LlamaModel, LanguageModel
from core.patterns.owl_disjoint import OwlDisjointWith
from core.patterns.owl_first_rest import OwlFirstRestPattern
from core.patterns.owl_restriction import OwlRestrictionPattern
from core.sampler import Sampler
from core.verbalizer import Vocabulary, Verbalizer, VerbalizerModelUsageConfig

logger = logging.getLogger(__name__)


class Processor:
    """
    The processor that starts the verbalization process and outputs the results.
    """

    def __init__(self,
                 llm: LanguageModel = LlamaModel(base_url='http://localhost:11434/v1'),
                 patterns: list = None,
                 min_patterns_evaluated=0,
                 min_statements=2,
                 vocab_ignore: set = None,
                 vocab_rephrased: dict = None,
                 extra_context: str = ""):
        """
        :param llm: The LLM to use for verbalization. Use None to run without an LLM.
        :param patterns: Patterns to use. If not passed will use the default patterns.
        :param min_patterns_evaluated: The minimum number of patterns needed to be evaluated so that LLM paraphrasing is used.
        :param min_statements: The minimum number of statements needed so that LLM paraphrasing is used.
        :param vocab_ignore: A set of URIs to ignore (can be relations or objects)
        :param vocab_rephrased: A manual set of labels to provide for certain URIs.
        :param extra_context: Additional context to pass to the LLM as part of the system message.
        """
        self.llm = llm
        self.patterns = patterns or [OwlFirstRestPattern, OwlRestrictionPattern, OwlDisjointWith]
        self.verbalizer_model_usage_config = VerbalizerModelUsageConfig(
            min_patterns_evaluated,
            min_statements,
            extra_context
        )
        self.vocab_ignore = vocab_ignore
        self.vocab_rephrase = vocab_rephrased

    def process(self, name: str,
                file_path: str,
                output_dir: str = './output', chunk_size: int = 1000,
                data_sampler: Optional[Sampler] = None):
        """
        Start the verbalization process.
        :param name: Name of the directory to create under the output directory.
        :param file_path: Input file path.
        :param output_dir: Name of the output directory.
        :param chunk_size: Number of entries (rows) per file. default = 1000
        :param data_sampler: A sampling configuration, use to sample large ontologies.
        """
        # current timestamp
        now = datetime.datetime.utcnow()
        timestamp = int(now.timestamp())

        # make output directory
        out = f'{output_dir}/{name}/{timestamp}'
        Path(out).mkdir(parents=True, exist_ok=True)

        graph = self.get_graph_from_file(file_path)

        vocab = Vocabulary(graph, self.vocab_ignore, self.vocab_rephrase)

        verbalizer = Verbalizer(
            graph,
            vocabulary=vocab,
            patterns=self.patterns,
            language_model=self.llm,
            usage_config=self.verbalizer_model_usage_config
        )

        classes = self._get_classes(graph)
        individuals = self._get_individuals(graph)

        if sampler := data_sampler:
            samples = sampler.get_sample({'classes': classes, 'individuals': individuals})
            classes, individuals = samples['classes'], samples['individuals']

        dataset = []
        partition = 0
        for entry in tqdm(classes + individuals, desc='Verbalizing'):
            fragment, text, llm_text, count = verbalizer.verbalize(entry)

            dataset.append({
                'ontology': name,
                'root': entry,
                'fragment': fragment,
                'text': text,
                'llm_text': llm_text,
                'statements': count,
                'llm': self.llm.name if self.llm else 'None'
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
        return [result[0] for result in
                tqdm(graph.query(query), desc='Loading Classes') if isinstance(result[0], URIRef)]

    @staticmethod
    def _get_individuals(graph):
        query = """
                        SELECT ?o ?label
                        WHERE {
                            ?o a owl:NamedIndividual .
                            OPTIONAL {
                                ?o rdfs:label ?label
                            }
                            FILTER NOT EXISTS {
                                ?o owl:deprecated true
                            }
                        }
                    """
        return [result[0] for result in
                tqdm(graph.query(query), desc='Loading Instances') if isinstance(result[0], URIRef)]

    @staticmethod
    def get_graph_from_file(file_path: str) -> Graph:
        """
        Helper function to load graph from file.
        """
        graph = Graph()
        formats = ['xml', 'n3']
        logger.info(f'Loading File {file_path}')
        for file_format in formats:
            try:
                graph.parse(file_path, format=file_format)
                break
            except SAXParseException:
                pass
        logger.info(f'Done Loading.')
        return graph
