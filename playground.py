import logging
import os

from verbalizer.nlp import ChatGptModelParaphrase, LlamaModelParaphrase
from verbalizer.process import Processor
from verbalizer.sampler import Sampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ignore = {
    'http://www.w3.org/2002/07/owl#onDatatype',
    'http://www.w3.org/2000/01/rdf-schema#seeAlso',
    'http://www.w3.org/2000/01/rdf-schema#label',
    'http://www.w3.org/2000/01/rdf-schema#comment',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
    'http://www.w3.org/2000/01/rdf-schema#isDefinedBy',

    # OBO documentation related
    'http://purl.obolibrary.org/obo/IAO_0000111',
    'http://purl.obolibrary.org/obo/IAO_0000114',
    'http://purl.obolibrary.org/obo/IAO_0000116',
    'http://purl.obolibrary.org/obo/IAO_0000117',
    'http://purl.obolibrary.org/obo/IAO_0000119',
    'http://purl.obolibrary.org/obo/IAO_0000589',
    'http://purl.obolibrary.org/obo/IAO_0000233',

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

    'http://purl.obolibrary.org/obo/mondo#excluded_from_qc_check',

    'http://purl.org/sig/ont/fma/author',
    'http://purl.org/sig/ont/fma/primary_author_and_curator',
    'http://purl.org/sig/ont/fma/contributing_author',
    'http://purl.org/sig/ont/fma/language',
    'http://purl.org/sig/ont/fma/FMAID',
    'http://purl.org/sig/ont/fma/Date_entered_modified',
    'http://purl.org/sig/ont/fma/authority',
    'http://purl.org/sig/ont/fma/Talairach',
    'http://purl.org/sig/ont/fma/modification',

    # SWEET
    'http://data.bioontology.org/metadata/prefixIRI',
    'http://purl.org/dc/terms/contributor',
    'http://purl.org/dc/terms/creator',
    'http://purl.org/dc/terms/created',
    'http://purl.org/dc/terms/source',

    # FOAF
    'http://www.w3.org/2003/06/sw-vocab-status/ns#term_status',

    # SKOS
    'http://www.w3.org/2004/02/skos/core#exactMatch',
    'http://www.w3.org/2004/02/skos/core#closeMatch'

}

rephrased = {
    'http://www.w3.org/2002/07/owl#equivalentClass': 'is same as',
    'http://www.w3.org/2000/01/rdf-schema#subClassOf': 'is a type of',
    'http://www.w3.org/2002/07/owl#intersectionOf': 'all of',
    'http://www.w3.org/2002/07/owl#unionOf': 'any of',
    'http://www.w3.org/2002/07/owl#disjointWith': 'is different from',
    'http://www.w3.org/2002/07/owl#withRestrictions': 'must be',
    'http://purl.obolibrary.org/obo/IAO_0000115': 'has definition',
}

if __name__ == '__main__':
    llama_model = LlamaModelParaphrase('http://localhost:11434/v1', temperature=0.1)
    openai_model = ChatGptModelParaphrase(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-4o', temperature=0.7)
    models = [openai_model, llama_model]

    sampler = Sampler(sample_n=100, seed=42)

    for model in models:
        processor = Processor(llm=model, vocab_ignore=ignore, vocab_rephrased=rephrased, min_statements=1)
        processor.process('people', './data/people.ttl')
        processor.process('pizza', './data/pizza.ttl')
        processor.process('mondo', './data/mondo.owl', data_sampler=sampler)
        processor.process('fma', './data/fma.owl', data_sampler=sampler)
