import os

from core.nlp import ChatGptModel, LlamaModel
from core.process import Processor

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
    'http://purl.org/dc/terms/source',

    # FOAF
    'http://www.w3.org/2003/06/sw-vocab-status/ns#term_status'
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
    # extra="""
    # The sentence describes concepts taken from an ontology for situation-based access control to determine what role can access patients' medical records.
    # """

    # model = None
    llama_model = LlamaModel('http://localhost:11434/v1', temperature=0.1)
    openai_model = ChatGptModel(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-4o', temperature=0.3)
    processor = Processor(llm=openai_model, vocab_ignore=ignore, vocab_rephrased=rephrased, min_statements=1)
    # processor.process('envo', './data/envo.owl', chunk_size=500)
    # processor.process('sweet', './data/sweet.owl', chunk_size=500)
    # processor.process('doid', './data/doid.owl', chunk_size=500)
    processor.process('pizza', './data/pizza.ttl')
    # processor.process('people', './data/people.ttl')
    # processor.process('foaf', './data/foaf.owl')
