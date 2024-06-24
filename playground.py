import os

from core.nlp import ChatGptModel
from core.process import Processor

if __name__ == '__main__':
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
        'http://purl.obolibrary.org/obo/IAO_0000115': 'has definition',
        'http://www.w3.org/2002/07/owl#disjointWith': 'is different from'
    }

    # extra="""
    # The sentence describes concepts taken from an ontology for situation-based access control to determine what role can access patients' medical records.
    # """

    # model = None
    model = ChatGptModel(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-4o', temperature=0.3)
    processor = Processor(llm=model, vocab_ignore=ignore, vocab_rephrased=rephrased, min_statements=1)
    # processor.process('envo', './data/envo.owl', chunk_size=500)
    # processor.process('sweet', './data/sweet.owl', chunk_size=500)
    # processor.process('doid', './data/doid.owl', chunk_size=500)
    processor.process('pizza', './data/pizza.ttl')
