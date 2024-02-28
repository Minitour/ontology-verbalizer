### Day 1

Initial inspection of two ontologies selected at random: ENVO and ADO, it seems that loading these ontologies directly
into a graph is not ideal. These ontologies seem to contain a lot of "noise" which originates from the OBO conventions.
This noise comes in the form of additional nodes that are purely for documentation of the development of the ontology
such as information about contributors. In addition to that, properties come in the form that is not graph friendly,
which results in relations between classes that do not have any proper labels.

From this initial setup, it can be concluded that verbalization should not take place directly over this graph, but
rather over some pre-processed version of it where only relevant information retained. The next step would be to
determine what is the ideal format which we would like to accept in OWL/RDF.

### Day 2

Successfully loaded the data into Neo4J and slowly learning Cypher. It seems that a pre-processing of the ontology is
not necessary. It is important however to note that the ontology does have some unnecessary attributes such as comments
and authors that should have some rules for removal.

### Day 3

Trying to inspect the definitions using Neo4J browser to see how graph snippets could be represented as textual
sentences.

Consider the following example:

Below is an XML snippet of the class `mediterranean grassland`. We can see that it is a subclass of two things:

- `subtropical grassland`
- a restriction that can be read as a thing that has `has quality some value from mediterranean`

In addition to that, it also defines `equivalentClass` such that:

- a thing that is `subtropical grassland and has quality someValuesFrom mediterranean`

It also contains additional information such as a textual definition, the author and time of creation.

```xml

<owl:Class rdf:about="http://purl.obolibrary.org/obo/ENVO_01001810">
    <owl:equivalentClass>
        <owl:Class>
            <owl:intersectionOf
                    rdf:parseType="Collection"> <!-- a thing that is subtropical grassland and has quality someValuesFrom mediterranean-->
                <rdf:Description
                        rdf:about="http://purl.obolibrary.org/obo/ENVO_01001809"/> <!-- subtropical grassland -->
                <owl:Restriction>
                    <owl:onProperty rdf:resource="http://purl.obolibrary.org/obo/RO_0000086"/> <!-- has quality -->
                    <owl:someValuesFrom
                            rdf:resource="http://purl.obolibrary.org/obo/ENVO_01000207"/> <!-- mediterranean -->
                </owl:Restriction>
            </owl:intersectionOf>
        </owl:Class>
    </owl:equivalentClass>
    <rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/ENVO_01001809"/>
    <rdfs:subClassOf>
        <owl:Restriction>
            <owl:onProperty rdf:resource="http://purl.obolibrary.org/obo/RO_0000086"/>
            <owl:someValuesFrom rdf:resource="http://purl.obolibrary.org/obo/ENVO_01000207"/>
        </owl:Restriction>
    </rdfs:subClassOf>
    <obo:IAO_0000115>A grassland which is subject to mediterranean climatic conditions.</obo:IAO_0000115>
    <dc:creator rdf:resource="https://orcid.org/0000-0002-4366-3088"/>
    <dc:date rdf:datatype="http://www.w3.org/2001/XMLSchema#dateTime">2019-10-27T17:00:35Z</dc:date>
    <rdfs:label xml:lang="en">mediterranean grassland</rdfs:label>
</owl:Class>
```

Cypher query for reference:

```shell
MATCH (n:Class WHERE n.label =  'mediterranean grassland')
```

### Day 4

Verbalizer has reached a good state in terms of implementation. I integrated it with OpenAI and did some optimizations.
Running it over all the concepts in envo with GPT-3.5 costs `$0.5760345`. This number is of course not done on all
concepts but only concepts who have at least 2 statements and at least one pattern detected. This resulted in 2643
concepts which were verbalized with OpenAI. The rest were verbalized with simple logic of the verbalizer. This process
took about an hour to complete but can be optimized to run in parallel in the future.

This process was also done for SWEET ontology, but in SWEET's case, only about 600 concepts were verbalized by the LLM
because of its lack of expressiveness.

Overall, this results in a dataset of size ~3000 which can be used for the initial fine-tuning process.