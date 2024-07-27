import os

from verbalizer.nlp import LlamaModelParaphrase, ChatGptModelParaphrase

examples = [
    """
    default:Margherita
      rdf:type owl:Class ;

      rdfs:subClassOf default:NamedPizza ;
      rdfs:subClassOf
              [ rdf:type owl:Restriction ;
                owl:onProperty default:hasTopping ;
                owl:someValuesFrom default:MozzarellaTopping
              ] ;
      rdfs:subClassOf
              [ rdf:type owl:Restriction ;
                owl:onProperty default:hasTopping ;
                owl:someValuesFrom default:TomatoTopping
              ] ;
      rdfs:subClassOf
              [ rdf:type owl:Restriction ;
                owl:allValuesFrom
                        [ rdf:type owl:Class ;
                          owl:unionOf (default:MozzarellaTopping default:TomatoTopping)
                        ] ;
                owl:onProperty default:hasTopping
              ] ;
      owl:disjointWith default:LaReine , default:FruttiDiMare , default:Fiorentina , default:SloppyGiuseppe , default:AmericanHot , default:Veneziana , default:UnclosedPizza , default:Mushroom , default:PolloAdAstra , default:Caprina , default:Cajun , default:American , default:Giardiniera , default:PrinceCarlo , default:Capricciosa , default:Siciliana , default:Parmense , default:Rosa , default:Soho , default:QuattroFormaggi , default:FourSeasons , default:Napoletana .
    """,

    """
    default:InterestingPizza
      rdf:type owl:Class ;
      rdfs:comment "Note that this is a cardinality constraint on the hasTopping property and NOT a qualified cardinality constraint (QCR). A QCR would specify from which class the members in this relationship must be. eg has at least 3 toppings from PizzaTopping. This is currently not supported in OWL."@en ;

      owl:equivalentClass
              [ rdf:type owl:Class ;
                owl:intersectionOf ([ rdf:type owl:Restriction ;
                            owl:minCardinality "3"^^xsd:int ;
                            owl:onProperty default:hasTopping
                          ] default:Pizza)
              ] .
    """,

    """
    :Child rdf:type owl:Class ;
       owl:equivalentClass [ owl:intersectionOf ( :Person
                                                  [ rdf:type owl:Restriction ;
                                                    owl:onProperty :has_Age ;
                                                    owl:someValuesFrom [ rdf:type rdfs:Datatype ;
                                                                         owl:onDatatype xsd:integer ;
                                                                         owl:withRestrictions ( [ xsd:maxExclusive 18
                                                                                                ]
                                                                                              )
                                                                       ]
                                                  ]
                                                ) ;
                             rdf:type owl:Class
                           ] ;
       rdfs:subClassOf :Person .
    """,

    """
    :John_Doe rdf:type owl:NamedIndividual ,
                   :Person ;
          :has_Brother :Tom_Doe ;
          :has_Daughter :Mary_Doe ;
          :has_Friend :John_Smith ;
          :has_Sister :Sarah_Doe ;
          :has_Wife :Beth_Doe ;
          :has_Age 34 .
    """,

    """
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/MONDO_0006755">
        <rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/MONDO_0002254"/>
        <rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/MONDO_0003240"/>
        <obo:IAO_0000115>Abnormal thyroid function tests, low triiodothyronine with elevated reverse triiodothyronine, in the setting of non-thyroidal illness.</obo:IAO_0000115>
        <oboInOwl:hasDbXref>DOID:2856</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>EFO:1000931</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>ICD10CM:E07.81</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>ICD9:790.94</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>MEDGEN:41908</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>MESH:D005067</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>MedDRA:10015549</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>NCIT:C113170</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>SCTID:237542005</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>UMLS:C0015190</oboInOwl:hasDbXref>
        <oboInOwl:hasExactSynonym>euthyroid sick syndrome</oboInOwl:hasExactSynonym>
        <oboInOwl:hasExactSynonym>sick euthyroid syndrome</oboInOwl:hasExactSynonym>
        <oboInOwl:hasExactSynonym>sick-euthyroid syndrome</oboInOwl:hasExactSynonym>
        <oboInOwl:id>MONDO:0006755</oboInOwl:id>
        <oboInOwl:inSubset rdf:resource="http://purl.obolibrary.org/obo/mondo#otar"/>
        <rdfs:label>euthyroid sick syndrome</rdfs:label>
        <skos:closeMatch rdf:resource="http://identifiers.org/meddra/10015549"/>
        <skos:exactMatch rdf:resource="http://identifiers.org/medgen/41908"/>
        <skos:exactMatch rdf:resource="http://identifiers.org/mesh/D005067"/>
        <skos:exactMatch rdf:resource="http://identifiers.org/snomedct/237542005"/>
        <skos:exactMatch rdf:resource="http://linkedlifedata.com/resource/umls/id/C0015190"/>
        <skos:exactMatch rdf:resource="http://purl.bioontology.org/ontology/ICD10CM/E07.81"/>
        <skos:exactMatch rdf:resource="http://purl.obolibrary.org/obo/DOID_2856"/>
        <skos:exactMatch rdf:resource="http://purl.obolibrary.org/obo/NCIT_C113170"/>
    </owl:Class>
    """,

    """
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/MONDO_0009162">
        <rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/MONDO_0006025"/>
        <rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/MONDO_0018770"/>
        <rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/MONDO_0019287"/>
        <rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/MONDO_0100547"/>
        <obo:IAO_0000115>Ellis-van Creveld syndrome (EVC) is a skeletal and ectoderlam dysplasia characterized by a tetrad of short stature, postaxial polydactyly, ectodermal dysplasia, and congenital heart defects.</obo:IAO_0000115>
        <obo:IAO_0000233 rdf:datatype="http://www.w3.org/2001/XMLSchema#anyURI">https://github.com/monarch-initiative/mondo/issues/4948</obo:IAO_0000233>
        <oboInOwl:hasDbXref>DOID:12714</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>GARD:1301</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>ICD9:756.55</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>MEDGEN:8584</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>MESH:D004613</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>MedDRA:10008724</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>NCIT:C84684</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>NORD:1083</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>OMIM:225500</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>Orphanet:289</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>SCTID:62501005</oboInOwl:hasDbXref>
        <oboInOwl:hasDbXref>UMLS:C0013903</oboInOwl:hasDbXref>
        <oboInOwl:hasExactSynonym>Chondroectodermal dysplasia</oboInOwl:hasExactSynonym>
        <oboInOwl:hasExactSynonym>EVC</oboInOwl:hasExactSynonym>
        <oboInOwl:hasExactSynonym>Ellis Van Creveld Syndrome</oboInOwl:hasExactSynonym>
        <oboInOwl:hasExactSynonym>Ellis Van Creveld syndrome</oboInOwl:hasExactSynonym>
        <oboInOwl:hasExactSynonym>Ellis-VAN Creveld syndrome</oboInOwl:hasExactSynonym>
        <oboInOwl:hasExactSynonym>Ellis-van Creveld syndrome</oboInOwl:hasExactSynonym>
        <oboInOwl:hasExactSynonym>Mesoectodermal dysplasia</oboInOwl:hasExactSynonym>
        <oboInOwl:hasExactSynonym>mesodermic dysplasia</oboInOwl:hasExactSynonym>
        <oboInOwl:id>MONDO:0009162</oboInOwl:id>
        <oboInOwl:inSubset rdf:resource="http://purl.obolibrary.org/obo/mondo#gard_rare"/>
        <oboInOwl:inSubset rdf:resource="http://purl.obolibrary.org/obo/mondo#nord_rare"/>
        <oboInOwl:inSubset rdf:resource="http://purl.obolibrary.org/obo/mondo#ordo_disorder"/>
        <oboInOwl:inSubset rdf:resource="http://purl.obolibrary.org/obo/mondo#ordo_malformation_syndrome"/>
        <oboInOwl:inSubset rdf:resource="http://purl.obolibrary.org/obo/mondo#orphanet_rare"/>
        <oboInOwl:inSubset rdf:resource="http://purl.obolibrary.org/obo/mondo#otar"/>
        <oboInOwl:inSubset rdf:resource="http://purl.obolibrary.org/obo/mondo#rare"/>
        <rdfs:label>Ellis-van Creveld syndrome</rdfs:label>
        <rdfs:seeAlso rdf:datatype="http://www.w3.org/2001/XMLSchema#anyURI">https://rarediseases.info.nih.gov/diseases/1301/ellis-van-creveld-syndrome</rdfs:seeAlso>
        <skos:closeMatch rdf:resource="http://identifiers.org/meddra/10008724"/>
        <skos:exactMatch rdf:resource="http://identifiers.org/medgen/8584"/>
        <skos:exactMatch rdf:resource="http://identifiers.org/mesh/D004613"/>
        <skos:exactMatch rdf:resource="http://identifiers.org/snomedct/62501005"/>
        <skos:exactMatch rdf:resource="http://linkedlifedata.com/resource/umls/id/C0013903"/>
        <skos:exactMatch rdf:resource="http://purl.obolibrary.org/obo/DOID_12714"/>
        <skos:exactMatch rdf:resource="http://purl.obolibrary.org/obo/NCIT_C84684"/>
        <skos:exactMatch rdf:resource="http://www.orpha.net/ORDO/Orphanet_289"/>
        <skos:exactMatch rdf:resource="https://omim.org/entry/225500"/>
    </owl:Class>
    """,

    """
    <owl:Class rdf:about="http://purl.org/sig/ont/fma/fma83332">
        <rdfs:label xml:lang="en">Subendocardial branch of septal division of left branch of atrioventricular bundle</rdfs:label>
        <rdfs:subClassOf rdf:resource="http://purl.org/sig/ont/fma/fma6266"/>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/regional_part_of"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma13880"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/constitutional_part"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma83393"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/regional_part_of"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma9476"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/constitutional_part"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma83112"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/continuous_with"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma83109"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/continuous_with"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma83331"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <fma:FMAID rdf:datatype="http://www.w3.org/2001/XMLSchema#integer">83332</fma:FMAID>
        <fma:definition rdf:datatype="http://www.w3.org/2001/XMLSchema#string">Subdivision of conducting system of heart which is interpersed with the regular cardiac myocyte on the left stratum of the interventricular septum.</fma:definition>
        <fma:preferred_name rdf:datatype="http://www.w3.org/2001/XMLSchema#string">Subendocardial branch of septal division of left branch of atrioventricular bundle</fma:preferred_name>
    </owl:Class>
    """,
    """
    <owl:Class rdf:about="http://purl.org/sig/ont/fma/fma7101">
        <rdfs:label xml:lang="en">Left ventricle</rdfs:label>
        <owl:equivalentClass>
            <owl:Class>
                <owl:intersectionOf rdf:parseType="Collection">
                    <rdf:Description rdf:about="http://purl.org/sig/ont/fma/fma7100"/>
                    <owl:Restriction>
                        <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/regional_part_of"/>
                        <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma7166"/>
                    </owl:Restriction>
                </owl:intersectionOf>
            </owl:Class>
        </owl:equivalentClass>
        <rdfs:subClassOf rdf:resource="http://purl.org/sig/ont/fma/fma7100"/>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/lymphatic_drainage"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma5834"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/arterial_supply"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma13905"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/constitutional_part"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma235574"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/venous_drainage"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma12846"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/orientation"/>
                <owl:someValuesFrom>
                    <owl:Class>
                        <owl:intersectionOf rdf:parseType="Collection">
                            <owl:Restriction>
                                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/related_object"/>
                                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma86184"/>
                            </owl:Restriction>
                            <owl:Restriction>
                                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/anatomical_coordinate"/>
                                <owl:hasValue rdf:datatype="http://www.w3.org/2001/XMLSchema#string">Anteroinferior</owl:hasValue>
                            </owl:Restriction>
                            <owl:Restriction>
                                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/laterality"/>
                                <owl:hasValue rdf:datatype="http://www.w3.org/2001/XMLSchema#string">Left</owl:hasValue>
                            </owl:Restriction>
                        </owl:intersectionOf>
                    </owl:Class>
                </owl:someValuesFrom>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/regional_part"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma9473"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/arterial_supply"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma3862"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/anteroinferior_to"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma7097"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/arterial_supply"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma3909"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/constitutional_part"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma7235"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/bounded_by"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma13245"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/continuous_with"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma7097"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/direct_right_of"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma7310"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/regional_part"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma9564"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/nerve_supply"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma6649"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/anteroinferior_to"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma3736"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/constitutional_part"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma9466"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/constitutional_part"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma7133"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/constitutional_part"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma9498"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/constitutional_part"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma9556"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/arterial_supply"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma3840"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/direct_right_of"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma7098"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/arterial_supply"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma76994"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/constitutional_part"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma7136"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/constitutional_part"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma86093"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/arterial_supply"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma3924"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/arterial_supply"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma13906"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/arterial_supply"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma3931"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/arterial_supply"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma3895"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/arterial_supply"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma3930"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/arterial_supply"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma3878"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/regional_part_of"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma7166"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/continuous_with"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma3736"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/arterial_supply"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma3902"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/constitutional_part"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma7236"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/nerve_supply"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma6650"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/anterosuperior_to"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma7098"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/venous_drainage"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma4706"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/superior_to"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma13295"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/continuous_with"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma7133"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/direct_left_of"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma3736"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/orientation"/>
                <owl:someValuesFrom>
                    <owl:Class>
                        <owl:intersectionOf rdf:parseType="Collection">
                            <owl:Restriction>
                                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/related_object"/>
                                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma7164"/>
                            </owl:Restriction>
                            <owl:Restriction>
                                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/anatomical_coordinate"/>
                                <owl:hasValue rdf:datatype="http://www.w3.org/2001/XMLSchema#string">Posteroinferior</owl:hasValue>
                            </owl:Restriction>
                            <owl:Restriction>
                                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/laterality"/>
                                <owl:hasValue rdf:datatype="http://www.w3.org/2001/XMLSchema#string">Left</owl:hasValue>
                            </owl:Restriction>
                        </owl:intersectionOf>
                    </owl:Class>
                </owl:someValuesFrom>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/posterosuperior_to"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma7097"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/regional_part_of"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma83580"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/arterial_supply"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma3860"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/direct_right_of"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma7097"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/medial_to"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma7310"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/arterial_supply"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma13914"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <rdfs:subClassOf>
            <owl:Restriction>
                <owl:onProperty rdf:resource="http://purl.org/sig/ont/fma/constitutional_part"/>
                <owl:someValuesFrom rdf:resource="http://purl.org/sig/ont/fma/fma86088"/>
            </owl:Restriction>
        </rdfs:subClassOf>
        <fma:FMAID rdf:datatype="http://www.w3.org/2001/XMLSchema#integer">7101</fma:FMAID>
        <fma:definition rdf:datatype="http://www.w3.org/2001/XMLSchema#string">Left cardiac chamber which is continuous with the aorta.</fma:definition>
        <fma:preferred_name rdf:datatype="http://www.w3.org/2001/XMLSchema#string">Left ventricle</fma:preferred_name>
        <fma:synonym rdf:datatype="http://www.w3.org/2001/XMLSchema#string">Left ventricle of heart</fma:synonym>
        <fma:non-English_equivalent rdf:datatype="http://www.w3.org/2001/XMLSchema#string">Linker Ventrikel</fma:non-English_equivalent>
        <fma:non-English_equivalent rdf:datatype="http://www.w3.org/2001/XMLSchema#string">Ventricule gauche</fma:non-English_equivalent>
        <fma:non-English_equivalent rdf:datatype="http://www.w3.org/2001/XMLSchema#string">Ventriculus cordis sinister</fma:non-English_equivalent>
        <fma:non-English_equivalent rdf:datatype="http://www.w3.org/2001/XMLSchema#string">Ventriculus sinister</fma:non-English_equivalent>
        <fma:non-English_equivalent rdf:datatype="http://www.w3.org/2001/XMLSchema#string">Ventrículo izquierdo</fma:non-English_equivalent>
    </owl:Class>
    """
]

if __name__ == '__main__':
    llama_model = LlamaModelParaphrase('http://localhost:11434/v1', temperature=0.1)
    openai_model = ChatGptModelParaphrase(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-4o', temperature=0.7)
    models = [openai_model, llama_model]

    for model in models:
        print(f'Running on {model.name}:')
        print('------------------------')
        for example in examples:
            output = model.pseudo_to_text(example)
            print(output)
            print('------------------------')
