# OWL Ontology Verbalizer

## Installation

```shell
pip install ontology-verbalizer
```

## Usage

### Configure URIs to ignore or rename

```python
ignore = {
    'http://www.w3.org/2002/07/owl#onDatatype',
    'http://www.w3.org/2000/01/rdf-schema#seeAlso',
    'http://www.w3.org/2000/01/rdf-schema#label',
    'http://www.w3.org/2000/01/rdf-schema#comment',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
    'http://www.w3.org/2000/01/rdf-schema#isDefinedBy',
}

rephrased = {
    'http://www.w3.org/2002/07/owl#equivalentClass': 'is same as',
    'http://www.w3.org/2000/01/rdf-schema#subClassOf': 'is a type of',
    'http://www.w3.org/2002/07/owl#intersectionOf': 'all of',
    'http://www.w3.org/2002/07/owl#unionOf': 'any of',
    'http://www.w3.org/2002/07/owl#disjointWith': 'is different from',
    'http://www.w3.org/2002/07/owl#withRestrictions': 'must be'
}
```

### Configure LLM of choice

```python
from verbalizer.nlp import ChatGptModelParaphrase

model = ChatGptModelParaphrase(api_key='sk-xyz', model='gpt-4o', temperature=0.7)
```

### Perform Verbalization

```python
from verbalizer.process import Processor

ontology = 'pizza.ttl'
name = 'pizza'
processor = Processor(llm=model, vocab_ignore=ignore, vocab_rephrased=rephrased, min_statements=1)
processor.process(name, ontology, output_dir='/path/to/my/output')
```

<table border="1">
    <tr>
        <th>OWL Fragment</th>
        <th>Pseudo Text</th>
        <th>LLM Text</th>
    </tr>
    <tr>
        <td>
            <pre>
@prefix owl: &lt;http://www.w3.org/2002/07/owl#&gt; .
@prefix rdfs: &lt;http://www.w3.org/2000/01/rdf-schema#&gt; .

:deep_pan_base a owl:Class ;
  rdfs:subClassOf :pizza_base ;
  owl:disjointWith :thin_and_crispy_base .
            </pre>
        </td>
        <td>deep pan base is a type of a pizza base.<br>deep pan base is different from a thin and crispy base.</td>
        <td>A deep pan base is a type of pizza base that differs from a thin and crispy base.</td>
    </tr>
    <tr>
        <td>
            <pre>
@prefix owl: &lt;http://www.w3.org/2002/07/owl#&gt; .
@prefix rdf: &lt;http://www.w3.org/1999/02/22-rdf-syntax-ns#&gt; .
@prefix rdfs: &lt;http://www.w3.org/2000/01/rdf-schema#&gt; .

:capricciosa a owl:Class ;
  rdfs:subClassOf [ a owl:Restriction ;
  owl:onProperty :has_topping ;
  owl:someValuesFrom :peperonata_topping ],
  [ a owl:Restriction ;
  owl:onProperty :has_topping ;
  owl:someValuesFrom :olive_topping ],
  [ a owl:Restriction ;
  owl:onProperty :has_topping ;
  owl:someValuesFrom :caper_topping ],
  [ a owl:Restriction ;
  owl:onProperty :has_topping ;
  owl:someValuesFrom :mozzarella_topping ],
  [ a owl:Restriction ;
  owl:allValuesFrom [ a owl:Class ;
  owl:unionOf [ rdf:first :tomato_topping ;
  rdf:rest [ rdf:first :ham_topping ;
  rdf:rest [ rdf:first :mozzarella_topping ;
  rdf:rest [ rdf:first :anchovies_topping ;
  rdf:rest [ rdf:first :olive_topping ;
  rdf:rest [ rdf:first :peperonata_topping ;
  rdf:rest [ rdf:first :caper_topping ;
  rdf:rest :nil ] ] ] ] ] ] ] ] ;
  owl:onProperty :has_topping ],
  [ a owl:Restriction ;
  owl:onProperty :has_topping ;
  owl:someValuesFrom :anchovies_topping ],
  [ a owl:Restriction ;
  owl:onProperty :has_topping ;
  owl:someValuesFrom :ham_topping ],
  [ a owl:Restriction ;
  owl:onProperty :has_topping ;
  owl:someValuesFrom :tomato_topping ],
  :named_pizza ;
  owl:disjointWith :american,
  :american_hot,
  :cajun,
  :caprina,
  :fiorentina,
  :four_seasons,
  :frutti_di_mare,
  :giardiniera,
  :la_reine,
  :margherita,
  :mushroom,
  :napoletana,
  :parmense,
  :pollo_ad_astra,
  :prince_carlo,
  :quattro_formaggi,
  :rosa,
  :siciliana,
  :sloppy_giuseppe,
  :soho,
  :unclosed_pizza,
  :veneziana .
            </pre>
        </td>
        <td>
            capricciosa is a type of a named pizza.<br>
            capricciosa is a type of at least has topping some a caper topping.<br>
            capricciosa is a type of at least has topping some a ham topping.<br>
            capricciosa is a type of at least has topping some a mozzarella topping.<br>
            capricciosa is a type of at least has topping some a peperonata topping.<br>
            capricciosa is a type of at least has topping some a tomato topping.<br>
            capricciosa is a type of at least has topping some an anchovies topping.<br>
            capricciosa is a type of at least has topping some an olive topping.<br>
            capricciosa is a type of only has topping any of (a tomato topping, and a ham topping, and a mozzarella topping, and an anchovies topping, and an olive topping, and a peperonata topping, and a caper topping).<br>
            capricciosa is different from (a la reine, and a frutti di mare, and a fiorentina, and a margherita, and a sloppy giuseppe, and a veneziana, and an american hot, and an unclosed pizza, and a mushroom, and a pollo ad astra, and a caprina, and a cajun, and a giardiniera, and an american, and a prince carlo, and a siciliana, and a parmense, and a rosa, and a soho, and a quattro formaggi, and a four seasons, and a napoletana).
        </td>
        <td>
            Capricciosa is a specific type of pizza that includes various toppings. It features caper, ham, mozzarella, peperonata, tomato, anchovies, and olive toppings. This pizza exclusively contains these toppings and does not have any other toppings. Capricciosa is distinct from other pizzas such as la reine, frutti di mare, fiorentina, margherita, sloppy giuseppe, veneziana, american hot, unclosed pizza, mushroom, pollo ad astra, caprina, cajun, giardiniera, american, prince carlo, siciliana, parmense, rosa, soho, quattro formaggi, four seasons, and napoletana.
        </td>
    </tr>
    <tr>
        <td>
            <pre>
@prefix owl: &lt;http://www.w3.org/2002/07/owl#&gt; .
@prefix rdfs: &lt;http://www.w3.org/2000/01/rdf-schema#&gt; .

:chicken_topping a owl:Class ;
  rdfs:subClassOf [ a owl:Restriction ;
  owl:onProperty :has_spiciness ;
  owl:someValuesFrom :mild ],
  :meat_topping ;
  owl:disjointWith :ham_topping,
  :hot_spiced_beef_topping,
  :peperoni_sausage_topping .
            </pre>
        </td>
        <td>
            chicken topping is a type of a meat topping.<br>
            chicken topping is a type of at least has spiciness some a mild.<br>
            chicken topping is different from (a peperoni sausage topping, and a hot spiced beef topping, and a ham topping).
        </td>
        <td>
            Chicken topping is a type of meat topping that has at least some mild spiciness. It is different from pepperoni sausage topping, hot spiced beef topping, and ham topping.
        </td>
    </tr>
</table>