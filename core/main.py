from typing import Callable

import neo4j
from neo4j import GraphDatabase
from neo4j import exceptions
from rdflib import Graph
from rdflib_neo4j import Neo4jStore, Neo4jStoreConfig, HANDLE_VOCAB_URI_STRATEGY

neo4j_username = 'neo4j'
neo4j_password = 'your_password'
neo4j_host = 'localhost'
neo4j_port = 7687

files = {
    'ado': '../data/ado.owl',
    'envo': '../data/envo.owl'
}


def create_graph_database(name: str):
    uri = f"bolt://{neo4j_host}:{neo4j_port}"

    def create_database(tx, database_name):
        tx.run("CREATE DATABASE $name", name=database_name)

    try:
        with GraphDatabase.driver(uri, auth=(neo4j_username, neo4j_password)) as driver:
            with driver.session() as session:
                session.write_transaction(create_database, name)
    except neo4j.exceptions.ClientError as e:
        if not e.code == 'Neo.ClientError.Database.ExistingDatabaseFound':
            raise e


def import_data_from_file(resolver: Callable, db_name: str):
    # Define the Neo4j store configuration
    auth_data = {
        'user': neo4j_username,
        'pwd': neo4j_password, 'database': db_name,
        'uri': f'neo4j://{neo4j_host}:{neo4j_port}/'
    }
    config = Neo4jStoreConfig(auth_data=auth_data, handle_vocab_uri_strategy=HANDLE_VOCAB_URI_STRATEGY.IGNORE)

    # Create a graph and open the Neo4j store
    g = Graph(store=Neo4jStore(config=config))

    for s, p, o in resolver():
        try:
            g.add((s, p, o))
        except:
            print('error inserting entry')


if __name__ == '__main__':
    for k, v in files.items():
        create_graph_database(k)

        g = Graph()
        g.parse(v)

        import_data_from_file(resolver=lambda: g.query('select ?s ?p ?o where { ?s ?p ?o }'), db_name=k)
