"""
CSC111 Project 2: Global Trade Interdependence - Graph Builder

This module defines a Graph class for building a trade network graph and a Vertex class
to represent each country with its export and import relationships. The build_trade_graph
function constructs the graph from processed trade data.
"""

import networkx as nx
import pandas as pd


def build_trade_graph(trade_data: pd.DataFrame) -> nx.DiGraph:
    """
    Constructs a directed graph representing the global trade network.

    Args:
        trade_data: A pandas DataFrame containing the trade data with columns:
                    'exporter_id', 'exporter_name', 'importer_id', 'importer_name', 'value'

    Returns:
        A NetworkX DiGraph where:
        - Nodes represent countries with attributes:
            - 'name': The country name
            - 'total_exports': Total value of exports from this country
            - 'total_imports': Total value of imports to this country
        - Edges represent trade relationships with attribute:
            - 'value': The trade value in USD
            - 'weight': Same as value, used for algorithms that rely on the 'weight' attribute
    """
    # Create a new directed graph
    graph = nx.DiGraph()

    # Step 1: Add all countries as nodes
    exporters = trade_data[['exporter_id', 'exporter_name']].drop_duplicates()
    importers = trade_data[['importer_id', 'importer_name']].drop_duplicates()

    # Add exporter countries
    for _, row in exporters.iterrows():
        graph.add_node(
            row['exporter_id'],
            name=row['exporter_name'],
            total_exports=0.0,
            total_imports=0.0
        )

    # Add importer countries (if not already added as exporters)
    for _, row in importers.iterrows():
        if row['importer_id'] not in graph:
            graph.add_node(
                row['importer_id'],
                name=row['importer_name'],
                total_exports=0.0,
                total_imports=0.0
            )

    # Step 2: Add trade relationships as edges
    for _, row in trade_data.iterrows():
        exporter_id = row['exporter_id']
        importer_id = row['importer_id']
        value = float(row['value'])

        # Add the edge with the trade value as weight
        graph.add_edge(exporter_id, importer_id, value=value, weight=value)

        # Update the total exports and imports for the respective countries
        graph.nodes[exporter_id]['total_exports'] += value
        graph.nodes[importer_id]['total_imports'] += value

    return graph


if __name__ == "__main__":
    import doctest
    doctest.testmod()
