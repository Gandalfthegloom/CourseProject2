"""
CSC111 Project 2: Global Trade Interdependence - Graph Builder

This module defines a Graph class for building a trade network graph and a Vertex class
to represent each country with its export and import relationships. The build_trade_graph
function constructs the graph from processed trade data.
"""

from typing import Any
import pandas as pd
import networkx as nx

DISP_FILTER_ALPHA_SIG = float(1e-12)  # alpha significance for disparity filter


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


def build_undirected_version(graph_original: nx.DiGraph) -> nx.Graph:
    """
    Returns the undirected version of graph_original, which is the trade graph. Here, all edges that connect the
    same pair of vertices will be merged by adding their value and weight.
    """
    # Construct an undirected version of graph_original
    graph_bi = nx.Graph()
    graph_bi.add_nodes_from(graph_original.nodes(data=True))  # Copy the information of all vertices
    for vertex in graph_original.nodes:
        for nbr in graph_original.neighbors(vertex):  # iterate over out-neighbors
            # Check if the reciprocal edge exists
            if graph_original.has_edge(nbr, vertex):
                combined_weight = graph_original[vertex][nbr]['weight'] + graph_original[nbr][vertex]['weight']
            else:
                # If not, use the weight of the single edge
                combined_weight = graph_original[vertex][nbr]['weight']
            graph_bi.add_edge(vertex, nbr, value=combined_weight, weight=combined_weight)

    return graph_bi


def build_sparse_trade_graph(trade_data: pd.DataFrame, alpha_sig: float = DISP_FILTER_ALPHA_SIG) -> nx.DiGraph:
    """
    Constructs a sparse version of the directed graph representing the overview of global trade network.

    Args:
        trade_data: A pandas DataFrame containing the trade data with columns:
                    'exporter_id', 'exporter_name', 'importer_id', 'importer_name', 'value'
        alpha_sig: The alpha significance threshold for the disparity filter.
                   Set the parameter to 0 to disable disparity filter. Default is DISP_FILTER_ALPHA_SIG

    Returns:
        A NetworkX Graph where:
        - Nodes represent countries with attributes:
            - 'name': The country name
            - 'total_exports': Total value of exports from this country
            - 'total_imports': Total value of imports to this country
        - Edges represent major trade relationships with attribute:
            - 'value': The trade value in USD
            - 'weight': Same as value, used for algorithms that rely on the 'weight' attribute

    Preconditions:
        - 0 <= alpha_sig < 1
    """
    # Construct the original trade graph and create the new sparse graph
    graph_original = build_trade_graph(trade_data)
    graph = nx.DiGraph()
    graph.add_nodes_from(graph_original.nodes(data=True))  # Copy the information of all vertices

    # Construct an undirected version of graph_original
    graph_bi = build_undirected_version(graph_original)

    def add_edge_to_graph(node_from: Any, node_to: Any) -> None:
        """
        Add edges from graph_original to graph.

        Preconditions:
            - node_from in graph_original.nodes
            - node_from in graph.nodes
            - node_to in graph_original.nodes
            - node_to in graph.nodes
        """
        if graph_original.has_edge(node_from, node_to):
            graph.add_edge(node_from, node_to, value=graph_original[node_from][node_to]['value'],
                           weight=graph_original[node_from][node_to]['weight'])
        if graph_original.has_edge(node_to, node_from):
            graph.add_edge(node_to, node_from, value=graph_original[node_to][node_from]['value'],
                           weight=graph_original[node_to][node_from]['weight'])

    # Use disparity filter to make the graph sparse
    for vertex, data in graph.nodes(data=True):
        adjacent = list(graph_original[vertex])
        k = len(adjacent)

        # If the degree is 1, just connect it
        if k == 1:
            add_edge_to_graph(vertex, adjacent[0])
            continue

        # Total weight of edges that are connected to the vertex
        # Note that we treat each edge as undirected,
        # and the weight is simply sum of all edges that connects the same pair
        total_weight = data['total_exports'] + data['total_imports']
        for nbr in adjacent:
            # Weight contribution from each edge
            pij = graph_bi[vertex][nbr]['weight'] / total_weight
            p_val = (1 - pij) ** (k - 1)  # p-value from null model

            # If the edge is statistically significant, don't remove it
            if p_val < alpha_sig:
                add_edge_to_graph(vertex, nbr)

    # Use maximum spanning tree to connect all vertices using significant edges (if it's not connected yet)
    if not nx.is_weakly_connected(graph):
        # Invert all the edges such that MST algorithm catches the one with max weight
        for _, _, data in graph_bi.edges(data=True):
            data['weight'] = -data['weight']

        max_st = nx.minimum_spanning_tree(graph_bi)

        # Add the edges from maximum spanning tree to ensure connectivity
        for u, v in max_st.edges:
            add_edge_to_graph(u, v)

    return graph


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    import python_ta

    # Disable R0914 error if you don't want to decrease number of local variables
    python_ta.check_all(config={
        'extra-imports': ['pandas', 'networkx', 'typing'],
        'max-line-length': 120,
        'disable': ['R1705', 'C0200', 'W1114']
    })
