"""
CSC111 Project 2: Global Trade Interdependence - Graph Builder

This module defines a Graph class for building a trade network graph and a Vertex class
to represent each country with its export and import relationships. The build_trade_graph
function constructs the graph from processed trade data.
"""

import pandas as pd
import networkx as nx


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


def build_sparse_trade_graph(trade_data: pd.DataFrame, use_disparity_filter: bool = True,
                             alpha: float = 0.05) -> nx.DiGraph:
    """
    Constructs a sparse version of the directed graph representing the overview of global trade network.

    Args:
        trade_data: A pandas DataFrame containing the trade data with columns:
                    'exporter_id', 'exporter_name', 'importer_id', 'importer_name', 'value'
        use_disparity_filter: An option to either use disparity filter or not. Default is True
        alpha: The p-value for the disparity filter. Default is 0.05

    Returns:
        A NetworkX Graph where:
        - Nodes represent countries with attributes:
            - 'name': The country name
            - 'total_exports': Total value of exports from this country
            - 'total_imports': Total value of imports to this country
        - Edges represent major trade relationships with attribute:
            - 'value': The trade value in USD
            - 'weight': Same as value, used for algorithms that rely on the 'weight' attribute
    """
    # Construct the original trade graph and create the new sparse graph
    graph_original = build_trade_graph(trade_data)
    graph = nx.DiGraph()
    graph.add_nodes_from(graph_original.nodes(data=True))  # Copy the information of all vertices

    # Construct an undirected version of graph_original
    graph_bi = nx.Graph()
    graph_bi.add_nodes_from(graph_original.nodes(data=True))  # Copy the information of all vertices
    for vertex in graph_original.nodes:
        for nbr in graph_original.neighbors(vertex):  # iterate over out-neighbors
            # Check if the reciprocal edge exists
            if graph_original.has_edge(nbr, vertex):
                combined_weight = (graph_original[vertex][nbr]['weight'] +
                                   graph_original[nbr][vertex]['weight'])
            else:
                # If not, use the weight of the single edge
                combined_weight = graph_original[vertex][nbr]['weight']
            graph_bi.add_edge(vertex, nbr, weight=combined_weight)

    if use_disparity_filter:
        # Use disparity filter to make the graph sparse
        for vertex in graph.nodes:
            adjacent = list(graph_original[vertex])
            k = len(adjacent)

            # If the degree is 1, just connect it
            if k == 1:
                graph.add_edge(vertex, adjacent[0], value=graph_original[vertex][adjacent[0]]['value'],
                               weight=graph_original[vertex][adjacent[0]]['weight'])
                if graph_original.has_edge(adjacent[0], vertex):
                    graph.add_edge(adjacent[0], vertex, value=graph_original[adjacent[0]][vertex]['value'],
                                   weight=graph_original[adjacent[0]][vertex]['weight'])
                continue

            # Total weight of edges that are connected to the vertex
            # Note that we treat each edge as undirected, and the weight is simply sum of all edges that
            # connects the same pair
            total_weight = graph.nodes[vertex]['total_exports'] + graph.nodes[vertex]['total_imports']
            for nbr in adjacent:
                # Weight contribution from each edge
                pij = graph_bi[vertex][nbr]['weight'] / total_weight
                significance = (1 - pij) ** (k - 1)  # p-value from null model

                # If the edge is statistically significant, don't remove it
                if significance < alpha:
                    graph.add_edge(vertex, nbr, value=graph_original[vertex][nbr]['value'],
                                   weight=graph_original[vertex][nbr]['weight'])
                    if graph_original.has_edge(nbr, vertex):
                        graph.add_edge(nbr, vertex, value=graph_original[nbr][vertex]['value'],
                                       weight=graph_original[nbr][vertex]['weight'])

    # Use maximum spanning tree to connect all vertices using significant edges (if it's not connected yet)
    if not nx.is_weakly_connected(graph):
        # Invert all the edges such that MST algorithm catches the one with max weight
        for u, v, data in graph_bi.edges(data=True):
            data['weight'] = -data['weight']

        max_st = nx.minimum_spanning_tree(graph_bi)

        # Add the edges from maximum spanning tree to ensure connectivity
        for u, v, data in max_st.edges(data=True):
            if graph_original.has_edge(u, v):
                graph.add_edge(u, v, value=graph_original[u][v]['value'],
                               weight=graph_original[u][v]['weight'])
            if graph_original.has_edge(v, u):
                graph.add_edge(v, u, value=graph_original[v][u]['value'],
                               weight=graph_original[v][u]['weight'])

    return graph


if __name__ == "__main__":
    import doctest
    doctest.testmod()
