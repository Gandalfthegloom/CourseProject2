"""CSC111 Project 2: Global Trade Interdependence - Graph Builder

This module contains functions for creating and manipulating the trade network graph.
It transforms processed trade data into a directed graph where nodes represent countries
and edges represent trade relationships.
"""

from typing import Dict, List, Tuple, Any
import pandas as pd
import networkx as nx


def build_trade_graph(trade_data: pd.DataFrame) -> nx.DiGraph:
    """Construct a directed graph from the trade data.
    
    Args:
        trade_data: A pandas DataFrame containing the processed trade data
        
    Returns:
        A NetworkX DiGraph where:
        - Nodes represent countries (with attributes like name, total_exports, total_imports)
        - Edges represent trade relationships (with attributes like trade_value, normalized_value)
        
    Preconditions:
        - trade_data has columns: 'exporter_id', 'exporter_name', 'importer_id', 'importer_name', 'value'
    """
    # Create a new directed graph
    # Add nodes for each unique country
    # Add edges for each trade relationship
    # Calculate and add node attributes (total exports, imports, etc.)
    # Return the constructed graph
    pass


def normalize_edge_weights(graph: nx.DiGraph, attribute: str = 'value', new_attribute: str = 'normalized_value') -> nx.DiGraph:
    """Normalize the edge weights in the graph for better visualization.
    
    Args:
        graph: The trade network graph
        attribute: The edge attribute to normalize
        new_attribute: The name of the new normalized attribute
        
    Returns:
        The modified graph with normalized edge weights
    """
    # Find the maximum and minimum values for the specified attribute
    # Add a new normalized attribute to each edge
    # Return the modified graph
    pass


def filter_graph_by_threshold(graph: nx.DiGraph, threshold: float, attribute: str = 'value') -> nx.DiGraph:
    """Create a subgraph containing only edges with weights above a certain threshold.
    
    Args:
        graph: The trade network graph
        threshold: The minimum value for an edge to be included
        attribute: The edge attribute to compare against the threshold
        
    Returns:
        A subgraph containing only the edges that meet the threshold criterion
    """
    # Create a list of edges to keep
    # Create a new graph with the same nodes but only the filtered edges
    pass


def get_country_subgraph(graph: nx.DiGraph, country_id: str, include_imports: bool = True, include_exports: bool = True) -> nx.DiGraph:
    """Extract a subgraph centered on a specific country.
    
    Args:
        graph: The trade network graph
        country_id: The ID of the country to focus on
        include_imports: Whether to include edges representing imports to the country
        include_exports: Whether to include edges representing exports from the country
        
    Returns:
        A subgraph containing only the specified country and its direct trade partners
    """
    # Create a new graph
    # Add the specified country node
    # Add relevant trade partner nodes and edges based on the parameters
    pass


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    
    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['pandas', 'networkx', 'typing'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })