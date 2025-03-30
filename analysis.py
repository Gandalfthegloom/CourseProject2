"""
CSC111 Project 2: Global Trade Interdependence - Network Analysis

This module contains functions for analyzing the trade network graph and extracting
meaningful insights about global trade patterns and dependencies.
"""

from typing import Dict, List, Tuple, Any
import networkx as nx
import community as community_louvain
from graph_builder import build_undirected_version

FIXED_SEED = 111  # For any function that uses random algorithm


def analyze_trade_network(graph: nx.DiGraph) -> Dict[str, Any]:
    """Perform comprehensive analysis on the trade network.

    Args:
        graph: The trade network graph

    Returns:
        A dictionary containing various analysis results, including:
        - 'top_exporters': List of (country, value) tuples sorted by export volume
        - 'top_importers': List of (country, value) tuples sorted by import volume
        - 'trade_balance': Dictionary mapping countries to their trade balance
        - 'centrality_measures': Dictionary of various centrality metrics
        - 'strongest_relationships': List of (exporter, importer, value) representing strongest trade ties
        - 'trade_communities': List of country groupings that form trade communities

    Preconditions:
        - For all vertices in graph.nodes(data=True), the vertex must have the following attributes:
          'name', 'total_exports', 'total_imports'
        - For all edges in graph.edges(data=True), the edge must have the following attributes:
          'value', 'weight'
    """
    # Orchestrate the various analysis functions
    # Compile results into a single dictionary
    results = {
        'top_exporters': get_top_exporters(graph),
        'top_importers': get_top_importers(graph),
        'trade_balance': calculate_trade_balance(graph),
        'centrality_measures': calculate_centrality_measures(graph),
        'strongest_relationships': get_strongest_trade_relations(graph),
        'trade_communities': identify_trade_communities(graph)
    }
    return results


def get_top_exporters(graph: nx.DiGraph, n: int = 20) -> List[Tuple[str, float]]:
    """Get the top exporting countries by total export value.

    Args:
        graph: The trade network graph
        n: The number of top exporters to return

    Returns:
        A list of (country_name, export_value) tuples, sorted by export value in descending order

    Preconditions:
        - For all vertices in graph.nodes(data=True), the vertex must have the following attributes:
          'name', 'total_exports', 'total_imports'
        - For all edges in graph.edges(data=True), the edge must have the following attributes:
          'value', 'weight'
        - 1 <= n <= len(graph)
    """
    # Calculate total exports for each country node
    exporters = []

    for node, data in graph.nodes(data=True):
        if 'name' in data and 'total_exports' in data:
            exporters.append((data['name'], data['total_exports']))
        else:
            # Calculate total exports by summing outgoing edge weights
            total_exports = sum(edge_data.get('value', 0) for _, _, edge_data in graph.out_edges(node, data=True))
            country_name = data.get('name', node)
            exporters.append((country_name, total_exports))

    # Sort countries by export value in descending order
    exporters.sort(key=lambda x: x[1], reverse=True)

    # Return the top n countries
    return exporters[:n]


def get_top_importers(graph: nx.DiGraph, n: int = 20) -> List[Tuple[str, float]]:
    """Get the top importing countries by total import value.

    Args:
        graph: The trade network graph
        n: The number of top importers to return

    Returns:
        A list of (country_name, import_value) tuples, sorted by import value in descending order

    Preconditions:
        - For all vertices in graph.nodes(data=True), the vertex must have the following attributes:
          'name', 'total_exports', 'total_imports'
        - For all edges in graph.edges(data=True), the edge must have the following attributes:
          'value', 'weight'
        - 1 <= n <= len(graph)
    """
    # Calculate total imports for each country node
    importers = []

    for node, data in graph.nodes(data=True):
        if 'name' in data and 'total_imports' in data:
            importers.append((data['name'], data['total_imports']))
        else:
            # Calculate total imports by summing incoming edge weights
            total_imports = sum(edge_data.get('value', 0) for _, _, edge_data in graph.in_edges(node, data=True))
            country_name = data.get('name', node)
            importers.append((country_name, total_imports))

    # Sort countries by import value in descending order
    importers.sort(key=lambda x: x[1], reverse=True)

    # Return the top n countries
    return importers[:n]


def calculate_trade_balance(graph: nx.DiGraph) -> Dict[str, float]:
    """Calculate the trade balance (exports - imports) for each country.

    Args:
        graph: The trade network graph

    Returns:
        A dictionary mapping country IDs to their trade balance values

    Preconditions:
        - For all vertices in graph.nodes(data=True), the vertex must have the following attributes:
          'name', 'total_exports', 'total_imports'
        - For all edges in graph.edges(data=True), the edge must have the following attributes:
          'value', 'weight'
    """
    # For each country, calculate total exports and imports
    trade_balance = {}

    for node, data in graph.nodes(data=True):
        # Check if total exports and imports are already stored in node attributes
        if 'total_exports' in data and 'total_imports' in data:
            exports = data['total_exports']
            imports = data['total_imports']
        else:
            # Calculate from edge weights
            exports = sum(edge_data.get('value', 0) for _, _, edge_data in graph.out_edges(node, data=True))
            imports = sum(edge_data.get('value', 0) for _, _, edge_data in graph.in_edges(node, data=True))

        # Compute trade balance as exports - imports
        trade_balance[node] = exports - imports

    return trade_balance


def calculate_centrality_measures(graph: nx.DiGraph) -> dict:
    """Calculate various centrality measures for nodes in the trade network, such as in-degree, out-degree, and
    eigenvector centrality.

    Args:
        graph: A directed graph representing the global trade network

    Returns:
        A dictionary mapping country nodes to dictionaries of their centrality metrics

    Preconditions:
        - For all edges in graph.edges(data=True), the edge must have the following attributes:
          'value', 'weight'
    """
    centrality_measures = {}

    # Degree centrality - measures number of connections
    # For a directed graph, we can look at in-degree and out-degree separately
    in_degree = nx.in_degree_centrality(graph)  # Importance as an importer
    out_degree = nx.out_degree_centrality(graph)  # Importance as an exporter

    # Eigenvector centrality - measures connection to important nodes
    # Countries with high eigenvector centrality trade with other important countries
    try:
        eigenvector = nx.eigenvector_centrality(graph, weight='weight', max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = nx.eigenvector_centrality_numpy(graph, weight='weight')

    # Combine all measures into a single dictionary
    for node in graph.nodes():
        centrality_measures[node] = {
            'in_degree': in_degree.get(node, 0),
            'out_degree': out_degree.get(node, 0),
            'eigenvector': eigenvector.get(node, 0)
        }

    return centrality_measures


def get_strongest_trade_relations(graph: nx.DiGraph, n: int = 20) -> List[Tuple[str, str, float]]:
    """Identify the strongest bilateral trade relationships.

    Args:
        graph: The trade network graph
        n: The number of relationships to return

    Returns:
        A list of (exporter_name, importer_name, trade_value) tuples,
        sorted by trade value in descending order

    Preconditions:
        - For all vertices in graph.nodes(data=True), the vertex must have the following attributes:
          'name', 'total_exports', 'total_imports'
        - For all edges in graph.edges(data=True), the edge must have the following attributes:
          'value', 'weight'
        - 1 <= n <= len(graph)
    """
    # Get all edges from the graph with their trade values
    trade_relationships = []

    for source, target, edge_data in graph.edges(data=True):
        # Extract value from edge data
        trade_value = edge_data.get('value', 0)

        # Get country names
        source_name = graph.nodes[source].get('name', source)
        target_name = graph.nodes[target].get('name', target)

        trade_relationships.append((source_name, target_name, trade_value))

    # Sort edges by trade value in descending order
    trade_relationships.sort(key=lambda x: x[2], reverse=True)

    # Return the top n relationships
    return trade_relationships[:n]


def identify_trade_communities(graph: nx.DiGraph) -> dict:
    """Detect communities in the trade network using the Louvain algorithm.

    Note: Requires converting the directed graph to undirected for community detection.

    Args:
        graph: A directed graph representing the global trade network

    Returns:
        A dictionary mapping node IDs to community IDs

    Preconditions:
        - For all edges in graph.edges(data=True), the edge must have the following attributes:
          'value', 'weight'
    """

    # Convert directed graph to undirected for community detection
    undirected_graph = build_undirected_version(graph)

    # Use the Louvain method for community detection
    partition = community_louvain.best_partition(undirected_graph, weight='weight', random_state=FIXED_SEED)

    # Count the number of communities
    num_communities = len(set(partition.values()))
    print(f"Detected {num_communities} trade communities")

    return partition


def calculate_trade_dependencies(graph: nx.DiGraph, gdp_data: dict = None) -> dict:
    """Calculate trade dependency metrics for each country.

    Args:
        graph: A directed graph representing the global trade network
        gdp_data: Optional dictionary mapping country codes to their GDP values

    Returns:
        A dictionary with trade dependency metrics for each country

    Preconditions:
        - For all edges in graph.edges(data=True), the edge must have the following attributes:
          'value', 'weight'
    """
    dependency_metrics = {}

    for country in graph.nodes():
        # Export dependencies - which countries the current country depends on for exports
        export_edges = list(graph.out_edges(country, data=True))
        total_exports = sum(edge[2].get('weight', edge[2].get('value', 0)) for edge in export_edges)

        # Calculate export concentration (Herfindahl-Hirschman Index)
        if total_exports > 0:
            export_shares = [(edge[2].get('weight', edge[2].get('value', 0)) / total_exports) ** 2
                             for edge in export_edges]
            export_concentration = sum(export_shares)
        else:
            export_concentration = 0

        # Import dependencies - which countries the current country depends on for imports
        import_edges = list(graph.in_edges(country, data=True))
        total_imports = sum(edge[2].get('weight', edge[2].get('value', 0)) for edge in import_edges)

        # Calculate import concentration (Herfindahl-Hirschman Index)
        if total_imports > 0:
            import_shares = [(edge[2].get('weight', edge[2].get('value', 0)) / total_imports) ** 2
                             for edge in import_edges]
            import_concentration = sum(import_shares)
        else:
            import_concentration = 0

        # Calculate trade balance
        trade_balance = total_exports - total_imports

        # Calculate trade to GDP ratio if GDP data is available
        trade_to_gdp = None
        if gdp_data and country in gdp_data and gdp_data[country] > 0:
            trade_to_gdp = (total_exports + total_imports) / gdp_data[country]

        # Calculate trade vulnerability index
        # Higher value = more vulnerable to trade disruptions
        vulnerability_index = (export_concentration + import_concentration) / 2

        # Calculate trade diversity score (inverse of concentration)
        # Higher value = more diversified trade partnerships
        export_diversity = 1 - export_concentration if export_concentration < 1 else 0
        import_diversity = 1 - import_concentration if import_concentration < 1 else 0
        trade_diversity = (export_diversity + import_diversity) / 2

        # Store metrics
        dependency_metrics[country] = {
            'total_exports': total_exports,
            'total_imports': total_imports,
            'export_concentration': export_concentration,  # Higher values indicate higher dependency
            'import_concentration': import_concentration,  # Higher values indicate higher dependency
            'trade_balance': trade_balance,
            'trade_to_gdp': trade_to_gdp,
            'vulnerability_index': vulnerability_index,
            'trade_diversity': trade_diversity
        }

        # Identify top trading partners
        if export_edges:
            top_export_partners = sorted(export_edges,
                                         key=lambda x: x[2].get('weight', x[2].get('value', 0)),
                                         reverse=True)[:5]
            dependency_metrics[country]['top_export_partners'] = [
                (edge[1],
                 edge[2].get('weight', edge[2].get('value', 0)),
                 edge[2].get('weight', edge[2].get('value', 0)) / total_exports if total_exports > 0 else 0)
                for edge in top_export_partners
            ]

        if import_edges:
            top_import_partners = sorted(import_edges,
                                         key=lambda x: x[2].get('weight', x[2].get('value', 0)),
                                         reverse=True)[:5]
            dependency_metrics[country]['top_import_partners'] = [
                (edge[0],
                 edge[2].get('weight', edge[2].get('value', 0)),
                 edge[2].get('weight', edge[2].get('value', 0)) / total_imports if total_imports > 0 else 0)
                for edge in top_import_partners
            ]

    return dependency_metrics


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta

    # Disable R0914 error if you don't want to decrease number of local variables
    python_ta.check_all(config={
        'extra-imports': ['pandas', 'networkx', 'typing', 'community', 'graph_builder'],
        'allowed-io': ['identify_trade_communities'],
        'max-line-length': 120,
        'disable': ['R1705', 'C0200']
    })
