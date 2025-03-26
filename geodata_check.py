import pandas as pd
import networkx as nx
import plotly.express as px
from graph_builder import build_trade_graph
from data_processing import load_trade_data, get_country_coordinates


def find_missing_geographic_data(trade_graph, country_coordinates):
    """
    Identifies countries present in the trade graph but missing from the geographic data.

    Args:
        trade_graph: A NetworkX DiGraph representing the trade network
        country_coordinates: DataFrame containing country names and their coordinates

    Returns:
        A list of tuples containing (country_id, country_name) for countries missing geographic data
    """
    missing_countries = []

    # Create a set of country names from the geographic data for faster lookup
    available_countries = set(country_coordinates['country'].str.lower())

    # Check each node in the graph
    for country_id, data in trade_graph.nodes(data=True):
        country_name = data.get('name', '')

        # Check if the country name exists in the geographic data (case-insensitive comparison)
        if country_name.lower() not in available_countries:
            missing_countries.append((country_id, country_name))

    return missing_countries
