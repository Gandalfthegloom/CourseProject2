"""
CSC111 Winter 2025: Global Trade Interdependence Visualization

This module serves as the main entry point for running the Global Trade Interdependence
visualization program. It integrates components for data processing, graph building,
analysis, and visualization to create an interactive representation of global trade networks.
"""

# The four libraries below is added just to be safe (although it may look like it has no use)
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import dash

import data_processing
from graph_builder import build_sparse_trade_graph, build_trade_graph
import visualization
import analysis

TRADE_DATA = 'Data/bilateral_value_clean_23_withid.csv'  # Default data to be used
GDP_DATA = 'Data/world_bank_gdp/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_26433.csv'


def run_trade_dashboard(filename: str = TRADE_DATA, gdp: str = GDP_DATA) -> None:
    """Run the complete trade visualization dashboard.

    This function orchestrates the following steps:
    1. Load and process the trade data
    2. Build the trade network graph
    3. Perform network analysis
    4. Launch the integrated dashboard with multiple visualization options

    Preconditions:
        - filename is a valid CSV file
    """
    # Step 1: Load and process the trade data
    print("Loading and processing trade data...")
    trade_data = data_processing.load_trade_data(filename)
    country_coords = data_processing.get_country_coordinates()
    print(f"Loaded trade data for {len(trade_data)} trade relationships")
    print(f"Loaded coordinates for {len(country_coords)} countries")

    # Load GDP data for openness index calculation
    gdp_data = data_processing.load_gdp_data(gdp)
    print(f"Loaded GDP data for {len(gdp_data)} countries")

    # Step 2: Build the trade network graph
    print("Building trade network graph...")
    trade_graph = build_trade_graph(trade_data)
    filtered_trade_graph = build_sparse_trade_graph(trade_data)
    print(f"Created graph with {trade_graph.number_of_nodes()} nodes and {trade_graph.number_of_edges()} edges")

    # Step 3: Perform network analysis
    print("Performing network analysis...")
    analysis_results = analysis.analyze_trade_network(trade_graph)
    print("Analysis complete")

    # Step 4: Launch the integrated dashboard
    print("Launching visualization dashboard...")
    visualization.create_dashboard(filtered_trade_graph, trade_graph, country_coords, analysis_results, gdp_data)


def run_simple_visualization(filename: str = TRADE_DATA) -> None:
    """Run a simple, non-interactive visualization.

    This function follows the same steps as run_trade_dashboard but creates
    a static visualization that can be saved or displayed without a web server.

    Preconditions:
        - filename is a valid CSV file
    """
    # Step 1: Load and process the trade data
    trade_data = data_processing.load_trade_data(filename)
    country_coords = data_processing.get_country_coordinates()

    # Step 2: Build the trade network graph
    trade_graph = build_trade_graph(trade_data)
    filtered_trade_graph = build_sparse_trade_graph(trade_data)

    # Step 3: Perform network analysis
    analysis_results = analysis.analyze_trade_network(trade_graph)

    # Step 4: Create a static visualization
    print("Creating visualization...")
    viz = visualization.create_community_visualization(filtered_trade_graph, country_coords, analysis_results)
    viz.show()  # Display the visualization

    print("Visualization complete! You can save this figure using the export button in the top-right corner.")


def run_sample_analysis(filename: str = TRADE_DATA) -> None:
    """Run a sample analysis on the trade data without visualization.

    This function demonstrates the core analysis capabilities:
    1. Load and process the trade data
    2. Build the trade network graph
    3. Calculate and display key trade metrics and statistics

    Preconditions:
        - filename is a valid CSV file
    """
    # Step 1: Load and process the trade data
    trade_data = data_processing.load_trade_data(filename)

    # Step 2: Build the trade network graph
    trade_graph = build_trade_graph(trade_data)

    # Step 3: Calculate and display key metrics
    results = analysis.analyze_trade_network(trade_graph)

    # Print top exporters, importers, and other relevant statistics
    print("=== Top 10 Export Countries by Volume ===")
    for country, value in results['top_exporters'][:10]:
        print(f"{country}: ${value:,.2f}")

    print("\n=== Top 10 Import Countries by Volume ===")
    for country, value in results['top_importers'][:10]:
        print(f"{country}: ${value:,.2f}")

    print("\n=== Most Interdependent Country Pairs ===")
    for country1, country2, value in results['strongest_relationships'][:10]:
        print(f"{country1} - {country2}: ${value:,.2f}")


if __name__ == "__main__":
    # Run the integrated dashboard by default
    # run_trade_dashboard()

    # Run the integrated dashboard with sample data
    # run_trade_dashboard('data/sample.csv')

    # Alternative running modes (uncomment to use):
    run_simple_visualization()  # For a simpler, non-interactive visualization
    # run_sample_analysis()       # For analysis without visualization
