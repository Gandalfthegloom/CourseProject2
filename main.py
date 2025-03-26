"""CSC111 Winter 2025: Global Trade Interdependence Visualization

This module serves as the main entry point for running the Global Trade Interdependence 
visualization program. It integrates components for data processing, graph building, 
analysis, and visualization to create an interactive representation of global trade networks.
"""

import os
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

import data_processing
import graph_builder
import visualization
import analysis


def run_trade_visualization() -> None:
    """Run the complete trade visualization pipeline.
    
    This function orchestrates the following steps:
    1. Load and process the trade data
    2. Build the trade network graph
    3. Perform network analysis
    4. Create and display the interactive visualization
    """
    # Step 1: Load and process the trade data
    print("Loading and processing trade data...")
    trade_data = data_processing.load_trade_data('data/bilateral_value_clean_23_withid.csv')
    country_coords = data_processing.get_country_coordinates()
    
    # Step 2: Build the trade network graph
    print("Building trade network graph...")
    trade_graph = graph_builder.build_trade_graph(trade_data)
    
    # Step 3: Perform network analysis
    print("Performing network analysis...")
    analysis_results = analysis.analyze_trade_network(trade_graph)
    
    # Step 4: Create and display the interactive visualization
    print("Creating visualization...")
    viz = visualization.create_trade_visualization(trade_graph, country_coords, analysis_results)
    viz.show()
    
    print("Visualization complete!")


def run_sample_analysis() -> None:
    """Run a sample analysis on the trade data without full visualization.
    
    This function demonstrates the core analysis capabilities:
    1. Load and process the trade data
    2. Build the trade network graph
    3. Calculate and display key trade metrics and statistics
    """
    # Step 1: Load and process the trade data
    trade_data = data_processing.load_trade_data('data/bilateral_value_clean_23_withid.csv')
    
    # Step 2: Build the trade network graph
    trade_graph = graph_builder.build_trade_graph(trade_data)
    
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
    # Uncomment the function you want to run
    run_trade_visualization()
    # run_sample_analysis()