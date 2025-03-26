"""CSC111 Project 2: Global Trade Interdependence - Visualization

This module contains functions for creating interactive visualizations of the trade network.
It uses Plotly to generate an interactive world map with trade flows and country statistics.
"""

from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_trade_visualization(
        graph: nx.DiGraph,
        country_coords: pd.DataFrame,
        analysis_results: Dict[str, Any]
) -> go.Figure:
    """Create an interactive visualization of the global trade network.

    Args:
        graph: The trade network graph
        country_coords: A DataFrame containing countries and their coordinates
        analysis_results: The dictionary of analysis results from the analysis module

    Returns:
        A Plotly Figure object containing the interactive visualization
    """
    # Create a base world map
    fig = go.Figure()

    # Create a mapping of country names to coordinates
    country_coord_map = dict(zip(country_coords['country'],
                                 zip(country_coords['centroid_lat'], country_coords['centroid_lon'])))

    # Add nodes (countries) as scatter markers
    node_sizes = []
    node_colors = []
    node_texts = []
    node_lats = []
    node_lons = []

    for country_id, data in graph.nodes(data=True):
        country_name = data['name']
        if country_name in country_coord_map:
            lat, lon = country_coord_map[country_name]

            # Calculate node size based on total trade volume
            total_trade = data['total_exports'] + data['total_imports']
            node_size = np.log1p(total_trade) * 2  # Log scale to handle wide range of values

            # Calculate node color based on trade balance
            trade_balance = data['total_exports'] - data['total_imports']
            # Normalize between -1 and 1 for coloring
            if total_trade > 0:
                balance_ratio = trade_balance / total_trade
            else:
                balance_ratio = 0

            # Prepare hover text
            hover_text = (
                f"<b>{country_name}</b><br>"
                f"Total Exports: ${data['total_exports']:,.2f}<br>"
                f"Total Imports: ${data['total_imports']:,.2f}<br>"
                f"Trade Balance: ${trade_balance:,.2f}<br>"
            )

            node_sizes.append(node_size)
            node_colors.append(balance_ratio)
            node_texts.append(hover_text)
            node_lats.append(lat)
            node_lons.append(lon)

    # Add country nodes to the figure
    fig.add_trace(go.Scattergeo(
        lon=node_lons,
        lat=node_lats,
        text=node_texts,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale='RdBu',
            cmin=-1,
            cmax=1,
            colorbar=dict(
                title="Trade Balance Ratio<br>(Exports-Imports)/(Exports+Imports)"
            ),
            line=dict(width=1, color='black')
        ),
        hoverinfo='text',
        name='Countries'
    ))

    # Add edges (trade flows) for the top trading relationships
    # To avoid cluttering, we'll only show the top relationships
    top_edges = []
    for source, target, data in graph.edges(data=True):
        source_name = graph.nodes[source]['name']
        target_name = graph.nodes[target]['name']

        if source_name in country_coord_map and target_name in country_coord_map:
            top_edges.append((source_name, target_name, data['value']))

    # Sort edges by value and take top 100
    top_edges.sort(key=lambda x: x[2], reverse=True)
    top_edges = top_edges[:100]

    # Add arrows for each top trade relationship
    for source_name, target_name, value in top_edges:
        if source_name in country_coord_map and target_name in country_coord_map:
            start_lat, start_lon = country_coord_map[source_name]
            end_lat, end_lon = country_coord_map[target_name]

            # Calculate edge weight (line thickness) based on value
            weight = np.log1p(value) * 0.1

            # Add the trade flow arrow
            plot_trade_arrow(
                fig,
                start_lat, start_lon,
                end_lat, end_lon,
                weight,
                'rgba(255, 50, 50, 0.6)',  # Semi-transparent red for export flows
                f"<b>Trade Flow</b><br>{source_name} → {target_name}<br>Value: ${value:,.2f}"
            )

    # Configure the layout
    fig.update_layout(
        title='Global Trade Network Visualization',
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            showcountries=True,
            countrycolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(230, 250, 255)',
            showlakes=True,
            lakecolor='rgb(220, 240, 255)'
        ),
        height=800,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig


def visualize_country_trade(
        graph: nx.DiGraph,
        country_id: str,
        country_coords: pd.DataFrame,
        analysis_results: Dict[str, Any]
) -> go.Figure:
    """Create a visualization focused on a specific country's trade relationships.

    Args:
        graph: The trade network graph
        country_id: The ID of the country to focus on
        country_coords: A DataFrame containing countries and their coordinates
        analysis_results: The dictionary of analysis results from the analysis module

    Returns:
        A Plotly Figure object showing the selected country's trade relationships
    """
    # Create a subgraph containing only the selected country and its partners
    fig = go.Figure()

    country_name = graph.nodes[country_id]['name']

    # Create a mapping of country names to coordinates
    country_coord_map = dict(zip(country_coords['country'],
                                 zip(country_coords['centroid_lat'], country_coords['centroid_lon'])))

    # Check if the country exists in our coordinate dataset
    if country_name not in country_coord_map:
        print(f"Warning: Coordinates for {country_name} not found")
        return fig

    # Get the selected country's coordinates
    selected_lat, selected_lon = country_coord_map[country_name]

    # Add the selected country as a highlighted node
    fig.add_trace(go.Scattergeo(
        lon=[selected_lon],
        lat=[selected_lat],
        text=[f"<b>{country_name}</b>"],
        mode='markers',
        marker=dict(
            size=20,
            color='yellow',
            line=dict(width=2, color='black')
        ),
        hoverinfo='text',
        name=country_name
    ))

    # Add export partners (outgoing edges)
    export_lats = []
    export_lons = []
    export_texts = []
    export_values = []

    for _, target in graph.out_edges(country_id):
        target_name = graph.nodes[target]['name']
        if target_name in country_coord_map:
            target_lat, target_lon = country_coord_map[target_name]
            value = graph.edges[country_id, target]['value']

            export_lats.append(target_lat)
            export_lons.append(target_lon)
            export_texts.append(f"<b>Export to {target_name}</b><br>Value: ${value:,.2f}")
            export_values.append(value)

            # Add arrow for this trade relationship
            plot_trade_arrow(
                fig,
                selected_lat, selected_lon,
                target_lat, target_lon,
                np.log1p(value) * 0.5,
                'rgba(255, 50, 50, 0.6)',  # Red for exports
                f"<b>Export</b><br>{country_name} → {target_name}<br>Value: ${value:,.2f}"
            )

    # Add import partners (incoming edges)
    import_lats = []
    import_lons = []
    import_texts = []
    import_values = []

    for source, _ in graph.in_edges(country_id):
        source_name = graph.nodes[source]['name']
        if source_name in country_coord_map:
            source_lat, source_lon = country_coord_map[source_name]
            value = graph.edges[source, country_id]['value']

            import_lats.append(source_lat)
            import_lons.append(source_lon)
            import_texts.append(f"<b>Import from {source_name}</b><br>Value: ${value:,.2f}")
            import_values.append(value)

            # Add arrow for this trade relationship
            plot_trade_arrow(
                fig,
                source_lat, source_lon,
                selected_lat, selected_lon,
                np.log1p(value) * 0.5,
                'rgba(50, 50, 255, 0.6)',  # Blue for imports
                f"<b>Import</b><br>{source_name} → {country_name}<br>Value: ${value:,.2f}"
            )

    # Add export trading partners
    if export_lats:
        fig.add_trace(go.Scattergeo(
            lon=export_lons,
            lat=export_lats,
            text=export_texts,
            mode='markers',
            marker=dict(
                size=np.log1p(export_values) * 2,
                color='red',
                line=dict(width=1, color='black')
            ),
            hoverinfo='text',
            name='Export Partners'
        ))

    # Add import trading partners
    if import_lats:
        fig.add_trace(go.Scattergeo(
            lon=import_lons,
            lat=import_lats,
            text=import_texts,
            mode='markers',
            marker=dict(
                size=np.log1p(import_values) * 2,
                color='blue',
                line=dict(width=1, color='black')
            ),
            hoverinfo='text',
            name='Import Partners'
        ))

    # Configure the layout
    fig.update_layout(
        title=f'Trade Relationships for {country_name}',
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            showcountries=True,
            countrycolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(230, 250, 255)',
            center=dict(lon=selected_lon, lat=selected_lat),
            projection_scale=2  # Zoom in on the selected country
        ),
        height=800,
        margin=dict(l=0, r=0, t=50, b=0),
        legend_title_text='Partner Type'
    )

    return fig


def create_country_selector(
        graph: nx.DiGraph,
        country_coords: pd.DataFrame,
        analysis_results: Dict[str, Any]
) -> None:
    """Create an interactive dashboard with country selection dropdown.

    This function creates and runs a Dash application that allows users to select
    a country and view its trade relationships.

    Args:
        graph: The trade network graph
        country_coords: A DataFrame containing countries and their coordinates
        analysis_results: The dictionary of analysis results from the analysis module
    """
    try:
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output
    except ImportError:
        print("Dash is required for the interactive dashboard. Install with: pip install dash")
        return

    # Create a list of countries for the dropdown
    countries = [(node_id, data['name']) for node_id, data in graph.nodes(data=True)]
    countries.sort(key=lambda x: x[1])  # Sort by country name

    # Create a Dash application
    app = dash.Dash(__name__, title="Global Trade Network Explorer")

    app.layout = html.Div([
        html.H1("Global Trade Network Explorer"),
        html.Div([
            html.Label("Select a Country:"),
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': name, 'value': id_} for id_, name in countries],
                value=countries[0][0]  # Default to first country
            )
        ]),
        dcc.Graph(id='trade-graph')
    ])

    @app.callback(
        Output('trade-graph', 'figure'),
        [Input('country-dropdown', 'value')]
    )
    def update_graph(selected_country):
        # Create a visualization for the selected country
        return visualize_country_trade(graph, selected_country, country_coords, analysis_results)

    print("Starting dashboard. Navigate to http://127.0.0.1:8050/ to view the application.")
    app.run_server(debug=True)


def plot_trade_arrow(
        fig: go.Figure,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        weight: float,
        color: str,
        hover_text: str
) -> None:
    """Add a directional arrow representing a trade relationship to the map.

    Args:
        fig: The Plotly Figure to add the arrow to
        start_lat, start_lon: Coordinates of the exporting country
        end_lat, end_lon: Coordinates of the importing country
        weight: The trade value, used to determine arrow thickness
        color: The color of the arrow
        hover_text: Text to display when hovering over the arrow
    """
    # Use a great circle path for the trade flow
    # This creates a curved line that follows the earth's curvature

    # To avoid arrows crossing through the earth, we'll create a path with intermediate points
    n_points = 100
    lat_path = np.linspace(start_lat, end_lat, n_points)
    lon_path = np.linspace(start_lon, end_lon, n_points)

    # Add slight curve to make the path visually appealing
    # (This is a simple approximation, not a true great circle)
    mid_index = n_points // 2
    curve_factor = 0.2  # How much to curve the path

    # Calculate angle perpendicular to the direct path
    angle = np.arctan2(end_lat - start_lat, end_lon - start_lon) + np.pi / 2

    # Apply the curve
    for i in range(1, n_points - 1):
        # Scale factor peaks at the middle of the path
        scale = curve_factor * np.sin(np.pi * i / n_points)
        lat_path[i] += scale * np.sin(angle)
        lon_path[i] += scale * np.cos(angle)

    # Add the arrow path to the figure
    fig.add_trace(go.Scattergeo(
        lon=lon_path,
        lat=lat_path,
        mode='lines',
        line=dict(
            width=weight,
            color=color
        ),
        hoverinfo='text',
        text=hover_text,
        showlegend=False
    ))

    # Add an arrowhead at the end of the path
    # We'll use the second-to-last point to determine the direction
    arrow_lat = lat_path[-2:].tolist()
    arrow_lon = lon_path[-2:].tolist()

    fig.add_trace(go.Scattergeo(
        lon=arrow_lon,
        lat=arrow_lat,
        mode='lines',
        line=dict(
            width=weight * 1.5,  # Make the arrowhead slightly thicker
            color=color
        ),
        hoverinfo='text',
        text=hover_text,
        showlegend=False
    ))


def create_choropleth_map(
        graph: nx.DiGraph,
        metric: str,
        analysis_results: Dict[str, Any],
        title: str
) -> go.Figure:
    """Create a choropleth map showing a specific trade metric for each country.

    Args:
        graph: The trade network graph
        metric: The metric to visualize ('exports', 'imports', 'balance', etc.)
        analysis_results: The dictionary of analysis results from the analysis module
        title: The title for the visualization

    Returns:
        A Plotly Figure object containing the choropleth map
    """
    # Create a dictionary to map ISO country codes to values
    values = {}

    # ISO country code mapping (partial, extend as needed)
    # In a real application, you would use a complete ISO code mapping
    # This is just a simplified example
    iso_codes = {
        'United States': 'USA',
        'China': 'CHN',
        'Japan': 'JPN',
        'Germany': 'DEU',
        'United Kingdom': 'GBR',
        'France': 'FRA',
        'Canada': 'CAN',
        'Australia': 'AUS',
        'Brazil': 'BRA',
        'India': 'IND',
        # Add more mappings as needed
    }

    # Prepare the data based on the metric
    for node_id, data in graph.nodes(data=True):
        country_name = data['name']

        if country_name in iso_codes:
            if metric == 'exports':
                values[iso_codes[country_name]] = data['total_exports']
            elif metric == 'imports':
                values[iso_codes[country_name]] = data['total_imports']
            elif metric == 'balance':
                values[iso_codes[country_name]] = data['total_exports'] - data['total_imports']
            elif metric == 'total':
                values[iso_codes[country_name]] = data['total_exports'] + data['total_imports']

    # Create the choropleth map
    fig = go.Figure(data=go.Choropleth(
        locations=list(values.keys()),
        z=list(values.values()),
        text=[f"{code}: ${value:,.2f}" for code, value in values.items()],
        colorscale='Viridis',
        autocolorscale=False,
        colorbar_title=f"{metric.capitalize()} (USD)",
        marker_line_color='white',
        marker_line_width=0.5,
    ))

    # Configure the layout
    fig.update_layout(
        title=title,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig


if __name__ == '__main__':
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'extra-imports': ['pandas', 'networkx', 'plotly.graph_objects', 'typing', 'numpy', 'dash'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
