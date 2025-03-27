"""CSC111 Project 2: Global Trade Interdependence - Visualization

This module contains functions for creating interactive visualizations of the trade network.
It uses Plotly and Dash to generate an interactive dashboard with multiple visualization options.
"""

from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, callback
import random


def create_trade_visualization(
        graph: nx.DiGraph,
        country_coords: pd.DataFrame,
        analysis_results: Dict[str, Any]
) -> go.Figure:
    """Create an interactive visualization of the global trade network."""
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
            node_size = 7 + (total_trade / 1e9) ** 0.49   # Log scale to handle wide range of values

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
    # Create a dictionary to store edges by source country
    country_edges = {}
    for source, target, data in graph.edges(data=True):
        source_name = graph.nodes[source]['name']
        target_name = graph.nodes[target]['name']

        if source_name in country_coord_map and target_name in country_coord_map:
            if source_name not in country_edges:
                country_edges[source_name] = []
            country_edges[source_name].append((source_name, target_name, data['value']))

    # Get top 5 trading partners for each country
    top_edges = []
    for country, edges in country_edges.items():
        # Sort edges for this country by value
        edges.sort(key=lambda x: x[2], reverse=True)
        # Take top 5 or fewer if less than 5 exist
        top_country_edges = edges[:5]
        top_edges.extend(top_country_edges)

    # Add arrows for each top trade relationship
    for source_name, target_name, value in top_edges:
        if source_name in country_coord_map and target_name in country_coord_map:
            start_lat, start_lon = country_coord_map[source_name]
            end_lat, end_lon = country_coord_map[target_name]

            # Calculate edge weight (line thickness) based on value
            weight = 0.15 + (value / 1e9) ** 0.2

            # Add the trade flow arrow
            plot_trade_arrow(
                fig,
                start_lat, start_lon,
                end_lat, end_lon,
                weight,
                'rgba(255, 50, 50, 0.6)',  # Semi-transparent red for export flows
                f"<b>Trade Flow</b><br>{source_name} → {target_name}<br>Value: ${value:,.2f}"
            )

    # Configure the layout with a more vibrant background
    fig.update_layout(
        title='Global Trade Network Visualization',
        geo=dict(
            projection_type='natural earth',
            showland=True,
            landcolor='rgb(240, 248, 255)',  # Light blue background for land
            showcountries=True,
            countrycolor='rgb(173, 216, 230)',  # Lighter country borders
            showocean=True,
            oceancolor='rgb(230, 250, 255)',  # Softer ocean color
            showlakes=True,
            lakecolor='rgb(220, 240, 255)'
        ),
        height=800,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='rgba(230, 240, 255, 0.5)',  # Very light blue background
        plot_bgcolor='rgba(230, 240, 255, 0.5)'
    )

    return fig


def visualize_country_trade(
        graph: nx.DiGraph,
        country_id: str,
        country_coords: pd.DataFrame,
        analysis_results: Dict[str, Any]
) -> go.Figure:
    """Create a visualization focused on a specific country's trade relationships."""
    # Create a base figure
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


def create_choropleth_map(
        graph: nx.DiGraph,
        metric: str,
        country_coords: pd.DataFrame,
        title: str
) -> go.Figure:
    """Create a choropleth map showing a specific trade metric for each country."""
    # Extract values based on the selected metric
    countries = []
    values = []
    texts = []
    lats = []
    lons = []

    # Create a mapping of country names to coordinates
    country_coord_map = dict(zip(country_coords['country'],
                                 zip(country_coords['centroid_lat'], country_coords['centroid_lon'])))

    for node_id, data in graph.nodes(data=True):
        country_name = data['name']

        if country_name in country_coord_map:
            lat, lon = country_coord_map[country_name]

            # Determine the value based on the selected metric
            if metric == 'exports':
                value = data['total_exports']
                text = f"<b>{country_name}</b><br>Total Exports: ${value:,.2f}"
            elif metric == 'imports':
                value = data['total_imports']
                text = f"<b>{country_name}</b><br>Total Imports: ${value:,.2f}"
            elif metric == 'balance':
                value = data['total_exports'] - data['total_imports']
                text = f"<b>{country_name}</b><br>Trade Balance: ${value:,.2f}"
            else:  # 'total'
                value = data['total_exports'] + data['total_imports']
                text = f"<b>{country_name}</b><br>Total Trade: ${value:,.2f}"

            countries.append(country_name)
            values.append(value)
            texts.append(text)
            lats.append(lat)
            lons.append(lon)

    # Create a figure with bubbles on a map (alternative to choropleth)
    fig = go.Figure()

    # Add bubbles for each country
    fig.add_trace(go.Scattergeo(
        lon=lons,
        lat=lats,
        text=texts,
        mode='markers',
        marker=dict(
            size=np.log1p(np.abs(values)) * 2,  # Size based on value (log scale)
            color=values,
            colorscale='RdYlGn' if metric == 'balance' else 'Viridis',
            colorbar=dict(title=f"{metric.capitalize()} (USD)"),
            cmin=min(values) if min(values) < 0 else None,
            cmax=max(values),
            line=dict(width=1, color='black')
        ),
        hoverinfo='text',
        name=f'{metric.capitalize()} by Country'
    ))

    # Configure the layout
    fig.update_layout(
        title=title,
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


def create_dashboard(
        graph: nx.DiGraph,
        country_coords: pd.DataFrame,
        analysis_results: Dict[str, Any]
) -> None:
    """Create an integrated dashboard with visualization options."""
    # Create a list of countries for the dropdown
    countries = [(node_id, data['name']) for node_id, data in graph.nodes(data=True)]
    countries.sort(key=lambda x: x[1])  # Sort by country name

    # Find the ID for Afghanistan (or first country if Afghanistan not found)
    default_country = next((id_ for id_, name in countries if name.lower() == 'afghanistan'), countries[0][0])

    # Create a Dash application with full-screen configuration
    app = dash.Dash(__name__,
                    title="Global Trade Network Explorer",
                    external_stylesheets=[
                        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css'
                    ])

    # Define the overall app layout with a modern, full-screen design
    app.layout = html.Div([
        # Full-screen header with gradient and shadow
        html.Header([
            html.Div([
                html.H1([
                    html.I(className="fas fa-globe-americas", style={'marginRight': '10px'}),
                    "Global Trade Network Explorer"
                ], style={
                    'textAlign': 'center',
                    'color': 'white',
                    'fontFamily': 'Arial, sans-serif',
                    'fontSize': '2.5em',
                    'fontWeight': 'bold',
                    'textShadow': '2px 2px 4px rgba(0,0,0,0.3)'
                }),
            ], style={
                'background': 'linear-gradient(135deg, #2c3e50, #3498db)',
                'padding': '20px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.2)',
                'borderBottom': '3px solid #2980b9'
            })
        ], style={'width': '100%'}),

        # Full-screen content container
        html.Div([
            # Dynamic tabs with modern styling
            dcc.Tabs(id='main-tabs', value='global-network',  # Set default value to 'global-network'
                     style={'width': '100%', 'display': 'flex', 'justifyContent': 'center'},
                     children=[
                dcc.Tab(label='🌍 Global Network', value='global-network',
                        style={'padding': '15px', 'fontSize': '1.1em'}),
                dcc.Tab(label='🏴 Country Details', value='country-trade',
                        style={'padding': '15px', 'fontSize': '1.1em'}),
                dcc.Tab(label='📊 Trade Metrics', value='trade-metrics',
                        style={'padding': '15px', 'fontSize': '1.1em'})
            ],
            colors={
                "border": "#2c3e50",
                "primary": "#3498db",
                "background": "#ecf0f1"
            }),

            # Dynamic content area
            html.Div(id='tabs-content', style={
                'padding': '20px',
                'background': 'linear-gradient(to bottom right, #f0f4f8, #e6f2ff)',
                'minHeight': 'calc(100vh - 200px)'
            })
        ])
    ], style={
        'fontFamily': 'Arial, sans-serif',
        'margin': '0',
        'padding': '0',
        'width': '100vw',
        'height': '100vh',
        'overflow': 'hidden',
        'background': 'linear-gradient(to bottom right, #f0f4f8, #e6f2ff)'
    })

    # Callback to render tab content with suppress_callback_exceptions
    @app.callback(
        Output('tabs-content', 'children'),
        [Input('main-tabs', 'value')],
        prevent_initial_call=False  # Change this to False to load on initial page load
    )
    def render_content(tab):
        # Wrap each tab's content in a container with consistent styling
        if tab == 'global-network':
            return html.Div([
                html.H2("Global Trade Network Overview",
                        style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                dcc.Graph(
                    id='global-trade-graph',
                    figure=create_trade_visualization(graph, country_coords, analysis_results),
                    style={'height': '70vh', 'width': '100%'}
                )
            ])

        elif tab == 'country-trade':
            return html.Div([
                html.H2("Country-Specific Trade Relationships",
                        style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                html.Div([
                    html.Label("Select a Country:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='country-dropdown',
                        options=[{'label': f'{get_flag_emoji(name)} {name}', 'value': id_} for id_, name in countries],
                        value=default_country,  # Set default to Afghanistan or first country
                        style={'width': '50%', 'margin': '0 auto'}
                    )
                ], style={'textAlign': 'center', 'marginBottom': '20px'}),
                dcc.Graph(id='country-trade-graph',
                          figure=visualize_country_trade(graph, default_country, country_coords, analysis_results))
            ])

        elif tab == 'trade-metrics':
            return html.Div([
                html.H2("Global Trade Metrics",
                        style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                html.Div([
                    dcc.RadioItems(
                        id='metric-selector',
                        options=[
                            {'label': '📤 Total Exports', 'value': 'exports'},
                            {'label': '📥 Total Imports', 'value': 'imports'},
                            {'label': '⚖️ Trade Balance', 'value': 'balance'},
                            {'label': '💱 Total Trade Volume', 'value': 'total'}
                        ],
                        value='exports',  # Set default to total exports
                        style={
                            'display': 'flex',
                            'justifyContent': 'center',
                            'gap': '15px',
                            'marginBottom': '20px'
                        },
                        labelStyle={'cursor': 'pointer'}
                    )
                ]),
                dcc.Graph(
                    id='trade-metrics-graph',
                    figure=create_choropleth_map(
                        graph,
                        'exports',
                        country_coords,
                        '📤 Global Export Volume by Country'
                    )
                )
            ])

    # Callback for country trade graph
    @app.callback(
        Output('country-trade-graph', 'figure'),
        [Input('country-dropdown', 'value')],
        prevent_initial_call=True
    )
    def update_country_trade_graph(selected_country):
        """Update the graph for a specific country's trade details."""
        return visualize_country_trade(graph, selected_country, country_coords, analysis_results)

    # Callback for trade metrics graph
    @app.callback(
        Output('trade-metrics-graph', 'figure'),
        [Input('metric-selector', 'value')],
        prevent_initial_call=True
    )
    def update_trade_metrics_graph(selected_metric):
        """Update the trade metrics graph based on selected metric."""
        titles = {
            'exports': '📤 Global Export Volume by Country',
            'imports': '📥 Global Import Volume by Country',
            'balance': '⚖️ Trade Balance by Country',
            'total': '💱 Total Trade Volume by Country'
        }
        return create_choropleth_map(
            graph,
            selected_metric,
            country_coords,
            titles[selected_metric]
        )

    # Configure the app to enable callback exceptions
    app.config.suppress_callback_exceptions = True

    print("Starting trade dashboard! Navigate to http://127.0.0.1:8050/ to explore global trade!")
    app.run(debug=True)


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
    """Add a directional arrow representing a trade relationship to the map."""
    # Use a great circle path for the trade flow
    # This creates a curved line that follows the earth's curvature

    # To avoid arrows crossing through the earth, we'll create a path with intermediate points
    n_points = 100
    lat_path = np.linspace(start_lat, end_lat, n_points)
    lon_path = np.linspace(start_lon, end_lon, n_points)

    # Add slight curve to the path
    mid_index = n_points // 2
    # Curve factor determines how much the path bends
    curve_factor = 0.2

    # Calculate angle perpendicular to the direct path
    angle = np.arctan2(end_lat - start_lat, end_lon - start_lon) + np.pi / 2

    # Apply curve
    for i in range(1, n_points - 1):
        # Scale of curve reduces towards path ends
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
    fig.add_trace(go.Scattergeo(
        lon=lon_path[-2:],
        lat=lat_path[-2:],
        mode='lines',
        line=dict(
            width=weight * 1.5,  # Make arrowhead slightly thicker
            color=color
        ),
        hoverinfo='text',
        text=hover_text,
        showlegend=False
    ))


def get_flag_emoji(country_name):
    """Generate a flag emoji for a given country name."""
    flag_emojis = {
        'United States': '🇺🇸', 'China': '🇨🇳', 'Japan': '🇯🇵',
        'Germany': '🇩🇪', 'United Kingdom': '🇬🇧', 'France': '🇫🇷',
        'India': '🇮🇳', 'Brazil': '🇧🇷', 'Italy': '🇮🇹',
        'Canada': '🇨🇦', 'Russia': '🇷🇺', 'South Korea': '🇰🇷',
        # Add more countries as needed
    }
    return flag_emojis.get(country_name, '🌍')  # Default to globe if no flag found


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['pandas', 'networkx', 'plotly.graph_objects', 'typing', 'numpy', 'dash', 'random'],
        'allowed-io': ['create_dashboard'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
