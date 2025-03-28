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
from dash import dcc, html, dash_table, Input, Output, callback
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
            node_size = 7 + (total_trade / 1e9) ** 0.49  # Hybrid scaling ftw

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
                title=dict(
                    text="Trade Balance Ratio<br>(Exports-Imports)/(Exports+Imports)",
                    font=dict(size=12),
                    side="right"
                ),
                thickness=15,
                len=0.5,
                y=0.5,
                yanchor="middle",
                x=1.02,
                xanchor="left",
                outlinewidth=1,
                outlinecolor="black",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1", "-0.5", "0", "0.5", "1"]
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

    # Get top x trading partners for each country
    top_edges = []
    for country, edges in country_edges.items():
        # Sort edges for this country by value
        edges.sort(key=lambda x: x[2], reverse=True)
        # Take top 5 or fewer if less than 5 exist
        top_country_edges = edges[:100]
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


def create_community_visualization(
        graph: nx.DiGraph,
        country_coords: pd.DataFrame,
        analysis_results: Dict[str, Any]
) -> go.Figure:
    """Create an interactive visualization of the global trade network with community coloring.

    Countries and trade flows are colored based on their community membership.

    Args:
        graph: NetworkX DiGraph containing the trade network
        country_coords: DataFrame with country coordinates
        analysis_results: Dictionary containing analysis results including trade_communities

    Returns:
        A Plotly figure with the visualization
    """
    # Create a base world map
    fig = go.Figure()

    # Create a mapping of country names to coordinates
    country_coord_map = dict(zip(country_coords['country'],
                                 zip(country_coords['centroid_lat'], country_coords['centroid_lon'])))

    # Get community assignments from analysis results
    communities = analysis_results.get('trade_communities', {})

    if not communities:
        print("Warning: No community data found in analysis results")
        # Fall back to the regular visualization if no communities
        return create_trade_visualization(graph, country_coords, analysis_results)

    # Generate distinct colors for each community
    unique_communities = set(communities.values())
    num_communities = len(unique_communities)

    # Create a color map for communities - using a colorful scale
    community_colors = {}
    colorscales = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis']
    chosen_scale = colorscales[0]  # Default to Viridis

    # Use a continuous color scale for a large number of communities
    import plotly.colors as pc
    if num_communities <= 10:
        # For small number of communities, use discrete colors
        colors = pc.qualitative.Bold + pc.qualitative.Prism
        for i, community_id in enumerate(sorted(unique_communities)):
            community_colors[community_id] = colors[i % len(colors)]
    else:
        # For many communities, use a continuous color scale
        color_scale = getattr(pc.sequential, chosen_scale)
        for i, community_id in enumerate(sorted(unique_communities)):
            index = i / (num_communities - 1) if num_communities > 1 else 0.5
            community_colors[community_id] = pc.sample_colorscale(color_scale, index)[0]

    # Add nodes (countries) as scatter markers
    node_sizes = []
    node_colors = []
    node_texts = []
    node_lats = []
    node_lons = []
    node_communities = []

    for country_id, data in graph.nodes(data=True):
        country_name = data['name']
        if country_name in country_coord_map:
            lat, lon = country_coord_map[country_name]

            # Calculate node size based on total trade volume
            total_trade = data['total_exports'] + data['total_imports']
            node_size = 7 + (total_trade / 1e9) ** 0.49  # Hybrid scaling as in the original

            # Get community color
            community_id = communities.get(country_id, -1)  # Default to -1 if not in a community
            node_community = community_id

            # Prepare hover text
            hover_text = (
                f"<b>{country_name}</b><br>"
                f"Community: {community_id}<br>"
                f"Total Exports: ${data['total_exports']:,.2f}<br>"
                f"Total Imports: ${data['total_imports']:,.2f}<br>"
                f"Trade Balance: ${data['total_exports'] - data['total_imports']:,.2f}<br>"
            )

            node_sizes.append(node_size)
            node_colors.append(community_id)
            node_texts.append(hover_text)
            node_lats.append(lat)
            node_lons.append(lon)
            node_communities.append(community_id)

    # Create continuous color scale mapping for communities
    unique_communities_sorted = sorted(set(node_communities))
    community_to_index = {comm: i / (len(unique_communities_sorted) - 1) if len(unique_communities_sorted) > 1 else 0.5
                          for i, comm in enumerate(unique_communities_sorted)}

    # Map community IDs to values between 0 and 1 for coloring
    node_color_values = [community_to_index.get(comm, 0.5) for comm in node_communities]

    # Add country nodes to the figure
    fig.add_trace(go.Scattergeo(
        lon=node_lons,
        lat=node_lats,
        text=node_texts,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_color_values,
            colorscale=chosen_scale,
            colorbar=dict(
                title=dict(
                    text="Trade Communities",
                    font=dict(size=12),
                    side="right"
                ),
                thickness=15,
                len=0.5,
                y=0.5,
                yanchor="middle",
                x=1.02,
                xanchor="left",
                outlinewidth=1,
                outlinecolor="black",
                tickmode='array',
                tickvals=[community_to_index.get(comm, 0.5) for comm in sorted(unique_communities_sorted)],
                ticktext=[str(comm) for comm in sorted(unique_communities_sorted)]
            ),
            line=dict(width=1, color='black')
        ),
        hoverinfo='text',
        name='Countries'
    ))

    # Add edges (trade flows) colored by source community
    # Create a dictionary to store edges by source country
    country_edges = {}
    for source, target, data in graph.edges(data=True):
        source_name = graph.nodes[source]['name']
        target_name = graph.nodes[target]['name']

        if source_name in country_coord_map and target_name in country_coord_map:
            if source_name not in country_edges:
                country_edges[source_name] = []
            country_edges[source_name].append((source, source_name, target, target_name, data['value']))

    # Get top trading partners for each country
    top_edges = []
    for country, edges in country_edges.items():
        # Sort edges for this country by value
        edges.sort(key=lambda x: x[4], reverse=True)
        # Take top 100 or fewer if less than 100 exist (as in the original function)
        top_country_edges = edges[:100]
        top_edges.extend(top_country_edges)

    # Add arrows for each top trade relationship
    for source_id, source_name, target_id, target_name, value in top_edges:
        if source_name in country_coord_map and target_name in country_coord_map:
            start_lat, start_lon = country_coord_map[source_name]
            end_lat, end_lon = country_coord_map[target_name]

            # Calculate edge weight (line thickness) based on value
            weight = 0.15 + (value / 1e9) ** 0.2

            # Get community of source country for coloring
            source_community = communities.get(source_id, -1)
            edge_color = community_colors.get(source_community, 'grey')

            # Add the trade flow arrow
            plot_trade_arrow(
                fig,
                start_lat, start_lon,
                end_lat, end_lon,
                weight,
                edge_color,  # Color based on source country's community
                f"<b>Trade Flow</b><br>{source_name} → {target_name}<br>Value: ${value:,.2f}<br>Community: {source_community}"
            )

    # Configure the layout with a vibrant background (same as in create_trade_visualization)
    fig.update_layout(
        title='Global Trade Network by Communities',
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
    total_exports = graph.nodes[country_id]['total_exports']
    total_imports = graph.nodes[country_id]['total_imports']

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
    export_sizes = []

    for _, target in graph.out_edges(country_id):
        target_name = graph.nodes[target]['name']
        if target_name in country_coord_map:
            target_lat, target_lon = country_coord_map[target_name]
            value = graph.edges[country_id, target]['value']
            size = 12 + (500 * value / total_exports) ** 0.7  # for node size

            export_lats.append(target_lat)
            export_lons.append(target_lon)
            export_texts.append(f"<b>Export to {target_name}</b><br>Value: ${value:,.2f}")
            export_values.append(value)
            export_sizes.append(size)

            # Calculate edge weight (line thickness) based on value
            weight = 0.15 + (value / 1e7) ** 0.2

            # Add arrow for this trade relationship
            plot_trade_arrow(
                fig,
                selected_lat, selected_lon,
                target_lat, target_lon,
                weight,
                'rgba(255, 50, 50, 0.6)',  # Red for exports
                f"<b>Export</b><br>{country_name} → {target_name}<br>Value: ${value:,.2f}"
            )

    # Add import partners (incoming edges)
    import_lats = []
    import_lons = []
    import_texts = []
    import_values = []
    import_sizes = []

    for source, _ in graph.in_edges(country_id):
        source_name = graph.nodes[source]['name']
        if source_name in country_coord_map:
            source_lat, source_lon = country_coord_map[source_name]
            value = graph.edges[source, country_id]['value']
            size = 12 + (500 * value / total_imports) ** 0.7  # for node size

            import_lats.append(source_lat)
            import_lons.append(source_lon)
            import_texts.append(f"<b>Import from {source_name}</b><br>Value: ${value:,.2f}")
            import_values.append(value)
            import_sizes.append(size)

            weight = 0.15 + (value / 1e7) ** 0.2

            # Add arrow for this trade relationship
            plot_trade_arrow(
                fig,
                source_lat, source_lon,
                selected_lat, selected_lon,
                weight,
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
                size=export_sizes,
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
                size=import_sizes,
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
        title: str,
        gdp_data: Optional[pd.DataFrame] = None
) -> go.Figure:
    """Create a hybrid choropleth-scatter map showing trade metrics for all countries.

    This function uses a choropleth for standard countries and scatter points for
    territories that aren't recognized by Plotly's choropleth.

    Args:
        graph: NetworkX DiGraph containing trade data
        metric: Which trade metric to visualize ('exports', 'imports', 'balance', 'total', or 'openness')
        country_coords: DataFrame with country coordinates
        title: Title for the visualization
        gdp_data: Optional DataFrame with GDP data (required for 'openness' metric)

    Returns:
        A Plotly figure object with the visualization
    """
    # Create a mapping of country names to coordinates
    country_coord_map = dict(zip(country_coords['country'],
                                 zip(country_coords['centroid_lat'], country_coords['centroid_lon'])))

    # Define list of territories that should be displayed as scatter points
    # These are territories that Plotly's choropleth doesn't recognize
    non_standard_territories = {
        'Hong Kong', 'Macau', 'Chinese Taipei', 'Puerto Rico', 'Palestine',
        'Reunion', 'Aruba', 'Anguilla', 'Bonaire', 'Curaçao', 'Montserrat',
        'Wallis and Futuna', 'Tokelau', 'British Virgin Islands',
        'French South Antarctic Territory', 'Saint Barthélemy', 'Cocos (Keeling) Islands',
        'Christmas Island', 'Northern Mariana Islands', 'Norfolk Island',
        'Pitcairn Islands', 'Saint Martin', 'Saint Pierre and Miquelon'
        # Add more as you identify them
    }

    # Country name mapping to match Plotly's expected names
    country_name_mapping = {
        'United States': 'USA',
        'United Kingdom': 'UK',
        'Burma': 'Myanmar',
        'Czechia': 'Czech Republic',
        'Republic of the Congo': 'Congo',
        'Democratic Republic of the Congo': 'Democratic Republic of Congo',
        "Cote d'Ivoire": 'Ivory Coast',
        'Eswatini': 'Swaziland',
        'North Macedonia': 'Macedonia',
        'Timor-Leste': 'East Timor',
        'Cape Verde': 'Cabo Verde',
        'Saint Vincent and the Grenadines': 'St. Vincent and the Grenadines',
        'Saint Kitts and Nevis': 'St. Kitts and Nevis',
        'Saint Lucia': 'St. Lucia',
        'Antigua and Barbuda': 'Antigua',
        'Trinidad and Tobago': 'Trinidad',
        'Bosnia and Herzegovina': 'Bosnia'
    }

    # Create dictionaries to store standard and non-standard country data
    standard_countries = []
    standard_values = []
    standard_texts = []

    nonstandard_countries = []
    nonstandard_values = []
    nonstandard_texts = []
    nonstandard_lats = []
    nonstandard_lons = []

    # Prepare ISO code to country name mapping for GDP data
    iso_to_country = {}
    if gdp_data is not None:
        for _, row in gdp_data.iterrows():
            iso_to_country[row['country_code'].lower()] = row['country_name']

    # Set up the color scale based on metric
    if metric == 'balance':
        color_scale = 'RdBu'  # Red-Blue for negative-positive values
    elif metric == 'openness':
        color_scale = 'Viridis'  # A different color scale for openness
    else:
        color_scale = 'oxy'  # Blues for other metrics

    # Process each country in the graph
    for node_id, data in graph.nodes(data=True):
        country_name = data['name']

        # For openness metric, we need to match with GDP data using ISO codes
        if metric == 'openness' and gdp_data is not None:
            # Get the country's ISO code
            country_iso = node_id.lower()  # Assuming node_id is the ISO code

            # Try to find the GDP data for this country
            gdp_value = gdp_data.loc[gdp_data['country_code'].str.lower() == country_iso, 'gdp_2023'].iloc[0] if not \
                gdp_data[gdp_data['country_code'].str.lower() == country_iso].empty else None

            # Calculate openness index if GDP data is available
            if gdp_value is not None and gdp_value > 0:
                total_trade = data['total_exports'] + data['total_imports']
                value = (total_trade / gdp_value) * 100  # As percentage
                text = (f"<b>{country_name}</b><br>"
                        f"Trade Openness: {value:.2f}%<br>"
                        f"Total Trade: ${total_trade:,.2f}<br>"
                        f"GDP: ${gdp_value:,.2f}")
            else:
                # Skip countries with no GDP data for openness metric
                continue
        # Determine the value based on other selected metrics
        elif metric == 'exports':
            value = data['total_exports']
            text = f"<b>{country_name}</b><br>Total Exports: ${value:,.2f}"
        elif metric == 'imports':
            value = data['total_imports']
            text = f"<b>{country_name}</b><br>Total Imports: ${value:,.2f}"
        elif metric == 'balance':
            # Calculate trade balance ratio ==
            trade_balance = data['total_exports'] - data['total_imports']
            total_trade = data['total_exports'] + data['total_imports']

            # Normalize between -1 and 1 for coloring
            if total_trade > 0:
                value = trade_balance / total_trade  # This will be between -1 and 1
            else:
                value = 0

            text = (f"<b>{country_name}</b><br>"
                    f"Trade Balance: ${trade_balance:,.2f}<br>"
                    f"Total Trade: ${total_trade:,.2f}<br>"
                    f"Balance Ratio: {value:.2%}")
        else:  # 'total'
            value = data['total_exports'] + data['total_imports']
            text = f"<b>{country_name}</b><br>Total Trade: ${value:,.2f}"

        # Check if this is a non-standard territory or not in our coordinate map
        if country_name in non_standard_territories or country_name not in country_coord_map:
            # Only add if we have coordinates for it
            if country_name in country_coord_map:
                lat, lon = country_coord_map[country_name]
                nonstandard_countries.append(country_name)
                nonstandard_values.append(value)
                nonstandard_texts.append(text)
                nonstandard_lats.append(lat)
                nonstandard_lons.append(lon)
        else:
            # Standard country for choropleth
            location_name = country_name_mapping.get(country_name, country_name)
            standard_countries.append(location_name)
            standard_values.append(value)
            standard_texts.append(text)

    # Create the figure
    fig = go.Figure()

    # Configure color scale and range
    if metric == 'balance':
        # For balance, we want a diverging scale centered at 0
        max_abs_value = max(
            abs(min(standard_values + nonstandard_values, default=0)),
            abs(max(standard_values + nonstandard_values, default=0))
        )
        zmin = -max_abs_value
        zmax = max_abs_value

        # Balance-specific tick formatting
        tick_values = [-1, -0.5, 0, 0.5, 1]
        tick_texts = ['-100%', '-50%', '0%', '50%', '100%']
    elif metric == 'openness':
        # For openness, we want a scale from 0 to max (or a reasonable upper limit like 150%)
        zmin = 0
        zmax = min(150, max(standard_values + nonstandard_values, default=100))

        # Openness-specific tick formatting
        tick_values = [0, 25, 50, 75, 100, 125, 150]
        tick_texts = ['0%', '25%', '50%', '75%', '100%', '125%', '150%']
    else:
        # For exports, imports, and total metrics, use logarithmic scaling
        # Add a small constant to handle zero values before taking log
        constant = 1e6  # $1 million, to avoid log(0) issues

        # Apply logarithmic transformation to standard values
        if standard_values:
            log_standard_values = [np.log10(val + constant) if val > 0 else 0 for val in standard_values]
        else:
            log_standard_values = []

        # Apply logarithmic transformation to non-standard values
        if nonstandard_values:
            log_nonstandard_values = [np.log10(val + constant) if val > 0 else 0 for val in nonstandard_values]
        else:
            log_nonstandard_values = []

        # Set range from 0 to max of log values
        zmin = np.log10(constant)  # Log of the constant we added
        max_log_value = max(log_standard_values + log_nonstandard_values, default=np.log10(4e12))
        zmax = max_log_value

        # Create logarithmic tick values that represent intuitive dollar amounts
        max_value = max(standard_values + nonstandard_values, default=4e12)

        # Create tick marks at powers of 10
        magnitude = int(np.log10(max_value))
        tick_values = [np.log10(10 ** i + constant) for i in range(7, magnitude + 2)]
        tick_texts = ['$10M', '$100M', '$1B', '$10B', '$100B', '$1T', '$10T'][:magnitude - 5]

        # Replace the values in the data with their log versions for plotting
        standard_values = log_standard_values
        nonstandard_values = log_nonstandard_values

    # PART 1: Add choropleth for standard countries
    if standard_countries:
        fig.add_trace(go.Choropleth(
            locations=standard_countries,
            locationmode='country names',
            z=standard_values,
            text=standard_texts,
            colorscale=color_scale,
            zmin=zmin,
            zmax=zmax,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            colorbar=dict(
                title=f'{title.split(" ")[1]} ({metric.capitalize()})',
                thicknessmode='pixels',
                thickness=20,
                lenmode='pixels',
                len=300,
                yanchor='bottom',
                y=0,
                x=1.01,
                tickmode='array',
                tickvals=tick_values,
                ticktext=tick_texts
            ),
            hoverinfo='text',
            name='Countries'
        ))

    # PART 2: Add scatter points for non-standard territories
    if nonstandard_lats:
        # Calculate node size based on value
        # Using log scale to handle wide range of values
        node_sizes = [max(7, 5 + (abs(val) / 1e9) ** 0.5) for val in nonstandard_values]

        # For balance and openness metrics, we need to handle coloring differently
        if metric == 'balance':
            # Normalize between -1 and 1 for coloring
            node_colors = []
            for val in nonstandard_values:
                if val > 0:
                    # Positive is blue (1)
                    normalized = min(1, val / max_abs_value)
                else:
                    # Negative is red (-1)
                    normalized = max(-1, val / max_abs_value)
                node_colors.append(normalized)

            scatter_colorscale = 'RdBu'
            scatter_cmin = -1
            scatter_cmax = 1
        elif metric == 'openness':
            # Normalize between 0 and zmax for coloring
            node_colors = [min(1, val / zmax) for val in nonstandard_values]
            scatter_colorscale = 'Viridis'
            scatter_cmin = 0
            scatter_cmax = 1
        else:
            # For exports, imports, total - just normalize from 0 to max
            node_colors = [val / zmax for val in nonstandard_values]
            scatter_colorscale = 'oxy'
            scatter_cmin = 0
            scatter_cmax = 1

        fig.add_trace(go.Scattergeo(
            lon=nonstandard_lons,
            lat=nonstandard_lats,
            text=nonstandard_texts,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale=scatter_colorscale,
                cmin=scatter_cmin,
                cmax=scatter_cmax,
                line=dict(width=1, color='black')
            ),
            hoverinfo='text',
            name='Special Territories'
        ))

    # Configure the layout
    fig.update_layout(
        title=title,
        geo=dict(
            showframe=False,
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
        height=900,
        margin=dict(l=0, r=150, t=50, b=100),
        legend_title_text='Territory Type'
    )

    return fig


def create_dashboard(
        filtered_graph: nx.DiGraph,
        graph: nx.DiGraph,
        country_coords: pd.DataFrame,
        analysis_results: Dict[str, Any],
        gdp_data: Optional[pd.DataFrame] = None
) -> None:
    """Create an integrated dashboard with visualization options."""
    # Create a list of countries for the dropdown
    countries = [(node_id, data['name']) for node_id, data in filtered_graph.nodes(data=True)]
    countries.sort(key=lambda x: x[1])  # Sort by country name

    # Find the ID for Afghanistan (or first country if Afghanistan not found)
    default_country = next((id_ for id_, name in countries if name.lower() == 'afghanistan'), countries[0][0])

    # Create a Dash application with full-screen configuration
    app = dash.Dash(__name__,
                    title="Global Trade Network Explorer",
                    external_stylesheets=[
                        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css'
                    ])

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
                    'textShadow': '2px 2px 4px rgba(0,0,0,0.3)',
                }),
            ], style={
                'background': 'linear-gradient(135deg, #2c3e50, #3498db)',
                'padding': '10px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.2)',
                'borderBottom': '0.5px solid black'
            })
        ], style={'width': '100%'}),

        # Full-screen content container
        html.Div([
            dcc.Tabs(id='main-tabs', value='global-network',  # Set default value to 'global-network'
                     style={'width': '100%', 'display': 'flex', 'justifyContent': 'center'},
                     children=[
                         dcc.Tab(label='🌍 Global Network', value='global-network',
                                 style={
                                     'textAlign': 'center',
                                     'color': 'white',
                                     'fontFamily': 'Arial, sans-serif',
                                     'fontSize': '1.3em',
                                     'fontWeight': 'bold',
                                     'textShadow': '2px 2px 4px rgba(0,0,0,0.3)',
                                     'padding': '15px'
                                 }),
                         dcc.Tab(label='🏴 Country Details', value='country-trade',
                                 style={
                                     'textAlign': 'center',
                                     'color': 'white',
                                     'fontFamily': 'Arial, sans-serif',
                                     'fontSize': '1.3em',
                                     'fontWeight': 'bold',
                                     'textShadow': '2px 2px 4px rgba(0,0,0,0.3)',
                                     'padding': '15px'
                                 }),
                         dcc.Tab(label='📊 Trade Metrics', value='trade-metrics',
                                 style={
                                     'textAlign': 'center',
                                     'color': 'white',
                                     'fontFamily': 'Arial, sans-serif',
                                     'fontSize': '1.3em',
                                     'fontWeight': 'bold',
                                     'textShadow': '2px 2px 4px rgba(0,0,0,0.3)',
                                     'padding': '15px'
                                 }),
                     ],
                     colors={
                         "border": "#2c3e50",
                         "primary": "#3498db",
                         "background": "#84b0d1"
                     }),

            # Dynamic content area
            html.Div(id='tabs-content', style={
                'padding': '20px',
                'background': 'linear-gradient(to bottom right, #f0f4f8, #e6f2ff)',
                'minHeight': 'calc(100vh - 250px)',  # Increased bottom space
                'width': '100%'
            })
        ], style={
            'height': 'calc(100vh - 100px)',
            'width': '100%',
            'overflowY': 'auto'  # Only one scrollbar for the entire content
        })
    ],
        style={
            'textAlign': 'center',
            'fontFamily': 'Arial, sans-serif',
            'fontSize': '1 em',
            'fontWeight': 'bold',
            'textShadow': '0.5px 0.5px 1px rgba(0,0,0,0.3)',
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
            # Format data for tables
            top_exporters_data = analysis_results.get('top_exporters', [])[:10]
            top_exporters_table = [{'Rank': i+1, 'Country': country, 'Exports (USD)': f"${value:,.2f}"}
                                  for i, (country, value) in enumerate(top_exporters_data)]

            top_importers_data = analysis_results.get('top_importers', [])[:10]
            top_importers_table = [{'Rank': i+1, 'Country': country, 'Imports (USD)': f"${value:,.2f}"}
                                  for i, (country, value) in enumerate(top_importers_data)]

            strongest_relationships_data = analysis_results.get('strongest_relationships', [])[:10]
            strongest_relationships_table = [{'Rank': i+1, 'Exporter': exporter, 'Importer': importer, 'Trade Value (USD)': f"${value:,.2f}"}
                                           for i, (exporter, importer, value) in enumerate(strongest_relationships_data)]

            # Create a table design style
            table_style = {
                'overflowX': 'auto',
                'backgroundColor': 'white',
                'border': '1px solid #ddd',
                'borderRadius': '8px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
                'margin': '20px 0',
                'width': '100%'
            }

            header_style = {
                'backgroundColor': '#3498db',
                'color': 'white',
                'textAlign': 'center',
                'fontWeight': 'bold',
                'padding': '12px 15px'
            }

            cell_style = {
                'textAlign': 'center',
                'padding': '12px 15px',
                'borderBottom': '1px solid #ddd'
            }

            # Create the tab content
            return html.Div([
                html.H2("Global Trade Network Overview",
                        style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                # Trade network visualization
                dcc.Graph(
                    id='global-trade-graph',
                    figure=create_trade_visualization(filtered_graph, country_coords, analysis_results),
                    style={
                        'height': '70vh',
                        'width': '90%',
                        'marginLeft': 'auto',
                        'marginRight': 'auto',
                        'marginBottom': '40px'
                    }
                ),
                # Key insights section with tables
                html.Div([
                    html.H2("Key Global Trade Insights",
                           style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),

                    # Tables layout - 2 per row for desktop
                    html.Div([
                        # Left column - Top Exporters
                        html.Div([
                            html.H3("Top 10 Exporting Countries",
                                   style={'textAlign': 'center', 'color': '#2980b9', 'marginBottom': '15px'}),
                            html.Div([
                                dash_table.DataTable(
                                    id='top-exporters-table',
                                    columns=[{'name': col, 'id': col} for col in ['Rank', 'Country', 'Exports (USD)']],
                                    data=top_exporters_table,
                                    style_table=table_style,
                                    style_header=header_style,
                                    style_cell=cell_style,
                                    style_data_conditional=[
                                        {
                                            'if': {'row_index': 'odd'},
                                            'backgroundColor': 'rgb(248, 248, 248)'
                                        }
                                    ]
                                )
                            ])
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                        # Right column - Top Importers
                        html.Div([
                            html.H3("Top 10 Importing Countries",
                                   style={'textAlign': 'center', 'color': '#2980b9', 'marginBottom': '15px'}),
                            html.Div([
                                dash_table.DataTable(
                                    id='top-importers-table',
                                    columns=[{'name': col, 'id': col} for col in ['Rank', 'Country', 'Imports (USD)']],
                                    data=top_importers_table,
                                    style_table=table_style,
                                    style_header=header_style,
                                    style_cell=cell_style,
                                    style_data_conditional=[
                                        {
                                            'if': {'row_index': 'odd'},
                                            'backgroundColor': 'rgb(248, 248, 248)'
                                        }
                                    ]
                                )
                            ])
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
                    ], style={'marginBottom': '30px'}),

                    # Strongest relationships (full width)
                    html.Div([
                        html.H3("Strongest Bilateral Trade Relationships",
                               style={'textAlign': 'center', 'color': '#2980b9', 'marginBottom': '15px'}),
                        html.Div([
                            dash_table.DataTable(
                                id='strongest-relationships-table',
                                columns=[{'name': col, 'id': col} for col in ['Rank', 'Exporter', 'Importer', 'Trade Value (USD)']],
                                data=strongest_relationships_table,
                                style_table=table_style,
                                style_header=header_style,
                                style_cell=cell_style,
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': 'rgb(248, 248, 248)'
                                    }
                                ]
                            )
                        ])
                    ], style={'marginBottom': '30px'}),
                    html.H2("Global Trade Communities Overview",
                            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                    # Trade network visualization
                    dcc.Graph(
                        id='global-trade-graph',
                        figure=create_community_visualization(filtered_graph, country_coords, analysis_results),
                        style={
                            'height': '70vh',
                            'width': '90%',
                            'marginLeft': 'auto',
                            'marginRight': 'auto',
                            'marginBottom': '40px'
                        }
                    ),
                    # Trade communities (if available)
                    html.Div([
                        html.H3("Trade Communities Overview",
                               style={'textAlign': 'center', 'color': '#2980b9', 'marginBottom': '15px'}),
                        html.P("The analysis identified major trading communities or blocs in the global trade network.",
                              style={'textAlign': 'center', 'marginBottom': '15px'}),
                        html.Div([
                            html.Div([
                                html.H4(f"Community {i+1}", style={'color': '#3498db', 'marginBottom': '10px'}),
                                html.P(f"Number of countries: {len(community_countries)}",
                                      style={'marginBottom': '5px'}),
                                html.P("Key members: " + ", ".join(community_countries[:5]) +
                                      ("..." if len(community_countries) > 5 else ""),
                                      style={'fontStyle': 'italic'})
                            ], style={
                                'backgroundColor': 'white',
                                'borderRadius': '8px',
                                'padding': '15px',
                                'marginBottom': '15px',
                                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                            })
                            for i, community_countries in enumerate(get_community_countries(filtered_graph, analysis_results))
                            if i < 5  # Show only the top 5 communities
                        ])
                    ], style={'marginBottom': '30px'})
                ], style={
                    'backgroundColor': 'rgba(255, 255, 255, 0.8)',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                    'marginBottom': '50px',
                    'width': '90%',
                    'marginLeft': 'auto',
                    'marginRight': 'auto'
                })
            ])

        elif tab == 'country-trade':
            # Regenerate countries list within the callback to avoid scope issues
            current_countries = [(node_id, data['name']) for node_id, data in filtered_graph.nodes(data=True)]
            current_countries.sort(key=lambda x: x[1])  # Sort by country name

            # Find the ID for Afghanistan (or first country if Afghanistan not found)
            current_default = next((id_ for id_, name in current_countries if name.lower() == 'afghanistan'),
                                  current_countries[0][0] if current_countries else None)

            return html.Div([
                html.H2("Country-Specific Trade Relationships",
                        style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                html.Div([
                    html.Label("Select a Country:",
                               style={'fontWeight': 'bold', 'marginRight': '10px', 'fontSize': '0.95 em'}),
                    dcc.Dropdown(
                        id='country-dropdown',
                        options=[{'label': f'{get_flag_emoji(name)} {name}', 'value': id_} for id_, name in current_countries],
                        value=current_default,  # Set default to Afghanistan or first country
                        style={'width': '50%', 'margin': '0 auto'}
                    )
                ], style={'textAlign': 'center', 'marginBottom': '20px'}),
                dcc.Graph(id='country-trade-graph',
                          figure=visualize_country_trade(graph, current_default, country_coords, analysis_results),
                          style={
                              'height': '70vh',
                              'width': '90%',
                              'marginLeft': 'auto',
                              'marginRight': 'auto',
                              'marginBottom': '40px'
                          }
                          )
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
                            {'label': '💱 Total Trade Volume', 'value': 'total'},
                            {'label': '🔄 Trade Openness Index', 'value': 'openness'}
                        ],
                        value='exports',  # Set default to total exports
                        style={
                            'display': 'flex',
                            'justifyContent': 'center',
                            'gap': '15px',
                            'marginBottom': '20px',
                            'fontSize': '0.95 em'
                        },
                        labelStyle={'cursor': 'pointer'}
                    )
                ]),
                dcc.Graph(
                    id='trade-metrics-graph',
                    figure=create_choropleth_map(
                        filtered_graph,
                        'exports',
                        country_coords,
                        '📤 Global Export Volume by Country',
                        gdp_data
                    ),
                    style={
                        'height': '70vh',
                        'width': '95%',
                        'marginLeft': 'auto',
                        'marginRight': 'auto',
                        'marginBottom': '100px'
                    }
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
            'total': '💱 Total Trade Volume by Country',
            'openness': '🔄 Trade Openness Index by Country'
        }
        return create_choropleth_map(
            filtered_graph,
            selected_metric,
            country_coords,
            titles[selected_metric],
            gdp_data
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
    flag_emojis = {'Afghanistan': '🇦🇫', 'Albania': '🇦🇱', 'Algeria': '🇩🇿', 'Angola': '🇦🇴', 'Argentina': '🇦🇷',
                   'Australia': '🇦🇺', 'Austria': '🇦🇹', 'Bahrain': '🇧🇭', 'Bangladesh': '🇧🇩', 'Belgium': '🇧🇪',
                   'Benin': '🇧🇯', 'Bolivia': '🇧🇴', 'Bosnia and Herzegovina': '🇧🇦', 'Botswana': '🇧🇼', 'Brazil': '🇧🇷',
                   'Bulgaria': '🇧🇬', 'Burkina Faso': '🇧🇫', 'Burundi': '🇧🇮', 'Cambodia': '🇰🇭', 'Cameroon': '🇨🇲',
                   'Canada': '🇨🇦', 'Central African Republic': '🇨🇫', 'Chad': '🇹🇩', 'Chile': '🇨🇱', 'China': '🇨🇳',
                   'Colombia': '🇨🇴', 'Comoros': '🇰🇲', 'Democratic Republic of the Congo': '🇨🇩',
                   'Republic of the Congo': '🇨🇬', 'Costa Rica': '🇨🇷', "Cote d'Ivoire": '🇨🇮', 'Croatia': '🇭🇷',
                   'Cuba': '🇨🇺', 'Czechia': '🇨🇿', 'Denmark': '🇩🇰', 'Djibouti': '🇩🇯', 'Dominican Republic': '🇩🇴',
                   'Ecuador': '🇪🇨', 'Egypt': '🇪🇬', 'El Salvador': '🇸🇻', 'Equatorial Guinea': '🇬🇶', 'Eritrea': '🇪🇷',
                   'Ethiopia': '🇪🇹', 'Finland': '🇫🇮', 'France': '🇫🇷', 'Gabon': '🇬🇦', 'Gambia': '🇬🇲', 'Germany': '🇩🇪',
                   'Ghana': '🇬🇭', 'Greece': '🇬🇷', 'Guatemala': '🇬🇹', 'Guinea': '🇬🇳', 'Guinea-Bissau': '🇬🇼',
                   'Haiti': '🇭🇹', 'Honduras': '🇭🇳', 'Hong Kong': '🇭🇰', 'Hungary': '🇭🇺', 'Iceland': '🇮🇸', 'India': '🇮🇳',
                   'Indonesia': '🇮🇩', 'Iran': '🇮🇷', 'Iraq': '🇮🇶', 'Ireland': '🇮🇪', 'Israel': '🇮🇱', 'Italy': '🇮🇹',
                   'Jamaica': '🇯🇲', 'Japan': '🇯🇵', 'Jordan': '🇯🇴', 'Kenya': '🇰🇪', 'North Korea': '🇰🇵',
                   'South Korea': '🇰🇷', 'Kuwait': '🇰🇼', 'Lebanon': '🇱🇧', 'Lesotho': '🇱🇸', 'Liberia': '🇱🇷',
                   'Libya': '🇱🇾', 'Madagascar': '🇲🇬', 'Malawi': '🇲🇼', 'Malaysia': '🇲🇾', 'Mali': '🇲🇱',
                   'Mauritania': '🇲🇷', 'Mauritius': '🇲🇺', 'Mexico': '🇲🇽', 'Mongolia': '🇲🇳', 'Montenegro': '🇲🇪',
                   'Morocco': '🇲🇦', 'Mozambique': '🇲🇿', 'Burma': '🇲🇲', 'Namibia': '🇳🇦', 'Nepal': '🇳🇵',
                   'Netherlands': '🇳🇱', 'New Zealand': '🇳🇿', 'Nicaragua': '🇳🇮', 'Niger': '🇳🇪', 'Nigeria': '🇳🇬',
                   'Norway': '🇳🇴', 'Oman': '🇴🇲', 'Pakistan': '🇵🇰', 'Panama': '🇵🇦', 'Paraguay': '🇵🇾', 'Peru': '🇵🇪',
                   'Philippines': '🇵🇭', 'Poland': '🇵🇱', 'Portugal': '🇵🇹', 'Puerto Rico': '🇵🇷', 'Reunion': '🇷🇪',
                   'Romania': '🇷🇴', 'Rwanda': '🇷🇼', 'Sao Tome and Principe': '🇸🇹', 'Saudi Arabia': '🇸🇦',
                   'Senegal': '🇸🇳', 'Serbia': '🇷🇸', 'Sierra Leone': '🇸🇱', 'Singapore': '🇸🇬', 'Slovakia': '🇸🇰',
                   'Slovenia': '🇸🇮', 'Somalia': '🇸🇴', 'South Africa': '🇿🇦', 'Spain': '🇪🇸', 'Sri Lanka': '🇱🇰',
                   'Sudan': '🇸🇩', 'Eswatini': '🇸🇿', 'Sweden': '🇸🇪', 'Switzerland': '🇨🇭', 'Syria': '🇸🇾',
                   'Chinese Taipei': '🇹🇼', 'Tanzania': '🇹🇿', 'Thailand': '🇹🇭', 'Togo': '🇹🇬',
                   'Trinidad and Tobago': '🇹🇹', 'Tunisia': '🇹🇳', 'Turkey': '🇹🇷', 'Uganda': '🇺🇬',
                   'United Kingdom': '🇬🇧', 'United States': '🇺🇸', 'Uruguay': '🇺🇾', 'Venezuela': '🇻🇪', 'Vietnam': '🇻🇳',
                   'Palestine': '🇵🇸', 'Yemen': '🇾🇪', 'Zambia': '🇿🇲', 'Zimbabwe': '🇿🇼', 'Aruba': '🇦🇼', 'Anguilla': '🇦🇮',
                   'Andorra': '🇦🇩', 'United Arab Emirates': '🇦🇪', 'Armenia': '🇦🇲', 'American Samoa': '🇦🇸',
                   'French South Antarctic Territory': '🇹🇫', 'Antigua and Barbuda': '🇦🇬', 'Azerbaijan': '🇦🇿',
                   'Bonaire': '🇧🇶', 'Bahamas': '🇧🇸', 'Saint Barthélemy': '🇧🇱', 'Belarus': '🇧🇾', 'Belize': '🇧🇿',
                   'Bermuda': '🇧🇲', 'Barbados': '🇧🇧', 'Brunei': '🇧🇳', 'Bhutan': '🇧🇹', 'Cocos (Keeling) Islands': '🇨🇨',
                   'Cook Islands': '🇨🇰', 'Cape Verde': '🇨🇻', 'Curaçao': '🇨🇼', 'Christmas Island': '🇨🇽',
                   'Cayman Islands': '🇰🇾', 'Cyprus': '🇨🇾', 'Dominica': '🇩🇲', 'Estonia': '🇪🇪', 'Fiji': '🇫🇯',
                   'Falkland Islands': '🇫🇰', 'Micronesia': '🇫🇲', 'Georgia': '🇬🇪', 'Gibraltar': '🇬🇮', 'Grenada': '🇬🇩',
                   'Greenland': '🇬🇱', 'Guam': '🇬🇺', 'Guyana': '🇬🇾', 'British Indian Ocean Territory': '🇮🇴',
                   'Kazakhstan': '🇰🇿', 'Kyrgyzstan': '🇰🇬', 'Kiribati': '🇰🇮', 'Saint Kitts and Nevis': '🇰🇳',
                   'Laos': '🇱🇦', 'Saint Lucia': '🇱🇨', 'Lithuania': '🇱🇹', 'Luxembourg': '🇱🇺', 'Latvia': '🇱🇻',
                   'Macau': '🇲🇴', 'Saint Martin': '🇲🇫', 'Moldova': '🇲🇩', 'Maldives': '🇲🇻', 'Marshall Islands': '🇲🇭',
                   'North Macedonia': '🇲🇰', 'Malta': '🇲🇹', 'Northern Mariana Islands': '🇲🇵', 'Montserrat': '🇲🇸',
                   'New Caledonia': '🇳🇨', 'Norfolk Island': '🇳🇫', 'Niue': '🇳🇺', 'Nauru': '🇳🇷',
                   'Pitcairn Islands': '🇵🇳', 'Palau': '🇵🇼', 'Papua New Guinea': '🇵🇬', 'French Polynesia': '🇵🇫',
                   'Qatar': '🇶🇦', 'Russia': '🇷🇺', 'Saint Helena': '🇸🇭', 'Solomon Islands': '🇸🇧', 'San Marino': '🇸🇲',
                   'Saint Pierre and Miquelon': '🇵🇲', 'South Sudan': '🇸🇸', 'Suriname': '🇸🇷', 'Seychelles': '🇸🇨',
                   'Turks and Caicos Islands': '🇹🇨', 'Tajikistan': '🇹🇯', 'Tokelau': '🇹🇰', 'Turkmenistan': '🇹🇲',
                   'Timor-Leste': '🇹🇱', 'Tonga': '🇹🇴', 'Tuvalu': '🇹🇻', 'Ukraine': '🇺🇦', 'Uzbekistan': '🇺🇿',
                   'Saint Vincent and the Grenadines': '🇻🇨', 'British Virgin Islands': '🇻🇬', 'Vanuatu': '🇻🇺',
                   'Wallis and Futuna': '🇼🇫', 'Samoa': '🇼🇸'}

    return flag_emojis.get(country_name, '🌍')  # Default to globe if no flag found


def get_community_countries(graph: nx.DiGraph, analysis_results: Dict[str, Any]) -> List[List[str]]:
    """Get lists of countries in each trade community.

    Args:
        graph: The trade network graph
        analysis_results: Dictionary of analysis results including trade_communities

    Returns:
        List of lists, where each inner list contains the country names in one community
    """
    if 'trade_communities' not in analysis_results:
        # If no communities in results, return empty list
        return []

    communities = analysis_results['trade_communities']

    # Create a reverse mapping to group countries by community
    community_groups = {}
    for node_id, community_id in communities.items():
        if community_id not in community_groups:
            community_groups[community_id] = []

        # Get country name from node attributes
        country_name = graph.nodes[node_id].get('name', node_id)
        community_groups[community_id].append(country_name)

    # Sort communities by size (largest first)
    sorted_communities = sorted(community_groups.values(), key=len, reverse=True)

    return sorted_communities


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
