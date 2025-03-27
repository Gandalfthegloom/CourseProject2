"""CSC111 Project 2: Global Trade Interdependence - Data Processing

This module contains functions for loading and processing the preprocessed trade data.
It handles the cleaning, filtering, and preparation of data for graph construction.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import plotly.express as px


def load_trade_data(file_path: str) -> pd.DataFrame:
    """Load the preprocessed trade data from the given file path.

    Args:
        file_path: The path to the preprocessed trade data file (CSV format)

    Returns:
        A pandas DataFrame containing the cleaned trade data ready for graph construction

    Preconditions:
        - file_path refers to a valid CSV file with the expected columns:
          'exporter_id', 'exporter_name', 'importer_id', 'importer_name', 'value'
    """
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Ensure value column is numeric
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Drop rows with NaN values
    df = df.dropna(subset=['exporter_id', 'importer_id', 'value'])

    # Convert country IDs to strings
    df['exporter_id'] = df['exporter_id'].astype(str)
    df['importer_id'] = df['importer_id'].astype(str)

    print(f"Successfully loaded trade data: {len(df)} trade relationships")

    return df


def get_country_coordinates() -> pd.DataFrame:
    """Generate a mapping of country IDs to their geographical coordinates.

    This function creates a DataFrame mapping each country name to its latitude and longitude
    coordinates for visualization on a world map. It uses Gapminder's centroids (141 countries)
    and appends extra coordinates (for the remaining countries from the trade data) from an external JSON file.

    Returns:
        A DataFrame with columns 'country', 'centroid_lat', and 'centroid_lon'.
    """
    import plotly.express as px
    import pandas as pd
    import os
    import json

    # Load gapminder centroids
    gapminder = px.data.gapminder(centroids=True)
    country_centroids = gapminder[['country', 'centroid_lat', 'centroid_lon']].drop_duplicates()

    # Rename countries to match trade data names, if needed
    country_name_mappings = {
        'Czech Republic': 'Czechia',
        'Slovak Republic': 'Slovakia',
        'Swaziland': 'Eswatini',
        'Myanmar': 'Burma',
        'Korea, Rep.': 'South Korea',
        'Korea, Dem. Rep.': 'North Korea',
        'Hong Kong, China': 'Hong Kong',
        'Taiwan': 'Chinese Taipei',
        'West Bank and Gaza': 'Palestine',
        'Congo, Dem. Rep.': 'Democratic Republic of the Congo',
        'Congo, Rep.': 'Republic of the Congo',
        'Yemen, Rep.': 'Yemen'
    }

    for gap_name, trade_name in country_name_mappings.items():
        mask = country_centroids['country'] == gap_name
        if mask.any():
            country_centroids.loc[mask, 'country'] = trade_name
        else:
            print(f"Warning: {gap_name} not found in gapminder dataset")

    # Load extra coordinates from the JSON file located in the "Data" folder
    extra_path = os.path.join(os.path.dirname(__file__), 'Data', 'extra_country_coords.json')
    with open(extra_path, 'r', encoding='utf-8') as f:
        extra_coords = json.load(f)
    extra_df = pd.DataFrame.from_dict(extra_coords, orient='index')

    # Concatenate gapminder data with extra coordinates and drop duplicates
    full_coords = pd.concat([country_centroids, extra_df], ignore_index=True)
    full_coords = full_coords.drop_duplicates(subset=['country'], keep='first')
    return full_coords


def filter_by_trade_volume(data: pd.DataFrame, min_value: float) -> pd.DataFrame:
    """Filter the trade data to include only relationships above a certain value threshold.

    Args:
        data: The trade data DataFrame
        min_value: The minimum trade value (in USD) to include

    Returns:
        A filtered DataFrame containing only the trade relationships above the threshold
    """
    # Filter the DataFrame based on the 'value' column
    pass


def get_top_trading_partners(data: pd.DataFrame, country_id: str, n: int = 10) -> pd.DataFrame:
    """Get the top trading partners for a specific country.

    Args:
        data: The trade data DataFrame
        country_id: The ID of the country to find partners for
        n: The number of top partners to return

    Returns:
        A DataFrame containing the top n trading partners sorted by trade value
    """
    # Filter for exports from the specified country
    # Sort by value and return the top n
    pass


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    # import python_ta
    # python_ta.check_all(config={
    #     'extra-imports': ['pandas', 'typing'],
    #     'allowed-io': ['load_trade_data'],
    #     'max-line-length': 100,
    #     'disable': ['R1705', 'C0200']
    # })
