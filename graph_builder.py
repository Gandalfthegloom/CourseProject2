"""
CSC111 Project 2: Global Trade Interdependence - Graph Builder

This module defines a Graph class for building a trade network graph and a Vertex class
to represent each country with its export and import relationships. The build_trade_graph
function constructs the graph from processed trade data.
"""

from typing import Dict
import pandas as pd

class Vertex:
    def __init__(self, item: str):
        """
        Initialize a Vertex representing a country.

        Args:
            item: The country's name.
        """
        self.item = item  # Country name
        self.export_to: Dict[str, float] = {}
        self.import_from: Dict[str, float] = {}

    def __repr__(self) -> str:
        return f"Vertex({self.item})"


class Graph:
    def __init__(self):
        """
        Initialize an empty Graph.
        """
        self.vertices: Dict[str, Vertex] = {}

    def add_vertex(self, country_id: str, country_name: str) -> None:
        """
        Add a vertex representing a country to the graph.

        Args:
            country_id: A unique identifier for the country.
            country_name: The country's name.
        """
        if country_id not in self.vertices:
            self.vertices[country_id] = Vertex(country_name)
        else:
            # Optionally, update the country name if necessary.
            pass

    def add_edge(self, exporter_id: str, importer_id: str, value: float) -> None:
        """
        Add an edge representing a trade relationship from exporter to importer.

        This method updates:
          - The export_to dictionary for the exporter vertex.
          - The import_from dictionary for the importer vertex.

        Args:
            exporter_id: The unique identifier for the exporting country.
            importer_id: The unique identifier for the importing country.
            value: The trade value.
        """
        if exporter_id not in self.vertices or importer_id not in self.vertices:
            raise ValueError("Both exporter and importer must be added as vertices first.")

        exporter_vertex = self.vertices[exporter_id]
        importer_vertex = self.vertices[importer_id]

        exporter_vertex.export_to[importer_vertex.item] = value
        importer_vertex.import_from[exporter_vertex.item] = value

    def __repr__(self) -> str:
        return f"Graph(vertices={list(self.vertices.keys())})"


def build_trade_graph(trade_data: pd.DataFrame) -> Graph:
    """
    Constructs a trade graph from the processed trade data.

    Args:
        trade_data: A pandas DataFrame containing the trade data with columns:
                    'exporter_id', 'exporter_name', 'importer_id', 'importer_name', 'value'

    Returns:
        A Graph instance populated with vertices representing countries and edges representing
        trade relationships. Each vertex stores:
            - item: The country name.
            - export_to: A dictionary mapping partner country names (exports) to trade values.
            - import_from: A dictionary mapping partner country names (imports) to trade values.
    """
    graph = Graph()

    # Iterate through each row of the DataFrame and update the graph accordingly.
    for _, row in trade_data.iterrows():
        exporter_id = row['exporter_id']
        exporter_name = row['exporter_name']
        importer_id = row['importer_id']
        importer_name = row['importer_name']
        value = row['value']

        # Add exporter and importer vertices if they do not already exist.
        graph.add_vertex(exporter_id, exporter_name)
        graph.add_vertex(importer_id, importer_name)

        # Add the edge representing the trade relationship.
        graph.add_edge(exporter_id, importer_id, value)

    return graph


if __name__ == "__main__":
    import doctest
    doctest.testmod()
