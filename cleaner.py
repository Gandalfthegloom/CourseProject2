"""CSC111 Winter 2025: Global Trade Interdependence Visualization

This module serves as the data cleaning program that takes raw data from BACI and processes it for main.py.
"""


import pandas as pd


def value_trade_data(input_csv, output_csv, year=None):
    """
    Takes BACI trade data and filters it down to bilateral (country pair)
    trade values for specified year, or all years in data if unspecified.
    """

    # Read CSV file
    df = pd.read_csv(input_csv)

    if year is not None:
        df = df[df["year"] == year]

    # Group data by country pairs and sum the total trade value
    aggregated_df = df.groupby(
        ["exporter_name", "importer_id"],
        as_index=False
    )["value"].sum()

    # Export the cleaned data to a new CSV file.
    aggregated_df.to_csv(output_csv, index=False)
    print(f"Cleaned data exported to {output_csv}")


def value_trade_data_withid(input_csv, output_csv, year=None):
    """
    Takes BACI trade data and filters it down to bilateral (country pair)
    trade values for specified year, or all years in data if unspecified.
    same as value_trade_data but includes importer and exporters' 3 letter id.
    """

    # Read CSV file
    df = pd.read_csv(input_csv)

    if year is not None:
        df = df[df["year"] == year]

    # Group data by country pairs and sum the total trade value
    aggregated_df = df.groupby(
        ["exporter_id", "exporter_name", "importer_id", "importer_name"],
        as_index=False
    )["value"].sum()

    # Export the cleaned data to a new CSV file.
    aggregated_df.to_csv(output_csv, index=False)
    print(f"Cleaned data exported to {output_csv}")


def run_interactive():
    """
    Basically the UI script, enables user to run the cleaning script as desired through python console.
    """
    print("Welcome to the Trade Data Cleaning Tool.")
    print("Available cleaning filters:")
    print("1. Aggregate Trade Data")
    print("   - Aggregates bilateral trade data by grouping exporter/importer pairs and summing the trade value.")
    print("   - Optionally, you can filter the data by a specific year.")
    print("2. Aggregate Trade Data With ID")
    print("Same as 1, but includes the importers' and exporters' 3 digit ID")

    # Ask the user to choose a cleaning filter
    choice = input("Enter the number corresponding to the cleaning filter you'd like to use (e.g 1 or 2): ").strip()
    if choice not in ['1', '2']:
        print("Invalid choice. Please run the program again and select either 1 or 2.")

    # Ask user for the input & output CSV file paths
    input_csv = input("Enter the path to the input CSV file (no quotes!): ").strip()
    output_csv = input("Enter the desired path for the output CSV file (no quotes!): ").strip()

    # Ask if the user wants to filter by a specific year
    year_filter = input("Do you want to filter by a specific year? (yes/no): ").strip().lower()

    year = None
    if year_filter in ['yes', 'y']:
        year_input = input("Enter the year you want to filter by (e.g., 2023): ").strip()
        try:
            year = int(year_input)
        except ValueError:
            print("Invalid year input. The script will process all available years.")

    # Process the data
    print("Processing...")
    if choice == '1':
        value_trade_data(input_csv, output_csv, year)
    if choice == '2':
        value_trade_data_withid(input_csv, output_csv, year)

    print("Completed. Thank you for using the Trade Data Cleaning Tool.")
    print("We hope to see you again!")


if __name__ == "__main__":
    run_interactive()
