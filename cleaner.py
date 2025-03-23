import pandas as pd
import argparse


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
        ["exporter_name", "importer_name"],
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

    # Ask the user to choose a cleaning filter
    choice = input("Enter the number corresponding to the cleaning filter you'd like to use (1): ").strip()
    if choice not in ['1', '2']:
        print("Invalid choice. Please run the program again and select either 1 or 2.")

    # Ask user for the input & output CSV file paths
    input_csv = input("Enter the path to the input CSV file: ").strip()
    output_csv = input("Enter the desired path for the output CSV file: ").strip()

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
    # if choice == 1:
    #     value_trade_data(input_csv, output_csv, year)


if __name__ == "__main__":
    run_interactive()
