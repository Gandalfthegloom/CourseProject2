This project constructs and explores a directed, weighted network of global trade flows for 2023, applying graph‐theoretic analyses (centrality, community detection, trade‐dependency metrics) and presenting results via an interactive Dash dashboard. 
https://wtrademap.onrender.com
Link above! Disclaimer, it uses Render's free tier, so it might take a while to load and start up.

## Overview

A snapshot of world trade in 2023 is modeled as a directed graph, where nodes are countries and edges are bilateral trade volumes. Key features include:

* Data cleaning and aggregation of Observatory of Economic Complexity (OEC) trade records
* Integration of GDP data to compute trade‑to‑GDP ratios
* Application of the disparity filter and Louvain algorithm for backbone extraction and community detection
* Computation of export/import rankings, trade balances, centrality measures, and concentration indices
* An interactive, multi‑tab dashboard built with Plotly and Dash  &#x20;

## Datasets

All source data is packaged in **global\_trade\_datasets.zip**, containing:

1. `bilateral_value_clean_23_withid.csv`
2. `API_NY.GDP.MKTP.CD_DS2_en_csv_v2_26433.csv`
3. `extra_country_coords.json`

You can download the archive via your course’s MarkUs page, Send.UToronto.ca link, or the provided OneDrive URL.&#x20;

## Installation (Docker)

### 1. Clone repo
```
git clone https://github.com/your-username/global-trade-network.git
cd global-trade-network
```

### 2. Build the Docker image
```
docker build -t global-trade-app .
```

### 3. Run the container and map to local port 8050
```
docker run -p 8050:8050 global-trade-app
```
Then, click the links in the CLI!

### What’s happening under the hood?

The app and its dataset files are bundled into a Docker container.

All dependencies (Dash, Pandas, NetworkX, etc.) are installed during build.

The server is configured to run with host=0.0.0.0 and port from the PORT environment variable—making it flexible and cloud-ready.

## Installation (No Docker)

### 1. Clone the repository

```
git clone https://github.com/your‑username/global‑trade‑network.git
cd global-trade-network
```

### 2. Prepare your data folder

Unzip `global_trade_datasets.zip` into a folder named `Data` at the project root:

```
unzip global_trade_datasets.zip -d Data
```

### 3. Install dependencies  
This project requires Python 3.13. Install all necessary libraries via the provided script:  
```
python install_packages.py
```

(Or, if you prefer, `pip install -r requirements.txt`)&#x20;

## Usage

1. Launch the Dash application:

   ```
   python main.py
   ```

2. Open your browser and navigate to

   ```
   http://127.0.0.1:8050/
   ```

3. Explore the tabs for global network, community views, country‑specific maps, and choropleth visualizations.

## Project Structure

```
├── Data/                              # Unzipped datasets
│   ├── bilateral_value_clean_23_withid.csv
│   ├── API_NY.GDP.MKTP.CD_DS2_en_csv_v2_26433.csv
│   └── extra_country_coords.json
├── cleaner.py                         # Cleans and aggregates raw trade data
├── data_processing.py                 # Merges in GDP & coordinates, preprocesses for graph
├── graph_builder.py                   # Builds NetworkX graph, applies disparity filter
├── analysis.py                        # Computes rankings, centrality, dependency metrics
├── visualization.py                   # Defines Plotly/Dash dashboard layout & callbacks
├── install_packages.py                # Installs Python dependencies
├── requirements.txt                   # Pin‑listed libraries
├── main.py                            # Entry point to launch Dash server
└── global_trade_datasets.zip          # Archive of all input data
```
