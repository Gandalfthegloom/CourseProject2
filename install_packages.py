"""CSC111 Project: Package Installation Script

This script installs all required packages for the Global Trade Interdependence
project directly without using a requirements.txt file.
"""

import subprocess
import sys


def install_packages():
    """Install all required packages directly."""
    # Hard-coded list of required packages with version specifications
    packages = [
        # Data processing
        "pandas>=2.0.0",
        "numpy>=1.25.0",
        # Graph operations
        "networkx>=3.1",
        "python-louvain>=0.16",
        # Visualization
        "plotly>=5.22.0",
        "dash>=2.13.0",
        # Testing and code checking
        "python-ta~=2.9.1",
        "pytest>=7.4.0"
    ]

    print("Installing required packages...")

    # Install each package
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")

    print("Package installation process completed.")


if __name__ == '__main__':
    install_packages()
