# Use a lightweight base image with Python (version 3.x)
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data into the container
COPY . . 

# Expose the Dash default port
EXPOSE 8050

# Start the Dash app server
CMD ["python", "main.py"]
