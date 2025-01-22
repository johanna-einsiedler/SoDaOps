# Use a slim Python 3.10 image as the base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app
 
# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get install -y dos2unix \
    curl \
    && rm -rf /var/lib/apt/lists/* 

# Copy only the necessary files and folders into the container
COPY pyproject.toml /app/pyproject.toml
COPY requirements.txt /app/requirements.txt
COPY src /app/src
COPY configs /app/configs
COPY entrypoint.sh /app/entrypoint.sh
COPY .dvc/dockerconfig /app/.dvc/config
COPY data.dvc /app/data.dvc

# Install Python dependencies and the local package
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir .

# Ensure correct line endings
RUN dos2unix entrypoint.sh

# Ensure the entrypoint script is executable
RUN chmod +x entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
