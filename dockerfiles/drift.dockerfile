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
COPY .dvc/dockerconfig /app/.dvc/config
COPY data.dvc /app/data.dvc

# Install Python dependencies and the local package
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir .


# Set the entrypoint
ENTRYPOINT ["uvicorn", "drift_report:app", "--host", "0.0.0.0", "--port", "8000"]
