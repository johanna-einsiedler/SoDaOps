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
COPY requirements.txt /app/requirements_drift.txt
COPY src /app/src
COPY configs /app/configs

# Install Python dependencies and the local package
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir .

# retrieving data
RUN dvc init --no-scm
COPY .dvc/dockerconfig /app/.dvc/config
COPY data.dvc /app/data.dvc
RUN dvc config core.no_scm true
RUN dvc pull


# Set the entrypoint
ENTRYPOINT ["uvicorn", "src.tweet_sentiment_analysis.drift_report:app", "--host", "0.0.0.0", "--port", "8000"]
