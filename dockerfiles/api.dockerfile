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

# Install Python dependencies and the local package
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir .

EXPOSE $PORT

CMD exec uvicorn src.tweet_sentiment_analysis.api:app --port $PORT --host 0.0.0.0 --workers 1