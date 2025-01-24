# Use a slim Python 3.10 image as the base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app
 
# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get install -y dos2unix \
    curl \
    && rm -rf /var/lib/apt/lists/* 

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y google-cloud-sdk

# Copy only the necessary files and folders into the container
COPY pyproject.toml /app/pyproject.toml
COPY requirements_drift.txt /app/requirements.txt
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
RUN gsutil -m cp -r gs://sentiment_dtumlops/data /app/data
#RUN dvc pull -v || (echo "DVC pull failed" && exit 1)


# Set the entrypoint
ENTRYPOINT ["uvicorn", "src.tweet_sentiment_analysis.drift_report:app", "--host", "0.0.0.0", "--port", "8000"]
