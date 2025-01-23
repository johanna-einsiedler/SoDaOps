from pathlib import Path

from google.cloud import storage
from loguru import logger


def upload_to_gcs(local_path: Path, bucket_name: str, destination_blob_name: str):
    """
    Uploads a file or directory to a GCS bucket.
    Models are saved under a `model/` folder with timestamp-named subfolders.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    if local_path.is_dir():
        for file_path in local_path.glob("**/*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                blob = bucket.blob(f"{destination_blob_name}/{relative_path}")
                blob.upload_from_filename(str(file_path))
                logger.info(f"Uploaded {file_path} to gs://{bucket_name}/{destination_blob_name}/{relative_path}")
    else:
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(str(local_path))
        logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{destination_blob_name}")


def download_from_gcs(bucket_name: str, folder_name: str, destination_dir: str):
    """
    Downloads an entire folder from a GCS bucket to a local directory.

    Args:
        bucket_name (str): Name of the GCS bucket.
        folder_name (str): Name of the folder in the bucket (e.g., 'models/<timestamp>').
        destination_dir (str): Path to the local directory to download to.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_name)

    destination_dir_path = Path(destination_dir)
    destination_dir_path.mkdir(parents=True, exist_ok=True)

    for blob in blobs:
        # Determine the relative file path within the destination directory
        relative_path = Path(blob.name).relative_to(folder_name)
        local_file_path = destination_dir_path / relative_path

        if not blob.name.endswith("/"):  # Skip "directory" markers
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_file_path))
            logger.info(f"Downloaded {blob.name} to {local_file_path}")
