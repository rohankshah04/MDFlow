import boto3
import os
import tempfile
import shutil
import logging
import sys
import numpy as np  # Import numpy for handling .npz files

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to DEBUG for more detailed logs if needed
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)  # Adjust logging level here
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)


class S3DataLoader:
    def __init__(self, bucket_name, prefix='', batch_size=1):
        """
        Initialize the data loader.

        Args:
            bucket_name (str): Name of the S3 bucket.
            prefix (str): Prefix to filter folders/files in the bucket.
            batch_size (int): Number of data points to retrieve at a time.
        """
        self.s3 = boto3.client('s3')  # Initialize the S3 client
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.batch_size = batch_size
        self.keys = self._get_all_keys()
        self.current_index = 0
        self.downloaded_files = []

    def _get_all_keys(self):
        """Retrieve all keys (folders/files) in the bucket matching the prefix."""
        paginator = self.s3.get_paginator('list_objects_v2')
        keys = []
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Only include folders/files directly under the prefix
                    keys.append(obj['Key'])
        # Extract unique top-level folders under the prefix
        unique_folders = sorted(list(set([key.split('/')[0] + '/' for key in keys if '/' in key])))
        logger.info(f"Found {len(unique_folders)} folders in bucket '{self.bucket_name}' with prefix '{self.prefix}'")
        return unique_folders

    def get_next_batch(self):
        """Retrieve the next batch of data points."""
        if self.current_index >= len(self.keys):
            return None  # No more data to retrieve

        batch = self.keys[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size

        # Download each data point in the batch
        data = []
        for key in batch:
            local_folder = tempfile.mkdtemp()
            logger.info(f"Downloading {key} to {local_folder}")
            self._download_folder(key, local_folder)
            data.append(local_folder)  # Append local path for processing

        return data

    def _download_folder(self, s3_folder, local_folder):
        """Download a folder from S3 to a local directory."""
        paginator = self.s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=s3_folder):
            if 'Contents' in page:
                for obj in page['Contents']:
                    file_key = obj['Key']
                    # Skip if the key is a folder (ends with '/')
                    if file_key.endswith('/'):
                        continue
                    local_file_path = os.path.join(local_folder, os.path.relpath(file_key, start=s3_folder))
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    self.s3.download_file(self.bucket_name, file_key, local_file_path)
                    self.downloaded_files.append(local_file_path)
                    logger.debug(f"Downloaded {file_key} to {local_file_path}")

    def reset(self):
        """Reset the loader to start from the beginning."""
        self.current_index = 0

    def cleanup(self, local_path):
        """Delete the temporary local folder after processing."""
        shutil.rmtree(local_path, ignore_errors=True)
        logger.debug(f"Cleaned up local folder {local_path}")


def inspect_npz_file(npz_path):
    """
    Inspect a .npz file and log its contents, including the number of arrays.

    Args:
        npz_path (str): Path to the .npz file.
    """
    try:
        with np.load(npz_path) as data:
            num_arrays = len(data.files)
            logger.info(f"Inspecting .npz file: {npz_path}")
            logger.info(f"Number of arrays in {os.path.basename(npz_path)}: {num_arrays}")
            for array_name in data.files:
                array = data[array_name]
                logger.info(f" - {array_name}: shape={array.shape}, dtype={array.dtype}")
    except Exception as e:
        logger.error(f"Failed to inspect .npz file {npz_path}: {e}")


# Example Usage
if __name__ == "__main__":
    bucket_name = "mdflow.atlas"
    loader = S3DataLoader(bucket_name=bucket_name, batch_size=2)

    while True:
        batch = loader.get_next_batch()
        if batch is None:
            logger.info("All data points have been retrieved.")
            break

        for folder in batch:
            logger.info(f"Processing data from: {folder}")
            # Find all .npz files in the folder
            npz_files = [f for f in os.listdir(folder) if f.endswith('.npz')]
            if not npz_files:
                logger.warning(f"No .npz files found in folder: {folder}")
            for npz_file in npz_files:
                npz_path = os.path.join(folder, npz_file)
                inspect_npz_file(npz_path)
                # Add your data processing logic here
                # e.g., perform computations on the arrays

            # Clean up the local folder after processing
            loader.cleanup(folder)