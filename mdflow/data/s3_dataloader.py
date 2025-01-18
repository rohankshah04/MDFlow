import boto3
import os
import tempfile
import shutil
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
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

    def _get_all_keys(self):
        """Retrieve all keys (folders/files) in the bucket matching the prefix."""
        paginator = self.s3.get_paginator('list_objects_v2')
        keys = []
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Only include folders/files directly under the prefix
                    keys.append(obj['Key'])
        return sorted(list(set([key.split('/')[0] + '/' for key in keys if '/' in key])))

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
                    local_file_path = os.path.join(local_folder, os.path.relpath(file_key, start=s3_folder))
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    self.s3.download_file(self.bucket_name, file_key, local_file_path)

    def reset(self):
        """Reset the loader to start from the beginning."""
        self.current_index = 0

    def cleanup(self, local_path):
        """Delete the temporary local folder after processing."""
        shutil.rmtree(local_path, ignore_errors=True)


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
            # Add your data processing logic here
            # e.g., load files from folder, extract data, etc.

            # Clean up the local folder after processing
            loader.cleanup(folder)
