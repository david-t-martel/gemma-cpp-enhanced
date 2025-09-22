"""
Google Cloud Storage Integration for Model Storage.

This module provides functionality for storing and retrieving models from Google Cloud Storage,
with support for encryption, compression, and efficient streaming.
"""

import gzip
import hashlib
import io
import json
import mimetypes
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import BinaryIO
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from google.api_core import retry
from google.api_core.exceptions import GoogleAPIError
from google.cloud import storage
from google.cloud.exceptions import Conflict
from google.cloud.exceptions import NotFound
from google.cloud.exceptions import TooManyRequests
from google.cloud.storage import Blob
from google.cloud.storage import Bucket

from .auth import GCPAuthManager
from .config import GCPConfig


@dataclass
class StorageObject:
    """Represents a GCS object with metadata."""

    bucket: str
    name: str
    size: int
    content_type: str
    etag: str
    created: datetime
    updated: datetime
    metadata: dict[str, Any]
    md5_hash: str | None = None
    crc32c: str | None = None
    generation: int | None = None
    metageneration: int | None = None


class GCSError(Exception):
    """Base exception for GCS operations."""


class GCSUploadError(GCSError):
    """Error during upload operation."""


class GCSDownloadError(GCSError):
    """Error during download operation."""


class GCSStorageManager:
    """Manages Google Cloud Storage operations for model storage."""

    def __init__(self, config: GCPConfig | None = None, auth_manager: GCPAuthManager | None = None):
        """Initialize GCS Storage Manager.

        Args:
            config: GCP configuration
            auth_manager: Authentication manager instance
        """
        self.config = config or GCPConfig.from_env()
        self.auth_manager = auth_manager or GCPAuthManager(self.config)

        # Initialize storage client
        self._client = None
        self._bucket = None

        # Retry configuration
        self.retry_config = retry.Retry(
            predicate=retry.if_exception_type(TooManyRequests, GoogleAPIError),
            initial=1.0,
            maximum=60.0,
            multiplier=2.0,
            timeout=300.0,
        )

        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)

    @property
    def client(self) -> storage.Client:
        """Get or create storage client."""
        if self._client is None:
            self._client = storage.Client(
                project=self.config.project_id, credentials=self.auth_manager.credentials
            )
        return self._client

    @property
    def bucket(self) -> Bucket | None:
        """Get the configured bucket."""
        if self._bucket is None and self.config.gcs_bucket:
            try:
                self._bucket = self.client.bucket(self.config.gcs_bucket)
            except Exception as e:
                logger.error(f"Failed to get bucket {self.config.gcs_bucket}: {e}")
        return self._bucket

    def create_bucket(
        self, bucket_name: str, location: str | None = None, storage_class: str = "STANDARD"
    ) -> Bucket:
        """Create a new GCS bucket.

        Args:
            bucket_name: Name of the bucket to create
            location: Location for the bucket (defaults to config region)
            storage_class: Storage class (STANDARD, NEARLINE, COLDLINE, ARCHIVE)

        Returns:
            Created bucket object
        """
        if location is None:
            location = self.config.region.value

        bucket = self.client.bucket(bucket_name)
        bucket.storage_class = storage_class
        bucket.location = location

        # Enable versioning for model safety
        bucket.versioning_enabled = True

        # Set lifecycle rules for old versions
        bucket.lifecycle_rules = [
            {
                "action": {"type": "Delete"},
                "condition": {
                    "age": 90,  # Delete versions older than 90 days
                    "isLive": False,
                },
            }
        ]

        try:
            bucket = self.client.create_bucket(bucket, location=location)
            logger.info(f"Created bucket {bucket_name} in {location}")

            # Set encryption if configured
            if self.config.enable_encryption and self.config.kms_key_name:
                bucket.default_kms_key_name = self.config.kms_key_name
                bucket.patch()
                logger.info(f"Enabled encryption for bucket {bucket_name}")

            return bucket
        except Conflict:
            logger.warning(f"Bucket {bucket_name} already exists")
            return self.client.bucket(bucket_name)
        except Exception as e:
            raise GCSError(f"Failed to create bucket {bucket_name}: {e}")

    def upload_file(
        self,
        source_path: str | Path,
        destination_blob_name: str,
        bucket_name: str | None = None,
        compress: bool = False,
        metadata: dict[str, str] | None = None,
        chunk_size: int = 8 * 1024 * 1024,  # 8MB chunks
    ) -> StorageObject:
        """Upload a file to GCS.

        Args:
            source_path: Local file path to upload
            destination_blob_name: Destination blob name in GCS
            bucket_name: Bucket name (uses config default if not specified)
            compress: Whether to compress the file with gzip
            metadata: Custom metadata to attach to the blob
            chunk_size: Upload chunk size in bytes

        Returns:
            StorageObject with upload details
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        bucket_name = bucket_name or self.config.gcs_bucket
        if not bucket_name:
            raise ValueError("Bucket name must be specified")

        bucket = self.client.bucket(bucket_name)
        blob_name = f"{self.config.gcs_prefix}/{destination_blob_name}"
        blob = bucket.blob(blob_name)

        # Set metadata
        if metadata:
            blob.metadata = metadata
        else:
            blob.metadata = {}

        # Add upload metadata
        blob.metadata.update(
            {
                "uploaded_at": datetime.now(UTC).isoformat(),
                "original_filename": source_path.name,
                "file_size": str(source_path.stat().st_size),
            }
        )

        # Determine content type
        content_type, _ = mimetypes.guess_type(str(source_path))
        if content_type:
            blob.content_type = content_type

        try:
            if compress:
                # Compress and upload
                with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as tmp_file:
                    try:
                        with open(source_path, "rb") as f_in:
                            with gzip.open(tmp_file.name, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)

                        blob.content_encoding = "gzip"
                        blob.upload_from_filename(
                            tmp_file.name, retry=self.retry_config, chunk_size=chunk_size
                        )
                    finally:
                        os.unlink(tmp_file.name)
            else:
                # Direct upload
                blob.upload_from_filename(
                    str(source_path), retry=self.retry_config, chunk_size=chunk_size
                )

            # Reload to get updated metadata
            blob.reload()

            logger.info(f"Uploaded {source_path} to gs://{bucket_name}/{blob_name}")

            return StorageObject(
                bucket=bucket_name,
                name=blob_name,
                size=blob.size,
                content_type=blob.content_type or "application/octet-stream",
                etag=blob.etag,
                created=blob.time_created,
                updated=blob.updated,
                metadata=blob.metadata,
                md5_hash=blob.md5_hash,
                crc32c=blob.crc32c,
                generation=blob.generation,
                metageneration=blob.metageneration,
            )

        except Exception as e:
            raise GCSUploadError(f"Failed to upload {source_path}: {e}")

    def download_file(
        self,
        blob_name: str,
        destination_path: str | Path,
        bucket_name: str | None = None,
        decompress: bool = False,
        chunk_size: int = 8 * 1024 * 1024,  # 8MB chunks
    ) -> Path:
        """Download a file from GCS.

        Args:
            blob_name: Blob name in GCS
            destination_path: Local destination path
            bucket_name: Bucket name (uses config default if not specified)
            decompress: Whether to decompress gzipped files
            chunk_size: Download chunk size in bytes

        Returns:
            Path to downloaded file
        """
        bucket_name = bucket_name or self.config.gcs_bucket
        if not bucket_name:
            raise ValueError("Bucket name must be specified")

        destination_path = Path(destination_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        bucket = self.client.bucket(bucket_name)
        blob_name = f"{self.config.gcs_prefix}/{blob_name}"
        blob = bucket.blob(blob_name)

        try:
            if not blob.exists():
                raise NotFound(f"Blob not found: gs://{bucket_name}/{blob_name}")

            if decompress and blob.content_encoding == "gzip":
                # Download to temp file and decompress
                with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as tmp_file:
                    try:
                        blob.download_to_filename(
                            tmp_file.name, retry=self.retry_config, chunk_size=chunk_size
                        )

                        with gzip.open(tmp_file.name, "rb") as f_in:
                            with open(destination_path, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    finally:
                        os.unlink(tmp_file.name)
            else:
                # Direct download
                blob.download_to_filename(
                    str(destination_path), retry=self.retry_config, chunk_size=chunk_size
                )

            logger.info(f"Downloaded gs://{bucket_name}/{blob_name} to {destination_path}")
            return destination_path

        except Exception as e:
            raise GCSDownloadError(f"Failed to download {blob_name}: {e}")

    def stream_download(
        self,
        blob_name: str,
        bucket_name: str | None = None,
        chunk_size: int = 1024 * 1024,  # 1MB chunks
    ) -> Iterator[bytes]:
        """Stream download a file from GCS.

        Args:
            blob_name: Blob name in GCS
            bucket_name: Bucket name (uses config default if not specified)
            chunk_size: Streaming chunk size in bytes

        Yields:
            Chunks of file data
        """
        bucket_name = bucket_name or self.config.gcs_bucket
        if not bucket_name:
            raise ValueError("Bucket name must be specified")

        bucket = self.client.bucket(bucket_name)
        blob_name = f"{self.config.gcs_prefix}/{blob_name}"
        blob = bucket.blob(blob_name)

        try:
            if not blob.exists():
                raise NotFound(f"Blob not found: gs://{bucket_name}/{blob_name}")

            # Use a BytesIO buffer for streaming
            with io.BytesIO() as buffer:
                blob.download_to_file(buffer, retry=self.retry_config)
                buffer.seek(0)

                while True:
                    chunk = buffer.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        except Exception as e:
            raise GCSDownloadError(f"Failed to stream {blob_name}: {e}")

    def list_objects(
        self,
        prefix: str | None = None,
        bucket_name: str | None = None,
        delimiter: str | None = None,
        max_results: int | None = None,
    ) -> list[StorageObject]:
        """List objects in a bucket.

        Args:
            prefix: Prefix to filter objects
            bucket_name: Bucket name (uses config default if not specified)
            delimiter: Delimiter for hierarchical listing
            max_results: Maximum number of results to return

        Returns:
            List of StorageObject instances
        """
        bucket_name = bucket_name or self.config.gcs_bucket
        if not bucket_name:
            raise ValueError("Bucket name must be specified")

        bucket = self.client.bucket(bucket_name)

        # Add configured prefix
        full_prefix = f"{self.config.gcs_prefix}/{prefix}" if prefix else self.config.gcs_prefix

        objects = []
        blobs = bucket.list_blobs(prefix=full_prefix, delimiter=delimiter, max_results=max_results)

        for blob in blobs:
            objects.append(
                StorageObject(
                    bucket=bucket_name,
                    name=blob.name,
                    size=blob.size,
                    content_type=blob.content_type or "application/octet-stream",
                    etag=blob.etag,
                    created=blob.time_created,
                    updated=blob.updated,
                    metadata=blob.metadata or {},
                    md5_hash=blob.md5_hash,
                    crc32c=blob.crc32c,
                    generation=blob.generation,
                    metageneration=blob.metageneration,
                )
            )

        return objects

    def delete_object(self, blob_name: str, bucket_name: str | None = None) -> bool:
        """Delete an object from GCS.

        Args:
            blob_name: Blob name to delete
            bucket_name: Bucket name (uses config default if not specified)

        Returns:
            True if deleted successfully
        """
        bucket_name = bucket_name or self.config.gcs_bucket
        if not bucket_name:
            raise ValueError("Bucket name must be specified")

        bucket = self.client.bucket(bucket_name)
        blob_name = f"{self.config.gcs_prefix}/{blob_name}"
        blob = bucket.blob(blob_name)

        try:
            blob.delete()
            logger.info(f"Deleted gs://{bucket_name}/{blob_name}")
            return True
        except NotFound:
            logger.warning(f"Blob not found: gs://{bucket_name}/{blob_name}")
            return False
        except Exception as e:
            raise GCSError(f"Failed to delete {blob_name}: {e}")

    def copy_object(
        self,
        source_blob_name: str,
        destination_blob_name: str,
        source_bucket: str | None = None,
        destination_bucket: str | None = None,
    ) -> StorageObject:
        """Copy an object within or between buckets.

        Args:
            source_blob_name: Source blob name
            destination_blob_name: Destination blob name
            source_bucket: Source bucket (uses config default if not specified)
            destination_bucket: Destination bucket (uses source bucket if not specified)

        Returns:
            StorageObject for the copied object
        """
        source_bucket = source_bucket or self.config.gcs_bucket
        destination_bucket = destination_bucket or source_bucket

        if not source_bucket:
            raise ValueError("Source bucket must be specified")

        source_bucket_obj = self.client.bucket(source_bucket)
        source_blob_name = f"{self.config.gcs_prefix}/{source_blob_name}"
        source_blob = source_bucket_obj.blob(source_blob_name)

        destination_bucket_obj = self.client.bucket(destination_bucket)
        destination_blob_name = f"{self.config.gcs_prefix}/{destination_blob_name}"

        try:
            destination_blob = source_bucket_obj.copy_blob(
                source_blob, destination_bucket_obj, destination_blob_name
            )

            logger.info(
                f"Copied gs://{source_bucket}/{source_blob_name} to "
                f"gs://{destination_bucket}/{destination_blob_name}"
            )

            return StorageObject(
                bucket=destination_bucket,
                name=destination_blob_name,
                size=destination_blob.size,
                content_type=destination_blob.content_type or "application/octet-stream",
                etag=destination_blob.etag,
                created=destination_blob.time_created,
                updated=destination_blob.updated,
                metadata=destination_blob.metadata or {},
                md5_hash=destination_blob.md5_hash,
                crc32c=destination_blob.crc32c,
                generation=destination_blob.generation,
                metageneration=destination_blob.metageneration,
            )

        except Exception as e:
            raise GCSError(f"Failed to copy {source_blob_name}: {e}")

    def generate_signed_url(
        self,
        blob_name: str,
        expiration: timedelta = timedelta(hours=1),
        method: str = "GET",
        bucket_name: str | None = None,
    ) -> str:
        """Generate a signed URL for temporary access to an object.

        Args:
            blob_name: Blob name
            expiration: URL expiration time
            method: HTTP method (GET, PUT, DELETE)
            bucket_name: Bucket name (uses config default if not specified)

        Returns:
            Signed URL string
        """
        bucket_name = bucket_name or self.config.gcs_bucket
        if not bucket_name:
            raise ValueError("Bucket name must be specified")

        bucket = self.client.bucket(bucket_name)
        blob_name = f"{self.config.gcs_prefix}/{blob_name}"
        blob = bucket.blob(blob_name)

        try:
            url = blob.generate_signed_url(
                version="v4",
                expiration=expiration,
                method=method,
                credentials=self.auth_manager.credentials,
            )
            logger.debug(f"Generated signed URL for gs://{bucket_name}/{blob_name}")
            return url
        except Exception as e:
            raise GCSError(f"Failed to generate signed URL: {e}")

    def upload_directory(
        self,
        source_dir: str | Path,
        destination_prefix: str,
        bucket_name: str | None = None,
        parallel: bool = True,
        exclude_patterns: list[str] | None = None,
    ) -> list[StorageObject]:
        """Upload an entire directory to GCS.

        Args:
            source_dir: Local directory to upload
            destination_prefix: Destination prefix in GCS
            bucket_name: Bucket name (uses config default if not specified)
            parallel: Whether to upload files in parallel
            exclude_patterns: Patterns to exclude from upload

        Returns:
            List of uploaded StorageObjects
        """
        source_dir = Path(source_dir)
        if not source_dir.is_dir():
            raise ValueError(f"Source is not a directory: {source_dir}")

        # Collect files to upload
        files_to_upload = []
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                # Check exclusion patterns
                if exclude_patterns:
                    skip = False
                    for pattern in exclude_patterns:
                        if file_path.match(pattern):
                            skip = True
                            break
                    if skip:
                        continue

                relative_path = file_path.relative_to(source_dir)
                destination_name = f"{destination_prefix}/{relative_path}".replace("\\", "/")
                files_to_upload.append((file_path, destination_name))

        # Upload files
        uploaded_objects = []

        if parallel and len(files_to_upload) > 1:
            # Parallel upload
            futures = []
            for source_path, dest_name in files_to_upload:
                future = self.executor.submit(self.upload_file, source_path, dest_name, bucket_name)
                futures.append(future)

            for future in as_completed(futures):
                try:
                    obj = future.result()
                    uploaded_objects.append(obj)
                except Exception as e:
                    logger.error(f"Failed to upload file: {e}")
        else:
            # Sequential upload
            for source_path, dest_name in files_to_upload:
                try:
                    obj = self.upload_file(source_path, dest_name, bucket_name)
                    uploaded_objects.append(obj)
                except Exception as e:
                    logger.error(f"Failed to upload {source_path}: {e}")

        logger.info(f"Uploaded {len(uploaded_objects)} files from {source_dir}")
        return uploaded_objects

    def get_object_metadata(self, blob_name: str, bucket_name: str | None = None) -> dict[str, Any]:
        """Get metadata for an object.

        Args:
            blob_name: Blob name
            bucket_name: Bucket name (uses config default if not specified)

        Returns:
            Object metadata dictionary
        """
        bucket_name = bucket_name or self.config.gcs_bucket
        if not bucket_name:
            raise ValueError("Bucket name must be specified")

        bucket = self.client.bucket(bucket_name)
        blob_name = f"{self.config.gcs_prefix}/{blob_name}"
        blob = bucket.blob(blob_name)

        try:
            blob.reload()
            return {
                "name": blob.name,
                "size": blob.size,
                "content_type": blob.content_type,
                "etag": blob.etag,
                "created": blob.time_created.isoformat() if blob.time_created else None,
                "updated": blob.updated.isoformat() if blob.updated else None,
                "md5_hash": blob.md5_hash,
                "crc32c": blob.crc32c,
                "generation": blob.generation,
                "metadata": blob.metadata,
            }
        except NotFound:
            raise GCSError(f"Object not found: gs://{bucket_name}/{blob_name}")
        except Exception as e:
            raise GCSError(f"Failed to get metadata: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.executor.shutdown(wait=False)
