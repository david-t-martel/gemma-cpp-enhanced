"""
Gemma Model Download and Management.

This module handles downloading, caching, and managing Gemma models from Kaggle and Hugging Face,
with integration to Google Cloud Storage for model persistence.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from datetime import timezone
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from .auth import GCPAuthManager
from .config import GCPConfig
from .storage import GCSStorageManager
from .storage import StorageObject


class ModelSource(Enum):
    """Model download sources."""

    KAGGLE = "kaggle"
    HUGGINGFACE = "huggingface"
    GCS = "gcs"
    LOCAL = "local"


class ModelVariant(Enum):
    """Gemma model variants."""

    GEMMA_2B = "gemma-2b"
    GEMMA_2B_IT = "gemma-2b-it"  # Instruction-tuned
    GEMMA_7B = "gemma-7b"
    GEMMA_7B_IT = "gemma-7b-it"  # Instruction-tuned
    GEMMA_2_2B = "gemma-2-2b"
    GEMMA_2_2B_IT = "gemma-2-2b-it"
    GEMMA_2_9B = "gemma-2-9b"
    GEMMA_2_9B_IT = "gemma-2-9b-it"
    GEMMA_2_27B = "gemma-2-27b"
    GEMMA_2_27B_IT = "gemma-2-27b-it"
    CODEGEMMA_2B = "codegemma-2b"
    CODEGEMMA_7B = "codegemma-7b"
    CODEGEMMA_7B_IT = "codegemma-7b-it"


@dataclass
class ModelInfo:
    """Information about a Gemma model."""

    variant: ModelVariant
    source: ModelSource
    size_gb: float
    files: list[str]
    kaggle_path: str | None = None
    huggingface_repo: str | None = None
    gcs_path: str | None = None
    local_path: Path | None = None
    checksum: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Model registry with download information
MODEL_REGISTRY = {
    ModelVariant.GEMMA_2B: ModelInfo(
        variant=ModelVariant.GEMMA_2B,
        source=ModelSource.KAGGLE,
        size_gb=5.0,
        files=["model.safetensors", "config.json", "tokenizer.json"],
        kaggle_path="google/gemma/keras/gemma_2b_en",
        huggingface_repo="google/gemma-2b",
    ),
    ModelVariant.GEMMA_2B_IT: ModelInfo(
        variant=ModelVariant.GEMMA_2B_IT,
        source=ModelSource.KAGGLE,
        size_gb=5.0,
        files=["model.safetensors", "config.json", "tokenizer.json"],
        kaggle_path="google/gemma/keras/gemma_instruct_2b_en",
        huggingface_repo="google/gemma-2b-it",
    ),
    ModelVariant.GEMMA_7B: ModelInfo(
        variant=ModelVariant.GEMMA_7B,
        source=ModelSource.KAGGLE,
        size_gb=17.0,
        files=["model.safetensors", "config.json", "tokenizer.json"],
        kaggle_path="google/gemma/keras/gemma_7b_en",
        huggingface_repo="google/gemma-7b",
    ),
    ModelVariant.GEMMA_7B_IT: ModelInfo(
        variant=ModelVariant.GEMMA_7B_IT,
        source=ModelSource.KAGGLE,
        size_gb=17.0,
        files=["model.safetensors", "config.json", "tokenizer.json"],
        kaggle_path="google/gemma/keras/gemma_instruct_7b_en",
        huggingface_repo="google/gemma-7b-it",
    ),
    ModelVariant.GEMMA_2_2B: ModelInfo(
        variant=ModelVariant.GEMMA_2_2B,
        source=ModelSource.HUGGINGFACE,
        size_gb=5.4,
        files=["model.safetensors", "config.json", "tokenizer.json"],
        huggingface_repo="google/gemma-2-2b",
    ),
    ModelVariant.GEMMA_2_2B_IT: ModelInfo(
        variant=ModelVariant.GEMMA_2_2B_IT,
        source=ModelSource.HUGGINGFACE,
        size_gb=5.4,
        files=["model.safetensors", "config.json", "tokenizer.json"],
        huggingface_repo="google/gemma-2-2b-it",
    ),
    ModelVariant.GEMMA_2_9B: ModelInfo(
        variant=ModelVariant.GEMMA_2_9B,
        source=ModelSource.HUGGINGFACE,
        size_gb=18.5,
        files=["model.safetensors", "config.json", "tokenizer.json"],
        huggingface_repo="google/gemma-2-9b",
    ),
    ModelVariant.GEMMA_2_9B_IT: ModelInfo(
        variant=ModelVariant.GEMMA_2_9B_IT,
        source=ModelSource.HUGGINGFACE,
        size_gb=18.5,
        files=["model.safetensors", "config.json", "tokenizer.json"],
        huggingface_repo="google/gemma-2-9b-it",
    ),
    ModelVariant.GEMMA_2_27B: ModelInfo(
        variant=ModelVariant.GEMMA_2_27B,
        source=ModelSource.HUGGINGFACE,
        size_gb=54.0,
        files=["model.safetensors", "config.json", "tokenizer.json"],
        huggingface_repo="google/gemma-2-27b",
    ),
    ModelVariant.GEMMA_2_27B_IT: ModelInfo(
        variant=ModelVariant.GEMMA_2_27B_IT,
        source=ModelSource.HUGGINGFACE,
        size_gb=54.0,
        files=["model.safetensors", "config.json", "tokenizer.json"],
        huggingface_repo="google/gemma-2-27b-it",
    ),
    ModelVariant.CODEGEMMA_2B: ModelInfo(
        variant=ModelVariant.CODEGEMMA_2B,
        source=ModelSource.HUGGINGFACE,
        size_gb=5.0,
        files=["model.safetensors", "config.json", "tokenizer.json"],
        huggingface_repo="google/codegemma-2b",
    ),
    ModelVariant.CODEGEMMA_7B: ModelInfo(
        variant=ModelVariant.CODEGEMMA_7B,
        source=ModelSource.HUGGINGFACE,
        size_gb=17.0,
        files=["model.safetensors", "config.json", "tokenizer.json"],
        huggingface_repo="google/codegemma-7b",
    ),
    ModelVariant.CODEGEMMA_7B_IT: ModelInfo(
        variant=ModelVariant.CODEGEMMA_7B_IT,
        source=ModelSource.HUGGINGFACE,
        size_gb=17.0,
        files=["model.safetensors", "config.json", "tokenizer.json"],
        huggingface_repo="google/codegemma-7b-it",
    ),
}


class ModelDownloadError(Exception):
    """Error during model download."""


class GemmaModelManager:
    """Manages Gemma model downloads and caching."""

    def __init__(
        self,
        config: GCPConfig | None = None,
        auth_manager: GCPAuthManager | None = None,
        storage_manager: GCSStorageManager | None = None,
    ):
        """Initialize Gemma Model Manager.

        Args:
            config: GCP configuration
            auth_manager: Authentication manager
            storage_manager: GCS storage manager
        """
        self.config = config or GCPConfig.from_env()
        self.auth_manager = auth_manager or GCPAuthManager(self.config)
        self.storage_manager = storage_manager or GCSStorageManager(self.config, self.auth_manager)

        # Ensure cache directory exists
        self.config.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Model metadata cache
        self._metadata_cache = {}
        self._load_metadata_cache()

    def _load_metadata_cache(self):
        """Load cached model metadata."""
        metadata_file = self.config.model_cache_dir / "models_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, encoding="utf-8") as f:
                    self._metadata_cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata cache: {e}")

    def _save_metadata_cache(self):
        """Save model metadata cache."""
        metadata_file = self.config.model_cache_dir / "models_metadata.json"
        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(self._metadata_cache, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metadata cache: {e}")

    def list_available_models(self) -> list[ModelInfo]:
        """List all available Gemma models.

        Returns:
            List of available model information
        """
        return list(MODEL_REGISTRY.values())

    def get_model_info(self, variant: ModelVariant) -> ModelInfo:
        """Get information about a specific model variant.

        Args:
            variant: Model variant

        Returns:
            Model information
        """
        if variant not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model variant: {variant}")
        return MODEL_REGISTRY[variant]

    def is_model_cached(self, variant: ModelVariant) -> bool:
        """Check if a model is cached locally.

        Args:
            variant: Model variant

        Returns:
            True if model is cached
        """
        model_dir = self.config.model_cache_dir / variant.value
        if not model_dir.exists():
            return False

        model_info = self.get_model_info(variant)
        return all((model_dir / file_name).exists() for file_name in model_info.files)

    def download_from_kaggle(self, variant: ModelVariant, force: bool = False) -> Path:
        """Download model from Kaggle.

        Args:
            variant: Model variant to download
            force: Force re-download even if cached

        Returns:
            Path to downloaded model directory
        """
        if not self.config.kaggle_username or not self.config.kaggle_key:
            raise ModelDownloadError("Kaggle credentials not configured")

        model_info = self.get_model_info(variant)
        if not model_info.kaggle_path:
            raise ModelDownloadError(f"No Kaggle path for {variant.value}")

        model_dir = self.config.model_cache_dir / variant.value

        # Check if already cached
        if not force and self.is_model_cached(variant):
            logger.info(f"Model {variant.value} already cached")
            return model_dir

        # Create model directory
        model_dir.mkdir(parents=True, exist_ok=True)

        # Set Kaggle credentials
        os.environ["KAGGLE_USERNAME"] = self.config.kaggle_username
        os.environ["KAGGLE_KEY"] = self.config.kaggle_key

        try:
            # Download using Kaggle API
            logger.info(f"Downloading {variant.value} from Kaggle...")

            # Try to import kaggle
            try:
                import kaggle
            except ImportError:
                # Install kaggle if not available
                subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
                import kaggle

            # Download dataset
            kaggle.api.dataset_download_files(
                model_info.kaggle_path, path=str(model_dir), unzip=True, quiet=False
            )

            # Update metadata
            self._update_model_metadata(variant, model_dir)

            logger.info(f"Successfully downloaded {variant.value}")
            return model_dir

        except Exception as e:
            # Clean up partial download
            if model_dir.exists():
                shutil.rmtree(model_dir)
            raise ModelDownloadError(f"Failed to download from Kaggle: {e}")

    def download_from_huggingface(self, variant: ModelVariant, force: bool = False) -> Path:
        """Download model from Hugging Face.

        Args:
            variant: Model variant to download
            force: Force re-download even if cached

        Returns:
            Path to downloaded model directory
        """
        model_info = self.get_model_info(variant)
        if not model_info.huggingface_repo:
            raise ModelDownloadError(f"No Hugging Face repo for {variant.value}")

        model_dir = self.config.model_cache_dir / variant.value

        # Check if already cached
        if not force and self.is_model_cached(variant):
            logger.info(f"Model {variant.value} already cached")
            return model_dir

        # Create model directory
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Downloading {variant.value} from Hugging Face...")

            # Try to use huggingface_hub
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                # Install huggingface_hub if not available
                subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub"])
                from huggingface_hub import snapshot_download

            # Download model
            snapshot_download(
                repo_id=model_info.huggingface_repo,
                local_dir=str(model_dir),
                token=self.config.huggingface_token,
                local_dir_use_symlinks=False,
                resume_download=True,
            )

            # Update metadata
            self._update_model_metadata(variant, model_dir)

            logger.info(f"Successfully downloaded {variant.value}")
            return model_dir

        except Exception as e:
            # Clean up partial download
            if model_dir.exists():
                shutil.rmtree(model_dir)
            raise ModelDownloadError(f"Failed to download from Hugging Face: {e}")

    def download_from_gcs(self, variant: ModelVariant, gcs_path: str, force: bool = False) -> Path:
        """Download model from Google Cloud Storage.

        Args:
            variant: Model variant to download
            gcs_path: GCS path to model (e.g., "models/gemma-7b")
            force: Force re-download even if cached

        Returns:
            Path to downloaded model directory
        """
        model_dir = self.config.model_cache_dir / variant.value

        # Check if already cached
        if not force and self.is_model_cached(variant):
            logger.info(f"Model {variant.value} already cached")
            return model_dir

        # Create model directory
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Downloading {variant.value} from GCS...")

            model_info = self.get_model_info(variant)

            # Download each file
            for file_name in model_info.files:
                gcs_file_path = f"{gcs_path}/{file_name}"
                local_file_path = model_dir / file_name

                self.storage_manager.download_file(gcs_file_path, local_file_path)

            # Update metadata
            self._update_model_metadata(variant, model_dir)

            logger.info(f"Successfully downloaded {variant.value} from GCS")
            return model_dir

        except Exception as e:
            # Clean up partial download
            if model_dir.exists():
                shutil.rmtree(model_dir)
            raise ModelDownloadError(f"Failed to download from GCS: {e}")

    def upload_to_gcs(
        self, variant: ModelVariant, gcs_path: str | None = None
    ) -> list[StorageObject]:
        """Upload a cached model to GCS.

        Args:
            variant: Model variant to upload
            gcs_path: GCS path (defaults to model variant name)

        Returns:
            List of uploaded storage objects
        """
        if not self.is_model_cached(variant):
            raise ValueError(f"Model {variant.value} is not cached locally")

        model_dir = self.config.model_cache_dir / variant.value
        gcs_path = gcs_path or f"models/{variant.value}"

        logger.info(f"Uploading {variant.value} to GCS...")

        uploaded_objects = self.storage_manager.upload_directory(
            model_dir, gcs_path, exclude_patterns=["*.pyc", "__pycache__", ".DS_Store"]
        )

        # Update model info with GCS path
        model_info = self.get_model_info(variant)
        model_info.gcs_path = gcs_path

        logger.info(f"Successfully uploaded {variant.value} to GCS")
        return uploaded_objects

    def download_model(
        self, variant: ModelVariant, source: ModelSource | None = None, force: bool = False
    ) -> Path:
        """Download a model from the best available source.

        Args:
            variant: Model variant to download
            source: Preferred source (auto-detect if not specified)
            force: Force re-download even if cached

        Returns:
            Path to downloaded model directory
        """
        # Check if already cached
        if not force and self.is_model_cached(variant):
            logger.info(f"Model {variant.value} already cached")
            return self.config.model_cache_dir / variant.value

        model_info = self.get_model_info(variant)

        # Determine source
        if source is None:
            source = model_info.source

        # Try downloading from specified source
        if source == ModelSource.KAGGLE:
            return self.download_from_kaggle(variant, force)
        elif source == ModelSource.HUGGINGFACE:
            return self.download_from_huggingface(variant, force)
        elif source == ModelSource.GCS and model_info.gcs_path:
            return self.download_from_gcs(variant, model_info.gcs_path, force)
        else:
            # Try all available sources
            if model_info.kaggle_path and self.config.kaggle_username:
                try:
                    return self.download_from_kaggle(variant, force)
                except ModelDownloadError as e:
                    logger.warning(f"Kaggle download failed: {e}")

            if model_info.huggingface_repo:
                try:
                    return self.download_from_huggingface(variant, force)
                except ModelDownloadError as e:
                    logger.warning(f"Hugging Face download failed: {e}")

            if model_info.gcs_path:
                try:
                    return self.download_from_gcs(variant, model_info.gcs_path, force)
                except ModelDownloadError as e:
                    logger.warning(f"GCS download failed: {e}")

            raise ModelDownloadError(f"Failed to download {variant.value} from any source")

    def verify_model(self, variant: ModelVariant) -> bool:
        """Verify model integrity.

        Args:
            variant: Model variant to verify

        Returns:
            True if model is valid
        """
        if not self.is_model_cached(variant):
            return False

        model_dir = self.config.model_cache_dir / variant.value
        model_info = self.get_model_info(variant)

        # Check all required files exist
        for file_name in model_info.files:
            file_path = model_dir / file_name
            if not file_path.exists():
                logger.error(f"Missing file: {file_path}")
                return False

            # Check file size is reasonable
            file_size = file_path.stat().st_size
            if file_size < 1000:  # Less than 1KB is suspicious
                logger.error(f"File too small: {file_path} ({file_size} bytes)")
                return False

        # Check model config is valid JSON
        config_path = model_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid config.json: {e}")
                return False

        logger.info(f"Model {variant.value} verified successfully")
        return True

    def clean_cache(self, keep_variants: list[ModelVariant] | None = None):
        """Clean model cache.

        Args:
            keep_variants: List of variants to keep (clean all if None)
        """
        if keep_variants is None:
            keep_variants = []

        for variant in ModelVariant:
            if variant not in keep_variants:
                model_dir = self.config.model_cache_dir / variant.value
                if model_dir.exists():
                    logger.info(f"Removing {variant.value} from cache")
                    shutil.rmtree(model_dir)

        # Clean metadata for removed models
        for variant_str in list(self._metadata_cache.keys()):
            variant = ModelVariant(variant_str)
            if variant not in keep_variants:
                del self._metadata_cache[variant_str]

        self._save_metadata_cache()

    def get_model_path(self, variant: ModelVariant) -> Path | None:
        """Get local path to a cached model.

        Args:
            variant: Model variant

        Returns:
            Path to model directory if cached, None otherwise
        """
        if self.is_model_cached(variant):
            return self.config.model_cache_dir / variant.value
        return None

    def _update_model_metadata(self, variant: ModelVariant, model_dir: Path):
        """Update model metadata cache.

        Args:
            variant: Model variant
            model_dir: Model directory
        """
        metadata = {
            "variant": variant.value,
            "path": str(model_dir),
            "downloaded_at": datetime.now(UTC).isoformat(),
            "size_bytes": sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()),
            "files": [f.name for f in model_dir.iterdir() if f.is_file()],
        }

        self._metadata_cache[variant.value] = metadata
        self._save_metadata_cache()

    def get_cache_size(self) -> float:
        """Get total size of cached models in GB.

        Returns:
            Total cache size in GB
        """
        total_size = 0
        for model_dir in self.config.model_cache_dir.iterdir():
            if model_dir.is_dir():
                for file_path in model_dir.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size

        return total_size / (1024**3)  # Convert to GB

    def list_cached_models(self) -> list[tuple[ModelVariant, dict[str, Any]]]:
        """List all cached models with metadata.

        Returns:
            List of (variant, metadata) tuples
        """
        cached_models = []
        for variant in ModelVariant:
            if self.is_model_cached(variant):
                metadata = self._metadata_cache.get(variant.value, {})
                cached_models.append((variant, metadata))
        return cached_models
