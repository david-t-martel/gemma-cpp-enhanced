"""
Google Cloud Platform Integration Package.

This package provides comprehensive GCP integration including:
- Service account authentication
- Google Cloud Storage operations
- Gemma model management
- Configuration management
"""

from .auth import AuthenticationError
from .auth import GCPAuthManager
from .auth import TokenCache
from .config import AuthMethod
from .config import GCPConfig
from .config import GCPRegion
from .config import ServiceAccountConfig
from .config import WorkloadIdentityConfig
from .config import get_config
from .config import validate_config
from .gemma import MODEL_REGISTRY
from .gemma import GemmaModelManager
from .gemma import ModelDownloadError
from .gemma import ModelInfo
from .gemma import ModelSource
from .gemma import ModelVariant
from .gemma_download import GEMMA_MODELS
from .gemma_download import GemmaDownloader
from .storage import GCSDownloadError
from .storage import GCSError
from .storage import GCSStorageManager
from .storage import GCSUploadError
from .storage import StorageObject

__all__ = [
    "GEMMA_MODELS",
    "MODEL_REGISTRY",
    "AuthMethod",
    "AuthenticationError",
    # Auth
    "GCPAuthManager",
    # Config
    "GCPConfig",
    "GCPRegion",
    "GCSDownloadError",
    "GCSError",
    # Storage
    "GCSStorageManager",
    "GCSUploadError",
    # Simple Downloader
    "GemmaDownloader",
    # Gemma
    "GemmaModelManager",
    "ModelDownloadError",
    "ModelInfo",
    "ModelSource",
    "ModelVariant",
    "ServiceAccountConfig",
    "StorageObject",
    "TokenCache",
    "WorkloadIdentityConfig",
    "get_config",
    "validate_config",
]

__version__ = "1.0.0"
