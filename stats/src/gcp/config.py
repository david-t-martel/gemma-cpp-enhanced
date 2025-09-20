"""
GCP Configuration and Credentials Management.

This module handles Google Cloud Platform configuration, including service account
authentication, credential management, and regional endpoint configuration.
"""

import json
import os
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union


class AuthMethod(Enum):
    """GCP Authentication methods."""

    SERVICE_ACCOUNT = "service_account"
    APPLICATION_DEFAULT = "application_default"
    WORKLOAD_IDENTITY = "workload_identity"
    USER_CREDENTIALS = "user"
    COMPUTE_ENGINE = "compute_engine"


class GCPRegion(Enum):
    """Common GCP regions with endpoint information."""

    US_CENTRAL1 = "us-central1"
    US_EAST1 = "us-east1"
    US_EAST4 = "us-east4"
    US_WEST1 = "us-west1"
    US_WEST2 = "us-west2"
    US_WEST3 = "us-west3"
    US_WEST4 = "us-west4"
    EUROPE_WEST1 = "europe-west1"
    EUROPE_WEST2 = "europe-west2"
    EUROPE_WEST3 = "europe-west3"
    EUROPE_WEST4 = "europe-west4"
    EUROPE_NORTH1 = "europe-north1"
    ASIA_EAST1 = "asia-east1"
    ASIA_EAST2 = "asia-east2"
    ASIA_NORTHEAST1 = "asia-northeast1"
    ASIA_NORTHEAST2 = "asia-northeast2"
    ASIA_NORTHEAST3 = "asia-northeast3"
    ASIA_SOUTH1 = "asia-south1"
    ASIA_SOUTHEAST1 = "asia-southeast1"
    ASIA_SOUTHEAST2 = "asia-southeast2"
    AUSTRALIA_SOUTHEAST1 = "australia-southeast1"
    AUSTRALIA_SOUTHEAST2 = "australia-southeast2"
    SOUTHAMERICA_EAST1 = "southamerica-east1"
    NORTHAMERICA_NORTHEAST1 = "northamerica-northeast1"
    NORTHAMERICA_NORTHEAST2 = "northamerica-northeast2"


@dataclass
class ServiceAccountConfig:
    """Service account configuration."""

    client_email: str
    client_id: str
    private_key: str
    private_key_id: str
    project_id: str
    auth_uri: str = "https://accounts.google.com/o/oauth2/auth"
    token_uri: str = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url: str = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url: str | None = None
    type: str = "service_account"
    universe_domain: str = "googleapis.com"

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ServiceAccountConfig":
        """Load service account config from JSON key file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Service account key file not found: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            return cls(
                client_email=data["client_email"],
                client_id=data["client_id"],
                private_key=data["private_key"],
                private_key_id=data["private_key_id"],
                project_id=data["project_id"],
                auth_uri=data.get("auth_uri", cls.auth_uri),
                token_uri=data.get("token_uri", cls.token_uri),
                auth_provider_x509_cert_url=data.get(
                    "auth_provider_x509_cert_url", cls.auth_provider_x509_cert_url
                ),
                client_x509_cert_url=data.get("client_x509_cert_url"),
                type=data.get("type", cls.type),
                universe_domain=data.get("universe_domain", cls.universe_domain),
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid service account key file: {e}")

    @classmethod
    def from_env(cls) -> Optional["ServiceAccountConfig"]:
        """Load service account config from environment variables."""
        # Check for JSON key file path in env
        key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if key_path and Path(key_path).exists():
            return cls.from_json_file(key_path)

        # Check for individual service account fields in env
        required_fields = [
            "GCP_CLIENT_EMAIL",
            "GCP_CLIENT_ID",
            "GCP_PRIVATE_KEY",
            "GCP_PRIVATE_KEY_ID",
            "GCP_PROJECT_ID",
        ]

        if all(os.getenv(field) for field in required_fields):
            return cls(
                client_email=os.getenv("GCP_CLIENT_EMAIL"),
                client_id=os.getenv("GCP_CLIENT_ID"),
                private_key=os.getenv("GCP_PRIVATE_KEY").replace("\\n", "\n"),
                private_key_id=os.getenv("GCP_PRIVATE_KEY_ID"),
                project_id=os.getenv("GCP_PROJECT_ID"),
                auth_uri=os.getenv("GCP_AUTH_URI", cls.auth_uri),
                token_uri=os.getenv("GCP_TOKEN_URI", cls.token_uri),
                auth_provider_x509_cert_url=os.getenv(
                    "GCP_AUTH_PROVIDER_CERT_URL", cls.auth_provider_x509_cert_url
                ),
                client_x509_cert_url=os.getenv("GCP_CLIENT_CERT_URL"),
                type=os.getenv("GCP_AUTH_TYPE", cls.type),
                universe_domain=os.getenv("GCP_UNIVERSE_DOMAIN", cls.universe_domain),
            )

        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for use with Google client libraries."""
        return {
            "type": self.type,
            "project_id": self.project_id,
            "private_key_id": self.private_key_id,
            "private_key": self.private_key,
            "client_email": self.client_email,
            "client_id": self.client_id,
            "auth_uri": self.auth_uri,
            "token_uri": self.token_uri,
            "auth_provider_x509_cert_url": self.auth_provider_x509_cert_url,
            "client_x509_cert_url": self.client_x509_cert_url,
            "universe_domain": self.universe_domain,
        }


@dataclass
class WorkloadIdentityConfig:
    """Workload Identity Federation configuration."""

    project_number: str
    pool_id: str
    provider_id: str
    service_account_email: str
    credential_source: dict[str, Any] = field(default_factory=dict)
    subject_token_type: str = "urn:ietf:params:oauth:token-type:jwt"
    token_url: str = "https://sts.googleapis.com/v1/token"
    service_account_impersonation_url: str | None = None

    @classmethod
    def from_json_file(cls, path: str | Path) -> "WorkloadIdentityConfig":
        """Load workload identity config from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Workload identity config file not found: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            return cls(
                project_number=data["project_number"],
                pool_id=data["pool_id"],
                provider_id=data["provider_id"],
                service_account_email=data["service_account_email"],
                credential_source=data.get("credential_source", {}),
                subject_token_type=data.get("subject_token_type", cls.subject_token_type),
                token_url=data.get("token_url", cls.token_url),
                service_account_impersonation_url=data.get("service_account_impersonation_url"),
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid workload identity config file: {e}")


@dataclass
class GCPConfig:
    """Main GCP configuration."""

    project_id: str
    region: GCPRegion = GCPRegion.US_CENTRAL1
    auth_method: AuthMethod = AuthMethod.APPLICATION_DEFAULT
    service_account_config: ServiceAccountConfig | None = None
    workload_identity_config: WorkloadIdentityConfig | None = None

    # Storage configuration
    gcs_bucket: str | None = None
    gcs_prefix: str = "gemma-models"

    # Model configuration
    model_cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "gemma")
    kaggle_username: str | None = None
    kaggle_key: str | None = None
    huggingface_token: str | None = None

    # API configuration
    api_endpoint_override: str | None = None
    api_timeout: int = 30
    api_retry_count: int = 3
    api_retry_delay: float = 1.0

    # Security configuration
    enable_audit_logging: bool = True
    enable_encryption: bool = True
    kms_key_name: str | None = None

    @classmethod
    def from_env(cls) -> "GCPConfig":
        """Create configuration from environment variables."""
        project_id = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT"))
        if not project_id:
            raise ValueError(
                "GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT environment variable is required"
            )

        region_str = os.getenv("GCP_REGION", "us-central1")
        try:
            region = GCPRegion(region_str)
        except ValueError:
            logger.warning(f"Unknown region {region_str}, using us-central1")
            region = GCPRegion.US_CENTRAL1

        # Determine auth method
        auth_method = AuthMethod.APPLICATION_DEFAULT
        service_account_config = None
        workload_identity_config = None

        if os.getenv("GCP_AUTH_METHOD"):
            auth_method = AuthMethod(os.getenv("GCP_AUTH_METHOD"))

        if auth_method == AuthMethod.SERVICE_ACCOUNT:
            service_account_config = ServiceAccountConfig.from_env()
            if not service_account_config:
                raise ValueError("Service account configuration not found in environment")
        elif auth_method == AuthMethod.WORKLOAD_IDENTITY:
            wif_config_path = os.getenv("GCP_WORKLOAD_IDENTITY_CONFIG")
            if wif_config_path:
                workload_identity_config = WorkloadIdentityConfig.from_json_file(wif_config_path)

        # Model cache directory
        cache_dir = os.getenv("GEMMA_MODEL_CACHE_DIR")
        model_cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "gemma"

        return cls(
            project_id=project_id,
            region=region,
            auth_method=auth_method,
            service_account_config=service_account_config,
            workload_identity_config=workload_identity_config,
            gcs_bucket=os.getenv("GCP_GCS_BUCKET"),
            gcs_prefix=os.getenv("GCP_GCS_PREFIX", "gemma-models"),
            model_cache_dir=model_cache_dir,
            kaggle_username=os.getenv("KAGGLE_USERNAME"),
            kaggle_key=os.getenv("KAGGLE_KEY"),
            huggingface_token=os.getenv("HUGGINGFACE_TOKEN"),
            api_endpoint_override=os.getenv("GCP_API_ENDPOINT"),
            api_timeout=int(os.getenv("GCP_API_TIMEOUT", "30")),
            api_retry_count=int(os.getenv("GCP_API_RETRY_COUNT", "3")),
            api_retry_delay=float(os.getenv("GCP_API_RETRY_DELAY", "1.0")),
            enable_audit_logging=os.getenv("GCP_ENABLE_AUDIT_LOGGING", "true").lower() == "true",
            enable_encryption=os.getenv("GCP_ENABLE_ENCRYPTION", "true").lower() == "true",
            kms_key_name=os.getenv("GCP_KMS_KEY_NAME"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GCPConfig":
        """Load configuration from YAML file."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Process service account config if present
        service_account_config = None
        if "service_account" in data:
            sa_data = data["service_account"]
            if isinstance(sa_data, str):
                # Path to service account key file
                service_account_config = ServiceAccountConfig.from_json_file(sa_data)
            else:
                # Inline service account configuration
                service_account_config = ServiceAccountConfig(**sa_data)

        # Process workload identity config if present
        workload_identity_config = None
        if "workload_identity" in data:
            wi_data = data["workload_identity"]
            if isinstance(wi_data, str):
                # Path to workload identity config file
                workload_identity_config = WorkloadIdentityConfig.from_json_file(wi_data)
            else:
                # Inline workload identity configuration
                workload_identity_config = WorkloadIdentityConfig(**wi_data)

        # Convert region string to enum
        region = GCPRegion(data.get("region", "us-central1"))

        # Convert auth method string to enum
        auth_method = AuthMethod(data.get("auth_method", "application_default"))

        # Create config object
        return cls(
            project_id=data["project_id"],
            region=region,
            auth_method=auth_method,
            service_account_config=service_account_config,
            workload_identity_config=workload_identity_config,
            gcs_bucket=data.get("gcs_bucket"),
            gcs_prefix=data.get("gcs_prefix", "gemma-models"),
            model_cache_dir=Path(data.get("model_cache_dir", Path.home() / ".cache" / "gemma")),
            kaggle_username=data.get("kaggle_username"),
            kaggle_key=data.get("kaggle_key"),
            huggingface_token=data.get("huggingface_token"),
            api_endpoint_override=data.get("api_endpoint_override"),
            api_timeout=data.get("api_timeout", 30),
            api_retry_count=data.get("api_retry_count", 3),
            api_retry_delay=data.get("api_retry_delay", 1.0),
            enable_audit_logging=data.get("enable_audit_logging", True),
            enable_encryption=data.get("enable_encryption", True),
            kms_key_name=data.get("kms_key_name"),
        )

    def get_regional_endpoint(self, service: str) -> str:
        """Get regional endpoint for a GCP service."""
        region_value = self.region.value

        if self.api_endpoint_override:
            return self.api_endpoint_override

        # Common regional endpoints
        endpoints = {
            "storage": f"https://storage.{region_value}.googleapis.com",
            "compute": f"https://compute.{region_value}.googleapis.com",
            "aiplatform": f"https://{region_value}-aiplatform.googleapis.com",
            "vertexai": f"https://{region_value}-aiplatform.googleapis.com",
            "kms": f"https://cloudkms.{region_value}.googleapis.com",
            "logging": f"https://logging.{region_value}.googleapis.com",
            "monitoring": f"https://monitoring.{region_value}.googleapis.com",
        }

        return endpoints.get(service, f"https://{service}.{region_value}.googleapis.com")


def find_gcp_profile_service_accounts() -> list[Path]:
    """Find gcp-profile business service account files in common locations."""
    import platform

    # Define search patterns for gcp-profile service accounts
    patterns = ["gcp-profile*.json", "gcp-profile*credentials*.json", "*gcp-profile*.json"]

    # Define search locations based on platform
    locations = []

    if platform.system() == "Windows":
        # Windows-specific locations
        locations.extend(
            [
                Path.home() / ".config" / "gcloud",
                Path.home() / ".gcp",
                Path.home() / "Downloads",
                Path.home() / "Documents",
                Path(os.getenv("APPDATA", "")) / "gcloud" if os.getenv("APPDATA") else None,
            ]
        )
    else:
        # Linux/macOS locations
        locations.extend(
            [
                Path.home() / ".config" / "gcloud",
                Path.home() / ".gcp",
                Path.home() / "Downloads",
                Path.home() / "Documents",
                Path("/etc/gcp"),
            ]
        )

    # Add current project locations
    current_dir = Path.cwd()
    locations.extend(
        [
            current_dir,
            current_dir / "config",
            current_dir / ".gcp",
        ]
    )

    # Remove None values and ensure paths exist
    locations = [loc for loc in locations if loc and loc.exists()]

    found_files = []

    for location in locations:
        for pattern in patterns:
            try:
                # Search recursively but limit depth to avoid deep traversal
                matches = list(location.glob(pattern))
                # Also check one level deep
                matches.extend(location.glob(f"*/{pattern}"))

                for file_path in matches:
                    if file_path.is_file() and file_path.suffix == ".json":
                        # Validate it's actually a service account file
                        if _is_gcp_profile_service_account(file_path):
                            found_files.append(file_path)
            except (OSError, PermissionError) as e:
                logger.debug(f"Could not search in {location}: {e}")

    return found_files


def _is_gcp_profile_service_account(file_path: Path) -> bool:
    """Validate if a file is a gcp-profile business service account."""
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Check if it's a service account
        if data.get("type") != "service_account":
            return False

        # Check required fields
        required_fields = [
            "project_id",
            "private_key_id",
            "private_key",
            "client_email",
            "client_id",
        ]
        if not all(field in data for field in required_fields):
            return False

        # Check if it's likely a gcp-profile or business service account
        client_email = data.get("client_email", "").lower()

        # Look for gcp-profile or business indicators in email
        profile_indicators = ["gcp-profile", "business", "profile", "analytics", "ml", "gemma"]

        return any(indicator in client_email for indicator in profile_indicators)

    except (json.JSONDecodeError, OSError, KeyError):
        return False


def auto_detect_gcp_profile_config() -> GCPConfig | None:
    """Automatically detect and create configuration from gcp-profile service account."""
    logger.info("Searching for gcp-profile business service account...")

    found_accounts = find_gcp_profile_service_accounts()

    if not found_accounts:
        logger.debug("No gcp-profile service account files found")
        return None

    # Use the first found account (could be enhanced to let user choose)
    sa_file = found_accounts[0]
    logger.info(f"Found gcp-profile service account: {sa_file}")

    try:
        # Load the service account configuration
        sa_config = ServiceAccountConfig.from_json_file(sa_file)

        # Create GCP configuration using the service account
        config = GCPConfig(
            project_id=sa_config.project_id,
            region=GCPRegion.US_CENTRAL1,  # Default, can be overridden
            auth_method=AuthMethod.SERVICE_ACCOUNT,
            service_account_config=sa_config,
            # Set model cache directory
            model_cache_dir=Path.home() / ".cache" / "gemma",
            # Enable security features by default
            enable_audit_logging=True,
            enable_encryption=True,
        )

        logger.info(f"Auto-configured GCP with project: {config.project_id}")
        logger.info(f"Service account: {sa_config.client_email}")

        return config

    except Exception as e:
        logger.error(f"Failed to auto-configure from {sa_file}: {e}")
        return None


@lru_cache(maxsize=1)
def get_config() -> GCPConfig:
    """Get or create GCP configuration singleton."""
    # Try loading from environment first
    try:
        config = GCPConfig.from_env()
        logger.info("Loaded GCP configuration from environment")
        return config
    except ValueError as e:
        logger.debug(f"Could not load config from environment: {e}")

    # Try loading from default config file
    config_paths = [
        Path("config/gcp-config.yaml"),
        Path.home() / ".gcp" / "config.yaml",
        Path("/etc/gcp/config.yaml"),
    ]

    for config_path in config_paths:
        if config_path.exists():
            try:
                config = GCPConfig.from_yaml(config_path)
                logger.info(f"Loaded GCP configuration from {config_path}")
                return config
            except Exception as e:
                logger.debug(f"Could not load config from {config_path}: {e}")

    # Try auto-detecting gcp-profile service account
    auto_config = auto_detect_gcp_profile_config()
    if auto_config:
        logger.info("Auto-detected gcp-profile configuration")
        return auto_config

    # Return minimal default config as last resort
    logger.warning("No GCP configuration found, using defaults")
    return GCPConfig(
        project_id="default-project",
        region=GCPRegion.US_CENTRAL1,
        auth_method=AuthMethod.APPLICATION_DEFAULT,
    )


def validate_config(config: GCPConfig) -> bool:
    """Validate GCP configuration."""
    errors = []

    # Validate project ID
    if not config.project_id or config.project_id == "default-project":
        errors.append("Valid GCP project ID is required")

    # Validate authentication
    if config.auth_method == AuthMethod.SERVICE_ACCOUNT:
        if not config.service_account_config:
            errors.append(
                "Service account configuration is required for SERVICE_ACCOUNT auth method"
            )
    elif config.auth_method == AuthMethod.WORKLOAD_IDENTITY:
        if not config.workload_identity_config:
            errors.append(
                "Workload identity configuration is required for WORKLOAD_IDENTITY auth method"
            )

    # Validate storage configuration if specified
    if config.gcs_bucket and not config.gcs_bucket.replace("-", "").replace("_", "").isalnum():
        errors.append(f"Invalid GCS bucket name: {config.gcs_bucket}")

    # Validate model sources
    if config.kaggle_username and not config.kaggle_key:
        errors.append("Kaggle key is required when Kaggle username is specified")

    if errors:
        for error in errors:
            logger.error(f"Config validation error: {error}")
        return False

    return True
