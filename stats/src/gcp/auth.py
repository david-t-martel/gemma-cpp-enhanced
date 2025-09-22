"""
GCP Service Account Authentication Manager.

This module handles authentication with Google Cloud Platform using various methods
including service accounts, Application Default Credentials, and Workload Identity Federation.
"""

import json
import threading
import time
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from functools import wraps
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import google.auth
from google.auth import compute_engine
from google.auth import default
from google.auth import external_account
from google.auth import impersonated_credentials
from google.auth import jwt
from google.auth.exceptions import DefaultCredentialsError
from google.auth.exceptions import RefreshError
from google.auth.transport import requests as auth_requests
from google.oauth2 import service_account

from .config import AuthMethod
from .config import GCPConfig
from .config import ServiceAccountConfig
from .config import WorkloadIdentityConfig


class AuthenticationError(Exception):
    """Authentication related errors."""


class TokenCache:
    """Thread-safe token cache with automatic refresh."""

    def __init__(self, refresh_threshold: timedelta = timedelta(minutes=5)):
        self._cache: dict[str, tuple[str, datetime]] = {}
        self._lock = threading.Lock()
        self.refresh_threshold = refresh_threshold

    def get(self, key: str) -> str | None:
        """Get token from cache if valid."""
        with self._lock:
            if key in self._cache:
                token, expiry = self._cache[key]
                if expiry - datetime.now(UTC) > self.refresh_threshold:
                    return token
                else:
                    # Token is expired or about to expire
                    del self._cache[key]
        return None

    def set(self, key: str, token: str, expiry: datetime) -> None:
        """Store token in cache."""
        with self._lock:
            self._cache[key] = (token, expiry)

    def clear(self, key: str | None = None) -> None:
        """Clear cache for specific key or all keys."""
        with self._lock:
            if key:
                self._cache.pop(key, None)
            else:
                self._cache.clear()


def retry_on_refresh_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry operations on token refresh errors."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RefreshError as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Token refresh failed (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        time.sleep(delay * (2**attempt))  # Exponential backoff
                    else:
                        logger.error(f"Token refresh failed after {max_retries} attempts")
            raise AuthenticationError(f"Failed to refresh credentials: {last_error}")

        return wrapper

    return decorator


class GCPAuthManager:
    """Manages GCP authentication and credential lifecycle."""

    def __init__(self, config: GCPConfig | None = None):
        """Initialize authentication manager.

        Args:
            config: GCP configuration. If not provided, will attempt to load from environment.
        """
        self.config = config or GCPConfig.from_env()
        self._credentials = None
        self._token_cache = TokenCache()
        self._lock = threading.Lock()
        self._request = auth_requests.Request()

    @property
    def credentials(self):
        """Get or create credentials based on configured auth method."""
        if self._credentials is None:
            with self._lock:
                if self._credentials is None:  # Double-check pattern
                    self._credentials = self._create_credentials()

        # Refresh if needed
        if (
            self._credentials
            and hasattr(self._credentials, "expired")
            and self._credentials.expired
        ):
            self.refresh_credentials()

        return self._credentials

    def _create_credentials(self):
        """Create credentials based on authentication method."""
        logger.info(f"Creating credentials using {self.config.auth_method.value} method")

        if self.config.auth_method == AuthMethod.SERVICE_ACCOUNT:
            return self._create_service_account_credentials()
        elif self.config.auth_method == AuthMethod.APPLICATION_DEFAULT:
            return self._create_adc_credentials()
        elif self.config.auth_method == AuthMethod.WORKLOAD_IDENTITY:
            return self._create_workload_identity_credentials()
        elif self.config.auth_method == AuthMethod.COMPUTE_ENGINE:
            return self._create_compute_engine_credentials()
        elif self.config.auth_method == AuthMethod.USER_CREDENTIALS:
            return self._create_user_credentials()
        else:
            raise AuthenticationError(
                f"Unsupported authentication method: {self.config.auth_method}"
            )

    def _create_service_account_credentials(self):
        """Create service account credentials."""
        if not self.config.service_account_config:
            raise AuthenticationError("Service account configuration is required")

        sa_config = self.config.service_account_config

        # Create credentials from service account info
        credentials = service_account.Credentials.from_service_account_info(
            sa_config.to_dict(),
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/devstorage.full_control",
            ],
        )

        logger.info(f"Created service account credentials for {sa_config.client_email}")
        return credentials

    def _create_adc_credentials(self):
        """Create Application Default Credentials."""
        try:
            credentials, project = default(
                scopes=[
                    "https://www.googleapis.com/auth/cloud-platform",
                    "https://www.googleapis.com/auth/devstorage.full_control",
                ]
            )

            # Update project ID if not set
            if project and not self.config.project_id:
                self.config.project_id = project

            logger.info(f"Created Application Default Credentials for project {project}")
            return credentials
        except DefaultCredentialsError as e:
            raise AuthenticationError(f"Failed to create Application Default Credentials: {e}")

    def _create_workload_identity_credentials(self):
        """Create Workload Identity Federation credentials."""
        if not self.config.workload_identity_config:
            raise AuthenticationError("Workload Identity configuration is required")

        wif_config = self.config.workload_identity_config

        # Create external account credentials
        audience = (
            f"//iam.googleapis.com/projects/{wif_config.project_number}/"
            f"locations/global/workloadIdentityPools/{wif_config.pool_id}/"
            f"providers/{wif_config.provider_id}"
        )

        credentials_config = {
            "type": "external_account",
            "audience": audience,
            "subject_token_type": wif_config.subject_token_type,
            "token_url": wif_config.token_url,
            "credential_source": wif_config.credential_source,
            "service_account_impersonation_url": wif_config.service_account_impersonation_url,
        }

        credentials = external_account.Credentials.from_info(
            credentials_config,
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/devstorage.full_control",
            ],
        )

        logger.info(f"Created Workload Identity credentials for {wif_config.service_account_email}")
        return credentials

    def _create_compute_engine_credentials(self):
        """Create Compute Engine credentials."""
        credentials = compute_engine.Credentials(
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/devstorage.full_control",
            ]
        )
        logger.info("Created Compute Engine credentials")
        return credentials

    def _create_user_credentials(self):
        """Create user credentials (for development/testing)."""
        # This typically requires OAuth2 flow
        # For simplicity, falling back to ADC
        logger.warning("User credentials method requested, falling back to ADC")
        return self._create_adc_credentials()

    @retry_on_refresh_error(max_retries=3)
    def refresh_credentials(self) -> None:
        """Refresh credentials if expired."""
        if self._credentials:
            try:
                self._credentials.refresh(self._request)
                logger.debug("Successfully refreshed credentials")
            except Exception as e:
                logger.error(f"Failed to refresh credentials: {e}")
                # Try to recreate credentials
                with self._lock:
                    self._credentials = None
                    self._credentials = self._create_credentials()

    def get_access_token(self) -> str:
        """Get current access token."""
        # Check cache first
        cache_key = f"{self.config.auth_method.value}:{self.config.project_id}"
        cached_token = self._token_cache.get(cache_key)
        if cached_token:
            return cached_token

        # Get fresh token
        credentials = self.credentials
        if not credentials.token:
            self.refresh_credentials()

        # Cache the token
        if hasattr(credentials, "expiry") and credentials.expiry:
            self._token_cache.set(cache_key, credentials.token, credentials.expiry)

        return credentials.token

    def impersonate_service_account(self, target_service_account: str, scopes: list | None = None):
        """Impersonate a service account.

        Args:
            target_service_account: Email of the service account to impersonate
            scopes: OAuth2 scopes for the impersonated credentials

        Returns:
            Impersonated credentials
        """
        source_credentials = self.credentials

        if scopes is None:
            scopes = [
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/devstorage.full_control",
            ]

        impersonated_creds = impersonated_credentials.Credentials(
            source_credentials=source_credentials,
            target_principal=target_service_account,
            target_scopes=scopes,
            lifetime=3600,  # 1 hour
        )

        logger.info(f"Created impersonated credentials for {target_service_account}")
        return impersonated_creds

    def create_jwt_credentials(self, audience: str) -> str:
        """Create a self-signed JWT for authentication.

        Args:
            audience: The audience claim for the JWT

        Returns:
            Signed JWT token
        """
        if self.config.auth_method != AuthMethod.SERVICE_ACCOUNT:
            raise AuthenticationError("JWT creation requires service account credentials")

        if not self.config.service_account_config:
            raise AuthenticationError("Service account configuration is required")

        sa_config = self.config.service_account_config

        # Create JWT credentials
        credentials = jwt.Credentials.from_service_account_info(
            sa_config.to_dict(), audience=audience
        )

        # Get the JWT token
        credentials.refresh(self._request)
        return credentials.token

    def validate_credentials(self) -> bool:
        """Validate that credentials are working.

        Returns:
            True if credentials are valid, False otherwise
        """
        try:
            credentials = self.credentials
            if not credentials.token:
                self.refresh_credentials()

            # Try to use the credentials
            import google.cloud.storage

            client = google.cloud.storage.Client(
                project=self.config.project_id, credentials=credentials
            )
            # Try to list buckets (will fail if no permissions, but validates auth)
            try:
                list(client.list_buckets(max_results=1))
            except Exception:
                # Permission error is fine, we just want to validate auth
                pass

            logger.info("Credentials validated successfully")
            return True

        except Exception as e:
            logger.error(f"Credential validation failed: {e}")
            return False

    def get_project_id(self) -> str:
        """Get the current project ID."""
        if self.config.project_id and self.config.project_id != "default-project":
            return self.config.project_id

        # Try to get from credentials
        credentials = self.credentials
        if hasattr(credentials, "project_id"):
            return credentials.project_id

        # Try to get from metadata server (if on GCE)
        try:
            import requests

            response = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/project/project-id",
                headers={"Metadata-Flavor": "Google"},
                timeout=1,
            )
            if response.status_code == 200:
                return response.text
        except Exception:
            pass

        return self.config.project_id

    def get_authenticated_session(self):
        """Get an authenticated requests session.

        Returns:
            auth_requests.AuthorizedSession
        """
        from google.auth.transport.requests import AuthorizedSession

        return AuthorizedSession(self.credentials)

    def save_credentials_to_file(self, filepath: Path) -> None:
        """Save current credentials to a file (service account only).

        Args:
            filepath: Path to save the credentials
        """
        if self.config.auth_method != AuthMethod.SERVICE_ACCOUNT:
            raise AuthenticationError("Can only save service account credentials")

        if not self.config.service_account_config:
            raise AuthenticationError("No service account configuration available")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.config.service_account_config.to_dict(), f, indent=2)

        # Set restrictive permissions
        import os
        import stat

        os.chmod(filepath, stat.S_IRUSR | stat.S_IWUSR)

        logger.info(f"Saved credentials to {filepath}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clear sensitive data."""
        self._token_cache.clear()
        self._credentials = None
