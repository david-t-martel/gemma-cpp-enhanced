"""Comprehensive unit tests for JWT authentication module."""

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import jwt
import pytest
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError

from src.server.auth import (
    TokenData,
    TokenResponse,
    JWTAuthenticator
)


class TestTokenData:
    """Test TokenData model."""

    def test_token_data_valid(self):
        """Test creating valid TokenData."""
        now = datetime.now(UTC)
        future = now + timedelta(hours=1)

        token_data = TokenData(
            sub="user123",
            exp=future,
            iat=now,
            jti="token-id-123",
            scope="api",
            api_key_hash="hash123"
        )

        assert token_data.sub == "user123"
        assert token_data.exp == future
        assert token_data.iat == now
        assert token_data.jti == "token-id-123"
        assert token_data.scope == "api"
        assert token_data.api_key_hash == "hash123"

    def test_token_data_default_scope(self):
        """Test TokenData with default scope."""
        now = datetime.now(UTC)
        future = now + timedelta(hours=1)

        token_data = TokenData(
            sub="user123",
            exp=future,
            iat=now,
            jti="token-id-123",
            api_key_hash="hash123"
        )

        assert token_data.scope == "api"

    def test_token_data_missing_fields(self):
        """Test TokenData with missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            TokenData(sub="user123")

        assert "exp" in str(exc_info.value)
        assert "iat" in str(exc_info.value)
        assert "jti" in str(exc_info.value)
        assert "api_key_hash" in str(exc_info.value)

    def test_token_data_serialization(self):
        """Test TokenData serialization."""
        now = datetime.now(UTC)
        future = now + timedelta(hours=1)

        token_data = TokenData(
            sub="user123",
            exp=future,
            iat=now,
            jti="token-id-123",
            api_key_hash="hash123"
        )

        data_dict = token_data.model_dump()
        assert data_dict["sub"] == "user123"
        assert data_dict["scope"] == "api"


class TestTokenResponse:
    """Test TokenResponse model."""

    def test_token_response_valid(self):
        """Test creating valid TokenResponse."""
        now = datetime.now(UTC)

        response = TokenResponse(
            access_token="jwt.token.here",
            expires_in=3600,
            issued_at=now
        )

        assert response.access_token == "jwt.token.here"
        assert response.token_type == "Bearer"
        assert response.expires_in == 3600
        assert response.issued_at == now

    def test_token_response_custom_token_type(self):
        """Test TokenResponse with custom token type."""
        now = datetime.now(UTC)

        response = TokenResponse(
            access_token="jwt.token.here",
            token_type="Custom",
            expires_in=3600,
            issued_at=now
        )

        assert response.token_type == "Custom"

    def test_token_response_serialization(self):
        """Test TokenResponse serialization."""
        now = datetime.now(UTC)

        response = TokenResponse(
            access_token="jwt.token.here",
            expires_in=3600,
            issued_at=now
        )

        data_dict = response.model_dump()
        assert data_dict["access_token"] == "jwt.token.here"
        assert data_dict["token_type"] == "Bearer"


class TestHashingFunctions:
    """Test API key hashing functions."""

    def test_hash_api_key(self):
        """Test API key hashing."""
        api_key = "test-api-key-123"
        hashed = hash_api_key(api_key)

        assert isinstance(hashed, str)
        assert len(hashed) == 64  # SHA-256 hex digest length
        assert hashed != api_key  # Should be different from original

    def test_hash_api_key_consistent(self):
        """Test that hashing is consistent."""
        api_key = "test-api-key-456"
        hash1 = hash_api_key(api_key)
        hash2 = hash_api_key(api_key)

        assert hash1 == hash2

    def test_hash_api_key_different_inputs(self):
        """Test that different inputs produce different hashes."""
        hash1 = hash_api_key("key1")
        hash2 = hash_api_key("key2")

        assert hash1 != hash2

    def test_verify_api_key_correct(self):
        """Test verifying correct API key."""
        api_key = "test-api-key-789"
        hashed = hash_api_key(api_key)

        assert verify_api_key(api_key, hashed) is True

    def test_verify_api_key_incorrect(self):
        """Test verifying incorrect API key."""
        api_key = "test-api-key-789"
        wrong_key = "wrong-api-key"
        hashed = hash_api_key(api_key)

        assert verify_api_key(wrong_key, hashed) is False

    def test_verify_api_key_empty(self):
        """Test verifying empty API key."""
        hashed = hash_api_key("test-key")

        assert verify_api_key("", hashed) is False
        assert verify_api_key(None, hashed) is False


class TestJWTAuthentication:
    """Test JWTAuthentication class."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock()
        settings.jwt_secret_key = "test-secret-key"
        settings.jwt_algorithm = "HS256"
        settings.jwt_expiration_hours = 24
        settings.api_keys = ["test-api-key-1", "test-api-key-2"]
        return settings

    @pytest.fixture
    def auth(self, mock_settings):
        """Create JWTAuthentication instance."""
        with patch('src.server.auth.Settings', return_value=mock_settings):
            return JWTAuthentication()

    def test_jwt_auth_initialization(self, auth, mock_settings):
        """Test JWTAuthentication initialization."""
        assert auth.secret_key == "test-secret-key"
        assert auth.algorithm == "HS256"
        assert auth.expiration_hours == 24
        assert len(auth.valid_api_key_hashes) == 2

    def test_generate_token(self, auth):
        """Test token generation."""
        api_key = "test-api-key-1"

        token_response = auth.generate_token(api_key)

        assert isinstance(token_response, TokenResponse)
        assert token_response.token_type == "Bearer"
        assert token_response.expires_in == 24 * 3600  # 24 hours in seconds
        assert len(token_response.access_token) > 0

    def test_generate_token_invalid_api_key(self, auth):
        """Test token generation with invalid API key."""
        with pytest.raises(HTTPException) as exc_info:
            auth.generate_token("invalid-api-key")

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_verify_token_valid(self, auth):
        """Test verifying valid token."""
        api_key = "test-api-key-1"
        token_response = auth.generate_token(api_key)

        token_data = auth.verify_token(token_response.access_token)

        assert isinstance(token_data, TokenData)
        assert token_data.scope == "api"
        assert token_data.api_key_hash == hash_api_key(api_key)

    def test_verify_token_invalid(self, auth):
        """Test verifying invalid token."""
        with pytest.raises(HTTPException) as exc_info:
            auth.verify_token("invalid.jwt.token")

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_verify_token_expired(self, auth, mock_settings):
        """Test verifying expired token."""
        # Create token with short expiration
        mock_settings.jwt_expiration_hours = -1  # Expired

        api_key = "test-api-key-1"
        token_response = auth.generate_token(api_key)

        with pytest.raises(HTTPException) as exc_info:
            auth.verify_token(token_response.access_token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_verify_token_malformed(self, auth):
        """Test verifying malformed token."""
        malformed_tokens = [
            "not.a.jwt",
            "too.few.parts",
            "malformed.jwt.token.with.too.many.parts",
            "",
            None
        ]

        for token in malformed_tokens:
            with pytest.raises(HTTPException) as exc_info:
                auth.verify_token(token)

            assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_verify_token_revoked_api_key(self, auth, mock_settings):
        """Test verifying token with revoked API key."""
        api_key = "test-api-key-1"
        token_response = auth.generate_token(api_key)

        # Remove API key from valid keys (simulate revocation)
        mock_settings.api_keys = ["test-api-key-2"]
        auth.valid_api_key_hashes = {hash_api_key(key) for key in mock_settings.api_keys}

        with pytest.raises(HTTPException) as exc_info:
            auth.verify_token(token_response.access_token)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_current_user_valid_token(self, auth):
        """Test getting current user with valid token."""
        api_key = "test-api-key-1"
        token_response = auth.generate_token(api_key)

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token_response.access_token
        )

        token_data = auth.get_current_user(credentials)

        assert isinstance(token_data, TokenData)
        assert token_data.scope == "api"

    def test_get_current_user_invalid_scheme(self, auth):
        """Test getting current user with invalid scheme."""
        credentials = HTTPAuthorizationCredentials(
            scheme="Basic",
            credentials="some-token"
        )

        with pytest.raises(HTTPException) as exc_info:
            auth.get_current_user(credentials)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_current_user_no_credentials(self, auth):
        """Test getting current user with no credentials."""
        with pytest.raises(HTTPException) as exc_info:
            auth.get_current_user(None)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


class TestAuthenticationMiddleware:
    """Test authentication middleware integration."""

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = Mock(spec=Request)
        request.headers = {"authorization": "Bearer valid.jwt.token"}
        return request

    def test_get_current_user_dependency(self):
        """Test get_current_user dependency function."""
        mock_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="valid.jwt.token"
        )

        with patch('src.server.auth.JWTAuthentication') as MockAuth:
            mock_auth_instance = Mock()
            mock_auth_instance.get_current_user.return_value = TokenData(
                sub="user123",
                exp=datetime.now(UTC) + timedelta(hours=1),
                iat=datetime.now(UTC),
                jti="token-id",
                api_key_hash="hash123"
            )
            MockAuth.return_value = mock_auth_instance

            result = get_current_user(mock_credentials)

            assert isinstance(result, TokenData)
            mock_auth_instance.get_current_user.assert_called_once_with(mock_credentials)

    def test_bearer_security_scheme(self):
        """Test HTTPBearer security scheme configuration."""
        from src.server.auth import bearer_scheme

        assert bearer_scheme is not None
        assert hasattr(bearer_scheme, 'scheme_name')


class TestAuthenticationSecurity:
    """Test authentication security aspects."""

    def test_jwt_secret_security(self):
        """Test JWT secret key security."""
        with patch('src.server.auth.Settings') as MockSettings:
            settings = Mock()

            # Test that weak secrets are detected (if implemented)
            weak_secrets = ["123", "password", "secret", ""]
            for secret in weak_secrets:
                settings.jwt_secret_key = secret
                MockSettings.return_value = settings

                # Initialize auth - should work but log warning
                auth = JWTAuthentication()
                assert auth.secret_key == secret

    def test_token_jti_uniqueness(self):
        """Test that JTI (JWT ID) is unique for each token."""
        with patch('src.server.auth.Settings') as MockSettings:
            settings = Mock()
            settings.jwt_secret_key = "test-secret"
            settings.jwt_algorithm = "HS256"
            settings.jwt_expiration_hours = 24
            settings.api_keys = ["test-key"]
            MockSettings.return_value = settings

            auth = JWTAuthentication()

            token1 = auth.generate_token("test-key")
            token2 = auth.generate_token("test-key")

            # Decode tokens to check JTI
            payload1 = jwt.decode(token1.access_token, "test-secret", algorithms=["HS256"])
            payload2 = jwt.decode(token2.access_token, "test-secret", algorithms=["HS256"])

            assert payload1["jti"] != payload2["jti"]

    def test_api_key_hashing_security(self):
        """Test API key hashing security."""
        api_key = "sensitive-api-key"
        hashed = hash_api_key(api_key)

        # Hash should be irreversible
        assert api_key not in hashed
        assert len(hashed) == 64  # SHA-256 length

        # Should use proper hashing algorithm
        expected_hash = hashlib.sha256(api_key.encode()).hexdigest()
        assert hashed == expected_hash

    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks in verification."""
        # This is a basic test - real timing attack resistance would need more sophisticated testing
        api_key = "test-key"
        hashed = hash_api_key(api_key)

        # Verify should handle both correct and incorrect keys
        assert verify_api_key(api_key, hashed) is True
        assert verify_api_key("wrong-key", hashed) is False
        assert verify_api_key("completely-different-length-key", hashed) is False


class TestAuthenticationIntegration:
    """Test authentication integration scenarios."""

    def test_full_authentication_flow(self):
        """Test complete authentication flow."""
        with patch('src.server.auth.Settings') as MockSettings:
            settings = Mock()
            settings.jwt_secret_key = "integration-test-secret"
            settings.jwt_algorithm = "HS256"
            settings.jwt_expiration_hours = 1
            settings.api_keys = ["integration-test-key"]
            MockSettings.return_value = settings

            auth = JWTAuthentication()

            # 1. Generate token
            token_response = auth.generate_token("integration-test-key")
            assert token_response.access_token

            # 2. Verify token
            token_data = auth.verify_token(token_response.access_token)
            assert token_data.sub

            # 3. Use token for authentication
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=token_response.access_token
            )

            user_data = auth.get_current_user(credentials)
            assert user_data.sub == token_data.sub

    def test_error_handling_integration(self):
        """Test error handling in authentication flow."""
        with patch('src.server.auth.Settings') as MockSettings:
            settings = Mock()
            settings.jwt_secret_key = "test-secret"
            settings.jwt_algorithm = "HS256"
            settings.jwt_expiration_hours = 24
            settings.api_keys = ["valid-key"]
            MockSettings.return_value = settings

            auth = JWTAuthentication()

            # Test invalid API key
            with pytest.raises(HTTPException):
                auth.generate_token("invalid-key")

            # Test invalid token
            with pytest.raises(HTTPException):
                auth.verify_token("invalid.token.here")

            # Test invalid credentials
            invalid_credentials = HTTPAuthorizationCredentials(
                scheme="Basic",
                credentials="invalid"
            )
            with pytest.raises(HTTPException):
                auth.get_current_user(invalid_credentials)