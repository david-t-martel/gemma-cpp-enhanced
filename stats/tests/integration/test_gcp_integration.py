#!/usr/bin/env python3
"""
Test script for GCP integration components.

This script tests the authentication, storage, and model management functionality.
"""

from datetime import UTC, timedelta
import json
import os
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gcp import (
    AuthMethod,
    GCPAuthManager,
    GCPConfig,
    GCPRegion,
    GCSStorageManager,
    GemmaModelManager,
    ModelSource,
    ModelVariant,
    ServiceAccountConfig,
    validate_config,
)


def test_config_creation():
    """Test configuration creation and validation."""
    print("Testing configuration creation...")

    # Test minimal config
    config = GCPConfig(
        project_id="test-project",
        region=GCPRegion.US_CENTRAL1,
        auth_method=AuthMethod.APPLICATION_DEFAULT,
    )

    assert config.project_id == "test-project"
    assert config.region == GCPRegion.US_CENTRAL1
    assert config.auth_method == AuthMethod.APPLICATION_DEFAULT

    # Test regional endpoint generation
    storage_endpoint = config.get_regional_endpoint("storage")
    assert "us-central1" in storage_endpoint

    print("✓ Configuration creation test passed")


def test_service_account_config():
    """Test service account configuration."""
    print("Testing service account configuration...")

    # Create a test service account config with mock/placeholder values
    # NOTE: These are mock values for testing only - never use real private keys in tests
    sa_config = ServiceAccountConfig(
        client_email="test@project.iam.gserviceaccount.com",
        client_id="123456789",
        private_key="-----BEGIN PRIVATE KEY-----\n[MOCK KEY DATA FOR TESTING]\n-----END PRIVATE KEY-----",
        private_key_id="MOCK_KEY_ID_123",
        project_id="test-project",
    )

    # Test conversion to dict
    sa_dict = sa_config.to_dict()
    assert sa_dict["client_email"] == "test@project.iam.gserviceaccount.com"
    assert sa_dict["type"] == "service_account"

    print("✓ Service account configuration test passed")


def test_auth_manager():
    """Test authentication manager (without actual credentials)."""
    print("Testing authentication manager...")

    # Create config with mock credentials
    config = GCPConfig(
        project_id="test-project",
        region=GCPRegion.US_CENTRAL1,
        auth_method=AuthMethod.APPLICATION_DEFAULT,
    )

    # Create auth manager
    auth_manager = GCPAuthManager(config)

    # Test project ID retrieval
    project_id = auth_manager.get_project_id()
    assert project_id == "test-project"

    print("✓ Authentication manager test passed")


def test_model_registry():
    """Test model registry and information."""
    print("Testing model registry...")

    from gcp.gemma import MODEL_REGISTRY

    # Check that all variants are in registry
    for variant in ModelVariant:
        assert variant in MODEL_REGISTRY
        model_info = MODEL_REGISTRY[variant]
        assert model_info.variant == variant
        assert model_info.size_gb > 0
        assert len(model_info.files) > 0

    # Test specific model info
    gemma_2b = MODEL_REGISTRY[ModelVariant.GEMMA_2B]
    assert gemma_2b.size_gb == 5.0
    assert "model.safetensors" in gemma_2b.files

    print(f"✓ Model registry test passed ({len(MODEL_REGISTRY)} models)")


def test_model_manager():
    """Test model manager functionality."""
    print("Testing model manager...")

    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        config = GCPConfig(
            project_id="test-project",
            region=GCPRegion.US_CENTRAL1,
            auth_method=AuthMethod.APPLICATION_DEFAULT,
            model_cache_dir=Path(temp_dir),
        )

        manager = GemmaModelManager(config)

        # Test listing available models
        models = manager.list_available_models()
        assert len(models) > 0

        # Test model info retrieval
        info = manager.get_model_info(ModelVariant.GEMMA_2B)
        assert info.variant == ModelVariant.GEMMA_2B

        # Test cache checking (should be empty)
        assert not manager.is_model_cached(ModelVariant.GEMMA_2B)

        # Test cache size
        cache_size = manager.get_cache_size()
        assert cache_size == 0

        # Test listing cached models (should be empty)
        cached = manager.list_cached_models()
        assert len(cached) == 0

        print("✓ Model manager test passed")


def test_storage_object():
    """Test storage object data class."""
    print("Testing storage object...")

    from datetime import datetime, timezone

    from gcp.storage import StorageObject

    obj = StorageObject(
        bucket="test-bucket",
        name="test-object",
        size=1024,
        content_type="application/octet-stream",
        etag="etag123",
        created=datetime.now(UTC),
        updated=datetime.now(UTC),
        metadata={"key": "value"},
    )

    assert obj.bucket == "test-bucket"
    assert obj.size == 1024
    assert obj.metadata["key"] == "value"

    print("✓ Storage object test passed")


def test_config_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")

    # Valid config
    valid_config = GCPConfig(
        project_id="valid-project",
        region=GCPRegion.US_CENTRAL1,
        auth_method=AuthMethod.APPLICATION_DEFAULT,
    )
    assert validate_config(valid_config)

    # Invalid config (no project ID)
    invalid_config = GCPConfig(
        project_id="default-project",  # This is the default invalid value
        region=GCPRegion.US_CENTRAL1,
        auth_method=AuthMethod.APPLICATION_DEFAULT,
    )
    assert not validate_config(invalid_config)

    print("✓ Configuration validation test passed")


def test_yaml_config():
    """Test YAML configuration loading."""
    print("Testing YAML configuration...")

    # Check if config file exists
    config_path = Path("config/gcp-config.yaml")
    if config_path.exists():
        try:
            config = GCPConfig.from_yaml(config_path)
            print(f"  Loaded config for project: {config.project_id}")
            print(f"  Region: {config.region.value}")
            print(f"  Auth method: {config.auth_method.value}")
            print("✓ YAML configuration test passed")
        except Exception as e:
            print(f"⚠ Could not load YAML config: {e}")
    else:
        print("⚠ No configuration file found at config/gcp-config.yaml")
        print("  Run scripts/setup-gcp.sh to create one")


def test_environment_config():
    """Test environment-based configuration."""
    print("Testing environment configuration...")

    # Set test environment variables
    os.environ["GCP_PROJECT_ID"] = "env-test-project"
    os.environ["GCP_REGION"] = "europe-west1"

    try:
        config = GCPConfig.from_env()
        assert config.project_id == "env-test-project"
        assert config.region == GCPRegion.EUROPE_WEST1
        print("✓ Environment configuration test passed")
    except ValueError as e:
        print(f"⚠ Environment configuration test skipped: {e}")
    finally:
        # Clean up
        os.environ.pop("GCP_PROJECT_ID", None)
        os.environ.pop("GCP_REGION", None)


def main():
    """Run all tests."""
    print("=" * 60)
    print("GCP Integration Test Suite")
    print("=" * 60)
    print()

    tests = [
        test_config_creation,
        test_service_account_config,
        test_auth_manager,
        test_model_registry,
        test_model_manager,
        test_storage_object,
        test_config_validation,
        test_yaml_config,
        test_environment_config,
    ]

    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        print()

    print("=" * 60)
    if failed == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ {failed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
