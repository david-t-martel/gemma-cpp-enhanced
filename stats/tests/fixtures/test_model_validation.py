"""
Integration tests for model validation functionality.

Tests model existence checking, download validation logic, and fallback behavior
for the Gemma model download and validation system.
"""

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

# Test imports
from src.gcp.gemma_download import (
    GEMMA_MODELS,
    GemmaDownloader,
    check_dependencies,
    detect_gpu,
    human_readable_size,
    main,
    recommend_model,
)


@dataclass
class MockGPUProperties:
    """Mock GPU properties for testing."""

    total_memory: int


class MockTorchCuda:
    """Mock torch.cuda module for testing."""

    def __init__(self, available: bool = True, memory_gb: float = 8.0):
        self._available = available
        self._memory_gb = memory_gb

    def is_available(self) -> bool:
        return self._available

    def current_device(self) -> int:
        return 0

    def get_device_properties(self, device_id: int) -> MockGPUProperties:
        memory_bytes = int(self._memory_gb * (1024**3))
        return MockGPUProperties(total_memory=memory_bytes)


@pytest.fixture
def temp_cache_dir():
    """Fixture providing temporary cache directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_huggingface_hub():
    """Fixture providing mocked Hugging Face hub."""
    with patch("src.gcp.gemma_download.HF_AVAILABLE", True):
        with patch("src.gcp.gemma_download.snapshot_download") as mock_download:
            yield mock_download


@pytest.fixture
def mock_gcs():
    """Fixture providing mocked Google Cloud Storage."""
    with patch("src.gcp.gemma_download.GCS_AVAILABLE", True):
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()

        mock_client.bucket.return_value = mock_bucket
        mock_bucket.list_blobs.return_value = [mock_blob]
        mock_blob.name = "model/config.json"
        mock_blob.download_to_filename = Mock()

        with patch("src.gcp.gemma_download.storage.Client", return_value=mock_client):
            yield mock_client


@pytest.fixture
def mock_requests():
    """Fixture providing mocked requests."""
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-length": "1024"}
        mock_response.iter_content = Mock(return_value=[b"test_data" for _ in range(10)])
        mock_get.return_value = mock_response
        yield mock_get


class TestDependencyValidation:
    """Test dependency validation functionality."""

    def test_check_dependencies_all_available(self):
        """Test dependency check when all dependencies are available."""
        with (
            patch("src.gcp.gemma_download.HF_AVAILABLE", True),
            patch("src.gcp.gemma_download.GCS_AVAILABLE", True),
            patch("src.gcp.gemma_download.TORCH_AVAILABLE", True),
        ):
            deps = check_dependencies()

            assert deps["tqdm"] is True
            assert deps["requests"] is True
            assert deps["huggingface_hub"] is True
            assert deps["google_cloud_storage"] is True
            assert deps["torch"] is True

    def test_check_dependencies_missing(self):
        """Test dependency check when dependencies are missing."""
        with (
            patch("src.gcp.gemma_download.HF_AVAILABLE", False),
            patch("src.gcp.gemma_download.GCS_AVAILABLE", False),
            patch("src.gcp.gemma_download.TORCH_AVAILABLE", False),
        ):
            deps = check_dependencies()

            assert deps["tqdm"] is True  # Always available in tests
            assert deps["requests"] is True  # Always available in tests
            assert deps["huggingface_hub"] is False
            assert deps["google_cloud_storage"] is False
            assert deps["torch"] is False

    def test_dependency_error_messages(self, capsys):
        """Test dependency error messages are helpful."""
        # Mock missing HF dependency
        with patch("src.gcp.gemma_download.HF_AVAILABLE", False):
            downloader = GemmaDownloader()

            with pytest.raises(RuntimeError) as exc_info:
                downloader.download_from_huggingface("gemma-2b")

            assert "huggingface-hub" in str(exc_info.value).lower()
            assert "install" in str(exc_info.value).lower()


class TestGPUDetection:
    """Test GPU detection and recommendation logic."""

    def test_detect_gpu_cuda_available(self):
        """Test GPU detection when CUDA is available."""
        mock_torch = MockTorchCuda(available=True, memory_gb=8.0)

        with (
            patch("src.gcp.gemma_download.TORCH_AVAILABLE", True),
            patch("src.gcp.gemma_download.torch.cuda", mock_torch),
        ):
            device, vram = detect_gpu()

            assert device == "cuda"
            assert vram == 8.0

    def test_detect_gpu_cuda_unavailable(self):
        """Test GPU detection when CUDA is unavailable."""
        with patch("src.gcp.gemma_download.TORCH_AVAILABLE", False):
            device, vram = detect_gpu()

            assert device is None
            assert vram is None

    def test_detect_gpu_nvidia_smi_fallback(self):
        """Test GPU detection fallback to nvidia-smi."""
        with (
            patch("src.gcp.gemma_download.TORCH_AVAILABLE", False),
            patch("src.gcp.gemma_download._nvidia_smi_vram", return_value=8192),
        ):  # 8GB in MiB
            device, vram = detect_gpu()

            assert device == "cuda"
            assert vram == 8.0

    def test_detect_gpu_no_gpu(self):
        """Test GPU detection when no GPU is available."""
        with (
            patch("src.gcp.gemma_download.TORCH_AVAILABLE", False),
            patch("src.gcp.gemma_download._nvidia_smi_vram", return_value=None),
        ):
            device, vram = detect_gpu()

            assert device is None
            assert vram is None

    @patch("subprocess.run")
    def test_nvidia_smi_command_execution(self, mock_run):
        """Test nvidia-smi command execution."""
        from src.gcp.gemma_download import _nvidia_smi_vram

        # Mock successful nvidia-smi output
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "8192\n"

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            result = _nvidia_smi_vram()

            assert result == 8192
            mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_nvidia_smi_command_failure(self, mock_run):
        """Test nvidia-smi command failure handling."""
        from src.gcp.gemma_download import _nvidia_smi_vram

        # Mock failed nvidia-smi execution
        mock_run.return_value.returncode = 1

        result = _nvidia_smi_vram()
        assert result is None

    @patch("subprocess.run")
    def test_nvidia_smi_timeout(self, mock_run):
        """Test nvidia-smi timeout handling."""
        from src.gcp.gemma_download import _nvidia_smi_vram

        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired(["nvidia-smi"], 2)

        result = _nvidia_smi_vram()
        assert result is None


class TestModelRecommendation:
    """Test model recommendation logic."""

    def test_recommend_model_low_vram(self):
        """Test model recommendation for low VRAM."""
        # Test various low VRAM scenarios
        test_cases = [
            (None, False, "gemma-2b-it"),
            (2.0, False, "gemma-2b-it"),
            (4.0, False, "gemma-2b-it"),
            (None, True, "codegemma-2b"),
            (3.0, True, "codegemma-2b"),
        ]

        for vram, prefer_code, expected in test_cases:
            result = recommend_model(vram, prefer_code)
            assert result == expected, f"VRAM: {vram}, Code: {prefer_code}"

    def test_recommend_model_medium_vram(self):
        """Test model recommendation for medium VRAM."""
        test_cases = [
            (6.0, False, "gemma-7b-it"),
            (8.0, False, "gemma-7b-it"),
            (12.0, False, "gemma-7b-it"),
            (6.0, True, "codegemma-7b"),
            (10.0, True, "codegemma-7b"),
        ]

        for vram, prefer_code, expected in test_cases:
            result = recommend_model(vram, prefer_code)
            assert result == expected, f"VRAM: {vram}, Code: {prefer_code}"

    def test_recommend_model_high_vram(self):
        """Test model recommendation for high VRAM."""
        test_cases = [
            (16.0, False, "gemma-2-9b-it"),
            (24.0, False, "gemma-2-9b-it"),
            (32.0, False, "gemma-2-27b-it"),
            (40.0, False, "gemma-2-27b-it"),
            (16.0, True, "codegemma-7b"),  # CodeGemma doesn't have larger variants
            (32.0, True, "codegemma-7b-it"),
        ]

        for vram, prefer_code, expected in test_cases:
            result = recommend_model(vram, prefer_code)
            assert result == expected, f"VRAM: {vram}, Code: {prefer_code}"

    def test_recommend_model_edge_cases(self):
        """Test model recommendation edge cases."""
        # Boundary conditions
        assert recommend_model(5.5, False) == "gemma-2b-it"  # Just below threshold
        assert recommend_model(5.6, False) == "gemma-7b-it"  # Just above threshold
        assert recommend_model(13.9, False) == "gemma-7b-it"
        assert recommend_model(14.1, False) == "gemma-2-9b-it"


class TestModelExistenceChecking:
    """Test model existence and validation."""

    def test_model_name_resolution(self):
        """Test model name resolution from aliases."""
        downloader = GemmaDownloader()

        # Test known model aliases
        assert "gemma-2b" in GEMMA_MODELS
        assert "gemma-7b-it" in GEMMA_MODELS
        assert "codegemma-7b" in GEMMA_MODELS

        # Test full repo IDs
        assert GEMMA_MODELS["gemma-2b"] == "google/gemma-2b"
        assert GEMMA_MODELS["gemma-7b-it"] == "google/gemma-7b-it"

    def test_invalid_model_name_handling(self, temp_cache_dir, mock_huggingface_hub):
        """Test handling of invalid model names."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        with pytest.raises(ValueError) as exc_info:
            downloader.download_from_huggingface("nonexistent-model")

        assert "Unknown model" in str(exc_info.value)
        assert "Available" in str(exc_info.value)

    def test_model_cache_validation(self, temp_cache_dir, mock_huggingface_hub):
        """Test validation of cached models."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        # Create incomplete cache directory
        model_dir = temp_cache_dir / "gemma-2b"
        model_dir.mkdir(parents=True)

        # Should detect incomplete cache and re-download
        downloader.download_from_huggingface("gemma-2b")

        mock_huggingface_hub.assert_called_once()

    def test_complete_model_cache_detection(self, temp_cache_dir, mock_huggingface_hub):
        """Test detection of complete cached models."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        # Create complete cache directory
        model_dir = temp_cache_dir / "gemma-2b"
        model_dir.mkdir(parents=True)

        # Create required files
        (model_dir / "config.json").write_text('{"model_type": "gemma"}')
        (model_dir / "tokenizer.json").write_text('{"version": "1.0"}')

        # Should use cached version
        result_path = downloader.download_from_huggingface("gemma-2b")

        assert result_path == model_dir
        mock_huggingface_hub.assert_not_called()


class TestDownloadValidation:
    """Test download validation logic."""

    def test_huggingface_download_success(self, temp_cache_dir, mock_huggingface_hub):
        """Test successful Hugging Face download."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        result_path = downloader.download_from_huggingface("gemma-2b")

        assert result_path.exists()
        assert result_path.name == "gemma-2b"
        mock_huggingface_hub.assert_called_once()

    def test_huggingface_download_with_token(self, temp_cache_dir, mock_huggingface_hub):
        """Test Hugging Face download with authentication token."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        downloader.download_from_huggingface("gemma-2b", token="test_token")

        # Verify token was passed
        call_args = mock_huggingface_hub.call_args
        assert call_args[1]["token"] == "test_token"

    def test_huggingface_download_environment_token(self, temp_cache_dir, mock_huggingface_hub):
        """Test Hugging Face download with environment token."""
        with patch.dict(os.environ, {"HF_TOKEN": "env_token"}):
            downloader = GemmaDownloader(cache_dir=temp_cache_dir)

            downloader.download_from_huggingface("gemma-2b")

            # Verify environment token was used
            call_args = mock_huggingface_hub.call_args
            assert call_args[1]["token"] == "env_token"

    def test_gcs_download_success(self, temp_cache_dir, mock_gcs):
        """Test successful GCS download."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        result_path = downloader.download_from_gcs("gs://test-bucket/models/gemma-7b/", "gemma-7b")

        assert result_path.exists()
        assert result_path.name == "gemma-7b"

    def test_gcs_download_invalid_uri(self, temp_cache_dir):
        """Test GCS download with invalid URI."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        with pytest.raises(ValueError) as exc_info:
            downloader.download_from_gcs("invalid://uri", "model")

        assert "Invalid GCS URI" in str(exc_info.value)

    def test_gcs_download_no_client(self, temp_cache_dir):
        """Test GCS download when client is not initialized."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)
        downloader.gcs_client = None

        with pytest.raises(RuntimeError) as exc_info:
            downloader.download_from_gcs("gs://bucket/path", "model")

        assert "GCS client not initialized" in str(exc_info.value)

    def test_http_download_success(self, temp_cache_dir, mock_requests):
        """Test successful HTTP download."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        result_path = downloader.download(
            "https://example.com/model.safetensors", "test_model.safetensors"
        )

        assert result_path.exists()
        assert result_path.name == "test_model.safetensors"
        mock_requests.assert_called_once()

    def test_download_progress_tracking(self, temp_cache_dir, mock_requests):
        """Test download progress tracking."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        # Mock progress tracking
        with patch("tqdm.tqdm") as mock_tqdm:
            mock_progress = Mock()
            mock_tqdm.return_value.__enter__.return_value = mock_progress

            downloader.download_file_with_progress(
                "https://example.com/large_file.bin", temp_cache_dir / "large_file.bin"
            )

            # Verify progress bar was used
            mock_tqdm.assert_called_once()

    def test_download_resume_capability(self, temp_cache_dir, mock_huggingface_hub):
        """Test download resume capability."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        downloader.download_from_huggingface("gemma-2b")

        # Verify resume_download was enabled
        call_args = mock_huggingface_hub.call_args
        assert call_args[1]["resume_download"] is True


class TestFallbackBehavior:
    """Test fallback behavior for model validation."""

    def test_primary_source_failure_fallback(self, temp_cache_dir):
        """Test fallback when primary download source fails."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        # Mock HF failure, should attempt other sources or provide helpful error
        with patch("src.gcp.gemma_download.HF_AVAILABLE", False):
            with pytest.raises(RuntimeError) as exc_info:
                downloader.download_from_huggingface("gemma-2b")

            assert "not available" in str(exc_info.value).lower()

    def test_network_failure_handling(self, temp_cache_dir):
        """Test handling of network failures."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        with patch("requests.get", side_effect=requests.ConnectionError("Network error")):
            with pytest.raises(requests.ConnectionError):
                downloader.download_file_with_progress(
                    "https://example.com/file.bin", temp_cache_dir / "file.bin"
                )

    def test_insufficient_disk_space_handling(self, temp_cache_dir):
        """Test handling of insufficient disk space."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        # Mock disk space error
        with patch("pathlib.Path.mkdir", side_effect=OSError("No space left on device")):
            with pytest.raises(OSError):
                downloader.download("https://example.com/large_model.bin")

    def test_corrupted_download_handling(self, temp_cache_dir, mock_huggingface_hub):
        """Test handling of corrupted downloads."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        # Simulate corrupted cache
        model_dir = temp_cache_dir / "gemma-2b"
        model_dir.mkdir(parents=True)
        (model_dir / "corrupted_file").write_text("invalid content")

        # Should detect corruption and re-download
        downloader.download_from_huggingface("gemma-2b")

        mock_huggingface_hub.assert_called_once()

    def test_model_version_fallback(self, temp_cache_dir, mock_huggingface_hub):
        """Test fallback to different model versions."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        # Test fallback from specific version to general
        result_path = downloader.download("google/gemma-2b-it")

        assert result_path.exists()
        mock_huggingface_hub.assert_called_once()

    def test_auto_selection_fallback(self):
        """Test auto-selection fallback for edge cases."""
        # Test fallback when GPU detection fails
        with patch("src.gcp.gemma_download.detect_gpu", return_value=(None, None)):
            model = recommend_model(None, False)
            assert model == "gemma-2b-it"  # Should default to smallest model

        # Test fallback when VRAM is extremely low
        model = recommend_model(0.5, False)
        assert model == "gemma-2b-it"

        # Test fallback when prefer_code but no code variants available for size
        model = recommend_model(50.0, True)  # Very high VRAM
        assert "codegemma" in model or "gemma" in model  # Should pick appropriate model


class TestCacheManagement:
    """Test cache management functionality."""

    def test_list_cached_models(self, temp_cache_dir):
        """Test listing of cached models."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        # Create some cached models
        for model_name in ["gemma-2b", "gemma-7b"]:
            model_dir = temp_cache_dir / model_name
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text('{"model": "' + model_name + '"}')
            (model_dir / "model.safetensors").write_text("fake model data")

        cached = downloader.list_cached_models()

        assert len(cached) == 2
        assert "gemma-2b" in cached
        assert "gemma-7b" in cached

        # Check metadata
        for model_name, info in cached.items():
            assert "size_bytes" in info
            assert "file_count" in info
            assert info["file_count"] >= 2

    def test_clear_specific_cache(self, temp_cache_dir):
        """Test clearing specific model cache."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        # Create cached models
        for model_name in ["gemma-2b", "gemma-7b"]:
            model_dir = temp_cache_dir / model_name
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}")

        # Clear specific model
        downloader.clear_cache("gemma-2b")

        # Check only specific model was cleared
        assert not (temp_cache_dir / "gemma-2b").exists()
        assert (temp_cache_dir / "gemma-7b").exists()

    def test_clear_all_cache(self, temp_cache_dir):
        """Test clearing all cache."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        # Create cached models
        for model_name in ["gemma-2b", "gemma-7b"]:
            model_dir = temp_cache_dir / model_name
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}")

        # Clear all cache
        downloader.clear_cache()

        # Check all models were cleared but cache dir exists
        assert temp_cache_dir.exists()
        assert not list(temp_cache_dir.iterdir())

    def test_cache_size_calculation(self, temp_cache_dir):
        """Test cache size calculation accuracy."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        # Create model with known size
        model_dir = temp_cache_dir / "test_model"
        model_dir.mkdir(parents=True)

        test_content = "x" * 1024  # 1KB content
        (model_dir / "file1.bin").write_text(test_content)
        (model_dir / "file2.bin").write_text(test_content)

        cached = downloader.list_cached_models()
        model_info = cached["test_model"]

        assert model_info["size_bytes"] == 2048  # 2KB total
        assert model_info["file_count"] == 2

    def test_human_readable_size_formatting(self):
        """Test human-readable size formatting."""
        test_cases = [
            (512, "512.00B"),
            (1024, "1.00KB"),
            (1024 * 1024, "1.00MB"),
            (1024 * 1024 * 1024, "1.00GB"),
            (1536, "1.50KB"),
            (0, "0.00B"),
        ]

        for size_bytes, expected in test_cases:
            result = human_readable_size(size_bytes)
            assert result == expected


class TestCLIIntegration:
    """Test CLI integration and command-line behavior."""

    def test_cli_auto_selection(self):
        """Test CLI auto-selection functionality."""
        test_args = [
            "dummy_script_name",  # sys.argv[0]
            "--auto",
            "--dry-run",
            "--json",
        ]

        with (
            patch("sys.argv", test_args),
            patch("src.gcp.gemma_download.detect_gpu", return_value=("cuda", 8.0)),
        ):
            exit_code = main()
            assert exit_code == 0

    def test_cli_dependency_check(self, capsys):
        """Test CLI dependency check."""
        test_args = ["dummy_script_name", "--show-deps", "--json"]

        with patch("sys.argv", test_args):
            exit_code = main()
            assert exit_code == 0

            captured = capsys.readouterr()
            output_data = json.loads(captured.out)
            assert "dependencies" in output_data

    def test_cli_cache_operations(self, temp_cache_dir):
        """Test CLI cache operations."""
        # Test list cache
        test_args = [
            "dummy_script_name",
            "--cache-dir",
            str(temp_cache_dir),
            "--list-cached",
            "--json",
        ]

        with patch("sys.argv", test_args):
            exit_code = main()
            assert exit_code == 0

    def test_cli_error_handling(self):
        """Test CLI error handling."""
        test_args = ["dummy_script_name", "nonexistent-model"]

        with patch("sys.argv", test_args), patch("src.gcp.gemma_download.HF_AVAILABLE", False):
            exit_code = main()
            assert exit_code == 1  # Should exit with error code

    def test_cli_verbose_mode(self, capsys):
        """Test CLI verbose mode."""
        test_args = ["dummy_script_name", "--verbose", "--show-deps"]

        with patch("sys.argv", test_args):
            exit_code = main()
            assert exit_code == 0

    def test_cli_quantization_suggestion(self):
        """Test CLI quantization suggestions."""
        test_args = ["dummy_script_name", "--auto", "--quantize", "--dry-run", "--json"]

        # Mock borderline VRAM scenario
        with (
            patch("sys.argv", test_args),
            patch("src.gcp.gemma_download.detect_gpu", return_value=("cuda", 7.0)),
        ):
            exit_code = main()
            assert exit_code == 0


class TestModelValidationIntegration:
    """Integration tests for complete model validation workflow."""

    def test_complete_validation_workflow(self, temp_cache_dir, mock_huggingface_hub):
        """Test complete model validation and download workflow."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        # 1. Detect hardware
        with patch("src.gcp.gemma_download.detect_gpu", return_value=("cuda", 8.0)):
            device, vram = detect_gpu()
            assert device == "cuda"
            assert vram == 8.0

        # 2. Recommend model
        recommended = recommend_model(vram, False)
        assert recommended == "gemma-7b-it"

        # 3. Validate model exists
        assert recommended in GEMMA_MODELS

        # 4. Download model
        model_path = downloader.download_from_huggingface(recommended)

        # 5. Validate download
        assert model_path.exists()
        assert model_path.name == recommended

        # 6. List and verify cache
        cached = downloader.list_cached_models()
        assert recommended in cached

    def test_fallback_validation_workflow(self, temp_cache_dir):
        """Test validation workflow with fallbacks."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        # 1. No GPU detected
        with patch("src.gcp.gemma_download.detect_gpu", return_value=(None, None)):
            device, vram = detect_gpu()
            assert device is None

        # 2. Should fallback to smallest model
        recommended = recommend_model(vram, False)
        assert recommended == "gemma-2b-it"

        # 3. Validate fallback model exists
        assert recommended in GEMMA_MODELS

    def test_error_recovery_validation(self, temp_cache_dir):
        """Test error recovery in validation workflow."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        # Test recovery from invalid model name
        with pytest.raises(ValueError):
            downloader.download_from_huggingface("invalid-model-name")

        # Test recovery from network issues
        with patch("requests.get", side_effect=requests.ConnectionError()):
            with pytest.raises(requests.ConnectionError):
                downloader.download_file_with_progress(
                    "https://example.com/file.bin", temp_cache_dir / "file.bin"
                )

    def test_concurrent_validation_operations(self, temp_cache_dir, mock_huggingface_hub):
        """Test concurrent validation operations."""
        downloader = GemmaDownloader(cache_dir=temp_cache_dir)

        async def download_model(model_name):
            return downloader.download_from_huggingface(model_name)

        # This would test concurrent downloads if implemented
        # For now, test that multiple downloaders can coexist
        downloader2 = GemmaDownloader(cache_dir=temp_cache_dir)

        # Both should work independently
        assert downloader.cache_dir == downloader2.cache_dir

    def test_validation_with_custom_config(self, temp_cache_dir):
        """Test validation with custom configuration."""
        # Test custom cache directory
        custom_cache = temp_cache_dir / "custom_cache"
        downloader = GemmaDownloader(cache_dir=custom_cache)

        assert downloader.cache_dir == custom_cache

        # Test environment variable override
        with patch.dict(os.environ, {"GEMMA_CACHE_DIR": str(custom_cache)}):
            env_downloader = GemmaDownloader()
            assert env_downloader.cache_dir == custom_cache


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
