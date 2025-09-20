#!/usr/bin/env python
"""Consolidated Gemma model downloader with multiple strategies and real GCP integration."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, ClassVar

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from huggingface_hub import login, snapshot_download

    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("⚠️ huggingface_hub not installed. Install with: uv pip install huggingface-hub")

try:
    from google.cloud import aiplatform, storage

    HAS_GCP = True
except ImportError:
    HAS_GCP = False
    print(
        "⚠️ Google Cloud libraries not installed. Install with: uv pip install google-cloud-storage google-cloud-aiplatform"
    )

try:
    import kagglehub

    HAS_KAGGLE = True
except ImportError:
    HAS_KAGGLE = False


class GemmaDownloader:
    """Unified Gemma model downloader with multiple backend strategies."""

    # Known alternative models that don't require license acceptance
    ALTERNATIVE_MODELS: ClassVar[dict[str, dict[str, Any]]] = {
        "microsoft/phi-2": {"size_gb": 2.7, "params": "2.7B", "type": "instruct"},
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {"size_gb": 1.1, "params": "1.1B", "type": "chat"},
        "stabilityai/stablelm-3b-4e1t": {"size_gb": 3.0, "params": "3B", "type": "base"},
        "EleutherAI/pythia-2.8b": {"size_gb": 2.8, "params": "2.8B", "type": "base"},
    }

    # Model name validation regex - alphanumeric, hyphens, underscores, and forward slash only
    MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-/]+$")
    MAX_MODEL_NAME_LENGTH = 128

    # Allowed file extensions for model files
    ALLOWED_EXTENSIONS: ClassVar[set[str]] = {
        ".bin",
        ".safetensors",
        ".pth",
        ".h5",
        ".json",
        ".txt",
        ".md",
    }

    def __init__(self, cache_dir: str = "./models"):
        # Sanitize and validate cache directory path
        self.cache_dir = self._sanitize_path(Path(cache_dir).resolve())
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration from environment with validation
        self.hf_token = self._validate_token(
            os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        )
        self.gcp_creds_path = self._validate_credentials_path(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )

        # Remove hardcoded project ID - must come from environment
        self.gcp_project = os.getenv("GCP_PROJECT_ID")
        if not self.gcp_project:
            print("⚠️ GCP_PROJECT_ID not set in environment. GCP features will be disabled.")
        # Validate project ID format
        elif not re.match(r"^[a-z][a-z0-9\-]{4,28}[a-z0-9]$", self.gcp_project):
            print(
                "⚠️ Invalid GCP_PROJECT_ID format. Must be 6-30 lowercase letters, digits, or hyphens."
            )
            self.gcp_project = None

        self.gcp_region = self._validate_gcp_region(os.getenv("GCP_REGION", "us-central1"))

        # Initialize GCP client if available
        self.gcs_client = None
        self.vertex_initialized = False

        if HAS_GCP and self.gcp_creds_path and self.gcp_project:
            try:
                # Set up GCS client with validated credentials
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.gcp_creds_path
                self.gcs_client = storage.Client(project=self.gcp_project)
                print("✅ GCS client initialized")

                # Initialize Vertex AI
                aiplatform.init(project=self.gcp_project, location=self.gcp_region)
                self.vertex_initialized = True
                print("✅ Vertex AI initialized")
            except Exception:
                # Don't expose detailed error information
                print("⚠️ GCP initialization failed. Check credentials and permissions.")

    def _sanitize_path(self, path: Path) -> Path:
        """Sanitize and validate file paths to prevent path traversal attacks."""
        try:
            # Resolve to absolute path and check for traversal attempts
            resolved = path.resolve()

            # Ensure the path doesn't contain dangerous patterns
            path_str = str(resolved)
            if ".." in path_str or "~" in path_str:
                raise ValueError("Path traversal attempt detected")

            # On Windows, check for UNC paths and device names
            if os.name == "nt":
                if path_str.startswith("\\\\"):
                    raise ValueError("UNC paths not allowed")
                # Check for reserved device names
                device_names = {
                    "CON",
                    "PRN",
                    "AUX",
                    "NUL",
                    "COM1",
                    "COM2",
                    "COM3",
                    "COM4",
                    "COM5",
                    "COM6",
                    "COM7",
                    "COM8",
                    "COM9",
                    "LPT1",
                    "LPT2",
                    "LPT3",
                    "LPT4",
                    "LPT5",
                    "LPT6",
                    "LPT7",
                    "LPT8",
                    "LPT9",
                }
                if resolved.name.upper() in device_names:
                    raise ValueError("Reserved device name not allowed")

            return resolved
        except Exception as exc:
            raise ValueError(f"Invalid path: {path}") from exc

    def _validate_token(self, token: str | None) -> str | None:
        """Validate API token format."""
        if not token:
            return None

        # Basic validation - token should be alphanumeric with some special chars
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", token):
            print("⚠️ Invalid token format detected")
            return None

        # Limit token length
        if len(token) > 512:
            print("⚠️ Token exceeds maximum length")
            return None

        return token

    def _validate_credentials_path(self, path: str | None) -> str | None:
        """Validate GCP credentials file path."""
        if not path:
            return None

        try:
            creds_path = self._sanitize_path(Path(path))

            # Check if file exists and is readable
            if not creds_path.exists():
                print("⚠️ Credentials file not found")
                return None

            if not creds_path.is_file():
                print("⚠️ Credentials path is not a file")
                return None

            # Validate file extension
            if creds_path.suffix not in [".json", ".p12"]:
                print("⚠️ Invalid credentials file extension")
                return None

            # Check file permissions (warn if world-readable)
            if os.name != "nt":  # Unix-like systems
                mode = creds_path.stat().st_mode
                if mode & 0o004:
                    print("⚠️ Warning: Credentials file is world-readable")

            return str(creds_path)
        except Exception:
            print("⚠️ Invalid credentials path")
            return None

    def _validate_gcp_region(self, region: str) -> str:
        """Validate GCP region format."""
        # GCP regions follow pattern: continent-direction-number
        if not re.match(r"^[a-z]+-[a-z]+[0-9]+$", region):
            print("⚠️ Invalid GCP region format, using default: us-central1")
            return "us-central1"
        return region

    def _validate_model_name(self, model_name: str) -> str:
        """Validate and sanitize model name to prevent injection attacks."""
        # Check length
        if len(model_name) > self.MAX_MODEL_NAME_LENGTH:
            raise ValueError(f"Model name too long (max {self.MAX_MODEL_NAME_LENGTH} characters)")

        # Check format
        if not self.MODEL_NAME_PATTERN.match(model_name):
            raise ValueError(
                "Invalid model name format. Only alphanumeric, hyphens, underscores, and forward slashes allowed"
            )

        # Prevent directory traversal in model names
        if ".." in model_name or model_name.startswith(("/", "\\")):
            raise ValueError("Invalid model name: path traversal detected")

        return model_name

    def download_model(self, model_name: str = "gemma-2b-it", force: bool = False) -> Path | None:
        """Download model using all available strategies."""

        # Validate model name
        try:
            model_name = self._validate_model_name(model_name)
        except ValueError as e:
            print(f"❌ Invalid model name: {e}")
            return None

        # Sanitize model path
        model_path = self._sanitize_path(self.cache_dir / model_name.replace("/", "_"))
        if (
            model_path.exists()
            and not force
            and (any(model_path.glob("*.bin")) or any(model_path.glob("*.safetensors")))
        ):
            print("✅ Model already cached")
            return model_path

        print(f"\n{'=' * 60}")
        print(f"🚀 Downloading {model_name}")
        print(f"{'=' * 60}\n")

        # Strategy 1: Try GCP Model Registry (real buckets)
        if self.gcs_client:
            print("1️⃣ Attempting GCP Model Registry download...")
            gcp_result = self._download_from_gcp_registry(model_name)
            if gcp_result:
                return gcp_result

        # Strategy 2: Try Vertex AI Model Garden
        if self.vertex_initialized:
            print("\n2️⃣ Attempting Vertex AI Model Garden...")
            vertex_result = self._download_from_vertex_ai(model_name)
            if vertex_result:
                return vertex_result

        # Strategy 3: Try Hugging Face with proper authentication
        if HAS_HF and self.hf_token:
            print("\n3️⃣ Attempting Hugging Face download...")
            hf_result = self._download_from_huggingface(model_name)
            if hf_result:
                return hf_result

        # Strategy 4: Try Kaggle (original Gemma source)
        if HAS_KAGGLE:
            print("\n4️⃣ Attempting Kaggle download...")
            kaggle_result = self._download_from_kaggle(model_name)
            if kaggle_result:
                return kaggle_result

        # Strategy 5: Suggest alternatives
        print("\n❌ All download strategies failed for Gemma")
        print("\n💡 Alternative models (no license required):")
        for alt_model, info in self.ALTERNATIVE_MODELS.items():
            print(f"   • {alt_model}: {info['params']} params, {info['size_gb']}GB, {info['type']}")

        # Try downloading an alternative
        if not model_name.startswith("google/gemma"):
            print(f"\n5️⃣ Attempting alternative model download: {model_name}")
            return self._download_alternative_model(model_name)

        return None

    def _download_from_gcp_registry(self, model_name: str) -> Path | None:
        """Download from GCP Model Registry buckets."""

        if not self.gcs_client or not self.gcp_project:
            return None

        # List of GCP buckets to try (dynamically generated, no hardcoded values)
        gcp_buckets = [
            f"gs://vertex-ai-{self.gcp_region}-models",
            f"gs://{self.gcp_project}-models",
            f"gs://ml-models-{self.gcp_project}",
            "gs://tfhub-modules",  # TensorFlow Hub models
            "gs://gresearch",  # Google Research public bucket
        ]

        for bucket_uri in gcp_buckets:
            try:
                bucket_name = bucket_uri.replace("gs://", "")
                # Validate bucket name format
                if not re.match(r"^[a-z0-9][a-z0-9\-_.]{1,61}[a-z0-9]$", bucket_name):
                    continue

                print(f"   Checking bucket: {bucket_name}")

                bucket = self.gcs_client.bucket(bucket_name)
                if not bucket.exists():
                    continue

                # Sanitize prefix to prevent injection
                safe_model_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", model_name)
                prefix = (
                    f"gemma/{safe_model_name}/" if "gemma" in model_name else f"{safe_model_name}/"
                )
                blobs = list(bucket.list_blobs(prefix=prefix, max_results=10))

                if blobs:
                    print(f"   ✅ Found {len(blobs)} files")
                    model_path = self._sanitize_path(self.cache_dir / model_name.replace("/", "_"))
                    model_path.mkdir(parents=True, exist_ok=True)

                    for blob in blobs:
                        # Validate blob name and prevent path traversal
                        blob_filename = Path(blob.name).name
                        if ".." in blob_filename or "/" in blob_filename or "\\" in blob_filename:
                            print(f"   ⚠️ Skipping suspicious file: {blob_filename}")
                            continue

                        # Check file extension
                        file_ext = Path(blob_filename).suffix.lower()
                        if file_ext and file_ext not in self.ALLOWED_EXTENSIONS:
                            print(f"   ⚠️ Skipping file with disallowed extension: {blob_filename}")
                            continue

                        local_file = self._sanitize_path(model_path / blob_filename)
                        print(f"   Downloading: {blob_filename}")
                        blob.download_to_filename(str(local_file))

                    return model_path

            except Exception:
                # Don't expose detailed error information
                print("   ❌ Bucket access failed")

        return None

    def _download_from_vertex_ai(self, model_name: str) -> Path | None:
        """Download from Vertex AI Model Garden."""
        if not self.vertex_initialized:
            return None

        try:
            # Sanitize model name for filter
            safe_model_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", model_name)

            # List available models in Model Registry
            models = aiplatform.Model.list(
                filter=f'display_name:"{safe_model_name}"',
                order_by="create_time desc",
                project=self.gcp_project,
                location=self.gcp_region,
            )

            if models:
                model = models[0]
                print(f"   Found model: {model.display_name}")

                # Download model artifacts
                if hasattr(model, "artifact_uri") and model.artifact_uri:
                    # Validate artifact URI format
                    if not model.artifact_uri.startswith("gs://"):
                        print("   ⚠️ Invalid artifact URI format")
                        return None

                    # Download from artifact URI
                    return self._download_from_gcs_uri(model.artifact_uri, model_name)
            else:
                print("   No models found")

        except Exception:
            # Don't expose detailed error information
            print("   ❌ Vertex AI access failed")

        return None

    def _download_from_gcs_uri(self, gcs_uri: str, model_name: str) -> Path | None:
        """Download from a specific GCS URI."""
        if not self.gcs_client:
            return None

        try:
            # Validate GCS URI format
            if not gcs_uri.startswith("gs://"):
                return None

            # Parse and validate GCS URI components
            uri_parts = gcs_uri[5:].split("/", 1)  # Remove "gs://"
            if not uri_parts:
                return None

            bucket_name = uri_parts[0]
            prefix = uri_parts[1] if len(uri_parts) > 1 else ""

            # Validate bucket name
            if not re.match(r"^[a-z0-9][a-z0-9\-_.]{1,61}[a-z0-9]$", bucket_name):
                print("   ⚠️ Invalid bucket name format")
                return None

            # Sanitize prefix
            prefix = re.sub(r"[^a-zA-Z0-9_\-/]", "_", prefix)

            bucket = self.gcs_client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix, max_results=100))

            if blobs:
                model_path = self._sanitize_path(self.cache_dir / model_name.replace("/", "_"))
                model_path.mkdir(parents=True, exist_ok=True)

                for blob in blobs:
                    # Validate blob name and prevent path traversal
                    blob_filename = Path(blob.name).name
                    if ".." in blob_filename or blob_filename.startswith("/"):
                        print("   ⚠️ Skipping suspicious file")
                        continue

                    # Check file extension
                    file_ext = Path(blob_filename).suffix.lower()
                    if file_ext and file_ext not in self.ALLOWED_EXTENSIONS:
                        continue

                    local_file = self._sanitize_path(model_path / blob_filename)
                    print(f"   Downloading: {blob_filename}")
                    blob.download_to_filename(str(local_file))

                return model_path

        except Exception:
            # Don't expose detailed error information
            print("   ❌ GCS download failed")

        return None

    def _download_from_huggingface(self, model_name: str) -> Path | None:
        """Download from Hugging Face with proper authentication."""
        if not HAS_HF or not self.hf_token:
            return None

        try:
            # Login with token
            login(token=self.hf_token, add_to_git_credential=False)
            print("   ✅ Logged in to Hugging Face")

            # Sanitize and map model names
            safe_model_name = re.sub(r"[^a-zA-Z0-9_\-/]", "_", model_name)
            hf_model_id = (
                f"google/{safe_model_name}" if "/" not in safe_model_name else safe_model_name
            )

            # Validate model ID format
            if not re.match(r"^[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-\.]+$", hf_model_id):
                print("   ❌ Invalid model ID format")
                return None

            # Download model with sanitized paths
            local_dir = self._sanitize_path(self.cache_dir / model_name.replace("/", "_"))
            model_path = snapshot_download(
                repo_id=hf_model_id,
                cache_dir=str(self.cache_dir),
                token=self.hf_token,
                resume_download=True,
                local_dir=str(local_dir),
            )

            print("   ✅ Downloaded successfully")
            return Path(model_path)

        except Exception as e:
            error_msg = str(e)
            # Don't expose full error details
            if "401" in error_msg or "403" in error_msg:
                print("   ❌ Authentication failed. Please check your token and model access.")
            elif "404" in error_msg:
                print("   ❌ Model not found")
            else:
                print("   ❌ Download failed")

        return None

    def _download_from_kaggle(self, model_name: str) -> Path | None:
        """Download from Kaggle (Gemma's original source)."""
        if not HAS_KAGGLE:
            return None

        try:
            # Sanitize model name for Kaggle path
            safe_model_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", model_name)
            kaggle_path = f"google/gemma/keras/{safe_model_name}"
            print("   Downloading from Kaggle")

            model_path = kagglehub.model_download(kaggle_path)
            print("   ✅ Downloaded successfully")

            # Copy to cache directory with path validation
            target_path = self._sanitize_path(self.cache_dir / model_name.replace("/", "_"))
            source_path = self._sanitize_path(Path(model_path))

            if source_path != target_path:
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)

            return target_path

        except Exception:
            # Don't expose detailed error information
            print("   ❌ Kaggle download failed")

        return None

    def _download_alternative_model(self, model_name: str) -> Path | None:
        """Download alternative open models that don't require license."""
        if not HAS_HF:
            print("   ❌ Hugging Face not available for alternative models")
            return None

        try:
            # Validate model name format
            if not re.match(r"^[a-zA-Z0-9_\-]+/[a-zA-Z0-9_\-\.]+$", model_name):
                print("   ❌ Invalid model name format")
                return None

            # Don't require login for open models
            local_dir = self._sanitize_path(self.cache_dir / model_name.replace("/", "_"))
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=str(self.cache_dir),
                resume_download=True,
                local_dir=str(local_dir),
            )

            print("   ✅ Alternative model downloaded")
            return Path(model_path)

        except Exception:
            # Don't expose detailed error information
            print("   ❌ Alternative download failed")

        return None

    def verify_model(self, model_path: Path) -> bool:
        """Verify that model files are present and valid."""
        try:
            # Sanitize path
            model_path = self._sanitize_path(model_path)

            if not model_path.exists():
                return False

            required_files = ["config.json", "tokenizer_config.json"]
            model_files = ["*.bin", "*.safetensors", "*.pth", "*.h5"]

            # Check for config files
            for file_name in required_files:
                file_path = model_path / file_name
                if not file_path.exists():
                    print(f"   ⚠️ Missing: {file_name}")
                    return False

                # Validate JSON files
                if file_name.endswith(".json"):
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            json.load(f)
                    except (OSError, json.JSONDecodeError):
                        print(f"   ⚠️ Invalid JSON: {file_name}")
                        return False

            # Check for model weights
            has_weights = any(list(model_path.glob(pattern)) for pattern in model_files)

            if not has_weights:
                print("   ⚠️ No model weight files found")
                return False

            print("   ✅ Model files verified")
            return True

        except Exception:
            return False

    def list_cached_models(self) -> dict[str, Any]:
        """List all cached models."""
        cached = {}

        try:
            for model_dir in self.cache_dir.iterdir():
                if model_dir.is_dir():
                    try:
                        # Sanitize directory path
                        model_dir = self._sanitize_path(model_dir)

                        # Calculate size safely
                        size_bytes = 0
                        file_count = 0
                        for f in model_dir.rglob("*"):
                            if f.is_file():
                                try:
                                    size_bytes += f.stat().st_size
                                    file_count += 1
                                except (OSError, PermissionError):
                                    continue

                        size_gb = size_bytes / (1024**3)

                        # Check validity
                        valid = self.verify_model(model_dir)

                        # Use sanitized name for display
                        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", model_dir.name)
                        cached[safe_name] = {
                            "size_gb": round(size_gb, 2),
                            "valid": valid,
                            "files": file_count,
                        }

                    except Exception:
                        # Skip problematic directories
                        continue

        except Exception:
            print("   ⚠️ Error listing cached models")

        return cached


def archive_old_scripts():
    """Archive deprecated download scripts."""
    archive_dir = Path("archived_download_scripts")
    archive_dir.mkdir(exist_ok=True)

    old_scripts = [
        "download_gemma.py",
        "download_gemma_gcp.py",
        "download_gemma_vertex.py",
        "download_gemma_alternative.py",
    ]

    for script in old_scripts:
        if Path(script).exists():
            shutil.move(script, archive_dir / script)
            print(f"   📦 Archived: {script}")


def main():
    """Main entry point with CLI."""
    parser = argparse.ArgumentParser(description="Unified Gemma model downloader")
    parser.add_argument("model", nargs="?", default="gemma-2b-it", help="Model name to download")
    parser.add_argument("--cache-dir", default="./models", help="Cache directory")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--list", action="store_true", help="List cached models")
    parser.add_argument("--verify", action="store_true", help="Verify cached models")
    parser.add_argument("--archive", action="store_true", help="Archive old download scripts")
    parser.add_argument("--alternative", action="store_true", help="Use alternative model")

    args = parser.parse_args()

    # Validate cache directory input
    try:
        cache_dir = Path(args.cache_dir).resolve()
        # Basic validation to prevent obvious issues
        if str(cache_dir).startswith("/etc") or str(cache_dir).startswith("/sys"):
            print("❌ Invalid cache directory location")
            return
    except Exception:
        print("❌ Invalid cache directory path")
        return

    # Archive old scripts if requested
    if args.archive:
        print("\n📦 Archiving old download scripts...")
        archive_old_scripts()
        print("✅ Archived old scripts to archived_download_scripts/")
        return

    # Initialize downloader with validated cache directory
    downloader = GemmaDownloader(cache_dir=str(cache_dir))

    # List cached models
    if args.list:
        print("\n📋 Cached Models:")
        cached = downloader.list_cached_models()
        if cached:
            for name, info in cached.items():
                status = "✅" if info["valid"] else "❌"
                # Don't expose full paths in output
                print(f"   {status} {name}: {info['size_gb']}GB, {info['files']} files")
        else:
            print("   No cached models found")
        return

    # Verify models
    if args.verify:
        print("\n🔍 Verifying cached models...")
        cached = downloader.list_cached_models()
        valid_count = sum(1 for info in cached.values() if info["valid"])
        print(f"   {valid_count}/{len(cached)} models are valid")
        return

    # Use alternative model if requested
    if args.alternative:
        print("\n🔄 Using alternative model...")
        model_name = "microsoft/phi-2"  # Default alternative
        print(f"   Selected: {model_name}")
    else:
        model_name = args.model

    # Download model
    result = downloader.download_model(model_name, force=args.force)

    if result:
        print(f"\n{'=' * 60}")
        print("✅ Success! Model downloaded successfully")
        print(f"{'=' * 60}")

        # Verify the download
        if downloader.verify_model(result):
            print("\n🎉 Model verification passed!")
            print("\nYou can now run:")
            print("  uv run python main.py")
            print("  uv run python test_react_agent_live.py")
        else:
            print("\n⚠️ Model verification failed. Some files may be missing.")
    else:
        print(f"\n{'=' * 60}")
        print("❌ Download failed")
        print(f"{'=' * 60}")
        print("\nTroubleshooting:")
        print("1. Check network connectivity")
        print("2. Verify GCP credentials are valid")
        print("3. Ensure GCP_PROJECT_ID is set in environment")
        print("4. Check token permissions and model access")
        print("5. Try an alternative model with --alternative flag")
        print("\nThe system can still run without a model:")
        print("  uv run python main.py --lightweight --no-tools")


if __name__ == "__main__":
    main()
