"""
Gemma / CodeGemma Model Downloader & Hardware-Aware Selector.

Enhancements added:
* Auto hardware detection (GPU VRAM) to recommend best-fitting model (``--auto``)
* Optional preference for CodeGemma variants for code-heavy tasks (``--prefer-code``)
* Dependency validation with actionable install hints
* Dry-run mode to preview selection logic without downloading (``--dry-run``)
* Cache inspection / clearing unchanged
* Robust error messaging & partial download safeguards

            # If authentication / permission issue and GCS mirror configured, attempt fallback
            err_text = str(dl_err)
            if any(code in err_text for code in ("401", "403")) and self.gcs_client is not None:
                mirror_base = os.getenv("GEMMA_GCS_MIRROR")
                if mirror_base:
                    # Compose GCS URI; if mirror_base already includes model name, use as is
                    if mirror_base.endswith("/"):
                        gcs_uri = mirror_base + model_name  # type: ignore[arg-type]
                    else:
                        gcs_uri = mirror_base + "/" + model_name  # type: ignore[arg-type]
                    logger.warning(
                        "HF auth failed (%s). Attempting GCS mirror fallback: %s", dl_err, gcs_uri
                    )
                    try:
                        return self.download_from_gcs(gcs_uri, model_name, cache_dir)  # type: ignore[arg-type]
                    except Exception as gcs_exc:  # pragma: no cover
                        logger.error("GCS fallback also failed: %s", gcs_exc)
            raise RuntimeError(f"Failed to download {repo_id}: {dl_err}") from dl_err
    < 6 GB VRAM  -> gemma-2b-it / codegemma-2b
    6-14 GB      -> gemma-7b-it / codegemma-7b
    14-30 GB     -> gemma-2-9b-it (if available)
    >= 30 GB     -> gemma-2-27b-it (largest)

If no GPU is detected or VRAM < 5 GB, defaults to 2B instruction variant.

Environment overrides:
    GEMMA_CACHE_DIR sets default cache root (if provided).
    HF_TOKEN / HUGGINGFACE_TOKEN used if --token omitted.

This module intentionally keeps download logic local (no background threads) -
``huggingface_hub.snapshot_download`` already supports resume and atomic writes.
"""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from ..shared.logging.logger import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover
    import torch  # type: ignore

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover
    pass

# Try importing Google Cloud dependencies lazily
try:  # pragma: no cover
    from google.cloud import storage  # type: ignore
    from google.oauth2 import service_account  # type: ignore

    GCS_AVAILABLE = True
except Exception:  # pragma: no cover
    GCS_AVAILABLE = False
    storage = None  # type: ignore
    service_account = None  # type: ignore

# Try importing Hugging Face Hub
try:
    from huggingface_hub import HfApi
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    HF_AVAILABLE = True
except Exception:  # pragma: no cover
    HF_AVAILABLE = False
    HfApi = None  # type: ignore
    snapshot_download = None  # type: ignore
    HfHubHTTPError = Exception  # type: ignore


def check_dependencies() -> dict[str, bool]:
    deps = {
        "tqdm": True,
        "requests": True,
        "huggingface_hub": HF_AVAILABLE,
        "google_cloud_storage": GCS_AVAILABLE,
        "torch": TORCH_AVAILABLE,
    }
    return deps


def _nvidia_smi_vram() -> int | None:  # pragma: no cover
    """Get GPU VRAM via nvidia-smi command if available."""
    try:
        # Find nvidia-smi executable for security
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return None

        # Safe fixed argument list (no user input) for nvidia-smi query
        completed = subprocess.run(
            [nvidia_smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if completed.returncode != 0:
            return None
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            return None
        return int(lines[0])
    except Exception as exc:
        logger.debug("nvidia-smi VRAM detection failed: %s", exc)
        return None


def detect_gpu() -> tuple[str | None, float | None]:  # pragma: no cover
    if TORCH_AVAILABLE and "torch" in globals() and torch.cuda.is_available():  # type: ignore
        try:
            idx = torch.cuda.current_device()  # type: ignore
            props = torch.cuda.get_device_properties(idx)  # type: ignore
            vram_gb = props.total_memory / (1024**3)
            return ("cuda", float(vram_gb))
        except Exception as exc:  # pragma: no cover
            logger.debug("torch CUDA VRAM detection failed: %s", exc)
    vram_mib = _nvidia_smi_vram()
    if vram_mib:
        return ("cuda", vram_mib / 1024.0)
    return (None, None)


def recommend_model(vram_gb: float | None, prefer_code: bool) -> str:
    base = "codegemma" if prefer_code else "gemma"
    if vram_gb is None or vram_gb < 5.5:
        return f"{base}-2b-it" if not prefer_code else f"{base}-2b"
    if vram_gb < 14:
        return f"{base}-7b-it" if not prefer_code else f"{base}-7b"
    if vram_gb < 30:
        return "gemma-2-9b-it" if not prefer_code else "codegemma-7b"
    return "gemma-2-27b-it" if not prefer_code else "codegemma-7b-it"


def human_readable_size(bytes_val: int) -> str:
    value = float(bytes_val)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}PB"


# Import centralized model configuration
try:
    from ..shared.config.model_configs import MODEL_REGISTRY

    # Generate GEMMA_MODELS from centralized config with backward compatibility
    GEMMA_MODELS = {}
    for spec in MODEL_REGISTRY.values():
        # Only include Gemma family models, avoid duplicates from aliases
        if spec.family.value == "gemma" and not any(
            existing_hf_id == spec.hf_model_id for existing_hf_id in GEMMA_MODELS.values()
        ):
            # Create short alias from full HF model ID
            short_name = spec.hf_model_id.replace("google/", "")
            GEMMA_MODELS[short_name] = spec.hf_model_id

    # Add legacy aliases for backward compatibility if not already present
    legacy_models = {
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-2-2b-it": "google/gemma-2-2b-it",
        "gemma-2-9b": "google/gemma-2-9b",
        "gemma-2-9b-it": "google/gemma-2-9b-it",
        "gemma-2-27b": "google/gemma-2-27b",
        "gemma-2-27b-it": "google/gemma-2-27b-it",
        "codegemma-2b": "google/codegemma-2b",
        "codegemma-7b": "google/codegemma-7b",
        "codegemma-7b-it": "google/codegemma-7b-it",
    }

    for short_name, hf_id in legacy_models.items():
        if short_name not in GEMMA_MODELS:
            GEMMA_MODELS[short_name] = hf_id

except ImportError:
    # Fallback to hardcoded models if centralized config is not available
    GEMMA_MODELS = {
        "gemma-2b": "google/gemma-2b",
        "gemma-2b-it": "google/gemma-2b-it",
        "gemma-7b": "google/gemma-7b",
        "gemma-7b-it": "google/gemma-7b-it",
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-2-2b-it": "google/gemma-2-2b-it",
        "gemma-2-9b": "google/gemma-2-9b",
        "gemma-2-9b-it": "google/gemma-2-9b-it",
        "gemma-2-27b": "google/gemma-2-27b",
        "gemma-2-27b-it": "google/gemma-2-27b-it",
        "codegemma-2b": "google/codegemma-2b",
        "codegemma-7b": "google/codegemma-7b",
        "codegemma-7b-it": "google/codegemma-7b-it",
    }


class GemmaDownloader:
    """Simple Gemma model downloader with caching."""

    def __init__(self, cache_dir: Path | None = None, service_account_path: Path | None = None):
        """Initialize the downloader.

        Args:
            cache_dir: Directory to cache models (default: ./models)
            service_account_path: Path to GCP service account JSON
        """
        env_cache = os.getenv("GEMMA_CACHE_DIR")
        # Precedence: explicit argument > env var > default
        self.cache_dir = cache_dir or (Path(env_cache) if env_cache else Path("./models"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GCS client if credentials are available
        self.gcs_client = self._init_gcs_client(service_account_path)

    def _init_gcs_client(self, service_account_path: Path | None) -> Any:
        """Initialize Google Cloud Storage client.

        Returns None if GCS libs or credentials are unavailable. No exceptions propagate.
        """
        if not GCS_AVAILABLE:
            return None
        try:  # pragma: no cover
            if (
                service_account_path
                and service_account_path.exists()
                and service_account is not None
                and storage is not None
            ):
                credentials = service_account.Credentials.from_service_account_file(
                    str(service_account_path)
                )  # type: ignore[attr-defined]
                return storage.Client(credentials=credentials)  # type: ignore[attr-defined]
            if storage is not None:
                # Try environment / default
                return storage.Client()  # type: ignore[attr-defined]
            return None
        except Exception as exc:  # pragma: no cover
            logger.debug("GCS client init failed: %s", exc)
            return None

    def download_from_huggingface(
        self, model_name: str, cache_dir: Path | None = None, token: str | None = None
    ) -> Path:
        """Download model from Hugging Face Hub.

        Args:
            model_name: Model name (e.g., "gemma-2b" or "google/gemma-2b")
            cache_dir: Override cache directory
            token: Hugging Face token

        Returns:
            Path to downloaded model directory
        """
        if not HF_AVAILABLE:
            raise RuntimeError(
                "Hugging Face Hub not available. Install with: uv pip install huggingface-hub"
            )

        # Resolve model repo ID
        if model_name in GEMMA_MODELS:
            repo_id = GEMMA_MODELS[model_name]
        elif "/" in model_name:
            repo_id = model_name
        else:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(GEMMA_MODELS.keys())}")

        cache_dir = cache_dir or self.cache_dir
        model_dir = cache_dir / model_name.replace("/", "--")

        # Check if already cached
        if model_dir.exists() and any(model_dir.iterdir()):
            present = {p.name for p in model_dir.iterdir() if p.is_file()}
            has_config = "config.json" in present
            # Accept any reasonable tokenizer artifact name used across Hugging Face repos
            has_tokenizer = any(
                name in present
                for name in (
                    "tokenizer.json",  # fast tokenizer json
                    "tokenizer.model",  # sentencepiece model (Gemma commonly uses this)
                    "tokenizer_config.json",  # tokenizer config sometimes present instead
                )
            )
            if has_config and has_tokenizer:
                logger.info(f"Model {model_name} already cached at {model_dir}")
                return model_dir
            logger.warning(
                "Model directory %s missing expected tokenizer/config files (found: %s); re-downloading",
                model_dir,
                sorted(present),
            )
            shutil.rmtree(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {model_name} from Hugging Face...")

        # Preflight: verify access (helps fail fast on gated models without token)
        hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        try:
            if HfApi is None:
                raise RuntimeError("huggingface_hub not available for preflight.")
            api = HfApi()
            _ = api.model_info(repo_id, token=hf_token)
        except Exception as pre_exc:
            # If explicitly an auth error, give clearer guidance
            if "401" in str(pre_exc) or "403" in str(pre_exc):
                raise RuntimeError(
                    f"Authentication required or access denied for {repo_id}. Accept the license and supply a valid token. Original: {pre_exc}"
                ) from pre_exc
            # Non-fatal (e.g., rate limit) -> proceed to snapshot attempt
            logger.debug("Preflight model_info failed (continuing to download): %s", pre_exc)

        try:
            if snapshot_download is None:
                raise RuntimeError("huggingface_hub not available for download.")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_dir),
                token=hf_token,
            )
        except HfHubHTTPError as dl_err:  # type: ignore
            # Clean up partial directory on auth issues/errors
            if model_dir.exists():
                with contextlib.suppress(Exception):  # pragma: no cover
                    shutil.rmtree(model_dir)
            raise RuntimeError(f"Failed to download {repo_id}: {dl_err}") from dl_err

        logger.info(f"Successfully downloaded {model_name} to {model_dir}")
        return model_dir

    def download_from_gcs(
        self, gcs_uri: str, model_name: str, cache_dir: Path | None = None
    ) -> Path:
        """Download model from Google Cloud Storage.

        Args:
            gcs_uri: GCS URI (e.g., "gs://my-bucket/models/gemma-7b/")
            model_name: Local model name for caching
            cache_dir: Override cache directory

        Returns:
            Path to downloaded model directory
        """
        if not self.gcs_client:
            raise RuntimeError("GCS client not initialized. Check credentials.")

        cache_dir = cache_dir or self.cache_dir
        model_dir = cache_dir / model_name

        # Check if already cached
        if model_dir.exists() and any(model_dir.iterdir()):
            logger.info(f"Model {model_name} already cached at {model_dir}")
            return model_dir

        # Parse GCS URI
        parsed = urlparse(gcs_uri)
        if parsed.scheme != "gs":
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")

        bucket_name = parsed.netloc
        prefix = parsed.path.lstrip("/")

        logger.info(f"Downloading {model_name} from {gcs_uri}...")

        # Create model directory
        model_dir.mkdir(parents=True, exist_ok=True)

        # List and download all blobs with the prefix
        bucket = self.gcs_client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            raise ValueError(f"No files found at {gcs_uri}")

        with tqdm(total=len(blobs), desc="Downloading files") as pbar:
            for blob in blobs:
                # Skip directories
                if blob.name.endswith("/"):
                    continue

                # Create relative path
                rel_path = blob.name[len(prefix) :].lstrip("/")
                if not rel_path:
                    continue

                local_path = model_dir / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)

                # Download file
                blob.download_to_filename(str(local_path))
                pbar.set_description(f"Downloaded {rel_path}")
                pbar.update(1)

        logger.info(f"Successfully downloaded {model_name} to {model_dir}")
        return model_dir

    def download_file_with_progress(self, url: str, local_path: Path) -> None:
        """Download a single file with progress bar."""
        logger.info(f"Downloading {url}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with (
            open(local_path, "wb") as f,
            tqdm(
                desc=local_path.name,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    def download(
        self,
        source: str,
        model_name: str | None = None,
        cache_dir: Path | None = None,
        **kwargs: Any,
    ) -> Path:
        """Download model from any supported source.

        Args:
            source: Source URI or model name
                - Hugging Face: "gemma-2b" or "google/gemma-2b"
                - GCS: "gs://bucket/path/to/model/"
                - Direct URL: "https://example.com/model.safetensors"
            model_name: Local name for caching (auto-detected if not provided)
            cache_dir: Override cache directory
            **kwargs: Additional arguments passed to specific downloaders

        Returns:
            Path to downloaded model directory or file
        """
        # Auto-detect model name
        if not model_name:
            if source.startswith("gs://"):
                model_name = Path(source.rstrip("/")).name
            elif "/" in source and not source.startswith("http"):
                model_name = source.split("/")[-1]
            else:
                model_name = source

        # Determine source type and download
        if source.startswith("gs://"):
            return self.download_from_gcs(source, model_name, cache_dir)
        elif source.startswith("http"):
            cache_dir = cache_dir or self.cache_dir
            local_path = cache_dir / model_name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.download_file_with_progress(source, local_path)
            return local_path
        else:
            # Assume Hugging Face
            return self.download_from_huggingface(source, cache_dir, **kwargs)

    def list_cached_models(self) -> dict[str, dict[str, Any]]:
        """List all cached models with metadata.

        Returns:
            Dictionary mapping model names to metadata
        """
        cached = {}

        for model_dir in self.cache_dir.iterdir():
            if not model_dir.is_dir():
                continue

            # Calculate total size
            total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())

            # Count files
            file_count = sum(1 for f in model_dir.rglob("*") if f.is_file())

            cached[model_dir.name] = {
                "path": str(model_dir),
                "size_bytes": total_size,
                "size_gb": total_size / (1024**3),
                "file_count": file_count,
                "files": [f.name for f in model_dir.iterdir() if f.is_file()][
                    :10
                ],  # First 10 files
            }

        return cached

    def clear_cache(self, model_name: str | None = None) -> None:
        """Clear model cache.

        Args:
            model_name: Specific model to clear (clear all if None)
        """
        if model_name:
            model_dir = self.cache_dir / model_name
            if model_dir.exists():
                shutil.rmtree(model_dir)
                logger.info(f"Cleared cache for {model_name}")
        elif self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared all cached models")


def _print_or_json(condition: bool, payload: dict[str, Any], text_fn: Callable[[], None]) -> None:
    if condition:
        print(json.dumps(payload))
    else:
        text_fn()


def _handle_show_deps(json_mode: bool) -> int:
    deps = check_dependencies()

    def _text() -> None:  # pragma: no cover - formatting only
        print("Dependency Status:")
        for k, ok in deps.items():
            print(f"  {k:20} : {'OK' if ok else 'MISSING'}")
        if not deps["huggingface_hub"]:
            print("Install with: uv pip install huggingface-hub")

    _print_or_json(json_mode, {"mode": "dependencies", "dependencies": deps}, _text)
    return 0


def _handle_list_cached(downloader: GemmaDownloader, json_mode: bool) -> int:
    cached = downloader.list_cached_models()
    total_bytes = sum(c["size_bytes"] for c in cached.values()) if cached else 0

    def _text() -> None:  # pragma: no cover - formatting only
        if cached:
            print("Cached models:")
            for name, info in cached.items():
                print(
                    f"  {name}: {info['size_gb']:.1f}GB ({info['file_count']} files) e.g. first files: {info['files']}"
                )
            print(f"Total cached size: {human_readable_size(total_bytes)}")
        else:
            print("No cached models found")

    _print_or_json(
        json_mode,
        {
            "mode": "list-cached",
            "models": cached,
            "total_size_bytes": total_bytes,
            "total_size_readable": human_readable_size(total_bytes),
        },
        _text,
    )
    return 0


def _handle_clear_cache(downloader: GemmaDownloader, target: str | None, json_mode: bool) -> int:
    downloader.clear_cache(target or None)
    _print_or_json(json_mode, {"mode": "clear-cache", "cleared": target or "ALL"}, lambda: None)
    return 0


def _handle_auto_select(args: argparse.Namespace, json_mode: bool) -> tuple[str, bool]:
    device, vram_gb = detect_gpu()
    if args.min_vram is not None:
        vram_gb = float(args.min_vram)
    selected = recommend_model(vram_gb, args.prefer_code)
    quant_suggestion = bool(
        args.quantize and vram_gb and 5.5 <= vram_gb < 8 and not args.prefer_code
    )

    def _text() -> None:  # pragma: no cover - formatting only
        print("Auto Selection:")
        print(f"  Detected device : {device or 'cpu'}")
        print(f"  Detected VRAM   : {f'{vram_gb:.2f} GB' if vram_gb else 'n/a'}")
        print(f"  Chosen model    : {selected}")
        if quant_suggestion:
            print("  Suggest: Use 8-bit quantization for 7B on borderline VRAM.")

    _print_or_json(
        json_mode,
        {
            "mode": "auto-selection",
            "device": device or "cpu",
            "vram_gb": vram_gb,
            "selected_model": selected,
            "quantize_suggestion": quant_suggestion,
        },
        _text,
    )
    return selected, args.dry_run


def _emit_error(json_mode: bool, err: Exception, err_type: str, hint: str | None = None) -> None:
    # Auto-augment hint for common issues
    err_text = str(err)
    if hint is None:
        if "401" in err_text and "Access to model" in err_text:
            hint = (
                "Gemma models are gated. Provide a Hugging Face token with access: set $env:HF_TOKEN='hf_xxx' "
                "or pass --token YOUR_TOKEN (after accepting the model license at https://huggingface.co/google/)."
            )
        elif "Unknown model" in err_text and "gemma" in err_text:
            hint = "Use --auto or one of: gemma-2b-it, gemma-7b-it, gemma-2-9b-it, gemma-2-27b-it (ensure access granted)."
    if json_mode:
        payload: dict[str, Any] = {"status": "error", "error": err_text, "type": err_type}
        if hint:
            payload["hint"] = hint
        print(json.dumps(payload))
    else:
        logger.error(f"{err_type} error: {err_text}")
        if hint:
            print(f"Hint: {hint}")


def main() -> int:
    """CLI entrypoint with hardware-aware auto selection."""
    parser = argparse.ArgumentParser(
        description="Download Gemma / CodeGemma models with auto hardware selection"
    )
    parser.add_argument(
        "source", nargs="?", help="Model source (model key, repo id, GCS URI, or URL)"
    )
    parser.add_argument("--model-name", help="Local model name for caching")
    parser.add_argument("--cache-dir", type=Path, help="Cache directory (or GEMMA_CACHE_DIR env)")
    parser.add_argument("--service-account", type=Path, help="GCP service account JSON file")
    parser.add_argument("--token", help="Hugging Face token (else HF_TOKEN/HUGGINGFACE_TOKEN env)")
    parser.add_argument("--list-cached", action="store_true", help="List cached models")
    parser.add_argument("--clear-cache", help="Clear cache for model (omit name to clear all)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--auto", action="store_true", help="Auto-detect hardware and pick best model"
    )
    parser.add_argument(
        "--prefer-code", action="store_true", help="Prefer CodeGemma variants during auto selection"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show decision and exit without downloading"
    )
    parser.add_argument(
        "--min-vram", type=float, help="Override detected VRAM in GB (testing / CI)"
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Emit recommendation to use 8-bit if borderline"
    )
    parser.add_argument(
        "--show-deps", action="store_true", help="Display dependency availability and exit"
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    parser.add_argument(
        "--gcs-fallback", help="GCS base URI (gs://bucket/path) used as mirror if HF auth fails"
    )
    args = parser.parse_args()

    json_mode = bool(args.json)
    exit_code = 0

    # Optional explicit GCS fallback mirror
    if args.gcs_fallback:
        os.environ["GEMMA_GCS_MIRROR"] = args.gcs_fallback.rstrip("/")

    if args.show_deps:
        return _handle_show_deps(json_mode)

    downloader = GemmaDownloader(
        cache_dir=args.cache_dir, service_account_path=args.service_account
    )

    if args.list_cached:
        return _handle_list_cached(downloader, json_mode)

    if args.clear_cache is not None:
        return _handle_clear_cache(downloader, args.clear_cache or None, json_mode)

    if args.auto and not args.source:
        selected, dry = _handle_auto_select(args, json_mode)
        args.source = selected
        if dry:
            return 0

    if not args.source:
        parser.error("Provide a model source or use --auto")

    try:
        model_path = downloader.download(
            args.source, model_name=args.model_name, cache_dir=args.cache_dir, token=args.token
        )
        _print_or_json(
            json_mode,
            {"status": "ok", "model_path": str(model_path)},
            lambda: print(f"Model downloaded to: {model_path}"),
        )
    except ValueError as err:
        _emit_error(json_mode, err, "value")
        exit_code = 1
    except RuntimeError as err:
        hint = "uv pip install huggingface-hub" if "huggingface-hub" in str(err).lower() else None
        _emit_error(json_mode, err, "runtime", hint)
        exit_code = 1
    except Exception as err:  # pragma: no cover
        _emit_error(json_mode, err, "unknown")
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
