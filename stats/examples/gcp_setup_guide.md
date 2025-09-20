# GCP Integration Setup Guide

This guide shows how to set up and use the minimal GCP integration for downloading Gemma models.

## Installation

Install the GCP dependencies:

```bash
# Install GCP integration dependencies
uv pip install --group gcp

# Or install individually
uv pip install google-cloud-storage google-auth huggingface-hub
```

## Authentication Setup

### Option 1: Service Account (Recommended for Production)

1. Create a service account in GCP Console
2. Download the JSON key file
3. Copy the template and fill in your credentials:

```bash
cp config/gcp_service_account.json.template config/my-service-account.json
# Edit config/my-service-account.json with your actual credentials
```

4. Set environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="config/my-service-account.json"
```

### Option 2: Application Default Credentials (Development)

```bash
# Install gcloud CLI if not already installed
# https://cloud.google.com/sdk/docs/install

# Login and set up application default credentials
gcloud auth application-default login
```

### Option 3: Compute Engine (When running on GCP)

No setup needed - credentials are automatically available.

## Basic Usage

### Using Python API

```python
from src.gcp.gemma_download import GemmaDownloader
from pathlib import Path

# Initialize downloader
downloader = GemmaDownloader(
    cache_dir=Path("./models"),
    service_account_path=Path("config/my-service-account.json")  # Optional
)

# Download from Hugging Face
model_path = downloader.download_from_huggingface("gemma-2b")
print(f"Model downloaded to: {model_path}")

# Download from GCS (if you have models stored there)
model_path = downloader.download_from_gcs(
    "gs://your-bucket/models/gemma-7b/",
    "gemma-7b-custom"
)

# Universal download method
model_path = downloader.download("gemma-2b-it")  # Auto-detects source
```

### Using CLI

```bash
# List available models (built into the code)
uv run python -c "from src.gcp.gemma_download import GEMMA_MODELS; print('\\n'.join(GEMMA_MODELS.keys()))"

# Download a model
uv run python -m src.gcp.gemma_download gemma-2b

# Download with custom cache directory
uv run python -m src.gcp.gemma_download gemma-2b --cache-dir ./my-models

# Download from GCS
uv run python -m src.gcp.gemma_download "gs://my-bucket/models/gemma-7b/" --model-name gemma-7b-custom

# List cached models
uv run python -m src.gcp.gemma_download --list-cached

# Clear specific model from cache
uv run python -m src.gcp.gemma_download --clear-cache gemma-2b

# Clear all cache
uv run python -m src.gcp.gemma_download --clear-cache ""

# Verbose logging
uv run python -m src.gcp.gemma_download gemma-2b --verbose
```

## Available Models

The downloader includes predefined configurations for common Gemma models:

- `gemma-2b` - Gemma 2B base model
- `gemma-2b-it` - Gemma 2B instruction-tuned
- `gemma-7b` - Gemma 7B base model
- `gemma-7b-it` - Gemma 7B instruction-tuned
- `gemma-2-2b` - Gemma 2 2B
- `gemma-2-2b-it` - Gemma 2 2B instruction-tuned
- `gemma-2-9b` - Gemma 2 9B
- `gemma-2-9b-it` - Gemma 2 9B instruction-tuned
- `gemma-2-27b` - Gemma 2 27B
- `gemma-2-27b-it` - Gemma 2 27B instruction-tuned
- `codegemma-2b` - CodeGemma 2B
- `codegemma-7b` - CodeGemma 7B
- `codegemma-7b-it` - CodeGemma 7B instruction-tuned

## Environment Variables

- `GOOGLE_APPLICATION_CREDENTIALS` - Path to service account JSON file
- `HF_TOKEN` - Hugging Face token (optional, for private models)

## Advanced Usage

### Custom GCS Download

```python
# Download specific files from GCS
downloader = GemmaDownloader()
model_path = downloader.download_from_gcs(
    "gs://my-bucket/custom-models/my-model/",
    "my-custom-model"
)
```

### Progress Monitoring

The downloader automatically shows progress bars for:
- Individual file downloads
- Multi-file GCS downloads
- Hugging Face model downloads

### Cache Management

```python
# Check what's cached
cached = downloader.list_cached_models()
for name, info in cached.items():
    print(f"{name}: {info['size_gb']:.1f}GB")

# Clear specific model
downloader.clear_cache("gemma-2b")

# Clear all models
downloader.clear_cache()
```

## Error Handling

The downloader gracefully handles:
- Missing credentials (falls back to available auth methods)
- Network failures (shows clear error messages)
- Interrupted downloads (can resume where left off)
- Missing dependencies (shows installation instructions)

## Integration with Existing Code

The minimal downloader is designed to work alongside the comprehensive `GemmaModelManager`:

```python
# Use both together
from src.gcp import GemmaDownloader, GemmaModelManager

# Simple download
downloader = GemmaDownloader()
model_path = downloader.download("gemma-2b")

# Advanced model management
manager = GemmaModelManager()
manager.verify_model(ModelVariant.GEMMA_2B)
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'google'**
   ```bash
   uv pip install --group gcp
   ```

2. **Authentication errors**
   ```bash
   # Check credentials
   gcloud auth application-default print-access-token

   # Or set service account
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
   ```

3. **Download failures**
   - Check network connectivity
   - Verify GCS bucket permissions
   - Check Hugging Face token for private models

### Logging

Enable verbose logging to debug issues:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Or use the CLI verbose flag:
```bash
uv run python -m src.gcp.gemma_download gemma-2b --verbose
```
