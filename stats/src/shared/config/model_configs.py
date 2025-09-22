"""Centralized model configurations for the Gemma LLM Stats project.

This module provides unified model configurations and defaults used across
all components of the system to eliminate duplication and ensure consistency.
"""

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator


class ModelFamily(str, Enum):
    """Supported model families."""

    GEMMA = "gemma"
    PHI = "phi"
    LLAMA = "llama"
    MISTRAL = "mistral"


class ModelSize(str, Enum):
    """Model size categories."""

    TINY = "tiny"  # <1B parameters
    SMALL = "small"  # 1-3B parameters
    MEDIUM = "medium"  # 3-8B parameters
    LARGE = "large"  # 8B+ parameters


class ModelType(str, Enum):
    """Model training types."""

    BASE = "base"
    INSTRUCT = "instruct"
    CHAT = "chat"


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a language model."""

    # Model identification
    name: str
    family: ModelFamily
    size: ModelSize
    type: ModelType

    # Model parameters
    parameter_count: int | None = None
    context_length: int = 2048

    # HuggingFace Hub information
    hf_model_id: str = ""
    hf_tokenizer_id: str | None = None

    # Model capabilities
    supports_tools: bool = False
    supports_vision: bool = False
    supports_code: bool = False

    # Performance characteristics
    min_memory_gb: float = 2.0
    recommended_memory_gb: float = 4.0
    supports_quantization: bool = True
    supports_cpu: bool = True
    supports_gpu: bool = True

    # Default inference parameters
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 50
    default_max_tokens: int = 512
    default_repetition_penalty: float = 1.1

    def __post_init__(self):
        """Set defaults after initialization."""
        if not self.hf_model_id:
            object.__setattr__(self, "hf_model_id", self.name)
        if not self.hf_tokenizer_id:
            object.__setattr__(self, "hf_tokenizer_id", self.hf_model_id)


# =============================================================================
# GEMMA MODEL CONFIGURATIONS
# =============================================================================

GEMMA_2B_IT = ModelSpec(
    name="google/gemma-2b-it",
    family=ModelFamily.GEMMA,
    size=ModelSize.SMALL,
    type=ModelType.INSTRUCT,
    parameter_count=2_000_000_000,
    context_length=8192,
    hf_model_id="google/gemma-2b-it",
    supports_tools=True,
    supports_code=True,
    min_memory_gb=2.5,
    recommended_memory_gb=4.0,
    default_temperature=0.7,
    default_max_tokens=1024,
)

GEMMA_2B_BASE = ModelSpec(
    name="google/gemma-2b",
    family=ModelFamily.GEMMA,
    size=ModelSize.SMALL,
    type=ModelType.BASE,
    parameter_count=2_000_000_000,
    context_length=8192,
    hf_model_id="google/gemma-2b",
    min_memory_gb=2.5,
    recommended_memory_gb=4.0,
)

GEMMA_7B_IT = ModelSpec(
    name="google/gemma-7b-it",
    family=ModelFamily.GEMMA,
    size=ModelSize.MEDIUM,
    type=ModelType.INSTRUCT,
    parameter_count=7_000_000_000,
    context_length=8192,
    hf_model_id="google/gemma-7b-it",
    supports_tools=True,
    supports_code=True,
    min_memory_gb=8.0,
    recommended_memory_gb=16.0,
    default_temperature=0.7,
    default_max_tokens=2048,
)

GEMMA_7B_BASE = ModelSpec(
    name="google/gemma-7b",
    family=ModelFamily.GEMMA,
    size=ModelSize.MEDIUM,
    type=ModelType.BASE,
    parameter_count=7_000_000_000,
    context_length=8192,
    hf_model_id="google/gemma-7b",
    min_memory_gb=8.0,
    recommended_memory_gb=16.0,
)

# Future Gemma models
GEMMA_9B_IT = ModelSpec(
    name="google/gemma-9b-it",
    family=ModelFamily.GEMMA,
    size=ModelSize.MEDIUM,
    type=ModelType.INSTRUCT,
    parameter_count=9_000_000_000,
    context_length=8192,
    hf_model_id="google/gemma-9b-it",
    supports_tools=True,
    supports_code=True,
    supports_vision=False,  # Will be True for Gemma3/PaliGemma
    min_memory_gb=10.0,
    recommended_memory_gb=20.0,
)

# =============================================================================
# PHI MODEL CONFIGURATIONS
# =============================================================================

PHI_2 = ModelSpec(
    name="microsoft/phi-2",
    family=ModelFamily.PHI,
    size=ModelSize.SMALL,
    type=ModelType.BASE,
    parameter_count=2_700_000_000,
    context_length=2048,
    hf_model_id="microsoft/phi-2",
    supports_code=True,
    min_memory_gb=3.0,
    recommended_memory_gb=6.0,
    default_temperature=0.3,
    default_max_tokens=1024,
)

PHI_3_MINI = ModelSpec(
    name="microsoft/Phi-3-mini-4k-instruct",
    family=ModelFamily.PHI,
    size=ModelSize.SMALL,
    type=ModelType.INSTRUCT,
    parameter_count=3_800_000_000,
    context_length=4096,
    hf_model_id="microsoft/Phi-3-mini-4k-instruct",
    supports_tools=True,
    supports_code=True,
    min_memory_gb=4.0,
    recommended_memory_gb=8.0,
)

# =============================================================================
# MODEL REGISTRY
# =============================================================================

# Registry of all available models
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # Gemma models
    "gemma-2b-it": GEMMA_2B_IT,
    "gemma-2b": GEMMA_2B_BASE,
    "gemma-7b-it": GEMMA_7B_IT,
    "gemma-7b": GEMMA_7B_BASE,
    "gemma-9b-it": GEMMA_9B_IT,
    # Phi models
    "phi-2": PHI_2,
    "phi-3-mini": PHI_3_MINI,
    # HuggingFace Hub IDs as aliases
    "google/gemma-2b-it": GEMMA_2B_IT,
    "google/gemma-2b": GEMMA_2B_BASE,
    "google/gemma-7b-it": GEMMA_7B_IT,
    "google/gemma-7b": GEMMA_7B_BASE,
    "microsoft/phi-2": PHI_2,
    "microsoft/Phi-3-mini-4k-instruct": PHI_3_MINI,
}

# Default models by category
DEFAULT_MODELS: dict[str, str] = {
    "default": "gemma-2b-it",
    "lightweight": "gemma-2b-it",
    "performance": "gemma-7b-it",
    "coding": "phi-2",
    "chat": "gemma-2b-it",
}

# Model recommendations by use case
RECOMMENDED_MODELS: dict[str, list[str]] = {
    "development": ["gemma-2b-it", "phi-2"],
    "production": ["gemma-7b-it", "gemma-9b-it"],
    "research": ["gemma-7b-it", "gemma-2b-base"],
    "education": ["gemma-2b-it", "phi-3-mini"],
    "low_resource": ["gemma-2b-it", "phi-2"],
    "high_quality": ["gemma-7b-it", "gemma-9b-it"],
}


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================


class InferenceConfig(BaseModel):
    """Configuration for model inference parameters."""

    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    top_k: int = Field(50, ge=1, description="Top-k sampling")
    max_tokens: int = Field(512, ge=1, le=8192, description="Maximum tokens to generate")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")
    do_sample: bool = Field(True, description="Whether to use sampling")
    num_return_sequences: int = Field(1, ge=1, le=10, description="Number of sequences to return")

    # Stop sequences
    stop_sequences: list[str] = Field(
        default_factory=list, description="Stop generation at these sequences"
    )

    # Advanced parameters
    length_penalty: float = Field(1.0, ge=0.0, le=2.0, description="Length penalty for beam search")
    early_stopping: bool = Field(False, description="Enable early stopping")

    @field_validator("stop_sequences")
    @classmethod
    def validate_stop_sequences(cls, v: list[str]) -> list[str]:
        """Validate stop sequences are reasonable."""
        if len(v) > 10:
            raise ValueError("Too many stop sequences (max 10)")
        for seq in v:
            if len(seq) > 50:
                raise ValueError("Stop sequence too long (max 50 chars)")
        return v


class QuantizationConfig(BaseModel):
    """Configuration for model quantization."""

    enabled: bool = Field(False, description="Enable quantization")
    method: str = Field("8bit", description="Quantization method (8bit, 4bit, int8)")
    device_map: str = Field("auto", description="Device mapping strategy")
    max_memory: dict[str, str] | None = Field(None, description="Per-device memory limits")

    # BitsAndBytes configuration
    load_in_8bit: bool = Field(False, description="Use 8-bit quantization")
    load_in_4bit: bool = Field(False, description="Use 4-bit quantization")
    bnb_4bit_compute_dtype: str = Field("float16", description="4-bit computation dtype")
    bnb_4bit_use_double_quant: bool = Field(False, description="Use double quantization")

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate quantization method."""
        valid_methods = {"8bit", "4bit", "int8", "int4", "fp16", "bf16"}
        if v not in valid_methods:
            raise ValueError(f"Invalid quantization method. Must be one of: {valid_methods}")
        return v


class ModelLoadConfig(BaseModel):
    """Configuration for model loading."""

    model_name: str = Field(..., description="Model name or path")
    device: str = Field("auto", description="Device to load model on")
    torch_dtype: str = Field("auto", description="PyTorch dtype")
    trust_remote_code: bool = Field(False, description="Trust remote code execution")
    cache_dir: Path | None = Field(None, description="Model cache directory")

    # Loading options
    low_cpu_mem_usage: bool = Field(True, description="Use low CPU memory during loading")
    use_safetensors: bool = Field(True, description="Use safetensors format when available")

    # Quantization
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)

    # Inference parameters
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device specification."""
        valid_devices = {"auto", "cpu", "cuda", "mps", "cuda:0", "cuda:1"}
        if v not in valid_devices and not v.startswith("cuda:"):
            raise ValueError(f"Invalid device: {v}")
        return v

    @field_validator("torch_dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        """Validate PyTorch dtype."""
        valid_dtypes = {"auto", "float32", "float16", "bfloat16", "int8"}
        if v not in valid_dtypes:
            raise ValueError(f"Invalid dtype: {v}")
        return v


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_model_spec(model_name: str) -> ModelSpec:
    """Get model specification by name.

    Args:
        model_name: Model name or HuggingFace model ID

    Returns:
        ModelSpec for the requested model

    Raises:
        ValueError: If model is not found
    """
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]

    # Try to find by partial match
    for key, spec in MODEL_REGISTRY.items():
        if model_name in key or key in model_name:
            return spec

    raise ValueError(
        f"Model '{model_name}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}"
    )


def get_default_model(category: str = "default") -> ModelSpec:
    """Get default model for a category.

    Args:
        category: Model category (default, lightweight, performance, etc.)

    Returns:
        ModelSpec for the default model in that category
    """
    model_name = DEFAULT_MODELS.get(category, DEFAULT_MODELS["default"])
    return get_model_spec(model_name)


def list_models_by_family(family: ModelFamily) -> list[ModelSpec]:
    """List all models in a family.

    Args:
        family: Model family to filter by

    Returns:
        List of ModelSpec objects for that family
    """
    return [spec for spec in MODEL_REGISTRY.values() if spec.family == family]


def list_models_by_size(size: ModelSize) -> list[ModelSpec]:
    """List all models of a given size.

    Args:
        size: Model size to filter by

    Returns:
        List of ModelSpec objects of that size
    """
    return [spec for spec in MODEL_REGISTRY.values() if spec.size == size]


def get_recommended_models(use_case: str) -> list[ModelSpec]:
    """Get recommended models for a use case.

    Args:
        use_case: Use case (development, production, etc.)

    Returns:
        List of recommended ModelSpec objects
    """
    model_names = RECOMMENDED_MODELS.get(use_case, [DEFAULT_MODELS["default"]])
    return [get_model_spec(name) for name in model_names]


def create_model_config(
    model_name: str, device: str = "auto", quantization: bool = False, **inference_kwargs
) -> ModelLoadConfig:
    """Create a model loading configuration.

    Args:
        model_name: Model name to load
        device: Device to load on
        quantization: Enable quantization
        **inference_kwargs: Additional inference parameters

    Returns:
        ModelLoadConfig object
    """
    spec = get_model_spec(model_name)

    # Set up quantization if requested
    quant_config = QuantizationConfig()
    if quantization:
        quant_config.enabled = True
        quant_config.load_in_8bit = True

    # Set up inference parameters with model defaults
    inference_config = InferenceConfig(
        temperature=inference_kwargs.get("temperature", spec.default_temperature),
        top_p=inference_kwargs.get("top_p", spec.default_top_p),
        top_k=inference_kwargs.get("top_k", spec.default_top_k),
        max_tokens=inference_kwargs.get("max_tokens", spec.default_max_tokens),
        repetition_penalty=inference_kwargs.get(
            "repetition_penalty", spec.default_repetition_penalty
        ),
        **{k: v for k, v in inference_kwargs.items() if k in InferenceConfig.__fields__},
    )

    return ModelLoadConfig(
        model_name=spec.hf_model_id,
        device=device,
        quantization=quant_config,
        inference=inference_config,
    )


def validate_model_requirements(model_name: str, available_memory_gb: float) -> tuple[bool, str]:
    """Validate if a model can run with available resources.

    Args:
        model_name: Model name to validate
        available_memory_gb: Available memory in GB

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        spec = get_model_spec(model_name)
    except ValueError as e:
        return False, str(e)

    if available_memory_gb < spec.min_memory_gb:
        return (
            False,
            f"Insufficient memory. Required: {spec.min_memory_gb}GB, Available: {available_memory_gb}GB",
        )

    if available_memory_gb < spec.recommended_memory_gb:
        return (
            True,
            f"Model will run but may be slow. Recommended: {spec.recommended_memory_gb}GB, Available: {available_memory_gb}GB",
        )

    return True, "Model requirements satisfied"
