"""Model and profile management system for gemma-cli.

This module provides comprehensive model discovery, validation, and performance
profile management with hardware detection and recommendation capabilities.
"""

import platform
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil
from pydantic import BaseModel, Field, ValidationError, field_validator
from rich.console import Console
from rich.table import Table

console = Console()


class ModelPreset(BaseModel):
    """Model preset configuration with metadata.

    Represents a complete model configuration including paths, format,
    performance characteristics, and recommended use cases.
    """

    name: str = Field(..., description="Unique model identifier")
    weights: str = Field(..., description="Path to model weights file (.sbs)")
    tokenizer: str = Field(..., description="Path to tokenizer file (.spm)")
    format: str = Field(..., description="Weight format (sfp, bf16, f32, nuq)")
    size_gb: float = Field(..., gt=0, description="Model size in GB")
    avg_tokens_per_sec: int = Field(..., gt=0, description="Average inference speed")
    quality: str = Field(..., description="Quality tier (high, medium, fast)")
    use_case: str = Field(..., description="Recommended use case")
    context_length: int = Field(default=8192, gt=0, description="Maximum context length")
    min_ram_gb: int = Field(default=4, gt=0, description="Minimum RAM requirement")

    @field_validator("weights", "tokenizer")
    @classmethod
    def validate_path_format(cls, v: str) -> str:
        """Validate path uses forward slashes or proper OS format."""
        if not v:
            raise ValueError("Path cannot be empty")
        # Convert to Path and back to normalize
        return str(Path(v))

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate model format is supported."""
        valid_formats = ["sfp", "bf16", "f32", "nuq"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Format must be one of: {valid_formats}")
        return v.lower()

    @field_validator("quality")
    @classmethod
    def validate_quality(cls, v: str) -> str:
        """Validate quality tier."""
        valid_qualities = ["high", "medium", "fast"]
        if v.lower() not in valid_qualities:
            raise ValueError(f"Quality must be one of: {valid_qualities}")
        return v.lower()


class PerformanceProfile(BaseModel):
    """Performance profile for inference tuning.

    Defines generation parameters optimized for different use cases
    (speed vs quality trade-offs).
    """

    name: str = Field(..., description="Profile identifier")
    max_tokens: int = Field(..., gt=0, le=32768, description="Maximum tokens to generate")
    temperature: float = Field(..., ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, gt=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(default=40, gt=0, description="Top-K sampling parameter")
    description: str = Field(..., description="Profile description")
    use_case: str = Field(default="general", description="Recommended use case")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Ensure temperature is in valid range."""
        if v < 0 or v > 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v


class HardwareInfo(BaseModel):
    """Detected hardware capabilities."""

    cpu_cores: int
    cpu_logical: int
    cpu_freq_mhz: float
    ram_total_gb: float
    ram_available_gb: float
    has_gpu: bool
    gpu_info: Optional[str] = None
    os_system: str
    os_release: str


class ModelManager:
    """Manages model presets, discovery, and validation.

    Provides centralized model management including:
    - Loading and saving model presets
    - Auto-discovery of models in standard locations
    - Model file validation
    - Default model configuration
    """

    def __init__(self, config_path: Path):
        """Initialize ModelManager.

        Args:
            config_path: Path to configuration file (config.toml)
        """
        self.config_path = config_path
        self.config_dir = config_path.parent
        self._models: Dict[str, ModelPreset] = {}
        self._default_model: Optional[str] = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from TOML file."""
        if not self.config_path.exists():
            console.print(f"[yellow]Config not found: {self.config_path}[/yellow]")
            return

        try:
            with open(self.config_path, "rb") as f:
                config = tomllib.load(f)

            # Load model presets if they exist
            if "model_presets" in config:
                for name, preset_data in config["model_presets"].items():
                    try:
                        preset = ModelPreset(name=name, **preset_data)
                        self._models[name] = preset
                    except ValidationError as e:
                        console.print(f"[red]Invalid preset '{name}': {e}[/red]")

            # Load default model
            if "model" in config and "default_model" in config["model"]:
                self._default_model = config["model"]["default_model"]

        except Exception as e:
            console.print(f"[red]Failed to load config: {e}[/red]")

    def list_models(self) -> List[ModelPreset]:
        """List all available model presets.

        Returns:
            List of ModelPreset objects sorted by name
        """
        return sorted(self._models.values(), key=lambda m: m.name)

    def get_model(self, name: str) -> Optional[ModelPreset]:
        """Get model preset by name.

        Args:
            name: Model identifier

        Returns:
            ModelPreset if found, None otherwise
        """
        return self._models.get(name)

    def get_default_model(self) -> Optional[ModelPreset]:
        """Get the default model preset.

        Returns:
            Default ModelPreset if set, None otherwise
        """
        if self._default_model:
            return self.get_model(self._default_model)
        return None

    def set_default_model(self, name: str) -> bool:
        """Set default model in configuration.

        Args:
            name: Model identifier

        Returns:
            True if successful, False otherwise
        """
        if name not in self._models:
            console.print(f"[red]Model '{name}' not found[/red]")
            return False

        self._default_model = name

        # Update config file
        try:
            config = {}
            if self.config_path.exists():
                with open(self.config_path, "rb") as f:
                    config = tomllib.load(f)

            if "model" not in config:
                config["model"] = {}
            config["model"]["default_model"] = name

            # Write back (note: this is simplified, production would preserve comments)
            self._write_config(config)
            console.print(f"[green]Default model set to: {name}[/green]")
            return True

        except Exception as e:
            console.print(f"[red]Failed to update config: {e}[/red]")
            return False

    def _write_config(self, config: Dict[str, Any]) -> None:
        """Write configuration to TOML file (simplified version)."""
        # Note: This is a basic implementation. Production version would use
        # a proper TOML writer that preserves formatting and comments.
        import tomli_w

        with open(self.config_path, "wb") as f:
            tomli_w.dump(config, f)

    def detect_models(self, search_paths: Optional[List[Path]] = None) -> List[Tuple[Path, Path]]:
        """Auto-detect models in standard locations.

        Searches for .sbs weight files and matching .spm tokenizer files.

        Args:
            search_paths: List of directories to search (default: standard locations)

        Returns:
            List of (weights_path, tokenizer_path) tuples
        """
        if search_paths is None:
            # Standard model locations
            search_paths = [
                Path("C:/codedev/llm/.models"),
                Path.home() / ".cache" / "gemma",
                Path.home() / "models",
            ]

        detected = []

        for search_path in search_paths:
            if not search_path.exists():
                continue

            # Find all .sbs files
            for weights_file in search_path.rglob("*.sbs"):
                # Look for tokenizer in same directory
                tokenizer_file = weights_file.parent / "tokenizer.spm"

                if tokenizer_file.exists():
                    detected.append((weights_file, tokenizer_file))
                else:
                    console.print(
                        f"[yellow]Warning: Found weights but no tokenizer: {weights_file}[/yellow]"
                    )

        return detected

    def validate_model(self, preset: ModelPreset) -> Tuple[bool, List[str]]:
        """Verify model files exist and are valid.

        Args:
            preset: ModelPreset to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check weights file
        weights_path = Path(preset.weights)
        if not weights_path.exists():
            errors.append(f"Weights file not found: {weights_path}")
        elif not weights_path.suffix == ".sbs":
            errors.append(f"Invalid weights file format: {weights_path.suffix}")
        else:
            # Check file size matches declared size (within 10% tolerance)
            actual_size_gb = weights_path.stat().st_size / (1024**3)
            size_diff = abs(actual_size_gb - preset.size_gb) / preset.size_gb
            if size_diff > 0.1:
                errors.append(
                    f"Size mismatch: declared {preset.size_gb}GB, "
                    f"actual {actual_size_gb:.2f}GB"
                )

        # Check tokenizer file
        tokenizer_path = Path(preset.tokenizer)
        if not tokenizer_path.exists():
            errors.append(f"Tokenizer file not found: {tokenizer_path}")
        elif not tokenizer_path.suffix == ".spm":
            errors.append(f"Invalid tokenizer format: {tokenizer_path.suffix}")

        return len(errors) == 0, errors

    def get_model_info(self, preset: ModelPreset) -> Dict[str, Any]:
        """Get detailed model information.

        Args:
            preset: ModelPreset to analyze

        Returns:
            Dictionary with detailed model information
        """
        weights_path = Path(preset.weights)
        tokenizer_path = Path(preset.tokenizer)

        info = {
            "name": preset.name,
            "format": preset.format,
            "quality": preset.quality,
            "use_case": preset.use_case,
            "size_gb": preset.size_gb,
            "context_length": preset.context_length,
            "min_ram_gb": preset.min_ram_gb,
            "avg_tokens_per_sec": preset.avg_tokens_per_sec,
            "weights_exists": weights_path.exists(),
            "tokenizer_exists": tokenizer_path.exists(),
        }

        if weights_path.exists():
            stat = weights_path.stat()
            info["weights_size_bytes"] = stat.st_size
            info["weights_size_gb"] = stat.st_size / (1024**3)
            info["weights_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

        if tokenizer_path.exists():
            stat = tokenizer_path.stat()
            info["tokenizer_size_bytes"] = stat.st_size
            info["tokenizer_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

        return info

    def display_models_table(self) -> None:
        """Display models in a rich table format."""
        models = self.list_models()

        if not models:
            console.print("[yellow]No model presets configured[/yellow]")
            return

        table = Table(title="Model Presets", show_header=True)
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Size", style="green")
        table.add_column("Format", style="yellow")
        table.add_column("Quality", style="magenta")
        table.add_column("Speed", style="blue")
        table.add_column("Use Case", style="dim")

        for model in models:
            is_default = model.name == self._default_model
            name_display = f"[bold]{model.name}[/bold] (default)" if is_default else model.name

            table.add_row(
                name_display,
                f"{model.size_gb:.1f} GB",
                model.format.upper(),
                model.quality.capitalize(),
                f"{model.avg_tokens_per_sec} tok/s",
                model.use_case,
            )

        console.print(table)


class ProfileManager:
    """Manages performance profiles for inference tuning.

    Provides profile management including:
    - Loading and saving profiles
    - Creating custom profiles
    - Profile recommendations based on use case
    """

    def __init__(self, config_path: Path):
        """Initialize ProfileManager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self._profiles: Dict[str, PerformanceProfile] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load performance profiles from configuration."""
        if not self.config_path.exists():
            return

        try:
            with open(self.config_path, "rb") as f:
                config = tomllib.load(f)

            if "performance_profiles" in config:
                for name, profile_data in config["performance_profiles"].items():
                    try:
                        profile = PerformanceProfile(name=name, **profile_data)
                        self._profiles[name] = profile
                    except ValidationError as e:
                        console.print(f"[red]Invalid profile '{name}': {e}[/red]")

        except Exception as e:
            console.print(f"[red]Failed to load profiles: {e}[/red]")

    def list_profiles(self) -> List[PerformanceProfile]:
        """List all performance profiles.

        Returns:
            List of PerformanceProfile objects sorted by name
        """
        return sorted(self._profiles.values(), key=lambda p: p.name)

    def get_profile(self, name: str) -> Optional[PerformanceProfile]:
        """Get profile by name.

        Args:
            name: Profile identifier

        Returns:
            PerformanceProfile if found, None otherwise
        """
        return self._profiles.get(name)

    def create_profile(
        self,
        name: str,
        max_tokens: int,
        temperature: float,
        top_p: float = 0.95,
        top_k: int = 40,
        description: str = "",
        use_case: str = "general",
    ) -> bool:
        """Create custom performance profile.

        Args:
            name: Profile identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-K sampling parameter
            description: Profile description
            use_case: Recommended use case

        Returns:
            True if successful, False otherwise
        """
        try:
            profile = PerformanceProfile(
                name=name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                description=description,
                use_case=use_case,
            )

            self._profiles[name] = profile
            self._save_profile(profile)
            console.print(f"[green]Profile '{name}' created successfully[/green]")
            return True

        except ValidationError as e:
            console.print(f"[red]Invalid profile parameters: {e}[/red]")
            return False

    def update_profile(self, name: str, **kwargs: Any) -> bool:
        """Update existing profile.

        Args:
            name: Profile identifier
            **kwargs: Fields to update

        Returns:
            True if successful, False otherwise
        """
        if name not in self._profiles:
            console.print(f"[red]Profile '{name}' not found[/red]")
            return False

        try:
            current = self._profiles[name]
            updated_data = current.model_dump()
            updated_data.update(kwargs)

            new_profile = PerformanceProfile(**updated_data)
            self._profiles[name] = new_profile
            self._save_profile(new_profile)

            console.print(f"[green]Profile '{name}' updated successfully[/green]")
            return True

        except ValidationError as e:
            console.print(f"[red]Invalid update parameters: {e}[/red]")
            return False

    def delete_profile(self, name: str) -> bool:
        """Remove custom profile.

        Args:
            name: Profile identifier

        Returns:
            True if successful, False otherwise
        """
        if name not in self._profiles:
            console.print(f"[red]Profile '{name}' not found[/red]")
            return False

        del self._profiles[name]
        self._remove_profile_from_config(name)
        console.print(f"[green]Profile '{name}' deleted[/green]")
        return True

    def _save_profile(self, profile: PerformanceProfile) -> None:
        """Save profile to configuration file."""
        try:
            config = {}
            if self.config_path.exists():
                with open(self.config_path, "rb") as f:
                    config = tomllib.load(f)

            if "performance_profiles" not in config:
                config["performance_profiles"] = {}

            config["performance_profiles"][profile.name] = profile.model_dump(
                exclude={"name"}
            )

            self._write_config(config)

        except Exception as e:
            console.print(f"[red]Failed to save profile: {e}[/red]")

    def _remove_profile_from_config(self, name: str) -> None:
        """Remove profile from configuration file."""
        try:
            if not self.config_path.exists():
                return

            with open(self.config_path, "rb") as f:
                config = tomllib.load(f)

            if "performance_profiles" in config and name in config["performance_profiles"]:
                del config["performance_profiles"][name]
                self._write_config(config)

        except Exception as e:
            console.print(f"[red]Failed to remove profile: {e}[/red]")

    def _write_config(self, config: Dict[str, Any]) -> None:
        """Write configuration to TOML file."""
        import tomli_w

        with open(self.config_path, "wb") as f:
            tomli_w.dump(config, f)

    def recommend_profile(self, hardware: "HardwareInfo") -> Optional[PerformanceProfile]:
        """Suggest profile based on hardware capabilities.

        Args:
            hardware: Detected hardware information

        Returns:
            Recommended PerformanceProfile or None
        """
        # Simple recommendation logic based on available RAM
        if hardware.ram_available_gb < 4:
            # Low memory: conservative settings
            return self.get_profile("fast") or self._create_fallback_profile("fast", 512, 0.5)
        elif hardware.ram_available_gb < 8:
            # Medium memory: balanced settings
            return self.get_profile("balanced") or self._create_fallback_profile(
                "balanced", 1024, 0.7
            )
        else:
            # High memory: quality settings
            return self.get_profile("quality") or self._create_fallback_profile(
                "quality", 2048, 0.9
            )

    def _create_fallback_profile(
        self, name: str, max_tokens: int, temperature: float
    ) -> PerformanceProfile:
        """Create a fallback profile if config doesn't have one."""
        return PerformanceProfile(
            name=name,
            max_tokens=max_tokens,
            temperature=temperature,
            description=f"Auto-generated {name} profile",
        )

    def display_profiles_table(self) -> None:
        """Display profiles in a rich table format."""
        profiles = self.list_profiles()

        if not profiles:
            console.print("[yellow]No performance profiles configured[/yellow]")
            return

        table = Table(title="Performance Profiles", show_header=True)
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Max Tokens", style="green")
        table.add_column("Temperature", style="yellow")
        table.add_column("Top-P", style="blue")
        table.add_column("Top-K", style="magenta")
        table.add_column("Description", style="dim")

        for profile in profiles:
            table.add_row(
                profile.name,
                str(profile.max_tokens),
                f"{profile.temperature:.2f}",
                f"{profile.top_p:.2f}",
                str(profile.top_k),
                profile.description,
            )

        console.print(table)


class HardwareDetector:
    """Detects hardware capabilities and recommends settings.

    Provides hardware detection including:
    - CPU information (cores, frequency)
    - Memory availability
    - GPU detection
    - Model and settings recommendations
    """

    def __init__(self) -> None:
        """Initialize HardwareDetector."""
        self._cached_info: Optional[HardwareInfo] = None

    def detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU information.

        Returns:
            Dictionary with CPU details
        """
        try:
            freq = psutil.cpu_freq()
            return {
                "physical_cores": psutil.cpu_count(logical=False) or 0,
                "logical_cores": psutil.cpu_count(logical=True) or 0,
                "frequency_mhz": freq.current if freq else 0.0,
                "max_frequency_mhz": freq.max if freq else 0.0,
                "processor": platform.processor(),
            }
        except Exception as e:
            console.print(f"[red]CPU detection failed: {e}[/red]")
            return {"physical_cores": 1, "logical_cores": 1, "frequency_mhz": 0.0}

    def detect_memory(self) -> Dict[str, Any]:
        """Get available RAM information.

        Returns:
            Dictionary with memory details
        """
        try:
            mem = psutil.virtual_memory()
            return {
                "total_gb": mem.total / (1024**3),
                "available_gb": mem.available / (1024**3),
                "used_gb": mem.used / (1024**3),
                "percent_used": mem.percent,
            }
        except Exception as e:
            console.print(f"[red]Memory detection failed: {e}[/red]")
            return {"total_gb": 0.0, "available_gb": 0.0}

    def detect_gpu(self) -> Tuple[bool, Optional[str]]:
        """Check for GPU availability.

        Returns:
            Tuple of (has_gpu, gpu_info_string)
        """
        gpu_info = None

        # Try CUDA (NVIDIA)
        try:
            import torch

            if torch.cuda.is_available():
                gpu_info = f"CUDA: {torch.cuda.get_device_name(0)}"
                return True, gpu_info
        except ImportError:
            pass

        # Try ROCm (AMD)
        try:
            import torch

            if hasattr(torch, "hip") and torch.hip.is_available():
                gpu_info = "ROCm: AMD GPU detected"
                return True, gpu_info
        except (ImportError, AttributeError):
            pass

        # Try Intel GPU detection (basic)
        try:
            if platform.system() == "Windows":
                import subprocess

                result = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if "Intel" in result.stdout:
                    gpu_info = "Intel GPU detected (no acceleration)"
                    return False, gpu_info  # Detection only, not accelerated
        except Exception:
            pass

        return False, None

    def get_hardware_info(self, refresh: bool = False) -> HardwareInfo:
        """Get comprehensive hardware information.

        Args:
            refresh: Force refresh of cached info

        Returns:
            HardwareInfo object with complete system details
        """
        if self._cached_info and not refresh:
            return self._cached_info

        cpu_info = self.detect_cpu()
        mem_info = self.detect_memory()
        has_gpu, gpu_info = self.detect_gpu()

        self._cached_info = HardwareInfo(
            cpu_cores=cpu_info["physical_cores"],
            cpu_logical=cpu_info["logical_cores"],
            cpu_freq_mhz=cpu_info["frequency_mhz"],
            ram_total_gb=mem_info["total_gb"],
            ram_available_gb=mem_info["available_gb"],
            has_gpu=has_gpu,
            gpu_info=gpu_info,
            os_system=platform.system(),
            os_release=platform.release(),
        )

        return self._cached_info

    def recommend_model(
        self, model_manager: ModelManager, hardware: Optional[HardwareInfo] = None
    ) -> Optional[ModelPreset]:
        """Suggest best model for hardware.

        Args:
            model_manager: ModelManager instance
            hardware: Hardware info (auto-detected if not provided)

        Returns:
            Recommended ModelPreset or None
        """
        if hardware is None:
            hardware = self.get_hardware_info()

        models = model_manager.list_models()
        if not models:
            return None

        # Filter models by RAM availability
        suitable_models = [
            m for m in models if m.min_ram_gb <= hardware.ram_available_gb
        ]

        if not suitable_models:
            console.print(
                f"[yellow]Warning: Available RAM ({hardware.ram_available_gb:.1f}GB) "
                f"may be insufficient for any configured model[/yellow]"
            )
            suitable_models = models  # Use all models anyway

        # Recommend based on quality if enough RAM, otherwise fastest
        if hardware.ram_available_gb >= 8:
            # Prefer high quality
            quality_models = [m for m in suitable_models if m.quality == "high"]
            if quality_models:
                return max(quality_models, key=lambda m: m.avg_tokens_per_sec)

        # Prefer fast models for lower RAM
        return max(suitable_models, key=lambda m: m.avg_tokens_per_sec)

    def recommend_settings(
        self, hardware: Optional[HardwareInfo] = None
    ) -> Dict[str, Any]:
        """Suggest optimal settings based on hardware.

        Args:
            hardware: Hardware info (auto-detected if not provided)

        Returns:
            Dictionary with recommended settings
        """
        if hardware is None:
            hardware = self.get_hardware_info()

        settings = {
            "max_tokens": 1024,
            "temperature": 0.7,
            "context_length": 4096,
            "batch_size": 1,
        }

        # Adjust based on available memory
        if hardware.ram_available_gb < 4:
            settings.update({
                "max_tokens": 512,
                "context_length": 2048,
                "batch_size": 1,
            })
        elif hardware.ram_available_gb < 8:
            settings.update({
                "max_tokens": 1024,
                "context_length": 4096,
                "batch_size": 1,
            })
        else:
            settings.update({
                "max_tokens": 2048,
                "context_length": 8192,
                "batch_size": 2 if hardware.cpu_cores >= 4 else 1,
            })

        # Adjust for CPU capabilities
        if hardware.cpu_cores >= 8:
            settings["num_threads"] = min(hardware.cpu_cores - 2, 12)
        else:
            settings["num_threads"] = max(1, hardware.cpu_cores - 1)

        return settings

    def display_hardware_info(self, hardware: Optional[HardwareInfo] = None) -> None:
        """Display hardware information in a rich table format.

        Args:
            hardware: Hardware info (auto-detected if not provided)
        """
        if hardware is None:
            hardware = self.get_hardware_info()

        table = Table(title="Hardware Information", show_header=True)
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Details", style="green")

        # CPU
        table.add_row(
            "CPU",
            f"{hardware.cpu_cores} cores / {hardware.cpu_logical} threads @ "
            f"{hardware.cpu_freq_mhz:.0f} MHz",
        )

        # Memory
        table.add_row(
            "RAM",
            f"{hardware.ram_total_gb:.1f} GB total / "
            f"{hardware.ram_available_gb:.1f} GB available",
        )

        # GPU
        gpu_status = hardware.gpu_info if hardware.has_gpu else "No GPU detected"
        table.add_row("GPU", gpu_status)

        # OS
        table.add_row("OS", f"{hardware.os_system} {hardware.os_release}")

        console.print(table)
