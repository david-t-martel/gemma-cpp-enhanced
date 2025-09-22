"""Gemma model-specific agent implementation."""

from enum import Enum
from typing import Any

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import pipeline

from ..shared.config.model_configs import DEFAULT_MODELS
from ..shared.config.model_configs import get_model_spec
from ..shared.logging import get_logger
from .core import BaseAgent
from .gemma_bridge import GemmaNativeBridge, create_native_bridge
from .tools import ToolRegistry

logger = get_logger(__name__)


class AgentMode(Enum):
    """Agent operation modes."""

    FULL = "full"  # Full model with direct PyTorch usage
    LIGHTWEIGHT = "lightweight"  # Pipeline-based for easier setup
    NATIVE = "native"  # Native C++ implementation for maximum performance


class UnifiedGemmaAgent(BaseAgent):
    """Unified Gemma agent supporting both full and lightweight modes."""

    def __init__(
        self,
        model_name: str | None = None,
        mode: AgentMode = AgentMode.LIGHTWEIGHT,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 5,
        verbose: bool = True,
        device: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        use_8bit: bool = False,
    ):
        """Initialize unified Gemma agent.

        Args:
            model_name: Name of the Gemma model to use
            mode: Agent operation mode (full or lightweight)
            tool_registry: Registry of available tools
            system_prompt: System prompt for the agent
            max_iterations: Maximum number of tool calling iterations
            verbose: Whether to print verbose output
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            use_8bit: Whether to use 8-bit quantization (full mode only)
        """
        super().__init__(tool_registry, system_prompt, max_iterations, verbose)

        self.mode = mode

        # Set default model based on mode
        if model_name is None:
            model_name = (
                DEFAULT_MODELS["default"]
                if mode == AgentMode.FULL
                else DEFAULT_MODELS["lightweight"]
            )

        # Get model specification and use defaults if not provided
        self.model_spec = get_model_spec(model_name)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens or self.model_spec.default_max_tokens
        self.temperature = temperature or self.model_spec.default_temperature
        self.top_p = top_p or self.model_spec.default_top_p

        # Initialize based on mode
        if self.mode == AgentMode.FULL:
            self._init_full_mode(device, use_8bit)
        elif self.mode == AgentMode.NATIVE:
            self._init_native_mode()
        else:
            self._init_lightweight_mode()

    def _init_full_mode(self, device: str | None, use_8bit: bool) -> None:
        """Initialize full mode with direct PyTorch usage."""
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.verbose:
            print(f"🚀 Initializing model (FULL mode): {self.model_name}")
            print(f"   Device: {self.device}")
            if use_8bit:
                print("   Using 8-bit quantization")

        # List of models to try in order of preference
        models_to_try = [self.model_name]

        # Add local model fallbacks
        local_model_paths = [
            "models/microsoft_phi-2",  # Local Phi-2 model
            "models/models--microsoft--phi-2/snapshots",  # Alternative Phi-2 path
            "microsoft/phi-2",  # Try direct download as last resort
        ]

        # Only add fallbacks if the original model isn't already a local path
        if not self.model_name.startswith("models/"):
            models_to_try.extend(local_model_paths)

        # Try each model until one works
        last_error = None
        for model_path in models_to_try:
            try:
                if self.verbose:
                    print(f"   Trying model: {model_path}")

                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore[no-untyped-call]
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Load model with optional 8-bit quantization
                model_kwargs: dict[str, Any] = {}
                if use_8bit and self.device == "cuda":
                    model_kwargs["load_in_8bit"] = True
                    model_kwargs["device_map"] = "auto"
                else:
                    model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32

                self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)  # type: ignore[no-untyped-call]

                if not use_8bit and self.device == "cuda" and self.model is not None:
                    self.model = self.model.to(self.device)  # type: ignore[arg-type]

                if self.verbose:
                    print(f"✅ Model loaded successfully with: {model_path}")

                # Update model name to reflect what we actually loaded
                self.model_name = model_path
                break

            except Exception as e:
                last_error = e
                if self.verbose:
                    print(f"   ❌ Failed with {model_path}: {e}")
                continue
        else:
            # All models failed - try CPU fallback with the first model
            logger.error(f"Failed to load any model in requested configuration. Last error: {last_error}")
            logger.info("Attempting CPU fallback with float32...")

            try:
                self.device = "cpu"
                model_path = models_to_try[0] if models_to_try else self.model_name
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # type: ignore[no-untyped-call]
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.float32
                )  # type: ignore[no-untyped-call]

                if self.verbose:
                    print(f"✅ CPU fallback successful with: {model_path}")

            except Exception as fallback_error:
                logger.error(f"CPU fallback also failed: {fallback_error}")
                raise RuntimeError(f"No usable model found. Models tried: {models_to_try}. Last error: {last_error}")

        # Pipeline is None in full mode
        self.pipeline = None

    def _init_lightweight_mode(self) -> None:
        """Initialize lightweight mode with pipeline."""
        if self.verbose:
            print(f"🚀 Initializing model (LIGHTWEIGHT mode): {self.model_name}")

        # List of models to try in order of preference
        models_to_try = [self.model_name]

        # Add local model fallbacks
        local_model_paths = [
            "models/microsoft_phi-2",  # Local Phi-2 model
            "models/models--microsoft--phi-2/snapshots",  # Alternative Phi-2 path
            "microsoft/phi-2",  # Try direct download as last resort
        ]

        # Only add fallbacks if the original model isn't already a local path
        if not self.model_name.startswith("models/"):
            models_to_try.extend(local_model_paths)

        # Create text generation pipeline with fallback
        last_error = None
        for model_path in models_to_try:
            try:
                if self.verbose:
                    print(f"   Trying model: {model_path}")

                # Type ignore needed for transformers pipeline typing
                self.pipeline = pipeline(
                    "text-generation",
                    model=model_path,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                )  # type: ignore[assignment,no-untyped-call]

                if self.verbose:
                    print(f"✅ Pipeline created successfully with: {model_path}")

                # Update model name to reflect what we actually loaded
                self.model_name = model_path
                break

            except Exception as e:
                last_error = e
                if self.verbose:
                    print(f"   ❌ Failed with {model_path}: {e}")
                continue
        else:
            # All models failed
            logger.error(f"Failed to create pipeline with any model. Last error: {last_error}")
            raise RuntimeError(f"No usable model found. Models tried: {models_to_try}. Last error: {last_error}")

        # Model and tokenizer are None in lightweight mode
        self.model = None  # type: ignore[assignment]
        self.tokenizer = None
        self.device = "auto"

    def _init_native_mode(self) -> None:
        """Initialize native mode with C++ Gemma bridge."""
        if self.verbose:
            print(f"🚀 Initializing model (NATIVE mode): {self.model_name}")

        try:
            # Create native bridge with current parameters
            self.native_bridge = create_native_bridge(
                model_type="4b-it",  # Use the Gemma 3 4B model
                verbose=self.verbose,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            
            # Test that the model loads properly
            if not self.native_bridge.load_native_model():
                raise RuntimeError("Failed to initialize native Gemma model")
            
            if self.verbose:
                print("✅ Native Gemma bridge initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize native mode: {e}")
            raise RuntimeError(f"Native mode initialization failed: {e}")

        # Set attributes for compatibility
        self.model = None  # type: ignore[assignment]
        self.tokenizer = None
        self.pipeline = None
        self.device = "native"

    def generate_response(self, prompt: str) -> str:
        """Generate a response using the appropriate mode.

        Args:
            prompt: Input prompt

        Returns:
            Model response
        """
        if self.mode == AgentMode.FULL:
            return self._generate_full_mode(prompt)
        elif self.mode == AgentMode.NATIVE:
            return self._generate_native_mode(prompt)
        else:
            return self._generate_lightweight_mode(prompt)

    def _generate_full_mode(self, prompt: str) -> str:
        """Generate response using full PyTorch mode."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048, padding=True
            )

            if self.device == "cuda" and not hasattr(self.model, "hf_device_map"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            response: str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part (remove the input prompt)
            if prompt in response:
                response = response[len(prompt) :].strip()
            # Sometimes the tokenizer modifies the prompt slightly
            # Try to find where the assistant's response starts
            elif "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: Failed to generate response - {e!s}"

    def _generate_lightweight_mode(self, prompt: str) -> str:
        """Generate response using pipeline mode."""
        try:
            # Generate response using pipeline
            if self.pipeline is None:
                raise RuntimeError("Pipeline not initialized")

            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                return_full_text=False,  # Return only generated text
            )

            response = outputs[0]["generated_text"]
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: Failed to generate response - {e!s}"

    def _generate_native_mode(self, prompt: str) -> str:
        """Generate response using native C++ Gemma bridge."""
        try:
            if not hasattr(self, 'native_bridge') or self.native_bridge is None:
                raise RuntimeError("Native bridge not initialized")
            
            # Generate response using the native bridge
            response = self.native_bridge.generate_text(
                prompt=prompt,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response in native mode: {e}")
            return f"Error: Failed to generate response - {e!s}"


# Legacy classes for backwards compatibility
class GemmaAgent(UnifiedGemmaAgent):
    """Legacy GemmaAgent - use UnifiedGemmaAgent with mode=AgentMode.FULL instead."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(mode=AgentMode.FULL, **kwargs)


class LightweightGemmaAgent(UnifiedGemmaAgent):
    """Legacy LightweightGemmaAgent - use UnifiedGemmaAgent with mode=AgentMode.LIGHTWEIGHT instead."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(mode=AgentMode.LIGHTWEIGHT, **kwargs)


def create_gemma_agent(
    lightweight: bool = True,
    model_name: str | None = None,
    mode: AgentMode | None = None,
    **kwargs: Any,
) -> UnifiedGemmaAgent:
    """Factory function to create a Gemma agent.

    Args:
        lightweight: Whether to use lightweight pipeline version (deprecated, use mode instead)
        model_name: Model name to use (auto-selected based on mode if None)
        mode: Agent operation mode (overrides lightweight parameter)
        **kwargs: Additional arguments for agent initialization

    Returns:
        Configured Gemma agent
    """
    # Determine mode
    if mode is not None:
        agent_mode = mode
    else:
        agent_mode = AgentMode.LIGHTWEIGHT if lightweight else AgentMode.FULL

    return UnifiedGemmaAgent(model_name=model_name, mode=agent_mode, **kwargs)
