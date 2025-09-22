"""Backend implementations for different model types."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import logging

from .exceptions import BackendError, InferenceError, ModelLoadError
from .models import ModelInfo

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Supported backend types."""
    GEMMA_NATIVE = "gemma_native"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    stream: bool = False


class ModelBackend(ABC):
    """Abstract base class for model backends."""
    
    def __init__(self, model_info: ModelInfo) -> None:
        """Initialize backend with model information.
        
        Args:
            model_info: Model metadata and configuration
        """
        self.model_info = model_info
        self._loaded = False
        self._model: Optional[Any] = None
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the model for inference."""
        pass
    
    @abstractmethod
    async def unload_model(self) -> None:
        """Unload the model to free resources."""
        pass
    
    @abstractmethod
    async def generate_text(self, 
                           prompt: str, 
                           config: GenerationConfig) -> str:
        """Generate text synchronously.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    async def generate_stream(self, 
                             prompt: str, 
                             config: GenerationConfig) -> AsyncIterator[str]:
        """Generate text with streaming.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            
        Yields:
            Text chunks
        """
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    @property
    def backend_type(self) -> BackendType:
        """Get backend type."""
        return BackendType(self.model_info.backend_type)


class GemmaNativeBackend(ModelBackend):
    """Backend for local Gemma models using C++ native implementation."""
    
    def __init__(self, model_info: ModelInfo) -> None:
        super().__init__(model_info)
        self._native_bridge: Optional[Any] = None
    
    async def load_model(self) -> None:
        """Load Gemma model using native bridge."""
        try:
            # Import here to avoid circular dependencies
            from ..agent.gemma_bridge import create_native_bridge
            
            # Determine model type from file name
            model_type = "2b-it"  # Default
            if "7b" in self.model_info.name.lower():
                model_type = "7b-it"
            elif "9b" in self.model_info.name.lower():
                model_type = "9b-it"
            elif "27b" in self.model_info.name.lower():
                model_type = "27b-it"
            
            self._native_bridge = create_native_bridge(
                model_type=model_type,
                verbose=False,
                max_tokens=self.model_info.default_max_tokens,
                temperature=self.model_info.default_temperature,
                top_p=self.model_info.default_top_p,
            )
            
            # Load the native model
            if not self._native_bridge.load_native_model():
                raise ModelLoadError(
                    f"Failed to load native Gemma model: {self.model_info.name}",
                    model_path=self.model_info.model_path,
                    backend="gemma_native"
                )
            
            self._loaded = True
            logger.info(f"Loaded Gemma native model: {self.model_info.name}")
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to initialize Gemma native backend: {e}",
                model_path=self.model_info.model_path,
                backend="gemma_native"
            )
    
    async def unload_model(self) -> None:
        """Unload Gemma model."""
        if self._native_bridge:
            # Native bridge doesn't have explicit cleanup method
            self._native_bridge = None
        
        self._loaded = False
        logger.info(f"Unloaded Gemma native model: {self.model_info.name}")
    
    async def generate_text(self, prompt: str, config: GenerationConfig) -> str:
        """Generate text using native Gemma bridge."""
        if not self._loaded or not self._native_bridge:
            raise InferenceError(
                "Model not loaded",
                model_name=self.model_info.name,
                backend="gemma_native"
            )
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._native_bridge.generate_text,
                prompt,
                config.max_tokens,
                config.temperature,
                config.top_p
            )
            
            return response
            
        except Exception as e:
            raise InferenceError(
                f"Native Gemma inference failed: {e}",
                model_name=self.model_info.name,
                backend="gemma_native"
            )
    
    async def generate_stream(self, prompt: str, config: GenerationConfig) -> AsyncIterator[str]:
        """Generate streaming text (simulated for native backend)."""
        # Native bridge doesn't support true streaming, so we simulate it
        text = await self.generate_text(prompt, config)
        
        # Split into chunks and yield with small delays
        words = text.split()
        chunk_size = max(1, len(words) // 10)  # ~10 chunks
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if i + chunk_size < len(words):
                chunk += " "
            yield chunk
            await asyncio.sleep(0.1)  # Small delay for streaming effect


class HuggingFaceBackend(ModelBackend):
    """Backend for HuggingFace transformers models."""
    
    def __init__(self, model_info: ModelInfo) -> None:
        super().__init__(model_info)
        self._pipeline: Optional[Any] = None
    
    async def load_model(self) -> None:
        """Load HuggingFace model."""
        try:
            import torch
            from transformers import pipeline
            
            # Determine model path/ID
            model_id = self.model_info.huggingface_id or self.model_info.model_path
            if not model_id:
                raise ModelLoadError(
                    f"No model path or HuggingFace ID specified for {self.model_info.name}",
                    backend="huggingface"
                )
            
            # Create pipeline in thread pool
            loop = asyncio.get_event_loop()
            self._pipeline = await loop.run_in_executor(
                None,
                lambda: pipeline(
                    "text-generation",
                    model=model_id,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                )
            )
            
            self._loaded = True
            logger.info(f"Loaded HuggingFace model: {self.model_info.name}")
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load HuggingFace model: {e}",
                model_path=self.model_info.model_path,
                backend="huggingface"
            )
    
    async def unload_model(self) -> None:
        """Unload HuggingFace model."""
        if self._pipeline:
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
            self._pipeline = None
        
        self._loaded = False
        logger.info(f"Unloaded HuggingFace model: {self.model_info.name}")
    
    async def generate_text(self, prompt: str, config: GenerationConfig) -> str:
        """Generate text using HuggingFace pipeline."""
        if not self._loaded or not self._pipeline:
            raise InferenceError(
                "Model not loaded",
                model_name=self.model_info.name,
                backend="huggingface"
            )
        
        try:
            # Run in thread pool
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                lambda: self._pipeline(
                    prompt,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=True,
                    return_full_text=False,
                )
            )
            
            return outputs[0]["generated_text"]
            
        except Exception as e:
            raise InferenceError(
                f"HuggingFace inference failed: {e}",
                model_name=self.model_info.name,
                backend="huggingface"
            )
    
    async def generate_stream(self, prompt: str, config: GenerationConfig) -> AsyncIterator[str]:
        """Generate streaming text (simulated)."""
        text = await self.generate_text(prompt, config)
        
        # Simulate streaming by yielding words with delays
        words = text.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)


class OpenAIBackend(ModelBackend):
    """Backend for OpenAI API models."""
    
    def __init__(self, model_info: ModelInfo) -> None:
        super().__init__(model_info)
        self._client: Optional[Any] = None
    
    async def load_model(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai
            import os
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ModelLoadError(
                    "OPENAI_API_KEY environment variable not set",
                    backend="openai"
                )
            
            self._client = openai.AsyncOpenAI(api_key=api_key)
            self._loaded = True
            logger.info(f"Initialized OpenAI client for: {self.model_info.name}")
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to initialize OpenAI client: {e}",
                backend="openai"
            )
    
    async def unload_model(self) -> None:
        """Clean up OpenAI client."""
        if self._client:
            await self._client.close()
            self._client = None
        
        self._loaded = False
        logger.info(f"Closed OpenAI client for: {self.model_info.name}")
    
    async def generate_text(self, prompt: str, config: GenerationConfig) -> str:
        """Generate text using OpenAI API."""
        if not self._loaded or not self._client:
            raise InferenceError(
                "OpenAI client not initialized",
                model_name=self.model_info.name,
                backend="openai"
            )
        
        try:
            response = await self._client.chat.completions.create(
                model=self.model_info.model_id or self.model_info.name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                stop=config.stop_sequences,
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            raise InferenceError(
                f"OpenAI API request failed: {e}",
                model_name=self.model_info.name,
                backend="openai"
            )
    
    async def generate_stream(self, prompt: str, config: GenerationConfig) -> AsyncIterator[str]:
        """Generate streaming text using OpenAI API."""
        if not self._loaded or not self._client:
            raise InferenceError(
                "OpenAI client not initialized",
                model_name=self.model_info.name,
                backend="openai"
            )
        
        try:
            stream = await self._client.chat.completions.create(
                model=self.model_info.model_id or self.model_info.name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                stop=config.stop_sequences,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise InferenceError(
                f"OpenAI streaming request failed: {e}",
                model_name=self.model_info.name,
                backend="openai"
            )


class AnthropicBackend(ModelBackend):
    """Backend for Anthropic Claude models."""
    
    def __init__(self, model_info: ModelInfo) -> None:
        super().__init__(model_info)
        self._client: Optional[Any] = None
    
    async def load_model(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic
            import os
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ModelLoadError(
                    "ANTHROPIC_API_KEY environment variable not set",
                    backend="anthropic"
                )
            
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
            self._loaded = True
            logger.info(f"Initialized Anthropic client for: {self.model_info.name}")
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to initialize Anthropic client: {e}",
                backend="anthropic"
            )
    
    async def unload_model(self) -> None:
        """Clean up Anthropic client."""
        if self._client:
            await self._client.close()
            self._client = None
        
        self._loaded = False
        logger.info(f"Closed Anthropic client for: {self.model_info.name}")
    
    async def generate_text(self, prompt: str, config: GenerationConfig) -> str:
        """Generate text using Anthropic API."""
        if not self._loaded or not self._client:
            raise InferenceError(
                "Anthropic client not initialized",
                model_name=self.model_info.name,
                backend="anthropic"
            )
        
        try:
            response = await self._client.messages.create(
                model=self.model_info.model_id or self.model_info.name,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                messages=[{"role": "user", "content": prompt}],
                stop_sequences=config.stop_sequences,
            )
            
            return response.content[0].text
            
        except Exception as e:
            raise InferenceError(
                f"Anthropic API request failed: {e}",
                model_name=self.model_info.name,
                backend="anthropic"
            )
    
    async def generate_stream(self, prompt: str, config: GenerationConfig) -> AsyncIterator[str]:
        """Generate streaming text using Anthropic API."""
        if not self._loaded or not self._client:
            raise InferenceError(
                "Anthropic client not initialized",
                model_name=self.model_info.name,
                backend="anthropic"
            )
        
        try:
            async with self._client.messages.stream(
                model=self.model_info.model_id or self.model_info.name,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                messages=[{"role": "user", "content": prompt}],
                stop_sequences=config.stop_sequences,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            raise InferenceError(
                f"Anthropic streaming request failed: {e}",
                model_name=self.model_info.name,
                backend="anthropic"
            )


# Backend factory
def create_backend(model_info: ModelInfo) -> ModelBackend:
    """Create appropriate backend for model.
    
    Args:
        model_info: Model information
        
    Returns:
        Configured backend instance
        
    Raises:
        BackendError: If backend type is not supported
    """
    backend_map = {
        BackendType.GEMMA_NATIVE: GemmaNativeBackend,
        BackendType.HUGGINGFACE: HuggingFaceBackend,
        BackendType.OPENAI: OpenAIBackend,
        BackendType.ANTHROPIC: AnthropicBackend,
    }
    
    try:
        backend_type = BackendType(model_info.backend_type)
    except ValueError:
        raise BackendError(f"Unsupported backend type: {model_info.backend_type}")
    
    if backend_type not in backend_map:
        raise BackendError(f"Backend not implemented: {backend_type.value}")
    
    return backend_map[backend_type](model_info)