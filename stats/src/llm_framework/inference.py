"""Inference engine for the LLM Framework."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import logging

from .backends import ModelBackend, GenerationConfig, create_backend
from .exceptions import InferenceError, ModelNotFoundError
from .models import ModelInfo, ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Request for text inference."""
    prompt: str
    model_name: str
    config: GenerationConfig = field(default_factory=GenerationConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Request tracking
    request_id: Optional[str] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Set default values after initialization."""
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class InferenceResponse:
    """Response from text inference."""
    text: str
    model_name: str
    request_id: Optional[str] = None
    
    # Performance metrics
    generation_time: Optional[float] = None
    tokens_generated: Optional[int] = None
    tokens_per_second: Optional[float] = None
    
    # Model info
    backend_type: Optional[str] = None
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Error info
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if inference was successful."""
        return self.error is None


class InferenceEngine:
    """Engine for managing model inference across different backends."""
    
    def __init__(self, 
                 model_registry: ModelRegistry,
                 max_concurrent_requests: int = 10,
                 default_timeout: float = 300.0) -> None:
        """Initialize inference engine.
        
        Args:
            model_registry: Registry of available models
            max_concurrent_requests: Maximum concurrent inference requests
            default_timeout: Default timeout for inference requests in seconds
        """
        self.model_registry = model_registry
        self.max_concurrent_requests = max_concurrent_requests
        self.default_timeout = default_timeout
        
        # Backend management
        self._backends: Dict[str, ModelBackend] = {}
        self._backend_locks: Dict[str, asyncio.Lock] = {}
        
        # Request management
        self._request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._active_requests: Dict[str, InferenceRequest] = {}
        
        # Performance tracking
        self._request_count = 0
        self._total_generation_time = 0.0
    
    async def __aenter__(self) -> "InferenceEngine":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.shutdown()
    
    async def generate_text(self, 
                           request: InferenceRequest,
                           timeout: Optional[float] = None) -> InferenceResponse:
        """Generate text for a single request.
        
        Args:
            request: Inference request
            timeout: Request timeout (uses default if None)
            
        Returns:
            Inference response
        """
        timeout = timeout or self.default_timeout
        request_id = request.request_id or f"req_{int(time.time() * 1000000)}"
        request.request_id = request_id
        
        try:
            # Acquire semaphore for concurrency control
            async with self._request_semaphore:
                self._active_requests[request_id] = request
                
                # Get or create backend
                backend = await self._get_backend(request.model_name)
                
                # Generate text with timeout
                start_time = time.time()
                text = await asyncio.wait_for(
                    backend.generate_text(request.prompt, request.config),
                    timeout=timeout
                )
                generation_time = time.time() - start_time
                
                # Create response
                response = InferenceResponse(
                    text=text,
                    model_name=request.model_name,
                    request_id=request_id,
                    generation_time=generation_time,
                    backend_type=backend.backend_type.value,
                    model_metadata=backend.model_info.to_dict(),
                )
                
                # Calculate performance metrics
                if text:
                    # Rough token estimation (words * 1.3)
                    estimated_tokens = len(text.split()) * 1.3
                    response.tokens_generated = int(estimated_tokens)
                    response.tokens_per_second = estimated_tokens / generation_time if generation_time > 0 else 0
                
                # Update performance tracking
                self._request_count += 1
                self._total_generation_time += generation_time
                
                return response
                
        except asyncio.TimeoutError:
            error_msg = f"Request timed out after {timeout} seconds"
            logger.error(f"Inference timeout for request {request_id}: {error_msg}")
            return InferenceResponse(
                text="",
                model_name=request.model_name,
                request_id=request_id,
                error=error_msg,
            )
            
        except Exception as e:
            error_msg = f"Inference failed: {e}"
            logger.error(f"Inference error for request {request_id}: {error_msg}")
            return InferenceResponse(
                text="",
                model_name=request.model_name,
                request_id=request_id,
                error=error_msg,
            )
            
        finally:
            # Clean up request tracking
            self._active_requests.pop(request_id, None)
    
    async def generate_stream(self, 
                             request: InferenceRequest,
                             timeout: Optional[float] = None) -> AsyncIterator[str]:
        """Generate streaming text for a request.
        
        Args:
            request: Inference request
            timeout: Request timeout (uses default if None)
            
        Yields:
            Text chunks
        """
        timeout = timeout or self.default_timeout
        request_id = request.request_id or f"stream_req_{int(time.time() * 1000000)}"
        request.request_id = request_id
        
        try:
            # Acquire semaphore for concurrency control
            async with self._request_semaphore:
                self._active_requests[request_id] = request
                
                # Get or create backend
                backend = await self._get_backend(request.model_name)
                
                # Generate streaming text with timeout
                async for chunk in asyncio.wait_for(
                    backend.generate_stream(request.prompt, request.config),
                    timeout=timeout
                ):
                    yield chunk
                    
        except asyncio.TimeoutError:
            logger.error(f"Streaming inference timeout for request {request_id}")
            yield f"[ERROR: Request timed out after {timeout} seconds]"
            
        except Exception as e:
            logger.error(f"Streaming inference error for request {request_id}: {e}")
            yield f"[ERROR: Streaming inference failed: {e}]"
            
        finally:
            # Clean up request tracking
            self._active_requests.pop(request_id, None)
    
    async def generate_batch(self, 
                            requests: List[InferenceRequest],
                            timeout: Optional[float] = None) -> List[InferenceResponse]:
        """Generate text for multiple requests concurrently.
        
        Args:
            requests: List of inference requests
            timeout: Request timeout (uses default if None)
            
        Returns:
            List of inference responses
        """
        tasks = [
            self.generate_text(request, timeout)
            for request in requests
        ]
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        result = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_response = InferenceResponse(
                    text="",
                    model_name=requests[i].model_name,
                    request_id=requests[i].request_id,
                    error=f"Batch inference failed: {response}",
                )
                result.append(error_response)
            else:
                result.append(response)
        
        return result
    
    async def _get_backend(self, model_name: str) -> ModelBackend:
        """Get or create backend for model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model backend instance
            
        Raises:
            ModelNotFoundError: If model is not found
            InferenceError: If backend creation/loading fails
        """
        # Check if backend already exists
        if model_name in self._backends:
            backend = self._backends[model_name]
            if backend.is_loaded:
                return backend
        
        # Get model info
        try:
            model_info = self.model_registry.get_model(model_name)
        except ModelNotFoundError:
            # Try to discover models dynamically
            discovered = self.model_registry.discover_models()
            for model in discovered:
                if model.name == model_name:
                    self.model_registry.register_model(model)
                    model_info = model
                    break
            else:
                raise ModelNotFoundError(model_name, list(self._backends.keys()))
        
        # Create lock for this model if needed
        if model_name not in self._backend_locks:
            self._backend_locks[model_name] = asyncio.Lock()
        
        # Load backend with lock to prevent race conditions
        async with self._backend_locks[model_name]:
            # Double-check if backend was created while waiting
            if model_name in self._backends and self._backends[model_name].is_loaded:
                return self._backends[model_name]
            
            try:
                # Create and load backend
                backend = create_backend(model_info)
                await backend.load_model()
                
                # Store backend
                self._backends[model_name] = backend
                logger.info(f"Created and loaded backend for model: {model_name}")
                
                return backend
                
            except Exception as e:
                raise InferenceError(
                    f"Failed to create/load backend for {model_name}: {e}",
                    model_name=model_name,
                    backend=model_info.backend_type
                )
    
    async def load_model(self, model_name: str) -> None:
        """Pre-load a model for faster inference.
        
        Args:
            model_name: Name of the model to load
        """
        await self._get_backend(model_name)
        logger.info(f"Pre-loaded model: {model_name}")
    
    async def unload_model(self, model_name: str) -> None:
        """Unload a model to free resources.
        
        Args:
            model_name: Name of the model to unload
        """
        if model_name in self._backends:
            backend = self._backends[model_name]
            await backend.unload_model()
            del self._backends[model_name]
            logger.info(f"Unloaded model: {model_name}")
    
    async def shutdown(self) -> None:
        """Shutdown the inference engine and clean up resources."""
        logger.info("Shutting down inference engine...")
        
        # Cancel active requests
        for request_id in list(self._active_requests.keys()):
            logger.warning(f"Cancelling active request: {request_id}")
        
        # Unload all backends
        for model_name in list(self._backends.keys()):
            try:
                await self.unload_model(model_name)
            except Exception as e:
                logger.error(f"Error unloading model {model_name}: {e}")
        
        logger.info("Inference engine shutdown complete")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Performance statistics dictionary
        """
        avg_generation_time = (
            self._total_generation_time / self._request_count
            if self._request_count > 0 else 0
        )
        
        return {
            "total_requests": self._request_count,
            "active_requests": len(self._active_requests),
            "loaded_models": len(self._backends),
            "average_generation_time": avg_generation_time,
            "total_generation_time": self._total_generation_time,
            "loaded_model_names": list(self._backends.keys()),
        }
    
    def get_active_requests(self) -> List[Dict[str, Any]]:
        """Get information about active requests.
        
        Returns:
            List of active request information
        """
        return [
            {
                "request_id": req.request_id,
                "model_name": req.model_name,
                "timestamp": req.timestamp,
                "prompt_length": len(req.prompt),
            }
            for req in self._active_requests.values()
        ]