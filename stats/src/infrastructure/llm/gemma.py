"""Google Gemma specific LLM implementation.

This module provides a concrete implementation of the LLM protocol
specifically for Google's Gemma models using HuggingFace Transformers.
"""

import asyncio
from collections.abc import AsyncGenerator
from threading import Thread
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import torch
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import GenerationConfig
from transformers import PreTrainedModel
from transformers import TextIteratorStreamer

from src.shared.logging import get_logger

logger = get_logger(__name__)

from src.shared.config.settings import Settings
from src.shared.exceptions import InferenceException
from src.shared.exceptions import ModelLoadException
from src.shared.exceptions import ResourceException
from src.shared.exceptions import StreamingException

from .base import BaseLLM


class GemmaLLM(BaseLLM):
    """Google Gemma implementation of the LLM protocol.

    This class provides a production-ready implementation for running
    Google Gemma models with optimizations and proper resource management.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the Gemma LLM implementation.

        Args:
            settings: Configuration settings
        """
        super().__init__(settings)
        self.logger = get_logger(f"{__name__}.GemmaLLM")

        # Gemma-specific attributes
        self._generation_config: GenerationConfig | None = None
        self._chat_template: str | None = None

        # Model optimization settings
        self._use_flash_attention = self.settings.performance.use_flash_attention
        self._use_bettertransformer = self.settings.performance.use_bettertransformer
        self._use_torch_compile = self.settings.performance.use_torch_compile

    async def _load_model_implementation(self) -> None:
        """Load the model implementation with fallback support."""
        # List of models to try in order of preference
        models_to_try = [self.model_name]

        # Add local model fallbacks if the original model isn't already a local path
        if not self.model_name.startswith("models/"):
            local_model_paths = [
                "models/microsoft_phi-2",  # Local Phi-2 model
                "models/models--microsoft--phi-2/snapshots",  # Alternative Phi-2 path
                "microsoft/phi-2",  # Try direct download as last resort
            ]
            models_to_try.extend(local_model_paths)

        last_error = None
        for model_name in models_to_try:
            try:
                device = self.device
                self.logger.info(f"Attempting to load model: {model_name} on {device}")

                # Configure quantization if needed
                quantization_config = self._get_quantization_config()

                # Configure model loading arguments
                model_kwargs = {
                    "torch_dtype": self._get_torch_dtype(),
                    "device_map": "auto" if device == "cuda" else None,
                    "trust_remote_code": True,
                }

                if quantization_config:
                    model_kwargs["quantization_config"] = quantization_config

                # Load model in executor to avoid blocking
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None, lambda: AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                )

                # Move to device if not using device_map
                if model_kwargs.get("device_map") is None:
                    self._model = self._model.to(device)

                # Apply optimizations
                await self._apply_optimizations()

                # Setup generation config
                self._setup_generation_config()

                # Get chat template if available
                if hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
                    self._chat_template = self._tokenizer.chat_template

                # Update settings to reflect what was actually loaded
                self.settings.model.name = model_name
                self.logger.info(f"Model loaded successfully: {model_name}")
                return

            except Exception as e:
                last_error = e
                self.logger.warning(f"Failed to load model {model_name}: {e}")
                continue

        # If all models failed, try CPU fallback with the first model
        self.logger.error(f"All model loading attempts failed. Last error: {last_error}")
        self.logger.info("Attempting CPU fallback...")

        try:
            model_name = models_to_try[0] if models_to_try else self.model_name
            self._device = "cpu"

            # Simple CPU configuration
            model_kwargs = {
                "torch_dtype": torch.float32,
                "device_map": None,
                "trust_remote_code": True,
            }

            # Load model in executor to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None, lambda: AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            )

            self._model = self._model.to("cpu")

            # Apply basic optimizations
            await self._apply_optimizations()

            # Setup generation config
            self._setup_generation_config()

            # Get chat template if available
            if hasattr(self._tokenizer, "chat_template") and self._tokenizer.chat_template:
                self._chat_template = self._tokenizer.chat_template

            self.settings.model.name = model_name
            self.logger.info(f"CPU fallback successful: {model_name}")

        except Exception as fallback_error:
            self.logger.error(f"CPU fallback also failed: {fallback_error}")
            raise ModelLoadException(f"No usable model found. Models tried: {models_to_try}. Last error: {last_error}")

    async def _generate_implementation(
        self,
        prompt: str,
        **generation_kwargs: Any,
    ) -> str:
        """Generate text using the Gemma model."""
        if not isinstance(self._model, PreTrainedModel):
            raise InferenceException("Model is not properly loaded")

        try:
            # Tokenize input
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_sequence_length,
            ).to(self.device)

            # Prepare generation arguments
            gen_kwargs = self._prepare_gemma_generation_kwargs(**generation_kwargs)

            # Generate in executor to avoid blocking
            loop = asyncio.get_event_loop()
            with torch.no_grad():
                outputs = await loop.run_in_executor(
                    None,
                    lambda: self._model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        **gen_kwargs,
                    ),
                )

            # Decode generated text
            generated_ids = outputs[0][inputs.input_ids.shape[-1] :]
            generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            return generated_text.strip()

        except torch.cuda.OutOfMemoryError:
            self.logger.error("CUDA out of memory during generation")
            # Clear cache and raise resource exception
            torch.cuda.empty_cache()
            raise ResourceException("Insufficient GPU memory for generation")

        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise InferenceException(f"Gemma generation failed: {e}")

    async def _generate_streaming_implementation(
        self,
        prompt: str,
        **generation_kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming text using the Gemma model."""
        if not isinstance(self._model, PreTrainedModel):
            raise StreamingException("Model is not properly loaded")

        try:
            # Tokenize input
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_sequence_length,
            ).to(self.device)

            # Setup streamer
            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_special_tokens=True,
                skip_prompt=True,
                timeout=30.0,
            )

            # Prepare generation arguments
            gen_kwargs = self._prepare_gemma_generation_kwargs(**generation_kwargs)
            gen_kwargs["streamer"] = streamer

            # Start generation in a separate thread
            generation_thread = Thread(
                target=self._model.generate,
                kwargs={
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    **gen_kwargs,
                },
            )
            generation_thread.start()

            # Yield chunks as they're generated
            try:
                for chunk in streamer:
                    if chunk:  # Skip empty chunks
                        yield chunk
            except Exception as e:
                self.logger.error(f"Streaming generation failed: {e}")
                raise StreamingException(f"Streaming generation failed: {e}")
            finally:
                # Ensure generation thread completes
                generation_thread.join(timeout=5.0)
                if generation_thread.is_alive():
                    self.logger.warning("Generation thread did not complete within timeout")

        except torch.cuda.OutOfMemoryError:
            self.logger.error("CUDA out of memory during streaming generation")
            torch.cuda.empty_cache()
            raise ResourceException("Insufficient GPU memory for streaming generation")

        except Exception as e:
            self.logger.error(f"Streaming generation setup failed: {e}")
            raise StreamingException(f"Streaming generation setup failed: {e}")

    async def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        """Convert messages to Gemma-specific prompt format."""
        if self._chat_template and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                # Use the model's chat template if available
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt
            except Exception as e:
                self.logger.warning(
                    f"Failed to use chat template: {e}, falling back to basic format"
                )

        # Fallback to Gemma-specific formatting
        prompt_parts = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                # Gemma handles system messages differently
                prompt_parts.append(f"<start_of_turn>system\n{content}<end_of_turn>\n")
            elif role == "user":
                prompt_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")
            elif role == "assistant":
                prompt_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>\n")

        # Add generation prompt
        prompt_parts.append("<start_of_turn>model\n")

        return "".join(prompt_parts)

    def _get_quantization_config(self) -> BitsAndBytesConfig | None:
        """Get quantization configuration based on settings."""
        precision = self.settings.performance.precision

        if precision == "int8":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif precision == "int4":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        return None

    def _get_torch_dtype(self) -> torch.dtype:
        """Get appropriate torch dtype based on settings."""
        precision = self.settings.performance.precision

        if precision == "float32":
            return torch.float32
        elif precision == "bfloat16":
            return torch.bfloat16
        else:  # float16 or quantized
            return torch.float16

    async def _apply_optimizations(self) -> None:
        """Apply various optimizations to the model."""
        if not isinstance(self._model, PreTrainedModel):
            return

        try:
            # Apply BetterTransformer optimization
            if self._use_bettertransformer:
                try:
                    self._model = self._model.to_bettertransformer()
                    self.logger.info("Applied BetterTransformer optimization")
                except Exception as e:
                    self.logger.warning(f"Failed to apply BetterTransformer: {e}")

            # Apply torch.compile optimization
            if self._use_torch_compile and torch.__version__ >= "2.0":
                try:
                    self._model = torch.compile(self._model)
                    self.logger.info("Applied torch.compile optimization")
                except Exception as e:
                    self.logger.warning(f"Failed to apply torch.compile: {e}")

            # Enable gradient checkpointing to save memory
            if hasattr(self._model, "gradient_checkpointing_enable"):
                self._model.gradient_checkpointing_enable()
                self.logger.info("Enabled gradient checkpointing")

        except Exception as e:
            self.logger.warning(f"Failed to apply some optimizations: {e}")

    def _setup_generation_config(self) -> None:
        """Setup generation configuration for the model."""
        try:
            self._generation_config = GenerationConfig(
                max_length=self.settings.model.max_length,
                temperature=self.settings.model.temperature,
                top_p=self.settings.model.top_p,
                top_k=self.settings.model.top_k,
                repetition_penalty=self.settings.model.repetition_penalty,
                do_sample=self.settings.model.do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

            # Set the generation config on the model
            if hasattr(self._model, "generation_config"):
                self._model.generation_config = self._generation_config

            self.logger.info("Generation configuration setup complete")

        except Exception as e:
            self.logger.warning(f"Failed to setup generation config: {e}")

    def _prepare_gemma_generation_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Prepare Gemma-specific generation parameters."""
        gen_kwargs = self._prepare_generation_kwargs(**kwargs)

        # Gemma-specific adjustments
        gen_kwargs["pad_token_id"] = self._tokenizer.eos_token_id
        gen_kwargs["eos_token_id"] = self._tokenizer.eos_token_id

        # Ensure we don't exceed model's context length
        if "max_new_tokens" not in gen_kwargs and "max_length" in gen_kwargs:
            # Convert max_length to max_new_tokens for better control
            gen_kwargs["max_new_tokens"] = min(
                gen_kwargs.pop("max_length"),
                self.max_sequence_length // 2,  # Reserve half for prompt
            )

        # Adjust for streaming if streamer is present
        if "streamer" in gen_kwargs:
            gen_kwargs["use_cache"] = True
            gen_kwargs["return_dict_in_generate"] = False

        return gen_kwargs

    async def get_model_info(self) -> dict[str, Any]:
        """Get Gemma-specific model information."""
        base_info = await super().get_model_info()

        gemma_info = {
            "model_type": "Gemma",
            "quantization": self.settings.performance.precision,
            "optimizations": {
                "flash_attention": self._use_flash_attention,
                "bettertransformer": self._use_bettertransformer,
                "torch_compile": self._use_torch_compile,
            },
            "has_chat_template": self._chat_template is not None,
        }

        # Add memory usage if on CUDA
        if self.device.startswith("cuda"):
            try:
                memory_info = {
                    "allocated": torch.cuda.memory_allocated(self.device) / 1024**3,  # GB
                    "reserved": torch.cuda.memory_reserved(self.device) / 1024**3,  # GB
                    "max_allocated": torch.cuda.max_memory_allocated(self.device) / 1024**3,  # GB
                }
                gemma_info["memory_usage"] = memory_info
            except Exception as e:
                self.logger.warning(f"Failed to get memory info: {e}")

        return {**base_info, **gemma_info}
