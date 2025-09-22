"""Demonstration of the LLM Framework capabilities."""

import asyncio
import logging
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_framework import (
    LLMFramework, 
    LLMConfig, 
    GenerationConfig, 
    ModelType,
    quick_generate
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_usage_demo():
    """Demonstrate basic framework usage."""
    print("=== Basic Usage Demo ===")
    
    # Quick generation (simplest usage)
    response = await quick_generate(
        "Explain the concept of machine learning in one sentence.",
        max_tokens=100
    )
    print(f"Quick generation result: {response}")
    
    # Framework with configuration
    config = LLMConfig(
        default_max_tokens=200,
        default_temperature=0.8,
        enable_fallbacks=True,
    )
    
    async with LLMFramework(config) as framework:
        # Generate text
        result = await framework.generate_text(
            "Write a haiku about programming:",
            model_name="gemma-2b-it",  # Prefer local model
            temperature=0.9
        )
        print(f"Haiku generation: {result}")


async def streaming_demo():
    """Demonstrate streaming text generation."""
    print("\n=== Streaming Demo ===")
    
    async with LLMFramework() as framework:
        print("Streaming response: ", end="", flush=True)
        
        async for chunk in await framework.generate_text(
            "Tell me a short story about a robot learning to paint:",
            max_tokens=300,
            stream=True
        ):
            print(chunk, end="", flush=True)
        
        print("\n")


async def batch_generation_demo():
    """Demonstrate batch text generation."""
    print("\n=== Batch Generation Demo ===")
    
    prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing.",
        "How do neural networks work?",
        "What is the future of robotics?"
    ]
    
    async with LLMFramework() as framework:
        results = await framework.generate_batch(
            prompts,
            max_tokens=150,
            temperature=0.7
        )
        
        for i, (prompt, result) in enumerate(zip(prompts, results), 1):
            print(f"{i}. Q: {prompt}")
            print(f"   A: {result}\n")


async def model_management_demo():
    """Demonstrate model discovery and management."""
    print("\n=== Model Management Demo ===")
    
    async with LLMFramework() as framework:
        # List all models
        all_models = framework.list_models()
        print(f"Total available models: {len(all_models)}")
        
        # List local models only
        local_models = framework.list_models(local_only=True)
        print(f"Local models: {len(local_models)}")
        for model in local_models:
            print(f"  - {model.name} ({model.size.value})")
        
        # List chat models
        chat_models = framework.list_models(model_type=ModelType.CHAT)
        print(f"Chat models: {len(chat_models)}")
        for model in chat_models:
            print(f"  - {model.name} ({model.backend_type})")
        
        # Get model info
        if local_models:
            model_info = framework.get_model_info(local_models[0].name)
            print(f"\nDetailed info for {model_info.name}:")
            print(f"  Type: {model_info.model_type.value}")
            print(f"  Backend: {model_info.backend_type}")
            print(f"  Max context: {model_info.capabilities.max_context_length}")
            print(f"  Supports streaming: {model_info.capabilities.supports_streaming}")
            print(f"  Memory requirement: {model_info.memory_requirement_gb}GB")


async def performance_demo():
    """Demonstrate performance monitoring."""
    print("\n=== Performance Demo ===")
    
    async with LLMFramework() as framework:
        # Generate some text to create stats
        await framework.generate_text("Hello, world!", max_tokens=50)
        await framework.generate_text("How are you?", max_tokens=50)
        
        # Get performance stats
        stats = framework.get_performance_stats()
        print("Performance Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


async def error_handling_demo():
    """Demonstrate error handling and fallbacks."""
    print("\n=== Error Handling Demo ===")
    
    config = LLMConfig(
        enable_fallbacks=True,
        fallback_models=["gemma-2b-it", "gpt-3.5-turbo"]
    )
    
    async with LLMFramework(config) as framework:
        try:
            # Try with a non-existent model
            result = await framework.generate_text(
                "This should fallback to another model.",
                model_name="non-existent-model",
                max_tokens=100
            )
            print(f"Fallback worked: {result}")
            
        except Exception as e:
            print(f"Error occurred: {e}")


async def configuration_demo():
    """Demonstrate different configuration options."""
    print("\n=== Configuration Demo ===")
    
    # Custom configuration
    config = LLMConfig(
        models_dir="/.models",
        auto_discover_models=True,
        max_concurrent_requests=5,
        default_timeout=60.0,
        default_max_tokens=256,
        default_temperature=0.8,
        enable_fallbacks=True,
        log_level="DEBUG"
    )
    
    print("Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Use framework with custom config
    async with LLMFramework(config) as framework:
        result = await framework.generate_text(
            "Test with custom configuration.",
            max_tokens=100
        )
        print(f"Result with custom config: {result}")


async def main():
    """Run all demos."""
    print("🚀 LLM Framework Demonstration")
    print("=" * 50)
    
    try:
        await basic_usage_demo()
        await streaming_demo()
        await batch_generation_demo()
        await model_management_demo()
        await performance_demo()
        await error_handling_demo()
        await configuration_demo()
        
        print("\n✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Demo error")


if __name__ == "__main__":
    asyncio.run(main())