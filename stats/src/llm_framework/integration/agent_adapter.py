"""Adapter to integrate LLM Framework with existing agent system."""

import asyncio
from typing import Any, Dict, List, Optional, Union
import logging

from ..core import LLMFramework, LLMConfig
from ..models import ModelType
from ..backends import GenerationConfig
from ..exceptions import LLMFrameworkError

# Import existing agent classes
try:
    from ...agent.core import BaseAgent
    from ...agent.gemma_agent import UnifiedGemmaAgent, AgentMode
    from ...agent.tools import ToolRegistry
    HAS_AGENT_SYSTEM = True
except ImportError:
    HAS_AGENT_SYSTEM = False
    BaseAgent = object
    UnifiedGemmaAgent = object
    AgentMode = None
    ToolRegistry = None

logger = logging.getLogger(__name__)


class FrameworkAgent(BaseAgent):
    """Agent that uses the LLM Framework as its backend."""
    
    def __init__(self, 
                 framework: Optional[LLMFramework] = None,
                 model_name: Optional[str] = None,
                 tool_registry: Optional[ToolRegistry] = None,
                 system_prompt: Optional[str] = None,
                 max_iterations: int = 5,
                 verbose: bool = True,
                 **generation_kwargs: Any) -> None:
        """Initialize framework-based agent.
        
        Args:
            framework: LLM Framework instance (created if None)
            model_name: Preferred model name
            tool_registry: Registry of available tools
            system_prompt: System prompt for the agent
            max_iterations: Maximum tool calling iterations
            verbose: Whether to print verbose output
            **generation_kwargs: Default generation parameters
        """
        if not HAS_AGENT_SYSTEM:
            raise ImportError("Agent system not available for integration")
        
        super().__init__(tool_registry, system_prompt, max_iterations, verbose)
        
        self.framework = framework
        self.model_name = model_name
        self.generation_kwargs = generation_kwargs
        self._framework_owned = framework is None
        
        # Initialize framework if not provided
        if self.framework is None:
            config = LLMConfig(**generation_kwargs)
            self.framework = LLMFramework(config)
    
    async def __aenter__(self) -> "FrameworkAgent":
        """Async context manager entry."""
        if self._framework_owned and not self.framework._initialized:
            await self.framework.initialize()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._framework_owned:
            await self.framework.shutdown()
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using the framework (sync wrapper)."""
        return asyncio.run(self.generate_response_async(prompt))
    
    async def generate_response_async(self, prompt: str) -> str:
        """Generate response using the framework."""
        try:
            response = await self.framework.generate_text(
                prompt=prompt,
                model_name=self.model_name,
                **self.generation_kwargs
            )
            return response
            
        except Exception as e:
            logger.error(f"Framework generation failed: {e}")
            return f"Error: {e}"


class AgentFrameworkAdapter:
    """Adapter to use existing agents with the LLM Framework."""
    
    def __init__(self, framework: LLMFramework) -> None:
        """Initialize adapter with framework.
        
        Args:
            framework: LLM Framework instance
        """
        self.framework = framework
    
    async def create_gemma_agent(self,
                                model_name: Optional[str] = None,
                                mode: Optional[str] = None,
                                **kwargs: Any) -> FrameworkAgent:
        """Create a Gemma-compatible agent using the framework.
        
        Args:
            model_name: Model name to use
            mode: Agent mode (ignored, framework handles this)
            **kwargs: Additional agent parameters
            
        Returns:
            Framework-based agent
        """
        # Use Gemma model if available
        if model_name is None:
            gemma_models = self.framework.list_models(local_only=True)
            gemma_models = [m for m in gemma_models if "gemma" in m.name.lower()]
            if gemma_models:
                model_name = gemma_models[0].name
            else:
                model_name = "gemma-2b-it"  # Fallback
        
        return FrameworkAgent(
            framework=self.framework,
            model_name=model_name,
            **kwargs
        )
    
    async def migrate_agent(self, 
                           agent: Union[UnifiedGemmaAgent, BaseAgent],
                           preserve_state: bool = True) -> FrameworkAgent:
        """Migrate existing agent to use the framework.
        
        Args:
            agent: Existing agent to migrate
            preserve_state: Whether to preserve agent state
            
        Returns:
            Migrated framework agent
        """
        # Extract configuration from existing agent
        kwargs = {}
        
        if hasattr(agent, 'tool_registry'):
            kwargs['tool_registry'] = agent.tool_registry
        
        if hasattr(agent, 'system_prompt'):
            kwargs['system_prompt'] = agent.system_prompt
        
        if hasattr(agent, 'max_iterations'):
            kwargs['max_iterations'] = agent.max_iterations
        
        if hasattr(agent, 'verbose'):
            kwargs['verbose'] = agent.verbose
        
        # Extract model information
        model_name = None
        if hasattr(agent, 'model_name'):
            model_name = agent.model_name
        
        # Extract generation parameters
        generation_params = {}
        if hasattr(agent, 'max_new_tokens'):
            generation_params['max_tokens'] = agent.max_new_tokens
        if hasattr(agent, 'temperature'):
            generation_params['temperature'] = agent.temperature
        if hasattr(agent, 'top_p'):
            generation_params['top_p'] = agent.top_p
        
        # Create framework agent
        framework_agent = FrameworkAgent(
            framework=self.framework,
            model_name=model_name,
            **kwargs,
            **generation_params
        )
        
        return framework_agent


class LegacyAgentWrapper:
    """Wrapper to make LLM Framework compatible with legacy agent interfaces."""
    
    def __init__(self, framework: LLMFramework, model_name: Optional[str] = None) -> None:
        """Initialize legacy wrapper.
        
        Args:
            framework: LLM Framework instance
            model_name: Default model name
        """
        self.framework = framework
        self.model_name = model_name
    
    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Legacy callable interface."""
        return asyncio.run(self._generate_async(prompt, **kwargs))
    
    async def _generate_async(self, prompt: str, **kwargs: Any) -> str:
        """Async generation wrapper."""
        model_name = kwargs.pop('model_name', self.model_name)
        
        return await self.framework.generate_text(
            prompt=prompt,
            model_name=model_name,
            **kwargs
        )
    
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Legacy generate method."""
        return self(prompt, **kwargs)
    
    def batch_generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """Legacy batch generation method."""
        return asyncio.run(self._batch_generate_async(prompts, **kwargs))
    
    async def _batch_generate_async(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """Async batch generation wrapper."""
        model_name = kwargs.pop('model_name', self.model_name)
        
        return await self.framework.generate_batch(
            prompts=prompts,
            model_name=model_name,
            **kwargs
        )


def create_framework_from_agent_config(agent_config: Dict[str, Any]) -> LLMFramework:
    """Create LLM Framework from legacy agent configuration.
    
    Args:
        agent_config: Legacy agent configuration
        
    Returns:
        Configured LLM Framework instance
    """
    # Map legacy config to framework config
    framework_config = {}
    
    # Model settings
    if 'model_name' in agent_config:
        # Framework will auto-select if not specified
        pass
    
    if 'max_new_tokens' in agent_config:
        framework_config['default_max_tokens'] = agent_config['max_new_tokens']
    
    if 'temperature' in agent_config:
        framework_config['default_temperature'] = agent_config['temperature']
    
    if 'top_p' in agent_config:
        framework_config['default_top_p'] = agent_config['top_p']
    
    # Performance settings
    if 'device' in agent_config:
        # Framework handles device selection automatically
        pass
    
    if 'verbose' in agent_config:
        framework_config['log_level'] = 'DEBUG' if agent_config['verbose'] else 'INFO'
    
    # Create framework
    config = LLMConfig(**framework_config)
    return LLMFramework(config)


# Backward compatibility functions
async def create_gemma_agent_async(**kwargs: Any) -> FrameworkAgent:
    """Create Gemma agent using the framework (async).
    
    Returns:
        Framework-based Gemma agent
    """
    config = LLMConfig()
    framework = LLMFramework(config)
    await framework.initialize()
    
    adapter = AgentFrameworkAdapter(framework)
    return await adapter.create_gemma_agent(**kwargs)


def create_gemma_agent(**kwargs: Any) -> FrameworkAgent:
    """Create Gemma agent using the framework (sync wrapper).
    
    Returns:
        Framework-based Gemma agent
    """
    return asyncio.run(create_gemma_agent_async(**kwargs))


# Factory function for easy migration
def upgrade_to_framework(agent: Optional[BaseAgent] = None, 
                        config: Optional[Dict[str, Any]] = None) -> LLMFramework:
    """Upgrade existing agent or config to use the LLM Framework.
    
    Args:
        agent: Existing agent to migrate (optional)
        config: Legacy configuration (optional)
        
    Returns:
        LLM Framework instance
    """
    if config:
        return create_framework_from_agent_config(config)
    elif agent:
        # Extract config from agent and create framework
        agent_config = {}
        
        if hasattr(agent, 'max_new_tokens'):
            agent_config['max_new_tokens'] = agent.max_new_tokens
        if hasattr(agent, 'temperature'):
            agent_config['temperature'] = agent.temperature
        if hasattr(agent, 'top_p'):
            agent_config['top_p'] = agent.top_p
        if hasattr(agent, 'verbose'):
            agent_config['verbose'] = agent.verbose
        
        return create_framework_from_agent_config(agent_config)
    else:
        # Create with defaults
        return LLMFramework()


if __name__ == "__main__":
    # Example usage and testing
    async def test_integration():
        """Test the integration components."""
        print("Testing LLM Framework Integration...")
        
        # Create framework
        config = LLMConfig(verbose=True)
        async with LLMFramework(config) as framework:
            # Test framework agent
            async with FrameworkAgent(framework) as agent:
                response = await agent.generate_response_async("Hello, world!")
                print(f"Framework agent response: {response}")
            
            # Test legacy wrapper
            wrapper = LegacyAgentWrapper(framework)
            legacy_response = wrapper("What is machine learning?")
            print(f"Legacy wrapper response: {legacy_response}")
        
        print("Integration test completed!")
    
    # Run test
    asyncio.run(test_integration())