"""
Integration module for connecting MCP Gemma with the existing stats/ Python agent framework.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add stats directory to path for imports
STATS_PATH = Path(__file__).parent.parent.parent / "stats"
sys.path.insert(0, str(STATS_PATH))

try:
    # Import from existing stats framework
    from src.agent.base_agent import BaseAgent
    from src.agent.tools import ToolRegistry
    from src.rag.memory_manager import MemoryManager
    from src.server.main import create_app
except ImportError as e:
    logging.warning(f"Could not import stats framework: {e}")
    BaseAgent = None
    ToolRegistry = None
    MemoryManager = None

# Import MCP Gemma client
sys.path.insert(0, str(Path(__file__).parent.parent))
from client import GemmaHTTPClient, GemmaStdioClient, GenerationRequest


class GemmaMCPAgent:
    """Agent that integrates MCP Gemma with the stats framework."""

    def __init__(
        self,
        client_type: str = "stdio",
        model_path: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        self.client_type = client_type
        self.client = None
        self.logger = logging.getLogger(__name__)

        # Initialize client based on type
        if client_type == "stdio":
            if not model_path:
                model_path = "/c/codedev/llm/.models/gemma2-2b-it-sfp.sbs"
            self.client = GemmaStdioClient(model_path=model_path, **kwargs)
        elif client_type == "http":
            if not base_url:
                base_url = "http://localhost:8080"
            self.client = GemmaHTTPClient(base_url=base_url, **kwargs)
        else:
            raise ValueError(f"Unsupported client type: {client_type}")

        # Memory integration
        self.memory_manager = None
        if MemoryManager:
            try:
                self.memory_manager = MemoryManager()
            except Exception as e:
                self.logger.warning(f"Could not initialize memory manager: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.client.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.disconnect()

    async def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response using Gemma MCP."""
        request = GenerationRequest(prompt=prompt, **kwargs)
        response = await self.client.generate_text(request)
        return response.text

    async def chat(self, message: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        """Chat-style interaction with context."""
        return await self.client.chat(message, context)

    async def store_memory(
        self, key: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store information in memory (both local and MCP)."""
        # Store in MCP server
        from client.base_client import MemoryEntry

        entry = MemoryEntry(key=key, content=content, metadata=metadata)
        mcp_result = await self.client.store_memory(entry)

        # Also store in local memory manager if available
        if self.memory_manager:
            try:
                await self.memory_manager.store(key, content, metadata)
            except Exception as e:
                self.logger.warning(f"Local memory storage failed: {e}")

        return mcp_result

    async def retrieve_memory(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve information from memory."""
        # Try MCP server first
        entry = await self.client.retrieve_memory(key)
        if entry:
            return {
                "content": entry.content,
                "metadata": entry.metadata,
                "timestamp": entry.timestamp,
                "id": entry.id,
            }

        # Fallback to local memory manager
        if self.memory_manager:
            try:
                return await self.memory_manager.retrieve(key)
            except Exception as e:
                self.logger.warning(f"Local memory retrieval failed: {e}")

        return None

    async def search_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memory by content."""
        results = []

        # Search MCP server
        try:
            mcp_entries = await self.client.search_memory(query, limit)
            for entry in mcp_entries:
                results.append(
                    {
                        "key": entry.key,
                        "content": entry.content,
                        "metadata": entry.metadata,
                        "timestamp": entry.timestamp,
                        "source": "mcp",
                    }
                )
        except Exception as e:
            self.logger.warning(f"MCP memory search failed: {e}")

        # Also search local memory manager
        if self.memory_manager and len(results) < limit:
            try:
                local_results = await self.memory_manager.search(query, limit - len(results))
                for result in local_results:
                    result["source"] = "local"
                    results.append(result)
            except Exception as e:
                self.logger.warning(f"Local memory search failed: {e}")

        return results[:limit]

    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from both MCP and local systems."""
        metrics = {"mcp": {}, "local": {}}

        # Get MCP metrics
        try:
            metrics["mcp"] = await self.client.get_metrics()
        except Exception as e:
            metrics["mcp"]["error"] = str(e)

        # Get local metrics if available
        if hasattr(self, "memory_manager") and self.memory_manager:
            try:
                metrics["local"]["memory"] = await self.memory_manager.get_stats()
            except Exception as e:
                metrics["local"]["memory_error"] = str(e)

        return metrics


class GemmaMCPTool:
    """Tool wrapper for integrating Gemma MCP into the stats framework tools."""

    def __init__(self, agent: GemmaMCPAgent):
        self.agent = agent
        self.logger = logging.getLogger(__name__)

    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using Gemma MCP."""
        return await self.agent.generate_response(prompt, **kwargs)

    async def chat_with_gemma(self, message: str, context: Optional[str] = None) -> str:
        """Chat with Gemma using context."""
        context_list = []
        if context:
            # Simple context parsing - could be enhanced
            context_list = [{"role": "user", "content": context}]

        return await self.agent.chat(message, context_list)

    async def store_knowledge(
        self, key: str, content: str, tags: Optional[List[str]] = None
    ) -> str:
        """Store knowledge in memory."""
        metadata = {"tags": tags} if tags else {}
        return await self.agent.store_memory(key, content, metadata)

    async def retrieve_knowledge(self, key: str) -> Optional[str]:
        """Retrieve knowledge from memory."""
        result = await self.agent.retrieve_memory(key)
        return result["content"] if result else None

    async def search_knowledge(self, query: str, limit: int = 5) -> List[str]:
        """Search knowledge base."""
        results = await self.agent.search_memory(query, limit)
        return [f"{r['key']}: {r['content'][:100]}..." for r in results]


def register_gemma_tools(tool_registry, agent: GemmaMCPAgent):
    """Register Gemma MCP tools with the stats framework tool registry."""
    if not ToolRegistry or not tool_registry:
        logging.warning("Tool registry not available")
        return

    tool = GemmaMCPTool(agent)

    # Register tools
    tool_registry.register("gemma_generate", tool.generate_text)
    tool_registry.register("gemma_chat", tool.chat_with_gemma)
    tool_registry.register("gemma_store", tool.store_knowledge)
    tool_registry.register("gemma_retrieve", tool.retrieve_knowledge)
    tool_registry.register("gemma_search", tool.search_knowledge)

    logging.info("Registered Gemma MCP tools with stats framework")


async def create_integrated_agent(config: Optional[Dict[str, Any]] = None) -> GemmaMCPAgent:
    """Create an integrated agent with default configuration."""
    if config is None:
        config = {
            "client_type": "stdio",
            "model_path": "/c/codedev/llm/.models/gemma2-2b-it-sfp.sbs",
            "debug": False,
        }

    agent = GemmaMCPAgent(**config)
    await agent.__aenter__()  # Initialize connection
    return agent


async def main():
    """Example usage of the integration."""
    logging.basicConfig(level=logging.INFO)

    # Create integrated agent
    async with GemmaMCPAgent() as agent:
        # Test text generation
        response = await agent.generate_response("Hello, how are you?")
        print(f"Generated: {response}")

        # Test memory operations
        await agent.store_memory("test_key", "This is test content", {"type": "test"})
        retrieved = await agent.retrieve_memory("test_key")
        print(f"Retrieved: {retrieved}")

        # Test search
        results = await agent.search_memory("test")
        print(f"Search results: {results}")

        # Test metrics
        metrics = await agent.get_metrics()
        print(f"Metrics: {metrics}")


if __name__ == "__main__":
    asyncio.run(main())
