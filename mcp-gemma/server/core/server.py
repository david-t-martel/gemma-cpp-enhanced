"""
Refactored MCP server following Single Responsibility Principle.
The server only coordinates between components, doesn't implement business logic.
"""

import logging
from typing import Any, Dict, List, Optional

from mcp import server
from mcp.server import Server
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    TextContent,
    Tool,
)

from .config import Configuration
from .contracts import (
    GenerationRequest,
    IGenerationService,
    IMemoryService,
    IMetricsService,
    IModelService,
    IRequestHandler,
    MemoryEntry,
)
from .handlers import (
    GenerationHandler,
    MemoryHandler,
    MetricsHandler,
    ModelHandler,
)


class MCPServer:
    """
    MCP Server implementation following Single Responsibility Principle.
    Only responsible for coordinating between components.
    """

    def __init__(
        self,
        config: Configuration,
        generation_service: IGenerationService,
        model_service: IModelService,
        memory_service: Optional[IMemoryService] = None,
        metrics_service: Optional[IMetricsService] = None,
    ):
        self.config = config
        self.generation_service = generation_service
        self.model_service = model_service
        self.memory_service = memory_service
        self.metrics_service = metrics_service

        # MCP server instance
        self.server = Server("gemma-mcp", "2.0.0")
        self.logger = logging.getLogger(__name__)

        # Request handlers using Chain of Responsibility
        self.handlers = self._create_handler_chain()

        # Setup MCP protocol handlers
        self._setup_mcp_handlers()

    def _create_handler_chain(self) -> List[IRequestHandler]:
        """Create the chain of request handlers."""
        handlers = []

        # Always include generation and model handlers
        handlers.append(GenerationHandler(self.generation_service, self.metrics_service))
        handlers.append(ModelHandler(self.model_service))

        # Conditionally add handlers based on services
        if self.memory_service:
            handlers.append(MemoryHandler(self.memory_service))

        if self.metrics_service:
            handlers.append(MetricsHandler(self.metrics_service))

        return handlers

    def _setup_mcp_handlers(self):
        """Setup MCP protocol handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """List available tools from all handlers."""
            tools = []

            for handler in self.handlers:
                handler_tools = handler.get_tools()
                tools.extend(handler_tools)

            return ListToolsResult(tools=tools)

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Route tool calls to appropriate handler."""
            # Find handler that can handle this tool
            for handler in self.handlers:
                if handler.can_handle(name):
                    try:
                        result = await handler.handle(name, arguments)
                        return CallToolResult(content=[TextContent(type="text", text=result)])
                    except Exception as e:
                        self.logger.error(f"Handler error for {name}: {e}")
                        return CallToolResult(
                            content=[TextContent(type="text", text=f"Error: {str(e)}")],
                            isError=True,
                        )

            # No handler found
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")], isError=True
            )

    def get_server(self) -> Server:
        """Get the MCP server instance."""
        return self.server

    async def initialize(self):
        """Initialize all services."""
        # Load initial model
        models = self.model_service.list_available_models()
        if models:
            await self.model_service.load_model(models[0])
        else:
            self.logger.warning("No models found during initialization")

    async def shutdown(self):
        """Cleanup resources."""
        self.logger.info("Shutting down MCP server")
        # Services can implement cleanup if needed
