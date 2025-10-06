#!/usr/bin/env python3
"""
Consolidated MCP Gemma Server - Combines all the best features from different implementations.

Features:
- C++ gemma.cpp integration with multiple backends (subprocess + native when available)
- Advanced conversation management with state persistence
- Multiple transport protocols (stdio, HTTP, WebSocket)
- Metrics and monitoring
- Memory backends (Redis, in-memory)
- Legacy compatibility for simple MCP server tools
"""

import asyncio
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# MCP imports
try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool
except ImportError:
    print("MCP not available. Install with: pip install mcp")
    sys.exit(1)

# Local imports
from server.base import BaseServer
from server.chat_handler import ConversationHandler, LegacyCompatibilityHandler
from server.handlers import GenerationHandler, MemoryHandler, MetricsHandler


class ConsolidatedMCPServer(BaseServer):
    """Consolidated MCP server with all features from different implementations."""

    def __init__(self, config):
        super().__init__(config)
        self.mcp_server = Server("consolidated-gemma-mcp")

        # Initialize handlers
        self.generation_handler = GenerationHandler(self)
        self.conversation_handler = ConversationHandler(self)
        self.memory_handler = MemoryHandler(self)
        self.metrics_handler = MetricsHandler(self)
        self.legacy_handler = LegacyCompatibilityHandler(self, self.conversation_handler)

        # Path to gemma-cli.py for subprocess fallback
        self.gemma_cli_path = Path(__file__).parent.parent.parent / "gemma" / "gemma-cli.py"

        # Setup MCP handlers
        self._setup_mcp_handlers()

    def _setup_mcp_handlers(self):
        """Setup MCP request handlers."""

        @self.mcp_server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools combining all implementations."""
            return [
                # Core generation tools
                Tool(
                    name="generate",
                    description="Generate text using Gemma model with advanced options",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Text prompt for generation",
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to generate",
                                "default": 512,
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Temperature for generation (0.0-2.0)",
                                "default": 0.7,
                            },
                            "stream": {
                                "type": "boolean",
                                "description": "Enable streaming response",
                                "default": False,
                            },
                        },
                        "required": ["prompt"],
                    },
                ),
                # Chat and conversation tools
                Tool(
                    name="chat",
                    description="Chat with Gemma model with conversation state",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Chat message",
                            },
                            "conversation_id": {
                                "type": "string",
                                "description": "Conversation ID (optional, creates new if not provided)",
                            },
                            "system_prompt": {
                                "type": "string",
                                "description": "System prompt for the conversation",
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to generate",
                                "default": 512,
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Temperature for generation",
                                "default": 0.8,
                            },
                        },
                        "required": ["message"],
                    },
                ),
                # Conversation management
                Tool(
                    name="list_conversations",
                    description="List all active conversations",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="get_conversation",
                    description="Get conversation details by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "conversation_id": {
                                "type": "string",
                                "description": "Conversation ID",
                            }
                        },
                        "required": ["conversation_id"],
                    },
                ),
                # Memory tools
                Tool(
                    name="store_memory",
                    description="Store information in long-term memory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Content to store",
                            },
                            "key": {
                                "type": "string",
                                "description": "Optional key for the memory",
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional metadata",
                            },
                        },
                        "required": ["content"],
                    },
                ),
                Tool(
                    name="search_memory",
                    description="Search long-term memory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum results to return",
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                ),
                # Model and system tools
                Tool(
                    name="list_models",
                    description="List available Gemma models",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="model_info",
                    description="Get information about a specific model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model": {
                                "type": "string",
                                "description": "Model name to get info for",
                            }
                        },
                        "required": ["model"],
                    },
                ),
                Tool(
                    name="server_status",
                    description="Get comprehensive server status and metrics",
                    inputSchema={"type": "object", "properties": {}},
                ),
                # Legacy compatibility tools
                Tool(
                    name="gemma_generate",
                    description="Legacy text generation (compatibility)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "model": {"type": "string", "default": "current"},
                            "max_tokens": {"type": "integer", "default": 512},
                            "temperature": {"type": "number", "default": 0.8},
                        },
                        "required": ["prompt"],
                    },
                ),
                Tool(
                    name="gemma_chat",
                    description="Legacy chat (compatibility)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {"type": "string"},
                            "conversation_id": {"type": "string"},
                            "system_prompt": {"type": "string"},
                        },
                        "required": ["message"],
                    },
                ),
                Tool(
                    name="gemma_models_list",
                    description="Legacy model listing (compatibility)",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="gemma_model_info",
                    description="Legacy model info (compatibility)",
                    inputSchema={
                        "type": "object",
                        "properties": {"model": {"type": "string"}},
                        "required": ["model"],
                    },
                ),
            ]

        @self.mcp_server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                # Track request metrics
                start_time = time.time()
                self.metrics["total_requests"] += 1

                result = None

                # Core generation tools
                if name == "generate":
                    result = await self.generation_handler.generate(**arguments)
                    result = {"generated_text": result}

                elif name == "chat":
                    result = await self.conversation_handler.chat(**arguments)

                elif name == "list_conversations":
                    result = {"conversations": self.conversation_handler.list_conversations()}

                elif name == "get_conversation":
                    conv = self.conversation_handler.get_conversation(arguments["conversation_id"])
                    result = {"conversation": conv} if conv else {"error": "Conversation not found"}

                # Memory tools
                elif name == "store_memory":
                    result = await self.memory_handler.store(**arguments)

                elif name == "search_memory":
                    result = await self.memory_handler.search(**arguments)

                # Model and system tools
                elif name == "list_models":
                    models = await self.legacy_handler.handle_models_list()
                    result = {"available_models": models}

                elif name == "model_info":
                    result = await self.legacy_handler.handle_model_info(arguments["model"])

                elif name == "server_status":
                    result = await self.metrics_handler.get_comprehensive_metrics()

                # Legacy compatibility tools
                elif name == "gemma_generate":
                    response = await self.legacy_handler.handle_gemma_generate(**arguments)
                    result = {"response": response}

                elif name == "gemma_chat":
                    result = await self.legacy_handler.handle_gemma_chat(**arguments)

                elif name == "gemma_models_list":
                    models = await self.legacy_handler.handle_models_list()
                    result = {
                        "available_models": models,
                        "models_directory": (
                            str(self.config.models_dir)
                            if hasattr(self.config, "models_dir")
                            else "Unknown"
                        ),
                    }

                elif name == "gemma_model_info":
                    result = await self.legacy_handler.handle_model_info(arguments["model"])

                else:
                    raise ValueError(f"Unknown tool: {name}")

                # Update metrics
                response_time = time.time() - start_time
                self.metrics["average_response_time"] = (
                    self.metrics["average_response_time"] * (self.metrics["total_requests"] - 1)
                    + response_time
                ) / self.metrics["total_requests"]

                return [TextContent(type="text", text=json.dumps(result, indent=2))]

            except Exception as e:
                self.logger.error(f"Error handling tool {name}: {e}")
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]

    async def run_stdio(self):
        """Run the server in stdio mode."""
        self.logger.info("Starting Consolidated Gemma MCP Server (stdio mode)")

        async with stdio_server() as (read_stream, write_stream):
            await self.mcp_server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="consolidated-gemma-mcp",
                    server_version="1.0.0",
                    capabilities={},
                ),
            )

    async def run_subprocess_fallback(self, prompt: str, **kwargs) -> str:
        """Fallback to subprocess when direct interface fails."""
        if not self.gemma_cli_path.exists():
            raise RuntimeError(f"gemma-cli.py not found at {self.gemma_cli_path}")

        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)

        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(prompt)
            temp_file = f.name

        try:
            # Run gemma-cli.py
            cmd = [
                sys.executable,
                str(self.gemma_cli_path),
                "--model",
                str(self.config.model_path),
                "--max-tokens",
                str(max_tokens),
                "--temperature",
                str(temperature),
                "--input-file",
                temp_file,
                "--non-interactive",
            ]

            if hasattr(self.config, "tokenizer_path") and self.config.tokenizer_path:
                cmd.extend(["--tokenizer", str(self.config.tokenizer_path)])

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.gemma_cli_path.parent,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8") if stderr else "Unknown error"
                raise RuntimeError(f"Gemma generation failed: {error_msg}")

            return stdout.decode("utf-8").strip()

        finally:
            # Clean up temp file
            try:
                import os

                os.unlink(temp_file)
            except OSError:
                pass


async def main():
    """Main entry point for consolidated server."""
    import argparse

    parser = argparse.ArgumentParser(description="Consolidated MCP Gemma Server")
    parser.add_argument("--model", required=True, help="Path to the Gemma model file")
    parser.add_argument("--tokenizer", help="Path to the tokenizer file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create minimal config
    class Config:
        def __init__(self):
            self.model_path = Path(args.model)
            self.tokenizer_path = Path(args.tokenizer) if args.tokenizer else None
            self.max_tokens = 2048
            self.temperature = 0.7

    config = Config()

    # Validate model file
    if not config.model_path.exists():
        print(f"Error: Model file not found: {config.model_path}")
        sys.exit(1)

    # Create and run server
    server = ConsolidatedMCPServer(config)

    try:
        await server.initialize()
        await server.run_stdio()
    except KeyboardInterrupt:
        print("\nShutdown complete")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
