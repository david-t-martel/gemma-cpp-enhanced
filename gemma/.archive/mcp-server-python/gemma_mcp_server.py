#!/usr/bin/env python3
"""
Gemma MCP Server - Model Context Protocol server for Gemma C++ inference engine
Provides text generation, chat, and model management capabilities via MCP.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Add the parent directory to path to import gemma-cli
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Resource,
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource
    )
except ImportError:
    print("MCP not available. Install with: pip install mcp")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gemma-mcp-server")

class GemmaMCPServer:
    """MCP Server for Gemma C++ inference engine."""

    def __init__(self):
        self.server = Server("gemma-mcp-server")
        self.gemma_cli_path = Path(__file__).parent.parent / "gemma-cli.py"
        self.models_dir = Path(__file__).parent.parent.parent / ".models"
        self.active_conversations: Dict[str, Any] = {}

        # Verify gemma-cli exists
        if not self.gemma_cli_path.exists():
            logger.error(f"gemma-cli.py not found at {self.gemma_cli_path}")
            raise FileNotFoundError("gemma-cli.py not found")

        # Setup MCP handlers
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup MCP request handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available Gemma tools."""
            return [
                Tool(
                    name="gemma_generate",
                    description="Generate text using Gemma model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Text prompt for generation"
                            },
                            "model": {
                                "type": "string",
                                "description": "Model to use (gemma-2b, gemma-7b, codegemma-2b)",
                                "default": "gemma-2b"
                            },
                            "max_tokens": {
                                "type": "integer",
                                "description": "Maximum tokens to generate",
                                "default": 512
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Temperature for generation (0.0-2.0)",
                                "default": 0.8
                            }
                        },
                        "required": ["prompt"]
                    }
                ),
                Tool(
                    name="gemma_chat",
                    description="Chat with Gemma model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Chat message"
                            },
                            "conversation_id": {
                                "type": "string",
                                "description": "Conversation ID (optional, creates new if not provided)"
                            },
                            "model": {
                                "type": "string",
                                "description": "Model to use",
                                "default": "gemma-2b"
                            },
                            "system_prompt": {
                                "type": "string",
                                "description": "System prompt for the conversation"
                            }
                        },
                        "required": ["message"]
                    }
                ),
                Tool(
                    name="gemma_models_list",
                    description="List available Gemma models",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="gemma_model_info",
                    description="Get information about a specific model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model": {
                                "type": "string",
                                "description": "Model name to get info for"
                            }
                        },
                        "required": ["model"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "gemma_generate":
                    return await self._handle_generate(arguments)
                elif name == "gemma_chat":
                    return await self._handle_chat(arguments)
                elif name == "gemma_models_list":
                    return await self._handle_models_list(arguments)
                elif name == "gemma_model_info":
                    return await self._handle_model_info(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error handling tool {name}: {e}")
                return [TextContent(
                    type="text",
                    text=f"Error: {str(e)}"
                )]

    async def _handle_generate(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle text generation."""
        prompt = args["prompt"]
        model = args.get("model", "gemma-2b")
        max_tokens = args.get("max_tokens", 512)
        temperature = args.get("temperature", 0.8)

        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(prompt)
            temp_file = f.name

        try:
            # Run gemma-cli.py
            cmd = [
                sys.executable, str(self.gemma_cli_path),
                "--model", model,
                "--max-tokens", str(max_tokens),
                "--temperature", str(temperature),
                "--input-file", temp_file,
                "--non-interactive"
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.gemma_cli_path.parent
            )

            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
                raise RuntimeError(f"Gemma generation failed: {error_msg}")

            output = stdout.decode('utf-8').strip()
            return [TextContent(type="text", text=output)]

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except OSError:
                pass

    async def _handle_chat(self, args: Dict[str, Any]) -> List[TextContent]:
        """Handle chat interaction."""
        message = args["message"]
        conversation_id = args.get("conversation_id", f"conv_{len(self.active_conversations)}")
        model = args.get("model", "gemma-2b")
        system_prompt = args.get("system_prompt", "You are a helpful AI assistant.")

        # Get or create conversation
        if conversation_id not in self.active_conversations:
            self.active_conversations[conversation_id] = {
                "messages": [{"role": "system", "content": system_prompt}],
                "model": model
            }

        conversation = self.active_conversations[conversation_id]
        conversation["messages"].append({"role": "user", "content": message})

        # Format conversation for Gemma
        prompt_parts = []
        for msg in conversation["messages"]:
            if msg["role"] == "system":
                prompt_parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")

        prompt_parts.append("Assistant:")
        full_prompt = "\n\n".join(prompt_parts)

        # Generate response
        response_args = {
            "prompt": full_prompt,
            "model": model,
            "max_tokens": 512,
            "temperature": 0.8
        }

        response = await self._handle_generate(response_args)
        response_text = response[0].text if response else "No response generated"

        # Add to conversation
        conversation["messages"].append({"role": "assistant", "content": response_text})

        return [TextContent(
            type="text",
            text=json.dumps({
                "response": response_text,
                "conversation_id": conversation_id,
                "message_count": len(conversation["messages"])
            }, indent=2)
        )]

    async def _handle_models_list(self, args: Dict[str, Any]) -> List[TextContent]:
        """List available models."""
        models = []

        if self.models_dir.exists():
            for model_file in self.models_dir.glob("*.sbs"):
                model_name = model_file.stem
                model_size = model_file.stat().st_size
                models.append({
                    "name": model_name,
                    "file": str(model_file),
                    "size_mb": round(model_size / (1024 * 1024), 2)
                })

        return [TextContent(
            type="text",
            text=json.dumps({
                "available_models": models,
                "models_directory": str(self.models_dir)
            }, indent=2)
        )]

    async def _handle_model_info(self, args: Dict[str, Any]) -> List[TextContent]:
        """Get model information."""
        model = args["model"]

        model_file = self.models_dir / f"{model}.sbs"
        if not model_file.exists():
            return [TextContent(
                type="text",
                text=f"Model {model} not found in {self.models_dir}"
            )]

        stat = model_file.stat()
        info = {
            "model": model,
            "file_path": str(model_file),
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": stat.st_mtime,
            "format": "SFP (Single File Pack)"
        }

        return [TextContent(
            type="text",
            text=json.dumps(info, indent=2)
        )]

    async def run(self):
        """Run the MCP server."""
        logger.info("Starting Gemma MCP Server")

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="gemma-mcp-server",
                    server_version="1.0.0",
                    capabilities={}
                )
            )

async def main():
    """Main entry point."""
    server = GemmaMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())