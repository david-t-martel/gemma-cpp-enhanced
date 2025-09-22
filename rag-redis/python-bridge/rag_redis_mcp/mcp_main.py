#!/usr/bin/env python3
"""
RAG Redis MCP Server - Python Bridge to Rust Implementation

This module provides a Python MCP server that acts as a bridge to the
high-performance Rust RAG Redis implementation.
"""
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import argparse

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Resource,
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
        LoggingLevel,
    )
except ImportError as e:
    print(f"Error: MCP SDK not available. Please install with: uv pip install mcp", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag-redis-mcp")

class RagRedisMCPServer:
    """Python MCP Server that bridges to Rust RAG Redis implementation."""
    
    def __init__(self, 
                 redis_url: str = "redis://127.0.0.1:6380",
                 rust_binary_path: Optional[str] = None,
                 log_level: str = "info"):
        self.redis_url = redis_url
        self.log_level = log_level
        
        # Auto-detect Rust binary path
        if rust_binary_path is None:
            rust_binary_path = self._find_rust_binary()
        
        self.rust_binary_path = rust_binary_path
        self.server = Server("rag-redis")
        
        # Register tools and resources
        self._register_tools()
        self._register_resources()
        
        logger.info(f"RAG Redis MCP Server initialized with Redis URL: {redis_url}")
        logger.info(f"Using Rust binary: {rust_binary_path}")
    
    def _find_rust_binary(self) -> str:
        """Find the Rust binary executable."""
        possible_paths = [
            "C:/codedev/llm/rag-redis/rag-redis-system/mcp-server/target/release/mcp-server.exe",
            "C:/codedev/llm/rag-redis/rag-redis-system/target/release/mcp-server.exe",
            "../rag-redis-system/mcp-server/target/release/mcp-server.exe",
            "../rag-redis-system/target/release/mcp-server.exe",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return str(Path(path).resolve())
        
        # If not found, return the most likely path
        return "C:/codedev/llm/rag-redis/rag-redis-system/mcp-server/target/release/mcp-server.exe"
    
    async def _call_rust_binary(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call the Rust binary with JSON-RPC style request."""
        try:
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": method,
                "params": params
            }
            
            # Prepare environment
            env = os.environ.copy()
            env.update({
                "REDIS_URL": self.redis_url,
                "RUST_LOG": self.log_level,
                "RAG_DATA_DIR": "C:/codedev/llm/rag-redis/data/rag",
                "EMBEDDING_CACHE_DIR": "C:/codedev/llm/rag-redis/cache/embeddings"
            })
            
            # Call the Rust binary
            process = await asyncio.create_subprocess_exec(
                self.rust_binary_path,
                "--stdin",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            # Send request and get response
            stdout, stderr = await process.communicate(
                input=json.dumps(request).encode()
            )
            
            if process.returncode != 0:
                logger.error(f"Rust binary failed: {stderr.decode()}")
                return {"error": f"Rust binary failed: {stderr.decode()}"}
            
            # Parse response
            try:
                response = json.loads(stdout.decode())
                return response.get("result", response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Rust response: {e}")
                return {"error": f"Invalid JSON response from Rust binary: {stdout.decode()}"}
                
        except Exception as e:
            logger.error(f"Error calling Rust binary: {e}")
            return {"error": str(e)}
    
    def _register_tools(self):
        """Register MCP tools."""
        
        @self.server.call_tool()
        async def ingest_document(content: str, metadata: Optional[Dict] = None, 
                                embedding_model: str = "all-MiniLM-L6-v2") -> List[TextContent]:
            """Ingest a document into the RAG system."""
            result = await self._call_rust_binary("ingest_document", {
                "content": content,
                "metadata": metadata or {},
                "embedding_model": embedding_model
            })
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        @self.server.call_tool()
        async def search(query: str, limit: int = 10, threshold: float = 0.7, 
                        filter: Optional[Dict] = None) -> List[TextContent]:
            """Search for documents using semantic similarity."""
            result = await self._call_rust_binary("search", {
                "query": query,
                "limit": limit,
                "threshold": threshold,
                "filter": filter or {}
            })
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        @self.server.call_tool()
        async def hybrid_search(query: str, vector_weight: float = 0.7, 
                              keyword_weight: float = 0.3, limit: int = 10) -> List[TextContent]:
            """Perform hybrid search combining vector and keyword matching."""
            result = await self._call_rust_binary("hybrid_search", {
                "query": query,
                "vector_weight": vector_weight,
                "keyword_weight": keyword_weight,
                "limit": limit
            })
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        @self.server.call_tool()
        async def research(query: str, sources: List[str] = None, 
                         max_results: int = 5, combine_with_local: bool = True) -> List[TextContent]:
            """Research a topic using local knowledge and external sources."""
            result = await self._call_rust_binary("research", {
                "query": query,
                "sources": sources or ["web"],
                "max_results": max_results,
                "combine_with_local": combine_with_local
            })
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        @self.server.call_tool()
        async def memory_store(content: str, memory_type: str = "short_term", 
                             importance: float = 0.5, ttl: Optional[int] = None,
                             tags: Optional[List[str]] = None) -> List[TextContent]:
            """Store information in the agent's memory system."""
            result = await self._call_rust_binary("memory_store", {
                "content": content,
                "memory_type": memory_type,
                "importance": importance,
                "ttl": ttl,
                "tags": tags or []
            })
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        @self.server.call_tool()
        async def memory_recall(query: Optional[str] = None, 
                              memory_types: List[str] = None,
                              limit: int = 10, min_importance: float = 0.0) -> List[TextContent]:
            """Recall information from the agent's memory."""
            result = await self._call_rust_binary("memory_recall", {
                "query": query,
                "memory_types": memory_types or ["all"],
                "limit": limit,
                "min_importance": min_importance
            })
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        @self.server.call_tool()
        async def health_check(verbose: bool = False) -> List[TextContent]:
            """Check the health status of the RAG system."""
            result = await self._call_rust_binary("health_check", {
                "verbose": verbose
            })
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
    
    def _register_resources(self):
        """Register MCP resources."""
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources."""
            return [
                Resource(
                    uri="rag://documents",
                    name="RAG Documents",
                    description="Access to all documents in the RAG system",
                    mimeType="application/json"
                ),
                Resource(
                    uri="rag://memory",
                    name="Agent Memory",
                    description="Access to the agent's memory system",
                    mimeType="application/json"
                ),
                Resource(
                    uri="rag://metrics",
                    name="System Metrics",
                    description="Performance and usage metrics",
                    mimeType="application/json"
                )
            ]
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> str:
            """Read a resource."""
            if uri == "rag://documents":
                result = await self._call_rust_binary("list_documents", {})
            elif uri == "rag://memory":
                result = await self._call_rust_binary("memory_recall", {})
            elif uri == "rag://metrics":
                result = await self._call_rust_binary("get_metrics", {})
            else:
                result = {"error": f"Unknown resource URI: {uri}"}
            
            return json.dumps(result, indent=2)

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG Redis MCP Server")
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6380",
                       help="Redis connection URL")
    parser.add_argument("--rust-binary", help="Path to Rust binary")
    parser.add_argument("--log-level", default="info",
                       choices=["debug", "info", "warning", "error"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Set log level
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if isinstance(numeric_level, int):
        logging.getLogger().setLevel(numeric_level)
    
    try:
        # Create server instance
        rag_server = RagRedisMCPServer(
            redis_url=args.redis_url,
            rust_binary_path=args.rust_binary,
            log_level=args.log_level
        )
        
        # Run the server
        await stdio_server(rag_server.server)
        
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())