"""Built-in tools for the LLM agent framework."""

import asyncio
import hashlib
import json
import mimetypes
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import aiofiles
import aiohttp
from pydantic import BaseModel
from pydantic import Field

from src.domain.tools.base import BaseTool
from src.domain.tools.base import ToolExecutionContext
from src.domain.tools.base import ToolResult
from src.domain.tools.base import tool
from src.domain.tools.schemas import EnhancedToolSchema
from src.domain.tools.schemas import ParameterSchema
from src.domain.tools.schemas import ResourceLimits
from src.domain.tools.schemas import SecurityLevel
from src.domain.tools.schemas import ToolCategory
from src.domain.tools.schemas import ToolType
from src.shared.logging import get_logger

logger = get_logger(__name__)


class FileOperationResult(BaseModel):
    """Result of file operation."""

    path: str
    operation: str
    success: bool
    size: int | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@tool(name="read_file", description="Read content from a file", category=ToolCategory.FILE_SYSTEM)
class ReadFileTool(BaseTool):
    """Tool for reading file content."""

    @property
    def schema(self) -> EnhancedToolSchema:
        return EnhancedToolSchema(
            name="read_file",
            description="Read content from a file with encoding detection",
            category=ToolCategory.FILE_SYSTEM,
            type=ToolType.BUILTIN,
            parameters=[
                ParameterSchema(
                    name="path",
                    type="string",
                    description="Path to the file to read",
                    required=True,
                    examples=["/path/to/file.txt", "config.json", "data.csv"],
                ),
                ParameterSchema(
                    name="encoding",
                    type="string",
                    description="File encoding (auto-detected if not specified)",
                    required=False,
                    default="auto",
                    examples=["utf-8", "latin-1", "ascii"],
                ),
                ParameterSchema(
                    name="max_size_mb",
                    type="number",
                    description="Maximum file size to read in MB",
                    required=False,
                    default=10,
                    minimum=0.1,
                    maximum=100,
                ),
            ],
            security=SecurityLevel(
                level="standard", sandbox_required=False, file_system_access=True
            ),
        )

    async def execute(self, context: ToolExecutionContext, **kwargs) -> ToolResult:
        """Execute file reading."""
        try:
            path = Path(kwargs["path"])
            encoding = kwargs.get("encoding", "auto")
            max_size_mb = kwargs.get("max_size_mb", 10)
            max_size_bytes = int(max_size_mb * 1024 * 1024)

            # Security checks
            if not path.exists():
                return ToolResult(
                    success=False,
                    error=f"File not found: {path}",
                    execution_time=0,
                    context=context,
                )

            if not path.is_file():
                return ToolResult(
                    success=False,
                    error=f"Path is not a file: {path}",
                    execution_time=0,
                    context=context,
                )

            # Check file size
            file_size = path.stat().st_size
            if file_size > max_size_bytes:
                return ToolResult(
                    success=False,
                    error=f"File too large: {file_size / (1024 * 1024):.1f}MB > {max_size_mb}MB",
                    execution_time=0,
                    context=context,
                )

            # Detect encoding if needed
            if encoding == "auto":
                encoding = await self._detect_encoding(path)

            # Read file content
            start_time = asyncio.get_event_loop().time()
            async with aiofiles.open(path, encoding=encoding) as f:
                content = await f.read()
            execution_time = asyncio.get_event_loop().time() - start_time

            # Get file metadata
            mime_type, _ = mimetypes.guess_type(str(path))
            metadata = {
                "size": file_size,
                "mime_type": mime_type,
                "encoding": encoding,
                "lines": content.count("\n") + 1 if content else 0,
            }

            return ToolResult(
                success=True,
                data={"content": content, "metadata": metadata},
                execution_time=execution_time,
                context=context,
            )

        except Exception as e:
            logger.error(f"File read error: {e}")
            return ToolResult(success=False, error=str(e), execution_time=0, context=context)

    async def _detect_encoding(self, path: Path) -> str:
        """Detect file encoding."""
        try:
            import chardet

            # Read first 8KB for encoding detection
            with open(path, "rb") as f:
                raw_data = f.read(8192)

            result = chardet.detect(raw_data)
            return result.get("encoding", "utf-8") or "utf-8"
        except ImportError:
            # Fallback to utf-8 if chardet not available
            return "utf-8"


@tool(name="write_file", description="Write content to a file", category=ToolCategory.FILE_SYSTEM)
class WriteFileTool(BaseTool):
    """Tool for writing file content."""

    @property
    def schema(self) -> EnhancedToolSchema:
        return EnhancedToolSchema(
            name="write_file",
            description="Write content to a file with backup and atomic operations",
            category=ToolCategory.FILE_SYSTEM,
            type=ToolType.BUILTIN,
            parameters=[
                ParameterSchema(
                    name="path",
                    type="string",
                    description="Path to the file to write",
                    required=True,
                ),
                ParameterSchema(
                    name="content",
                    type="string",
                    description="Content to write to the file",
                    required=True,
                ),
                ParameterSchema(
                    name="encoding",
                    type="string",
                    description="File encoding",
                    required=False,
                    default="utf-8",
                    enum=["utf-8", "latin-1", "ascii"],
                ),
                ParameterSchema(
                    name="create_backup",
                    type="boolean",
                    description="Create backup of existing file",
                    required=False,
                    default=True,
                ),
                ParameterSchema(
                    name="atomic",
                    type="boolean",
                    description="Use atomic write operation",
                    required=False,
                    default=True,
                ),
            ],
            security=SecurityLevel(
                level="standard", sandbox_required=True, file_system_access=True
            ),
        )

    async def execute(self, context: ToolExecutionContext, **kwargs) -> ToolResult:
        """Execute file writing."""
        try:
            path = Path(kwargs["path"])
            content = kwargs["content"]
            encoding = kwargs.get("encoding", "utf-8")
            create_backup = kwargs.get("create_backup", True)
            atomic = kwargs.get("atomic", True)

            start_time = asyncio.get_event_loop().time()

            # Create parent directories
            path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup if file exists
            backup_path = None
            if create_backup and path.exists():
                backup_path = path.with_suffix(f"{path.suffix}.backup")
                await self._create_backup(path, backup_path)

            # Write file
            if atomic:
                await self._atomic_write(path, content, encoding)
            else:
                async with aiofiles.open(path, "w", encoding=encoding) as f:
                    await f.write(content)

            execution_time = asyncio.get_event_loop().time() - start_time

            # Get file metadata
            file_size = path.stat().st_size
            mime_type, _ = mimetypes.guess_type(str(path))

            result_data = {
                "path": str(path),
                "size": file_size,
                "mime_type": mime_type,
                "encoding": encoding,
                "lines": content.count("\n") + 1 if content else 0,
            }

            if backup_path:
                result_data["backup_path"] = str(backup_path)

            return ToolResult(
                success=True, data=result_data, execution_time=execution_time, context=context
            )

        except Exception as e:
            logger.error(f"File write error: {e}")
            return ToolResult(success=False, error=str(e), execution_time=0, context=context)

    async def _create_backup(self, source: Path, backup: Path) -> None:
        """Create backup of existing file."""
        async with aiofiles.open(source, "rb") as src, aiofiles.open(backup, "wb") as dst:
            async for chunk in src:
                await dst.write(chunk)

    async def _atomic_write(self, path: Path, content: str, encoding: str) -> None:
        """Perform atomic write operation."""
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        try:
            async with aiofiles.open(temp_path, "w", encoding=encoding) as f:
                await f.write(content)
            temp_path.replace(path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise


@tool(
    name="list_directory", description="List directory contents", category=ToolCategory.FILE_SYSTEM
)
class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents."""

    @property
    def schema(self) -> EnhancedToolSchema:
        return EnhancedToolSchema(
            name="list_directory",
            description="List directory contents with detailed metadata",
            category=ToolCategory.FILE_SYSTEM,
            type=ToolType.BUILTIN,
            parameters=[
                ParameterSchema(
                    name="path",
                    type="string",
                    description="Path to the directory to list",
                    required=True,
                    default=".",
                ),
                ParameterSchema(
                    name="recursive",
                    type="boolean",
                    description="List files recursively",
                    required=False,
                    default=False,
                ),
                ParameterSchema(
                    name="include_hidden",
                    type="boolean",
                    description="Include hidden files and directories",
                    required=False,
                    default=False,
                ),
                ParameterSchema(
                    name="pattern",
                    type="string",
                    description="Glob pattern to filter files",
                    required=False,
                    examples=["*.py", "test_*.txt", "**/*.json"],
                ),
                ParameterSchema(
                    name="max_depth",
                    type="integer",
                    description="Maximum recursion depth",
                    required=False,
                    minimum=1,
                    maximum=10,
                ),
            ],
            security=SecurityLevel(
                level="standard", sandbox_required=False, file_system_access=True
            ),
        )

    async def execute(self, context: ToolExecutionContext, **kwargs) -> ToolResult:
        """Execute directory listing."""
        try:
            path = Path(kwargs["path"])
            recursive = kwargs.get("recursive", False)
            include_hidden = kwargs.get("include_hidden", False)
            pattern = kwargs.get("pattern")
            max_depth = kwargs.get("max_depth", 3)

            if not path.exists():
                return ToolResult(
                    success=False,
                    error=f"Directory not found: {path}",
                    execution_time=0,
                    context=context,
                )

            if not path.is_dir():
                return ToolResult(
                    success=False,
                    error=f"Path is not a directory: {path}",
                    execution_time=0,
                    context=context,
                )

            start_time = asyncio.get_event_loop().time()

            files = []
            directories = []

            if recursive:
                for item in self._walk_directory(path, max_depth, include_hidden):
                    item_info = await self._get_item_info(item, path)
                    if pattern and not Path(item).match(pattern):
                        continue

                    if item.is_file():
                        files.append(item_info)
                    elif item.is_dir():
                        directories.append(item_info)
            else:
                for item in path.iterdir():
                    if not include_hidden and item.name.startswith("."):
                        continue

                    if pattern and not item.match(pattern):
                        continue

                    item_info = await self._get_item_info(item, path)
                    if item.is_file():
                        files.append(item_info)
                    elif item.is_dir():
                        directories.append(item_info)

            execution_time = asyncio.get_event_loop().time() - start_time

            return ToolResult(
                success=True,
                data={
                    "path": str(path),
                    "files": sorted(files, key=lambda x: x["name"]),
                    "directories": sorted(directories, key=lambda x: x["name"]),
                    "total_files": len(files),
                    "total_directories": len(directories),
                },
                execution_time=execution_time,
                context=context,
            )

        except Exception as e:
            logger.error(f"Directory listing error: {e}")
            return ToolResult(success=False, error=str(e), execution_time=0, context=context)

    def _walk_directory(
        self, path: Path, max_depth: int, include_hidden: bool, current_depth: int = 0
    ):
        """Walk directory recursively with depth limit."""
        if current_depth >= max_depth:
            return

        try:
            for item in path.iterdir():
                if not include_hidden and item.name.startswith("."):
                    continue

                yield item

                if item.is_dir() and current_depth < max_depth - 1:
                    yield from self._walk_directory(
                        item, max_depth, include_hidden, current_depth + 1
                    )
        except (PermissionError, OSError):
            pass

    async def _get_item_info(self, item: Path, base_path: Path) -> dict[str, Any]:
        """Get detailed information about file or directory."""
        try:
            stat = item.stat()
            relative_path = item.relative_to(base_path)

            info = {
                "name": item.name,
                "path": str(item),
                "relative_path": str(relative_path),
                "type": "file" if item.is_file() else "directory",
                "size": stat.st_size if item.is_file() else None,
                "modified": stat.st_mtime,
                "created": stat.st_ctime if hasattr(stat, "st_ctime") else stat.st_mtime,
                "permissions": oct(stat.st_mode)[-3:],
            }

            if item.is_file():
                mime_type, encoding = mimetypes.guess_type(str(item))
                info.update(
                    {"mime_type": mime_type, "encoding": encoding, "extension": item.suffix.lower()}
                )

            return info

        except (OSError, PermissionError) as e:
            return {"name": item.name, "path": str(item), "error": str(e)}


@tool(
    name="web_search",
    description="Search the web for information",
    category=ToolCategory.WEB_SEARCH,
)
class WebSearchTool(BaseTool):
    """Tool for web searching."""

    @property
    def schema(self) -> EnhancedToolSchema:
        return EnhancedToolSchema(
            name="web_search",
            description="Search the web using multiple search engines",
            category=ToolCategory.WEB_SEARCH,
            type=ToolType.BUILTIN,
            parameters=[
                ParameterSchema(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                    min_length=1,
                    max_length=500,
                ),
                ParameterSchema(
                    name="engine",
                    type="string",
                    description="Search engine to use",
                    required=False,
                    default="duckduckgo",
                    enum=["duckduckgo", "bing", "google"],
                ),
                ParameterSchema(
                    name="num_results",
                    type="integer",
                    description="Number of results to return",
                    required=False,
                    default=10,
                    minimum=1,
                    maximum=50,
                ),
                ParameterSchema(
                    name="safe_search",
                    type="boolean",
                    description="Enable safe search filtering",
                    required=False,
                    default=True,
                ),
            ],
            security=SecurityLevel(level="standard", sandbox_required=False, network_access=True),
        )

    async def execute(self, context: ToolExecutionContext, **kwargs) -> ToolResult:
        """Execute web search."""
        try:
            query = kwargs["query"]
            engine = kwargs.get("engine", "duckduckgo")
            num_results = kwargs.get("num_results", 10)
            safe_search = kwargs.get("safe_search", True)

            start_time = asyncio.get_event_loop().time()

            # Perform search based on engine
            if engine == "duckduckgo":
                results = await self._search_duckduckgo(query, num_results, safe_search)
            else:
                # Fallback: Try DuckDuckGo as default for unknown engines
                logger.warning(f"Unknown search engine '{engine}', falling back to DuckDuckGo")
                results = await self._search_duckduckgo(query, num_results, safe_search)

            execution_time = asyncio.get_event_loop().time() - start_time

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "engine": engine,
                    "results": results,
                    "total_results": len(results),
                },
                execution_time=execution_time,
                context=context,
            )

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return ToolResult(success=False, error=str(e), execution_time=0, context=context)

    async def _search_duckduckgo(
        self, query: str, num_results: int, safe_search: bool
    ) -> list[dict[str, Any]]:
        """Search using DuckDuckGo."""
        # This is a placeholder implementation
        # In practice, you'd use a proper search API or library
        results = []

        try:
            async with aiohttp.ClientSession() as session:
                params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}

                if safe_search:
                    params["safe_search"] = "strict"

                async with session.get("https://api.duckduckgo.com/", params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Process results (simplified)
                        for i, result in enumerate(data.get("Results", [])[:num_results]):
                            results.append(
                                {
                                    "title": result.get("Text", ""),
                                    "url": result.get("FirstURL", ""),
                                    "snippet": result.get("Text", ""),
                                    "rank": i + 1,
                                }
                            )

        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")

        return results

    async def _search_fallback(self, query: str, num_results: int) -> list[dict[str, Any]]:
        """Fallback search implementation.

        This method is kept for backward compatibility but should not be used.
        Use DuckDuckGo search instead.
        """
        logger.error("Fallback search should not be called - use DuckDuckGo instead")
        raise NotImplementedError(
            "Fallback search is not implemented. Use DuckDuckGo search engine instead."
        )


@tool(name="http_request", description="Make HTTP requests", category=ToolCategory.WEB_SEARCH)
class HttpRequestTool(BaseTool):
    """Tool for making HTTP requests."""

    @property
    def schema(self) -> EnhancedToolSchema:
        return EnhancedToolSchema(
            name="http_request",
            description="Make HTTP requests with comprehensive options",
            category=ToolCategory.WEB_SEARCH,
            type=ToolType.BUILTIN,
            parameters=[
                ParameterSchema(
                    name="url", type="url", description="URL to make request to", required=True
                ),
                ParameterSchema(
                    name="method",
                    type="string",
                    description="HTTP method",
                    required=False,
                    default="GET",
                    enum=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
                ),
                ParameterSchema(
                    name="headers", type="object", description="HTTP headers", required=False
                ),
                ParameterSchema(
                    name="params", type="object", description="Query parameters", required=False
                ),
                ParameterSchema(
                    name="data", type="string", description="Request body data", required=False
                ),
                ParameterSchema(
                    name="json_data", type="object", description="JSON request body", required=False
                ),
                ParameterSchema(
                    name="timeout",
                    type="integer",
                    description="Request timeout in seconds",
                    required=False,
                    default=30,
                    minimum=1,
                    maximum=300,
                ),
                ParameterSchema(
                    name="follow_redirects",
                    type="boolean",
                    description="Follow HTTP redirects",
                    required=False,
                    default=True,
                ),
                ParameterSchema(
                    name="verify_ssl",
                    type="boolean",
                    description="Verify SSL certificates",
                    required=False,
                    default=True,
                ),
            ],
            security=SecurityLevel(level="standard", sandbox_required=False, network_access=True),
        )

    async def execute(self, context: ToolExecutionContext, **kwargs) -> ToolResult:
        """Execute HTTP request."""
        try:
            url = kwargs["url"]
            method = kwargs.get("method", "GET").upper()
            headers = kwargs.get("headers", {})
            params = kwargs.get("params", {})
            data = kwargs.get("data")
            json_data = kwargs.get("json_data")
            timeout = kwargs.get("timeout", 30)
            follow_redirects = kwargs.get("follow_redirects", True)
            verify_ssl = kwargs.get("verify_ssl", True)

            start_time = asyncio.get_event_loop().time()

            # Set up request parameters
            request_kwargs = {
                "method": method,
                "url": url,
                "headers": headers,
                "params": params,
                "timeout": aiohttp.ClientTimeout(total=timeout),
                "allow_redirects": follow_redirects,
                "ssl": verify_ssl,
            }

            if json_data:
                request_kwargs["json"] = json_data
            elif data:
                request_kwargs["data"] = data

            # Make request
            async with aiohttp.ClientSession() as session:
                async with session.request(**request_kwargs) as response:
                    response_text = await response.text()

                    # Try to parse as JSON
                    try:
                        response_json = await response.json()
                    except:
                        response_json = None

                    execution_time = asyncio.get_event_loop().time() - start_time

                    return ToolResult(
                        success=response.status < 400,
                        data={
                            "status_code": response.status,
                            "headers": dict(response.headers),
                            "text": response_text,
                            "json": response_json,
                            "url": str(response.url),
                            "method": method,
                            "size": len(response_text),
                        },
                        execution_time=execution_time,
                        context=context,
                    )

        except Exception as e:
            logger.error(f"HTTP request error: {e}")
            return ToolResult(success=False, error=str(e), execution_time=0, context=context)


@tool(name="hash_text", description="Generate hash of text", category=ToolCategory.DATA_PROCESSING)
class HashTextTool(BaseTool):
    """Tool for generating text hashes."""

    @property
    def schema(self) -> EnhancedToolSchema:
        return EnhancedToolSchema(
            name="hash_text",
            description="Generate cryptographic hash of text using various algorithms",
            category=ToolCategory.DATA_PROCESSING,
            type=ToolType.BUILTIN,
            parameters=[
                ParameterSchema(
                    name="text", type="string", description="Text to hash", required=True
                ),
                ParameterSchema(
                    name="algorithm",
                    type="string",
                    description="Hash algorithm to use",
                    required=False,
                    default="sha256",
                    enum=["md5", "sha1", "sha256", "sha512", "blake2b", "blake2s"],
                ),
                ParameterSchema(
                    name="encoding",
                    type="string",
                    description="Text encoding",
                    required=False,
                    default="utf-8",
                    enum=["utf-8", "ascii", "latin-1"],
                ),
                ParameterSchema(
                    name="output_format",
                    type="string",
                    description="Output format",
                    required=False,
                    default="hex",
                    enum=["hex", "base64", "bytes"],
                ),
            ],
            security=SecurityLevel(
                level="minimal",
                sandbox_required=False,
                network_access=False,
                file_system_access=False,
            ),
            idempotent=True,
            cacheable=True,
        )

    async def execute(self, context: ToolExecutionContext, **kwargs) -> ToolResult:
        """Execute text hashing."""
        try:
            text = kwargs["text"]
            algorithm = kwargs.get("algorithm", "sha256")
            encoding = kwargs.get("encoding", "utf-8")
            output_format = kwargs.get("output_format", "hex")

            start_time = asyncio.get_event_loop().time()

            # Encode text
            text_bytes = text.encode(encoding)

            # Create hash
            if algorithm == "md5":
                hasher = hashlib.md5()
            elif algorithm == "sha1":
                hasher = hashlib.sha1()
            elif algorithm == "sha256":
                hasher = hashlib.sha256()
            elif algorithm == "sha512":
                hasher = hashlib.sha512()
            elif algorithm == "blake2b":
                hasher = hashlib.blake2b()
            elif algorithm == "blake2s":
                hasher = hashlib.blake2s()
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            hasher.update(text_bytes)

            # Format output
            if output_format == "hex":
                hash_value = hasher.hexdigest()
            elif output_format == "base64":
                import base64

                hash_value = base64.b64encode(hasher.digest()).decode("ascii")
            elif output_format == "bytes":
                hash_value = hasher.digest().hex()
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            execution_time = asyncio.get_event_loop().time() - start_time

            return ToolResult(
                success=True,
                data={
                    "hash": hash_value,
                    "algorithm": algorithm,
                    "encoding": encoding,
                    "output_format": output_format,
                    "input_size": len(text_bytes),
                },
                execution_time=execution_time,
                context=context,
            )

        except Exception as e:
            logger.error(f"Hash generation error: {e}")
            return ToolResult(success=False, error=str(e), execution_time=0, context=context)


# Helper function to register all built-in tools
async def register_builtin_tools():
    """Register all built-in tools with the global registry."""
    from src.domain.tools.base import get_global_registry

    registry = get_global_registry()

    tools = [
        ReadFileTool(),
        WriteFileTool(),
        ListDirectoryTool(),
        WebSearchTool(),
        HttpRequestTool(),
        HashTextTool(),
    ]

    for tool in tools:
        await registry.register(tool)

    logger.info(f"Registered {len(tools)} built-in tools")
    return registry
