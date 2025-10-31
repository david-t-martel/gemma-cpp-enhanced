"""Optimized Gemma inference interface with process pooling and improved I/O."""

import asyncio
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from asyncio import Queue

from pydantic import BaseModel, Field, PositiveInt, NonNegativeFloat

from ..utils.profiler import PerformanceMonitor, lazy_property

logger = logging.getLogger(__name__)


class GemmaRuntimeParams(BaseModel):
    """Runtime parameters for Gemma model inference."""
    model_path: str = Field(..., description="Path to the model weights file (.sbs)")
    tokenizer_path: Optional[str] = Field(None, description="Path to tokenizer file (.spm)")
    gemma_executable: Optional[str] = Field(None, description="Path to gemma.exe binary")
    max_tokens: PositiveInt = Field(2048, description="Maximum tokens to generate", gt=0)
    temperature: NonNegativeFloat = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)
    debug_mode: bool = Field(False, description="Enable debug mode")


class OptimizedGemmaInterface:
    """
    Optimized Gemma interface with:
    - Larger buffer size for better throughput
    - Process reuse capability
    - Lazy executable discovery
    - Improved async I/O patterns
    """

    # Performance-tuned constants
    MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_PROMPT_LENGTH = 50_000  # 50KB
    BUFFER_SIZE = 65536  # 64KB buffer - 8x larger for better throughput
    FORBIDDEN_CHARS = {"\x00", "\x1b"}  # Security validation

    def __init__(self, params: GemmaRuntimeParams):
        """Initialize with optimized settings."""
        self.model_path = os.path.normpath(params.model_path)
        self.tokenizer_path = os.path.normpath(params.tokenizer_path) if params.tokenizer_path else None
        self._gemma_executable = params.gemma_executable  # Store for lazy eval
        self.max_tokens = params.max_tokens
        self.temperature = params.temperature
        self.process: Optional[subprocess.Popen] = None
        self.debug_mode = params.debug_mode
        self._executable_verified = False

    @lazy_property
    @PerformanceMonitor.track("gemma_executable_discovery")
    def gemma_executable(self) -> str:
        """Lazy discovery of gemma executable."""
        if self._gemma_executable:
            return os.path.normpath(self._gemma_executable)

        executable = self._find_gemma_executable()
        self._gemma_executable = executable
        return executable

    def _find_gemma_executable(self) -> str:
        """Find gemma executable (unchanged from original)."""
        exe_name = "gemma.exe" if os.name == "nt" else "gemma"

        # Check environment variable
        if gemma_path := os.environ.get("GEMMA_EXECUTABLE"):
            if Path(gemma_path).exists():
                return gemma_path

        # Search common build directories
        search_paths = [
            Path.cwd() / "build" / "Release" / exe_name,
            Path.cwd() / "build-avx2-sycl" / "bin" / "RELEASE" / exe_name,
            Path.cwd() / "build_wsl" / exe_name,
            Path.cwd().parent / "build" / "Release" / exe_name,
        ]

        for path in search_paths:
            if path.exists():
                return str(path.resolve())

        # Search system PATH
        import shutil
        if gemma_path := shutil.which(exe_name):
            return gemma_path

        raise FileNotFoundError(f"'{exe_name}' not found")

    def _verify_executable(self):
        """Verify executable exists (only once)."""
        if self._executable_verified:
            return

        if not os.path.exists(self.gemma_executable):
            raise FileNotFoundError(
                f"Gemma executable not found: {self.gemma_executable}\n"
                f"Set GEMMA_EXECUTABLE environment variable or place gemma.exe in build/Release/"
            )
        self._executable_verified = True

    def _build_command(self, prompt: str) -> list[str]:
        """Build command with security validation."""
        # Security validation
        if len(prompt) > self.MAX_PROMPT_LENGTH:
            raise ValueError(f"Prompt exceeds maximum length of {self.MAX_PROMPT_LENGTH}")

        forbidden_found = [char for char in self.FORBIDDEN_CHARS if char in prompt]
        if forbidden_found:
            raise ValueError(f"Prompt contains forbidden characters: {forbidden_found}")

        cmd = [self.gemma_executable, "--weights", self.model_path]

        if self.tokenizer_path:
            cmd.extend(["--tokenizer", self.tokenizer_path])

        cmd.extend([
            "--max_generated_tokens", str(self.max_tokens),
            "--temperature", str(self.temperature),
            "--prompt", prompt,
        ])

        return cmd

    @PerformanceMonitor.track("generate_response")
    async def generate_response(
        self,
        prompt: str,
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Generate response with optimized I/O.

        Performance improvements:
        - 64KB buffer instead of 8KB (8x improvement in syscalls)
        - Chunked reading for better async performance
        - Optimized string concatenation
        """
        self._verify_executable()
        cmd = self._build_command(prompt)

        if self.debug_mode:
            logger.debug(f"Command: {' '.join(cmd)}")

        try:
            # Create subprocess with optimized buffer settings
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                limit=self.BUFFER_SIZE * 2,  # Increase stream buffer limit
            )

            # Use list for efficient string building
            response_chunks: list[str] = []
            total_size = 0

            # Read with larger chunks for better performance
            if self.process.stdout:
                while True:
                    try:
                        # Read in optimized chunks
                        chunk = await self.process.stdout.read(self.BUFFER_SIZE)
                        if not chunk:
                            break

                        # Security check
                        total_size += len(chunk)
                        if total_size > self.MAX_RESPONSE_SIZE:
                            raise RuntimeError(f"Response exceeded maximum size")

                        output = chunk.decode("utf-8", errors="ignore")
                        if output:
                            response_chunks.append(output)
                            if stream_callback:
                                stream_callback(output)

                    except (OSError, UnicodeDecodeError) as e:
                        if self.debug_mode:
                            logger.debug(f"Read error: {e}")
                        break

            # Wait for completion
            return_code = await self.process.wait()

            if return_code != 0:
                stderr_output = ""
                if self.process.stderr:
                    stderr_bytes = await self.process.stderr.read()
                    stderr_output = stderr_bytes.decode("utf-8", errors="ignore")
                raise RuntimeError(f"Gemma process failed (code {return_code}): {stderr_output}")

            # Efficient string joining
            return "".join(response_chunks)

        finally:
            await self._cleanup_process()

    async def _cleanup_process(self) -> None:
        """Clean up subprocess."""
        if self.process:
            try:
                if self.process.returncode is None:
                    self.process.terminate()
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        self.process.kill()
                        await self.process.wait()
            except (OSError, ProcessLookupError) as e:
                if self.debug_mode:
                    logger.debug(f"Cleanup error: {e}")
            finally:
                self.process = None

    async def stop_generation(self) -> None:
        """Stop current generation."""
        await self._cleanup_process()

    def set_parameters(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> None:
        """Update generation parameters."""
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature

    def get_config(self) -> dict[str, Any]:
        """Get current configuration."""
        return {
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "executable": self._gemma_executable,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "debug_mode": self.debug_mode,
        }


class GemmaProcessPool:
    """
    Process pool for reusing Gemma processes (future enhancement).
    This would require gemma.exe to support a persistent server mode.
    """

    def __init__(self, max_processes: int = 2):
        self.max_processes = max_processes
        self.available: Queue[OptimizedGemmaInterface] = Queue(maxsize=max_processes)
        self.in_use: set[OptimizedGemmaInterface] = set()
        self.params: Optional[GemmaRuntimeParams] = None

    async def initialize(self, params: GemmaRuntimeParams):
        """Initialize the process pool."""
        self.params = params

        # Pre-create processes
        for _ in range(min(2, self.max_processes)):
            process = OptimizedGemmaInterface(params)
            await self.available.put(process)

    async def acquire(self) -> OptimizedGemmaInterface:
        """Get a process from the pool."""
        if not self.available.empty():
            process = await self.available.get()
        else:
            # Create new process on demand
            if not self.params:
                raise RuntimeError("Pool not initialized")
            process = OptimizedGemmaInterface(self.params)

        self.in_use.add(process)
        return process

    async def release(self, process: OptimizedGemmaInterface):
        """Return process to pool."""
        if process in self.in_use:
            self.in_use.remove(process)

            # Return to pool if space available
            if self.available.qsize() < self.max_processes:
                await self.available.put(process)

    async def shutdown(self):
        """Shutdown all processes in pool."""
        # Stop in-use processes
        for process in self.in_use:
            await process.stop_generation()

        # Stop available processes
        while not self.available.empty():
            process = await self.available.get()
            await process.stop_generation()

        self.in_use.clear()


# Export the optimized interface with the same name for compatibility
GemmaInterface = OptimizedGemmaInterface

__all__ = [
    'GemmaInterface',
    'OptimizedGemmaInterface',
    'GemmaRuntimeParams',
    'GemmaProcessPool',
]