"""Gemma inference interface using native Windows executable."""

import asyncio

import os

import subprocess

from collections.abc import Callable

from pathlib import Path

from typing import Optional

import logging



from pydantic import BaseModel, Field, PositiveInt, NonNegativeFloat



# Configure logging

logger = logging.getLogger(__name__)



class GemmaRuntimeParams(BaseModel):
    """Runtime parameters for Gemma model inference."""
    model_path: str = Field(..., description="Path to the model weights file (.sbs)")
    tokenizer_path: Optional[str] = Field(None, description="Path to tokenizer file (.spm), optional for single-file models")
    gemma_executable: Optional[str] = Field(None, description="Path to gemma.exe binary. If None, auto-discovered.")
    max_tokens: PositiveInt = Field(2048, description="Maximum tokens to generate", gt=0)
    temperature: NonNegativeFloat = Field(0.7, description="Sampling temperature (0.0 to 2.0)", ge=0.0, le=2.0)
    debug_mode: bool = Field(False, description="Enable debug mode with verbose output")


class GemmaInterface:
    """Interface for communicating with the native Windows gemma.exe."""

    # Security and performance constants
    MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10MB - prevent memory exhaustion
    MAX_PROMPT_LENGTH = 50_000  # 50KB - prevent command injection
    BUFFER_SIZE = 8192  # 8KB buffer - efficient I/O operations
    FORBIDDEN_CHARS = {"\x00", "\x1b"}  # Null bytes, escape sequences - security

    def __init__(
        self,
        params: GemmaRuntimeParams
    ) -> None:
        """
        Initialize the Gemma interface with structured runtime parameters.

        Args:
            params: An instance of GemmaRuntimeParams containing all necessary configuration.
        """
        self.model_path = os.path.normpath(params.model_path)
        self.tokenizer_path = os.path.normpath(params.tokenizer_path) if params.tokenizer_path else None

        # Auto-discover executable if not provided in params
        if params.gemma_executable is None:
            gemma_executable = self._find_gemma_executable()
        else:
            gemma_executable = params.gemma_executable

        self.gemma_executable = os.path.normpath(gemma_executable)
        self.max_tokens = params.max_tokens
        self.temperature = params.temperature
        self.process: Optional[subprocess.Popen] = None
        self.debug_mode = params.debug_mode

        # Verify executable exists
        if not os.path.exists(self.gemma_executable):
            raise FileNotFoundError(
                f"Gemma executable not found: {self.gemma_executable}\n"
                f"Set GEMMA_EXECUTABLE environment variable or place gemma.exe in build/Release/"
            )

    def _find_gemma_executable(self) -> str:
        """
        Find gemma executable in standard and configured locations.

        Search Order:
        1.  `GEMMA_EXECUTABLE` environment variable.
        2.  Pre-defined common build directories.
        3.  System's PATH.

        Returns:
            The absolute path to the gemma executable.

        Raises:
            FileNotFoundError: If the executable cannot be found.
        """
        exe_name = "gemma.exe" if os.name == "nt" else "gemma"
        
        # 1. Check environment variable
        if gemma_path := os.environ.get("GEMMA_EXECUTABLE"):
            if Path(gemma_path).exists():
                return gemma_path

        # TODO: [Deployment] Integrate uvx binary wrapper for gemma.exe execution.
        # This would provide a more controlled environment for the C++ executable.
        # TODO: [Executable Discovery] Enhance _find_gemma_executable to check for bundled gemma.exe
        # within the application's installation directory (e.g., PyInstaller output).

        # 2. Search common build directories
        # User-provided path is now the first search path
        search_paths = [
            Path.cwd() / "build" / "Release" / exe_name,
            Path.cwd() / "build-avx2-sycl" / "bin" / "RELEASE" / exe_name,
            Path.cwd() / "build_wsl" / exe_name,
            Path.cwd().parent / "build" / "Release" / exe_name,
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path.resolve())

        # 3. Search system PATH
        import shutil
        if gemma_path := shutil.which(exe_name):
            return gemma_path

        logger.error(
            f"'{exe_name}' not found. Searched environment variables, "
            f"common build directories, and system PATH. Please ensure gemma.exe is built "
            f"and its location is in the system PATH or set via GEMMA_EXECUTABLE."
        )
        raise FileNotFoundError(
            f"'{exe_name}' not found. Searched environment variables, "
            f"common build directories, and system PATH. Please ensure gemma.exe is built "
            f"and its location is in the system PATH or set via GEMMA_EXECUTABLE."
        )

    def _build_command(self, prompt: str) -> list[str]:
        """
        Build the command to execute gemma with security validation.

        Args:
            prompt: Input prompt for generation

        Returns:
            Command list ready for subprocess execution

        Raises:
            ValueError: If prompt exceeds MAX_PROMPT_LENGTH or contains forbidden characters
        """
        # Security validation: Check prompt length
        if len(prompt) > self.MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt exceeds maximum length of {self.MAX_PROMPT_LENGTH} bytes "
                f"(got {len(prompt)} bytes). This prevents potential command injection "
                "and ensures reasonable processing times."
            )

        # Security validation: Check for forbidden characters
        forbidden_found = [char for char in self.FORBIDDEN_CHARS if char in prompt]
        if forbidden_found:
            raise ValueError(
                f"Prompt contains forbidden characters: {forbidden_found}. "
                "Null bytes and escape sequences are not allowed to prevent "
                "command injection and terminal manipulation attacks."
            )

        cmd = [self.gemma_executable, "--weights", self.model_path]

        if self.tokenizer_path:
            cmd.extend(["--tokenizer", self.tokenizer_path])

        # Add generation parameters
        cmd.extend([
            "--max_generated_tokens",
            str(self.max_tokens),
            "--temperature",
            str(self.temperature),
            "--prompt",
            prompt,
        ])

        return cmd

    async def generate_response(
        self, prompt: str, stream_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Generate response from gemma with optional streaming callback.

        Security features:
        - Validates prompt length (max 50KB) to prevent command injection
        - Limits response size (max 10MB) to prevent memory exhaustion
        - Sanitizes input for forbidden characters (null bytes, escape sequences)

        Performance features:
        - Efficient buffered I/O (8KB chunks) instead of byte-by-byte reading

        Args:
            prompt: Input prompt for generation (max 50KB, no null bytes/escape sequences)
            stream_callback: Optional callback function called with each generated chunk

        Returns:
            Complete generated response text (max 10MB)

        Raises:
            ValueError: If prompt validation fails (length or forbidden characters)
            RuntimeError: If gemma process fails or response exceeds size limit
        """
        cmd = self._build_command(prompt)  # Validates prompt security constraints

        # Debug: print command if enabled
        if self.debug_mode:
            logger.debug(f"Command: {' '.join(cmd)}")

        try:
            # Use asyncio subprocess for proper async handling
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
            )

            response_parts: list[str] = []
            total_size = 0  # Track accumulated response size for security

            # Read output in real-time for streaming
            if self.process.stdout:
                while True:
                    try:
                        # Performance fix: Read in 8KB chunks instead of 1 byte at a time
                        # This reduces syscalls by ~8000x and dramatically improves performance
                        chunk = await self.process.stdout.read(self.BUFFER_SIZE)
                        if not chunk:
                            break

                        # Security check: Prevent memory exhaustion
                        total_size += len(chunk)
                        if total_size > self.MAX_RESPONSE_SIZE:
                            raise RuntimeError(
                                f"Response exceeded maximum size of {self.MAX_RESPONSE_SIZE} bytes. "
                                "This prevents memory exhaustion attacks or runaway generation."
                            )

                        output = chunk.decode("utf-8", errors="ignore")
                        if output:
                            response_parts.append(output)
                            if stream_callback:
                                stream_callback(output)

                    except (OSError, UnicodeDecodeError) as e:
                        if self.debug_mode:
                            logger.debug(f"Read error: {e}")
                        break

            # Wait for process completion
            return_code = await self.process.wait()

            if return_code != 0:
                stderr_output = ""
                if self.process.stderr:
                    stderr_bytes = await self.process.stderr.read()
                    stderr_output = stderr_bytes.decode("utf-8", errors="ignore")

                raise RuntimeError(
                    f"Gemma process failed (code {return_code}): {stderr_output}"
                )

            return "".join(response_parts)

        except (OSError, ValueError, RuntimeError) as e:
            raise e

        finally:
            await self._cleanup_process()

    async def _cleanup_process(self) -> None:
        """Clean up the subprocess properly."""
        if self.process:
            try:
                if self.process.returncode is None:
                    # Process still running
                    self.process.terminate()
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        # Force kill if it doesn't terminate gracefully
                        self.process.kill()
                        await self.process.wait()
            except (OSError, ProcessLookupError) as e:
                if self.debug_mode:
                    logger.debug(f"Cleanup error: {e}")
            finally:
                self.process = None

    async def stop_generation(self) -> None:
        """Stop current generation process."""
        await self._cleanup_process()

    def set_parameters(
        self, max_tokens: Optional[int] = None, temperature: Optional[float] = None
    ) -> None:
        """
        Update generation parameters.

        Args:
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature

    def get_config(self) -> dict[str, any]:
        """
        Get current configuration.

        Returns:
            Dictionary with current configuration
        """
        return {
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "executable": self.gemma_executable,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "debug_mode": self.debug_mode,
        }


def create_gemma_interface(params: GemmaRuntimeParams, use_optimized: bool = True) -> GemmaInterface:
    """
    Create a Gemma interface based on configuration settings.

    Args:
        params: Runtime parameters for Gemma model
        use_optimized: Whether to use optimized interface (default: True)

    Returns:
        GemmaInterface or OptimizedGemmaInterface instance
    """
    if use_optimized:
        try:
            from gemma_cli.core.optimized_gemma import OptimizedGemmaInterface
            logger.info("Using OptimizedGemmaInterface with streaming and process reuse")
            return OptimizedGemmaInterface(params=params)
        except ImportError as e:
            logger.warning(f"Failed to import OptimizedGemmaInterface: {e}, falling back to standard interface")
            return GemmaInterface(params=params)
    else:
        logger.info("Using standard GemmaInterface")
        return GemmaInterface(params=params)
