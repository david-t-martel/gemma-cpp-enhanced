"""Async CLI runner for testing terminal applications."""

import asyncio
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
import pyte
from dataclasses import dataclass


@dataclass
class CLIResult:
    """Result from CLI command execution."""
    stdout: str
    stderr: str
    returncode: int
    duration: float
    terminal_display: str


class AsyncCLIRunner:
    """Run CLI commands asynchronously with terminal emulation."""

    def __init__(self, width: int = 120, height: int = 40):
        self.width = width
        self.height = height

    async def run(
        self,
        command: List[str],
        input_text: Optional[str] = None,
        timeout: float = 30.0,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[Path] = None,
    ) -> CLIResult:
        """Run CLI command and capture output.

        Args:
            command: Command and arguments
            input_text: Text to send to stdin
            timeout: Command timeout in seconds
            env: Environment variables
            cwd: Working directory

        Returns:
            CLIResult with output and metadata
        """
        start_time = asyncio.get_event_loop().time()

        # Create terminal emulator
        screen = pyte.Screen(self.width, self.height)
        stream = pyte.Stream(screen)

        # Merge environment
        full_env = {**subprocess.os.environ}
        if env:
            full_env.update(env)

        # Start process
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE if input_text else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
            cwd=str(cwd) if cwd else None,
        )

        # Send input if provided
        if input_text:
            process.stdin.write(input_text.encode())
            await process.stdin.drain()
            process.stdin.close()

        # Capture output
        try:
            stdout_data, stderr_data = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(f"Command timed out after {timeout}s")

        duration = asyncio.get_event_loop().time() - start_time

        # Process stdout through terminal emulator
        stdout_text = stdout_data.decode("utf-8", errors="replace")
        stream.feed(stdout_text)
        terminal_display = "\n".join(screen.display)

        return CLIResult(
            stdout=stdout_text,
            stderr=stderr_data.decode("utf-8", errors="replace"),
            returncode=process.returncode,
            duration=duration,
            terminal_display=terminal_display,
        )

    async def run_interactive(
        self,
        command: List[str],
        interactions: List[tuple[float, str]],
        timeout: float = 60.0,
        env: Optional[Dict[str, str]] = None,
    ) -> CLIResult:
        """Run interactive CLI session with timed inputs.

        Args:
            command: Command to run
            interactions: List of (delay_seconds, input_text) tuples
            timeout: Total timeout
            env: Environment variables

        Returns:
            CLIResult with captured session
        """
        start_time = asyncio.get_event_loop().time()

        # Create terminal emulator
        screen = pyte.Screen(self.width, self.height)
        stream = pyte.Stream(screen)

        # Start process
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**subprocess.os.environ, **(env or {})},
        )

        stdout_buffer = []
        stderr_buffer = []

        # Execute interactions
        for delay, input_text in interactions:
            # Wait for delay
            await asyncio.sleep(delay)

            # Read any output
            while True:
                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(), timeout=0.1
                    )
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace")
                    stdout_buffer.append(decoded)
                    stream.feed(decoded)
                except asyncio.TimeoutError:
                    break

            # Send input
            process.stdin.write(f"{input_text}\n".encode())
            await process.stdin.drain()

        # Close stdin and wait for completion
        process.stdin.close()

        try:
            remaining_time = timeout - (asyncio.get_event_loop().time() - start_time)
            await asyncio.wait_for(process.wait(), timeout=remaining_time)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

        # Capture remaining output
        try:
            remaining_stdout, remaining_stderr = await asyncio.wait_for(
                process.communicate(), timeout=1.0
            )
            if remaining_stdout:
                decoded = remaining_stdout.decode("utf-8", errors="replace")
                stdout_buffer.append(decoded)
                stream.feed(decoded)
            if remaining_stderr:
                stderr_buffer.append(remaining_stderr.decode("utf-8", errors="replace"))
        except asyncio.TimeoutError:
            pass

        duration = asyncio.get_event_loop().time() - start_time
        terminal_display = "\n".join(screen.display)

        return CLIResult(
            stdout="".join(stdout_buffer),
            stderr="".join(stderr_buffer),
            returncode=process.returncode,
            duration=duration,
            terminal_display=terminal_display,
        )
