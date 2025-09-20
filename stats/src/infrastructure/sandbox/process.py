"""Process isolation sandbox for secure tool execution."""

import asyncio
import contextlib
import os
import platform
from pathlib import Path

# Import resource module conditionally (Unix only)
try:
    import resource
    HAS_RESOURCE_MODULE = True
except ImportError:
    # resource module is not available on Windows
    HAS_RESOURCE_MODULE = False
import signal
import subprocess
import tempfile
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import aiofiles
import psutil
from pydantic import BaseModel

from src.domain.tools.base import ToolExecutionContext
from src.domain.tools.base import ToolResult
from src.domain.tools.schemas import ResourceLimits
from src.domain.tools.schemas import SecurityLevel


class ProcessSandboxConfig(BaseModel):
    """Process sandbox configuration."""

    max_memory_mb: int = 256
    max_cpu_percent: float = 25.0
    max_execution_time: int = 30
    max_file_descriptors: int = 64
    max_processes: int = 1
    enable_network: bool = False
    temp_dir: str | None = None
    allowed_paths: list[str] = []
    blocked_paths: list[str] = []
    environment_variables: dict[str, str] = {}
    chroot_dir: str | None = None
    user_id: int | None = None
    group_id: int | None = None


class ProcessResult(BaseModel):
    """Result from process execution."""

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    peak_memory_mb: float
    cpu_time: float
    process_id: int | None = None
    resource_usage: dict[str, Any] = {}


class ProcessSandbox:
    """Process-based secure execution sandbox."""

    def __init__(self, config: ProcessSandboxConfig | None = None):
        self.config = config or ProcessSandboxConfig()
        self._is_unix = platform.system() in ("Linux", "Darwin")
        self._active_processes: dict[int, psutil.Process] = {}

    async def initialize(self) -> None:
        """Initialize the process sandbox."""
        logger.info("Process sandbox initialized")

        # Set up signal handlers for cleanup
        if self._is_unix:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle signals for cleanup."""
        logger.info(f"Received signal {signum}, cleaning up processes")
        asyncio.create_task(self.cleanup())

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        context: ToolExecutionContext | None = None,
        files: dict[str, str] | None = None,
        requirements: list[str] | None = None,
    ) -> ProcessResult:
        """Execute code in a secure process sandbox."""
        # Create temporary workspace
        with tempfile.TemporaryDirectory(dir=self.config.temp_dir) as temp_dir:
            temp_path = Path(temp_dir)

            # Write code file
            code_file = await self._write_code_file(temp_path, code, language)

            # Write additional files
            if files:
                await self._write_additional_files(temp_path, files)

            # Install requirements if needed
            if requirements and language == "python":
                await self._install_python_requirements(temp_path, requirements)

            # Prepare execution command
            command = self._get_execution_command(language, temp_path / code_file)

            # Execute with restrictions
            return await self._execute_command(command, temp_path, context)

    async def execute_command(
        self,
        command: str | list[str],
        context: ToolExecutionContext | None = None,
        working_dir: Path | None = None,
    ) -> ProcessResult:
        """Execute a command in a secure process sandbox."""
        if isinstance(command, str):
            command = [command]

        work_dir = working_dir or Path.cwd()
        return await self._execute_command(command, work_dir, context)

    async def _write_code_file(self, temp_path: Path, code: str, language: str) -> str:
        """Write code to a file."""
        extensions = {
            "python": "py",
            "javascript": "js",
            "node": "js",
            "java": "java",
            "rust": "rs",
            "go": "go",
            "bash": "sh",
            "shell": "sh",
            "powershell": "ps1",
            "c": "c",
            "cpp": "cpp",
            "csharp": "cs",
        }

        extension = extensions.get(language.lower(), "txt")
        code_file = f"main.{extension}"
        code_path = temp_path / code_file

        async with aiofiles.open(code_path, "w", encoding="utf-8") as f:
            await f.write(code)

        # Make executable if needed
        if extension in ["sh", "ps1"]:
            os.chmod(code_path, 0o755)

        return code_file

    async def _write_additional_files(self, temp_path: Path, files: dict[str, str]) -> None:
        """Write additional files needed for execution."""
        for filename, content in files.items():
            file_path = temp_path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(content)

    async def _install_python_requirements(self, temp_path: Path, requirements: list[str]) -> None:
        """Install Python requirements in the sandbox."""
        if not requirements:
            return

        # Create requirements.txt
        requirements_path = temp_path / "requirements.txt"
        async with aiofiles.open(requirements_path, "w", encoding="utf-8") as f:
            await f.write("\n".join(requirements))

        # Install requirements with pip
        try:
            command = [
                "python",
                "-m",
                "pip",
                "install",
                "--user",
                "--no-warn-script-location",
                "--disable-pip-version-check",
                "-r",
                str(requirements_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            _stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)

            if process.returncode != 0:
                logger.warning(f"Failed to install requirements: {stderr.decode()}")

        except Exception as e:
            logger.warning(f"Error installing Python requirements: {e}")

    def _get_execution_command(self, language: str, code_file: Path) -> list[str]:
        """Get execution command for the language."""
        commands = {
            "python": ["python", str(code_file)],
            "python3": ["python3", str(code_file)],
            "javascript": ["node", str(code_file)],
            "node": ["node", str(code_file)],
            "java": self._get_java_command(code_file),
            "rust": self._get_rust_command(code_file),
            "go": ["go", "run", str(code_file)],
            "bash": ["bash", str(code_file)],
            "shell": ["sh", str(code_file)],
            "powershell": ["powershell", "-File", str(code_file)],
            "c": self._get_c_command(code_file),
            "cpp": self._get_cpp_command(code_file),
            "csharp": self._get_csharp_command(code_file),
        }

        return commands.get(language.lower(), ["cat", str(code_file)])

    def _get_java_command(self, code_file: Path) -> list[str]:
        """Get Java compilation and execution command."""
        class_name = code_file.stem
        return ["sh", "-c", f"javac {code_file} && java -cp {code_file.parent} {class_name}"]

    def _get_rust_command(self, code_file: Path) -> list[str]:
        """Get Rust compilation and execution command."""
        executable = code_file.with_suffix("")
        return ["sh", "-c", f"rustc {code_file} -o {executable} && {executable}"]

    def _get_c_command(self, code_file: Path) -> list[str]:
        """Get C compilation and execution command."""
        executable = code_file.with_suffix("")
        return ["sh", "-c", f"gcc {code_file} -o {executable} && {executable}"]

    def _get_cpp_command(self, code_file: Path) -> list[str]:
        """Get C++ compilation and execution command."""
        executable = code_file.with_suffix("")
        return ["sh", "-c", f"g++ {code_file} -o {executable} && {executable}"]

    def _get_csharp_command(self, code_file: Path) -> list[str]:
        """Get C# compilation and execution command."""
        executable = code_file.with_suffix(".exe")
        return ["sh", "-c", f"csc {code_file} -out:{executable} && mono {executable}"]

    async def _execute_command(
        self, command: list[str], working_dir: Path, context: ToolExecutionContext | None
    ) -> ProcessResult:
        """Execute command with resource restrictions."""
        start_time = time.time()
        process = None
        psutil_process = None

        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(self.config.environment_variables)

            # Apply security restrictions
            restrictions = self._prepare_restrictions(context)

            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=working_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                preexec_fn=restrictions if self._is_unix else None,
            )

            # Track the process
            if process.pid:
                try:
                    psutil_process = psutil.Process(process.pid)
                    self._active_processes[process.pid] = psutil_process
                except psutil.NoSuchProcess:
                    pass

            # Wait for completion with timeout and resource monitoring
            timeout = self._get_timeout(context)
            stdout, stderr = await self._wait_with_monitoring(process, psutil_process, timeout)

            execution_time = time.time() - start_time
            exit_code = process.returncode or 0

            # Get resource usage
            resource_usage = await self._get_resource_usage(psutil_process)

            return ProcessResult(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                execution_time=execution_time,
                peak_memory_mb=resource_usage.get("peak_memory_mb", 0),
                cpu_time=resource_usage.get("cpu_time", 0),
                process_id=process.pid,
                resource_usage=resource_usage,
            )

        except TimeoutError:
            execution_time = time.time() - start_time
            await self._terminate_process(process, psutil_process)

            return ProcessResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr="Process execution timed out",
                execution_time=execution_time,
                peak_memory_mb=0,
                cpu_time=0,
                process_id=process.pid if process else None,
                resource_usage={},
            )

        except Exception as e:
            execution_time = time.time() - start_time
            await self._terminate_process(process, psutil_process)

            return ProcessResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                peak_memory_mb=0,
                cpu_time=0,
                process_id=process.pid if process else None,
                resource_usage={},
            )

        finally:
            # Cleanup process tracking
            if process and process.pid in self._active_processes:
                del self._active_processes[process.pid]

    def _prepare_restrictions(self, context: Optional[ToolExecutionContext]) -> Optional[Callable]:
        """Prepare process restrictions (Unix only)."""
        if not self._is_unix:
            return None

        def set_limits():
            try:
                if not HAS_RESOURCE_MODULE:
                    # Resource limits not available on Windows
                    return

                # Set memory limit
                memory_limit = self.config.max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

                # Set CPU time limit
                cpu_limit = self.config.max_execution_time
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))

                # Set file descriptor limit
                resource.setrlimit(
                    resource.RLIMIT_NOFILE,
                    (self.config.max_file_descriptors, self.config.max_file_descriptors),
                )

                # Set process limit
                resource.setrlimit(
                    resource.RLIMIT_NPROC, (self.config.max_processes, self.config.max_processes)
                )

                # Disable core dumps
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

                # Change user/group if specified
                if self.config.group_id is not None:
                    os.setgid(self.config.group_id)
                if self.config.user_id is not None:
                    os.setuid(self.config.user_id)

            except Exception as e:
                logger.warning(f"Failed to set resource limits: {e}")

        return set_limits

    def _get_timeout(self, context: ToolExecutionContext | None) -> int:
        """Get execution timeout."""
        if context and context.timeout:
            return min(context.timeout, self.config.max_execution_time)
        return self.config.max_execution_time

    async def _wait_with_monitoring(
        self,
        process: asyncio.subprocess.Process,
        psutil_process: psutil.Process | None,
        timeout: int,
    ) -> tuple[bytes, bytes]:
        """Wait for process completion with resource monitoring."""
        monitoring_interval = 0.1  # Check every 100ms
        elapsed = 0.0

        while elapsed < timeout:
            try:
                # Check if process is still running
                if process.returncode is not None:
                    # Process completed
                    stdout, stderr = await process.communicate()
                    return stdout, stderr

                # Monitor resource usage
                if psutil_process and psutil_process.is_running():
                    try:
                        # Check memory usage
                        memory_info = psutil_process.memory_info()
                        memory_mb = memory_info.rss / (1024 * 1024)

                        if memory_mb > self.config.max_memory_mb:
                            logger.warning(f"Process {process.pid} exceeded memory limit")
                            psutil_process.terminate()
                            raise ResourceWarning("Memory limit exceeded")

                        # Check CPU usage
                        cpu_percent = psutil_process.cpu_percent()
                        if cpu_percent > self.config.max_cpu_percent:
                            logger.warning(f"Process {process.pid} exceeded CPU limit")

                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # Wait a bit before next check
                await asyncio.sleep(monitoring_interval)
                elapsed += monitoring_interval

            except Exception as e:
                logger.warning(f"Error during process monitoring: {e}")
                break

        # Timeout reached
        raise TimeoutError("Process execution timed out")

    async def _terminate_process(
        self, process: asyncio.subprocess.Process | None, psutil_process: psutil.Process | None
    ) -> None:
        """Terminate a running process."""
        if psutil_process:
            try:
                # Terminate process tree
                children = psutil_process.children(recursive=True)
                for child in children:
                    with contextlib.suppress(psutil.NoSuchProcess):
                        child.terminate()

                psutil_process.terminate()

                # Wait a bit for graceful termination
                await asyncio.sleep(0.5)

                # Force kill if still running
                if psutil_process.is_running():
                    psutil_process.kill()
                    for child in children:
                        try:
                            if child.is_running():
                                child.kill()
                        except psutil.NoSuchProcess:
                            pass

            except psutil.NoSuchProcess:
                pass
            except Exception as e:
                logger.warning(f"Error terminating process: {e}")

        if process:
            try:
                if process.returncode is None:
                    process.terminate()
                    await asyncio.sleep(0.5)
                    if process.returncode is None:
                        process.kill()
            except Exception as e:
                logger.warning(f"Error terminating subprocess: {e}")

    async def _get_resource_usage(self, psutil_process: psutil.Process | None) -> dict[str, Any]:
        """Get process resource usage statistics."""
        usage = {}

        if psutil_process:
            try:
                # Memory info
                memory_info = psutil_process.memory_info()
                usage["peak_memory_mb"] = memory_info.rss / (1024 * 1024)
                usage["virtual_memory_mb"] = memory_info.vms / (1024 * 1024)

                # CPU info
                cpu_times = psutil_process.cpu_times()
                usage["cpu_time"] = cpu_times.user + cpu_times.system
                usage["cpu_percent"] = psutil_process.cpu_percent()

                # I/O info
                try:
                    io_counters = psutil_process.io_counters()
                    usage["read_bytes"] = io_counters.read_bytes
                    usage["write_bytes"] = io_counters.write_bytes
                except (psutil.AccessDenied, AttributeError):
                    pass

                # File descriptors
                with contextlib.suppress(psutil.AccessDenied, AttributeError):
                    usage["num_fds"] = psutil_process.num_fds()

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning(f"Failed to get resource usage: {e}")

        return usage

    async def list_active_processes(self) -> list[dict[str, Any]]:
        """List currently active sandbox processes."""
        active = []

        for pid, process in list(self._active_processes.items()):
            try:
                if process.is_running():
                    info = {
                        "pid": pid,
                        "name": process.name(),
                        "create_time": process.create_time(),
                        "memory_mb": process.memory_info().rss / (1024 * 1024),
                        "cpu_percent": process.cpu_percent(),
                    }
                    active.append(info)
                else:
                    # Remove dead process from tracking
                    del self._active_processes[pid]
            except (psutil.NoSuchProcess, KeyError):
                # Process is gone
                if pid in self._active_processes:
                    del self._active_processes[pid]

        return active

    async def terminate_process(self, pid: int) -> bool:
        """Terminate a specific sandbox process."""
        if pid in self._active_processes:
            process = self._active_processes[pid]
            try:
                await self._terminate_process(None, process)
                return True
            except Exception as e:
                logger.error(f"Failed to terminate process {pid}: {e}")
                return False
        return False

    async def cleanup(self) -> None:
        """Cleanup all active processes."""
        logger.info(f"Cleaning up {len(self._active_processes)} active processes")

        for pid, process in list(self._active_processes.items()):
            try:
                await self._terminate_process(None, process)
            except Exception as e:
                logger.warning(f"Error cleaning up process {pid}: {e}")

        self._active_processes.clear()
        logger.info("Process sandbox cleanup completed")

    def get_system_limits(self) -> dict[str, Any]:
        """Get current system resource limits."""
        limits = {}

        if self._is_unix and HAS_RESOURCE_MODULE:
            try:
                # Get current resource limits
                memory_limit = resource.getrlimit(resource.RLIMIT_AS)
                cpu_limit = resource.getrlimit(resource.RLIMIT_CPU)
                fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
                proc_limit = resource.getrlimit(resource.RLIMIT_NPROC)

                limits.update(
                    {
                        "memory_limit": memory_limit,
                        "cpu_limit": cpu_limit,
                        "file_descriptor_limit": fd_limit,
                        "process_limit": proc_limit,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to get system limits: {e}")

        # Add system info
        limits.update(
            {
                "available_memory_mb": psutil.virtual_memory().available / (1024 * 1024),
                "cpu_count": psutil.cpu_count(),
                "load_average": psutil.getloadavg() if hasattr(psutil, "getloadavg") else None,
            }
        )

        return limits
