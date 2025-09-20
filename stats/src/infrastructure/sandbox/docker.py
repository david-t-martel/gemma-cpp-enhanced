"""Docker-based code execution sandbox for secure tool execution."""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import aiofiles
import docker
from docker.errors import APIError
from docker.errors import ContainerError
from docker.errors import ImageNotFound
from pydantic import BaseModel

from src.domain.tools.base import ToolExecutionContext
from src.domain.tools.base import ToolResult
from src.domain.tools.schemas import ResourceLimits
from src.domain.tools.schemas import SecurityLevel


class DockerSandboxConfig(BaseModel):
    """Docker sandbox configuration."""

    image: str = "python:3.11-slim"
    working_dir: str = "/app"
    network_mode: str = "none"  # No network access by default
    memory_limit: str = "512m"
    cpu_count: float = 0.5
    timeout: int = 30
    remove_container: bool = True
    read_only_root_fs: bool = True
    no_new_privileges: bool = True
    user: str = "nobody"
    environment: dict[str, str] = {}
    volumes: dict[str, str] = {}
    cap_drop: list[str] = ["ALL"]
    cap_add: list[str] = []
    security_opt: list[str] = ["no-new-privileges:true"]


class SandboxResult(BaseModel):
    """Result from sandbox execution."""

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    resource_usage: dict[str, Any] = {}
    container_id: str | None = None
    image_used: str = ""


class DockerSandbox:
    """Docker-based secure execution sandbox."""

    def __init__(self, config: DockerSandboxConfig | None = None):
        self.config = config or DockerSandboxConfig()
        self._docker_client: docker.DockerClient | None = None
        self._available_images: dict[str, str] = {}

    async def initialize(self) -> None:
        """Initialize the Docker sandbox."""
        try:
            # Create Docker client
            self._docker_client = docker.from_env()

            # Test Docker connection
            self._docker_client.ping()
            logger.info("Docker sandbox initialized successfully")

            # Prepare base images
            await self._prepare_images()

        except Exception as e:
            logger.error(f"Failed to initialize Docker sandbox: {e}")
            raise RuntimeError(f"Docker sandbox initialization failed: {e}")

    async def _prepare_images(self) -> None:
        """Prepare and cache Docker images for different languages."""
        images = {
            "python": "python:3.11-slim",
            "node": "node:18-alpine",
            "java": "openjdk:11-jre-slim",
            "rust": "rust:1.70-slim",
            "go": "golang:1.20-alpine",
            "bash": "bash:5.2-alpine3.17",
        }

        for lang, image in images.items():
            try:
                # Pull image if not available
                self._docker_client.images.get(image)
                self._available_images[lang] = image
                logger.debug(f"Image {image} is available")
            except ImageNotFound:
                logger.info(f"Pulling Docker image: {image}")
                try:
                    self._docker_client.images.pull(image)
                    self._available_images[lang] = image
                    logger.info(f"Successfully pulled image: {image}")
                except Exception as e:
                    logger.warning(f"Failed to pull image {image}: {e}")

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        context: ToolExecutionContext | None = None,
        files: dict[str, str] | None = None,
        requirements: list[str] | None = None,
    ) -> SandboxResult:
        """Execute code in a secure Docker container."""
        if not self._docker_client:
            raise RuntimeError("Docker sandbox not initialized")

        # Determine the appropriate image
        image = self._get_image_for_language(language)

        # Create temporary directory for code and files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write code file
            code_file = await self._write_code_file(temp_path, code, language)

            # Write additional files
            if files:
                await self._write_additional_files(temp_path, files)

            # Create requirements file if needed
            if requirements and language == "python":
                await self._write_requirements_file(temp_path, requirements)

            # Prepare container configuration
            container_config = await self._prepare_container_config(
                image, temp_path, code_file, language, context
            )

            # Execute in container
            return await self._run_container(container_config, context)

    def _get_image_for_language(self, language: str) -> str:
        """Get appropriate Docker image for language."""
        return self._available_images.get(language.lower(), self.config.image)

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
        }

        extension = extensions.get(language.lower(), "txt")
        code_file = f"main.{extension}"
        code_path = temp_path / code_file

        async with aiofiles.open(code_path, "w", encoding="utf-8") as f:
            await f.write(code)

        return code_file

    async def _write_additional_files(self, temp_path: Path, files: dict[str, str]) -> None:
        """Write additional files needed for execution."""
        for filename, content in files.items():
            file_path = temp_path / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(content)

    async def _write_requirements_file(self, temp_path: Path, requirements: list[str]) -> None:
        """Write Python requirements.txt file."""
        requirements_path = temp_path / "requirements.txt"
        async with aiofiles.open(requirements_path, "w", encoding="utf-8") as f:
            await f.write("\n".join(requirements))

    async def _prepare_container_config(
        self,
        image: str,
        temp_path: Path,
        code_file: str,
        language: str,
        context: ToolExecutionContext | None,
    ) -> dict[str, Any]:
        """Prepare container configuration."""
        # Base configuration
        config = {
            "image": image,
            "working_dir": self.config.working_dir,
            "volumes": {str(temp_path): {"bind": self.config.working_dir, "mode": "ro"}},
            "network_mode": self.config.network_mode,
            "mem_limit": self.config.memory_limit,
            "cpu_count": self.config.cpu_count,
            "remove": self.config.remove_container,
            "read_only": self.config.read_only_root_fs,
            "user": self.config.user,
            "cap_drop": self.config.cap_drop,
            "cap_add": self.config.cap_add,
            "security_opt": self.config.security_opt,
            "environment": self.config.environment.copy(),
        }

        # Add execution command based on language
        config["command"] = self._get_execution_command(language, code_file)

        # Apply security restrictions from context
        if context and context.security_level:
            config = self._apply_security_restrictions(config, context.security_level)

        # Apply resource limits from context
        if context and context.timeout:
            config["timeout"] = min(context.timeout, self.config.timeout)

        return config

    def _get_execution_command(self, language: str, code_file: str) -> list[str]:
        """Get execution command for the language."""
        commands = {
            "python": ["python", code_file],
            "javascript": ["node", code_file],
            "node": ["node", code_file],
            "java": ["sh", "-c", f"javac {code_file} && java {code_file.replace('.java', '')}"],
            "rust": ["sh", "-c", f"rustc {code_file} -o main && ./main"],
            "go": ["go", "run", code_file],
            "bash": ["bash", code_file],
            "shell": ["sh", code_file],
        }
        return commands.get(language.lower(), ["cat", code_file])

    def _apply_security_restrictions(
        self, config: dict[str, Any], security_level: str
    ) -> dict[str, Any]:
        """Apply security restrictions based on security level."""
        if security_level == "maximum":
            config["network_mode"] = "none"
            config["read_only"] = True
            config["cap_drop"] = ["ALL"]
            config["cap_add"] = []
            config["user"] = "nobody"
            config["mem_limit"] = "256m"
            config["cpu_count"] = 0.25

        elif security_level == "strict":
            config["network_mode"] = "none"
            config["read_only"] = True
            config["cap_drop"] = ["ALL"]
            config["user"] = "nobody"

        elif security_level == "standard":
            config["network_mode"] = "none"
            config["cap_drop"] = ["NET_RAW", "SYS_ADMIN", "SYS_PTRACE"]

        # Minimal security level uses default settings

        return config

    async def _run_container(
        self, config: dict[str, Any], context: ToolExecutionContext | None
    ) -> SandboxResult:
        """Run the Docker container and capture results."""
        container = None
        start_time = time.time()

        try:
            # Create and start container
            container = self._docker_client.containers.create(**config)
            container.start()

            # Wait for completion with timeout
            timeout = config.get("timeout", self.config.timeout)
            try:
                exit_code = container.wait(timeout=timeout)["StatusCode"]
            except Exception:
                # Timeout or other error
                if container:
                    container.kill()
                raise TimeoutError(f"Container execution timed out after {timeout}s")

            # Get output
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8")

            # Get resource usage statistics
            stats = {}
            try:
                stats = container.stats(stream=False)
            except Exception as e:
                logger.warning(f"Failed to get container stats: {e}")

            execution_time = time.time() - start_time

            return SandboxResult(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                execution_time=execution_time,
                resource_usage=self._parse_resource_stats(stats),
                container_id=container.id if container else None,
                image_used=config["image"],
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Container execution failed: {e}")

            return SandboxResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                container_id=container.id if container else None,
                image_used=config.get("image", "unknown"),
            )

        finally:
            # Cleanup container
            if container:
                try:
                    if config.get("remove", True):
                        container.remove(force=True)
                    else:
                        container.stop()
                except Exception as e:
                    logger.warning(f"Failed to cleanup container: {e}")

    def _parse_resource_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Parse Docker container resource statistics."""
        if not stats:
            return {}

        try:
            resource_usage = {}

            # Memory usage
            if "memory" in stats:
                memory_stats = stats["memory"]
                resource_usage["memory_used_mb"] = memory_stats.get("usage", 0) / (1024 * 1024)
                resource_usage["memory_limit_mb"] = memory_stats.get("limit", 0) / (1024 * 1024)

            # CPU usage
            if "cpu" in stats:
                cpu_stats = stats["cpu"]
                # Calculate CPU percentage (simplified)
                cpu_usage = cpu_stats.get("cpu_usage", {})
                if "total_usage" in cpu_usage:
                    resource_usage["cpu_total_usage"] = cpu_usage["total_usage"]

            # Network I/O
            if "networks" in stats:
                total_rx = sum(net.get("rx_bytes", 0) for net in stats["networks"].values())
                total_tx = sum(net.get("tx_bytes", 0) for net in stats["networks"].values())
                resource_usage["network_rx_bytes"] = total_rx
                resource_usage["network_tx_bytes"] = total_tx

            # Block I/O
            if "block_io" in stats:
                block_io = stats["block_io"]
                resource_usage["block_read_bytes"] = sum(
                    stat.get("value", 0)
                    for stat in block_io.get("io_service_bytes_recursive", [])
                    if stat.get("op") == "Read"
                )
                resource_usage["block_write_bytes"] = sum(
                    stat.get("value", 0)
                    for stat in block_io.get("io_service_bytes_recursive", [])
                    if stat.get("op") == "Write"
                )

            return resource_usage

        except Exception as e:
            logger.warning(f"Failed to parse resource stats: {e}")
            return {}

    async def execute_shell_command(
        self, command: str, context: ToolExecutionContext | None = None
    ) -> SandboxResult:
        """Execute a shell command in a secure container."""
        # Use bash image for shell commands
        image = self._available_images.get("bash", "bash:5.2-alpine3.17")

        config = {
            "image": image,
            "command": ["sh", "-c", command],
            "working_dir": "/tmp",
            "network_mode": "none",
            "mem_limit": "256m",
            "cpu_count": 0.25,
            "remove": True,
            "read_only": False,  # Allow temporary file creation
            "user": "nobody",
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
        }

        # Apply context restrictions
        if context:
            if context.security_level:
                config = self._apply_security_restrictions(config, context.security_level)
            if context.timeout:
                config["timeout"] = min(context.timeout, 30)

        return await self._run_container(config, context)

    async def cleanup(self) -> None:
        """Cleanup Docker resources."""
        if self._docker_client:
            try:
                # Remove any leftover containers
                containers = self._docker_client.containers.list(
                    all=True, filters={"label": "sandbox=true"}
                )
                for container in containers:
                    try:
                        container.remove(force=True)
                    except Exception as e:
                        logger.warning(f"Failed to remove container {container.id}: {e}")

                self._docker_client.close()
                logger.info("Docker sandbox cleanup completed")

            except Exception as e:
                logger.error(f"Error during Docker sandbox cleanup: {e}")

    async def list_available_images(self) -> dict[str, str]:
        """List available Docker images for execution."""
        return self._available_images.copy()

    async def pull_image(self, image: str, tag: str = "latest") -> bool:
        """Pull a Docker image."""
        if not self._docker_client:
            raise RuntimeError("Docker sandbox not initialized")

        try:
            full_image = f"{image}:{tag}"
            logger.info(f"Pulling Docker image: {full_image}")
            self._docker_client.images.pull(image, tag=tag)
            return True
        except Exception as e:
            logger.error(f"Failed to pull image {image}:{tag}: {e}")
            return False

    async def build_custom_image(
        self, dockerfile: str, tag: str, context_files: dict[str, str] | None = None
    ) -> bool:
        """Build a custom Docker image."""
        if not self._docker_client:
            raise RuntimeError("Docker sandbox not initialized")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write Dockerfile
            dockerfile_path = temp_path / "Dockerfile"
            async with aiofiles.open(dockerfile_path, "w") as f:
                await f.write(dockerfile)

            # Write context files
            if context_files:
                for filename, content in context_files.items():
                    file_path = temp_path / filename
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    async with aiofiles.open(file_path, "w") as f:
                        await f.write(content)

            try:
                logger.info(f"Building custom Docker image: {tag}")
                _image, _build_logs = self._docker_client.images.build(
                    path=str(temp_path), tag=tag, rm=True, forcerm=True
                )

                logger.info(f"Successfully built custom image: {tag}")
                return True

            except Exception as e:
                logger.error(f"Failed to build custom image {tag}: {e}")
                return False
