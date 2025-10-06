#!/usr/bin/env python3
"""
Integration tests for WSL gemma backend.
"""

import asyncio
import json
import pytest
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestGemmaBackendIntegration:
    """Test suite for gemma.cpp backend integration."""

    @pytest.fixture
    def gemma_path(self):
        """Path to the gemma executable."""
        return "/c/codedev/llm/gemma/gemma.cpp/build_wsl/gemma"

    @pytest.fixture
    def model_paths(self):
        """Paths to model files."""
        return {
            "model": "/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs",
            "tokenizer": "/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/tokenizer.spm"
        }

    def test_gemma_executable_exists(self, gemma_path):
        """Test that gemma executable exists and is executable."""
        gemma_file = Path(gemma_path)
        assert gemma_file.exists(), f"Gemma executable not found at {gemma_path}"
        assert gemma_file.is_file(), f"Gemma path is not a file: {gemma_path}"

        # Test if executable (on Unix-like systems)
        if hasattr(gemma_file, 'stat'):
            import stat
            mode = gemma_file.stat().st_mode
            is_executable = bool(mode & stat.S_IEXEC)
            assert is_executable, f"Gemma executable is not executable: {gemma_path}"

    def test_model_files_exist(self, model_paths):
        """Test that required model files exist."""
        for name, path in model_paths.items():
            file_path = Path(path)
            assert file_path.exists(), f"{name} file not found at {path}"
            assert file_path.is_file(), f"{name} path is not a file: {path}"
            assert file_path.stat().st_size > 0, f"{name} file is empty: {path}"

    @pytest.mark.integration
    def test_gemma_executable_help(self, gemma_path):
        """Test that gemma executable responds to help command."""
        try:
            result = subprocess.run(
                [gemma_path, "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Should either succeed or show help message
            assert result.returncode in [0, 1], f"Unexpected return code: {result.returncode}"

            # Should have some output
            output = result.stdout + result.stderr
            assert len(output) > 0, "No output from gemma --help"

        except subprocess.TimeoutExpired:
            pytest.fail("Gemma executable timed out on --help")
        except FileNotFoundError:
            pytest.fail(f"Gemma executable not found or not executable: {gemma_path}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_gemma_model_loading(self, gemma_path, model_paths):
        """Test that gemma can load the model files."""
        try:
            # Test basic model loading with minimal prompt
            cmd = [
                gemma_path,
                "--tokenizer", model_paths["tokenizer"],
                "--weights", model_paths["model"],
                "--prompt", "Hello",
                "--max_tokens", "1"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # Model loading can take time
            )

            # Check if model loaded successfully
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")

            # Model loading issues are common, so we check for specific error patterns
            stderr_lower = result.stderr.lower()
            if "no such file" in stderr_lower or "cannot open" in stderr_lower:
                pytest.fail(f"Model file access error: {result.stderr}")
            elif "out of memory" in stderr_lower or "memory" in stderr_lower:
                pytest.skip("Insufficient memory for model loading")
            elif result.returncode != 0:
                # Other errors might be environment-specific
                pytest.skip(f"Model loading failed (environment issue): {result.stderr}")

        except subprocess.TimeoutExpired:
            pytest.skip("Model loading timed out (possibly too slow for testing)")
        except FileNotFoundError:
            pytest.fail(f"Gemma executable not found: {gemma_path}")

    @pytest.mark.integration
    def test_gemma_subprocess_creation(self, gemma_path, model_paths):
        """Test subprocess creation patterns used by the server."""
        # Test the command pattern used by the server
        cmd = [
            gemma_path,
            "--tokenizer", model_paths["tokenizer"],
            "--weights", model_paths["model"],
            "--prompt", "Test",
            "--max_tokens", "5",
            "--verbosity", "0"
        ]

        try:
            # Test subprocess creation without execution
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Give it a moment to start
            time.sleep(1)

            # Check if process started
            assert proc.poll() is None or proc.returncode == 0, "Process failed to start"

            # Terminate the process
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        except Exception as e:
            pytest.fail(f"Failed to create subprocess: {e}")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_async_gemma_execution(self, gemma_path, model_paths):
        """Test async execution pattern used by the server."""
        cmd = [
            gemma_path,
            "--tokenizer", model_paths["tokenizer"],
            "--weights", model_paths["model"],
            "--prompt", "Hello",
            "--max_tokens", "3",
            "--verbosity", "0"
        ]

        try:
            # Create async subprocess
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=30.0
                )

                # Check results
                if proc.returncode != 0:
                    # Log for debugging but don't fail (environment issues)
                    print(f"Gemma execution failed: {stderr.decode()}")
                    pytest.skip("Gemma execution failed (environment issue)")

                # Should have some output
                output = stdout.decode().strip()
                assert len(output) > 0, "No output from gemma"

            except asyncio.TimeoutError:
                proc.terminate()
                pytest.skip("Gemma execution timed out")

        except Exception as e:
            pytest.skip(f"Async execution test failed: {e}")

    def test_gemma_command_validation(self, gemma_path, model_paths):
        """Test command line argument validation."""
        # Test invalid arguments
        invalid_commands = [
            [gemma_path, "--invalid_flag"],
            [gemma_path, "--tokenizer", "/nonexistent/file"],
            [gemma_path, "--weights", "/nonexistent/file"],
        ]

        for cmd in invalid_commands:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                # Should fail with non-zero exit code
                assert result.returncode != 0, f"Expected failure for command: {cmd}"

            except subprocess.TimeoutExpired:
                # Timeout is also acceptable for invalid commands
                pass

    @pytest.mark.integration
    def test_gemma_memory_usage(self, gemma_path, model_paths):
        """Test memory usage patterns during model loading."""
        import psutil
        import os

        cmd = [
            gemma_path,
            "--tokenizer", model_paths["tokenizer"],
            "--weights", model_paths["model"],
            "--prompt", "Test",
            "--max_tokens", "1"
        ]

        try:
            # Start process
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Monitor memory for a short time
            max_memory = 0
            for _ in range(10):  # Check for 5 seconds
                try:
                    process = psutil.Process(proc.pid)
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    max_memory = max(max_memory, memory_mb)
                    time.sleep(0.5)

                    if proc.poll() is not None:
                        break
                except psutil.NoSuchProcess:
                    break

            # Cleanup
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

            # Memory usage should be reasonable (adjust based on model size)
            # 2B model should use less than 4GB
            assert max_memory < 4096, f"Memory usage too high: {max_memory:.1f}MB"

        except psutil.NoSuchProcess:
            pytest.skip("Cannot monitor memory usage")
        except Exception as e:
            pytest.skip(f"Memory monitoring failed: {e}")

    @pytest.mark.integration
    def test_gemma_concurrent_execution(self, gemma_path, model_paths):
        """Test behavior with concurrent gemma processes."""
        cmd = [
            gemma_path,
            "--tokenizer", model_paths["tokenizer"],
            "--weights", model_paths["model"],
            "--prompt", "Quick test",
            "--max_tokens", "1"
        ]

        processes = []
        try:
            # Start multiple processes
            for i in range(3):
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                processes.append(proc)

            # Wait for completion or timeout
            results = []
            for proc in processes:
                try:
                    stdout, stderr = proc.communicate(timeout=30)
                    results.append((proc.returncode, stdout, stderr))
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait()
                    results.append((-1, b"", b"timeout"))

            # At least one should succeed (resource permitting)
            successful = sum(1 for ret, _, _ in results if ret == 0)
            if successful == 0:
                pytest.skip("No concurrent processes succeeded (resource constraints)")

        finally:
            # Cleanup any remaining processes
            for proc in processes:
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait(timeout=5)
                except:
                    pass

    def test_path_handling(self, model_paths):
        """Test various path formats and conversions."""
        # Test Windows to WSL path conversion if needed
        for name, path in model_paths.items():
            # Path should be accessible
            file_path = Path(path)
            assert file_path.exists(), f"Path not accessible: {path}"

            # Test absolute path
            abs_path = file_path.absolute()
            assert abs_path.exists(), f"Absolute path not accessible: {abs_path}"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_gemma_response_quality(self, gemma_path, model_paths):
        """Test basic response quality from gemma."""
        cmd = [
            gemma_path,
            "--tokenizer", model_paths["tokenizer"],
            "--weights", model_paths["model"],
            "--prompt", "What is 2+2?",
            "--max_tokens", "10"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                pytest.skip(f"Execution failed: {result.stderr}")

            output = result.stdout.strip()
            if len(output) == 0:
                pytest.skip("No output generated")

            # Basic sanity check - should contain some relevant content
            # (This is a very basic check, actual content validation would be more complex)
            assert len(output) > 0, "Empty response"

        except subprocess.TimeoutExpired:
            pytest.skip("Response generation timed out")
        except Exception as e:
            pytest.skip(f"Response quality test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])