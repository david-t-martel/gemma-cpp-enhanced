
import asyncio
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from gemma_cli.core.gemma import GemmaInterface


@pytest.fixture
def gemma_interface(tmp_path):
    gemma_executable = tmp_path / "gemma.exe"
    gemma_executable.touch()
    return GemmaInterface(model_path="dummy_model.sbs", gemma_executable=str(gemma_executable))


@patch("shutil.which", return_value=None)
@patch("os.environ.get", return_value=None)
def test_find_gemma_executable(mock_os_environ_get, mock_shutil_which, tmp_path):
    gemma_executable = tmp_path / "build" / "Release" / "gemma.exe"
    gemma_executable.parent.mkdir(parents=True)
    gemma_executable.touch()
    with patch.object(Path, 'cwd', return_value=tmp_path):
        interface = GemmaInterface(model_path="dummy_model.sbs")
        assert interface.gemma_executable == str(gemma_executable.resolve())


def test_build_command(gemma_interface):
    prompt = "Hello"
    command = gemma_interface._build_command(prompt)
    assert command == [
        gemma_interface.gemma_executable,
        "--weights",
        gemma_interface.model_path,
        "--max_generated_tokens",
        str(gemma_interface.max_tokens),
        "--temperature",
        str(gemma_interface.temperature),
        "--prompt",
        prompt,
    ]


def test_build_command_invalid_prompt(gemma_interface):
    with pytest.raises(ValueError):
        gemma_interface._build_command("a" * 50_001)

    with pytest.raises(ValueError):
        gemma_interface._build_command("\x00")


@pytest.mark.asyncio
async def test_generate_response(gemma_interface):
    with patch("asyncio.create_subprocess_exec") as mock_create_subprocess_exec:
        mock_process = AsyncMock()
        mock_process.stdout.read.side_effect = [b"Hello", b" World", b""]
        mock_process.wait.return_value = 0
        mock_create_subprocess_exec.return_value = mock_process

        response = await gemma_interface.generate_response("Hi")
        assert response == "Hello World"


@pytest.mark.asyncio
async def test_generate_response_process_error(gemma_interface):
    with patch("asyncio.create_subprocess_exec") as mock_create_subprocess_exec:
        mock_process = AsyncMock()
        mock_process.stdout.read.side_effect = [b"Error", b""]
        mock_process.wait.return_value = 1
        mock_process.stderr.read.return_value = b"Error message"
        mock_create_subprocess_exec.return_value = mock_process

        with pytest.raises(RuntimeError):
            await gemma_interface.generate_response("Hi")
