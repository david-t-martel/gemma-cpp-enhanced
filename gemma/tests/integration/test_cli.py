
import sys
import sys
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

# Mock rich before importing cli
sys.modules["rich"] = MagicMock()
sys.modules["rich.console"] = MagicMock()
sys.modules["rich.live"] = MagicMock()
sys.modules["rich.panel"] = MagicMock()
sys.modules["rich.prompt"] = MagicMock()
sys.modules["rich.text"] = MagicMock()
sys.modules["rich.progress"] = MagicMock()

from gemma_cli.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_chat_command(runner, monkeypatch):
    monkeypatch.setattr("gemma_cli.core.gemma.GemmaInterface", MagicMock())
    monkeypatch.setattr("gemma_cli.core.conversation.ConversationManager", MagicMock())

    result = runner.invoke(cli, ["chat"], input="Hello\n/quit\n")
    assert result.exit_code == 0
    assert "Welcome" in result.output


def test_ask_command(runner, monkeypatch):
    monkeypatch.setattr("gemma_cli.core.gemma.GemmaInterface", MagicMock())

    result = runner.invoke(cli, ["ask", "What is the capital of France?"])
    assert result.exit_code == 0


    def test_ingest_command(runner, monkeypatch, tmp_path):


        monkeypatch.setattr("gemma_cli.rag.hybrid_rag.HybridRAGManager", MagicMock())


        doc = tmp_path / "doc.txt"


        doc.write_text("This is a test document.")


    


        result = runner.invoke(cli, ["ingest", str(doc)])


        assert result.exit_code == 0


    


    


    def test_memory_command(runner, monkeypatch):


        monkeypatch.setattr("gemma_cli.rag.hybrid_rag.HybridRAGManager", MagicMock())


    


        result = runner.invoke(cli, ["memory"])


        assert result.exit_code == 0
