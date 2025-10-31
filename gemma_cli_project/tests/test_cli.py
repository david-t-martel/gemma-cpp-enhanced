from typer.testing import CliRunner
from gemma_cli.main import app

runner = CliRunner()

def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Gemma CLI" in result.stdout

def test_status():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Gemma CLI" in result.stdout
