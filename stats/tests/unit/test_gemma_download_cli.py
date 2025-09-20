import json
from pathlib import Path
import subprocess
import sys

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "src" / "gcp" / "gemma_download.py"


def run_cmd(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args], capture_output=True, text=True, check=False
    )


def parse_json(stdout: str):
    stdout = stdout.strip()
    assert stdout, "No stdout produced"
    return json.loads(stdout.splitlines()[-1])


def test_auto_dry_run_json():
    # Ensure dry-run does not attempt network download
    proc = run_cmd(["--auto", "--dry-run", "--json"])
    assert proc.returncode == 0, proc.stderr
    data = parse_json(proc.stdout)
    assert data.get("mode") == "auto-selection"
    assert "selected_model" in data
    # VRAM may be None on CPU-only systems; device should still be present
    assert "device" in data


def test_show_deps_json():
    proc = run_cmd(["--show-deps", "--json"])
    assert proc.returncode == 0, proc.stderr
    data = parse_json(proc.stdout)
    assert data.get("mode") == "dependencies"
    assert "dependencies" in data
    assert isinstance(data["dependencies"], dict)


def test_list_cached_empty_json(tmp_path: Path):
    # Use isolated cache dir to ensure empty state
    proc = run_cmd(["--list-cached", "--cache-dir", str(tmp_path), "--json"])
    assert proc.returncode == 0, proc.stderr
    data = parse_json(proc.stdout)
    assert data.get("mode") == "list-cached"
    assert "models" in data
    assert isinstance(data["models"], dict)
