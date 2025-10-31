
import argparse
import json
import subprocess
import sys
from pathlib import Path

def apply_ruff_fixes(file_path: Path) -> dict:
    """
    Runs ruff check with --fix on the specified file and returns the output.
    """
    if not file_path.is_file():
        return {"success": False, "message": f"Error: File not found at {file_path}"}

    try:
        # Run ruff with --fix and JSON output format
        result = subprocess.run(
            ["ruff", "check", "--fix", "--output-format=json", str(file_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return {"success": True, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        return {"success": False, "message": f"Error running ruff: {e}", "stdout": e.stdout, "stderr": e.stderr}
    except FileNotFoundError:
        return {"success": False, "message": "Error: 'ruff' command not found. Please ensure ruff is installed and in your PATH."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ruff check with --fix on a specified Python file."
    )
    parser.add_argument(
        "paths",
        nargs='+',
        type=Path,
        help="Paths to Python files or directories to format with ruff."
    )
    args = parser.parse_args()

    python_files = []
    for path in args.paths:
        if path.is_file() and path.suffix == ".py":
            python_files.append(path)
        elif path.is_dir():
            for py_file in path.rglob("*.py"):
                python_files.append(py_file)

    if not python_files:
        print("No Python files found to format.")
        sys.exit(0)

    for file_path in python_files:
        print(f"Processing {file_path}...")
        result = apply_ruff_fixes(file_path)
        if result["success"]:
            print(f"Ruff --fix applied to {file_path}")
            if result["stdout"]:
                print("Ruff stdout:")
                print(result["stdout"])
        else:
            print(f"Error applying ruff --fix to {file_path}: {result["message"]}")
            if "stdout" in result:
                print("Ruff stdout:")
                print(result["stdout"])
            if "stderr" in result:
                print("Ruff stderr:")
                print(result["stderr"])
            sys.exit(1)
