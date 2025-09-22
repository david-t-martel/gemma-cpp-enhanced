#!/usr/bin/env python3
"""
Simple syntax test for the test automation scripts.
This verifies that the scripts have valid Python syntax.
"""

import ast
import sys
from pathlib import Path


def test_script_syntax(script_path):
    """Test if a Python script has valid syntax."""
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Parse the AST to check syntax
        ast.parse(source)
        return True, "OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    """Test all automation scripts."""
    project_root = Path(__file__).parent

    scripts = [
        "compile_test.py",
        "validate_backends.py",
        "run_comprehensive_tests.py"
    ]

    print("Testing script syntax...")

    all_passed = True
    for script in scripts:
        script_path = project_root / script
        if script_path.exists():
            passed, message = test_script_syntax(script_path)
            status = "PASS" if passed else "FAIL"
            print(f"  {script}: {status} - {message}")
            if not passed:
                all_passed = False
        else:
            print(f"  {script}: MISSING")
            all_passed = False

    if all_passed:
        print("\nAll scripts have valid syntax!")
        return 0
    else:
        print("\nSome scripts have syntax errors!")
        return 1


if __name__ == "__main__":
    sys.exit(main())