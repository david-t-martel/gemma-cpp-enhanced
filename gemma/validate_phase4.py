#!/usr/bin/env python3
"""Validation script for Phase 4 Python files."""

import ast
import sys
from pathlib import Path

# Test files
FILES_TO_TEST = [
    "src/gemma_cli/config/models.py",
    "src/gemma_cli/config/prompts.py",
    "src/gemma_cli/commands/model.py",
    "src/gemma_cli/cli.py",
]

REQUIRED_DEPS = [
    "psutil",
    "tomllib",  # Python 3.11+
    "tomli_w",
    "pyyaml",
    "pydantic",
    "rich",
    "click",
]

def validate_syntax(filepath: Path) -> tuple[bool, str]:
    """Validate Python syntax using AST."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code, filename=str(filepath))
        return True, "✅ PASS"
    except SyntaxError as e:
        return False, f"❌ FAIL - Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"⚠️ ERROR - {e}"

def check_imports(filepath: Path) -> tuple[bool, list[str]]:
    """Extract import statements from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(filepath))

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])

        return True, list(set(imports))
    except Exception as e:
        return False, [f"Error: {e}"]

def check_dependency_availability():
    """Check if required dependencies are installed."""
    missing = []
    available = []

    for dep in REQUIRED_DEPS:
        # Handle tomllib (built-in Python 3.11+)
        if dep == "tomllib":
            try:
                import tomllib
                available.append(dep)
            except ImportError:
                try:
                    import tomli as tomllib  # Fallback for <3.11
                    available.append(f"{dep} (via tomli)")
                except ImportError:
                    missing.append(dep)
            continue

        # Standard dependency check
        try:
            __import__(dep)
            available.append(dep)
        except ImportError:
            missing.append(dep)

    return available, missing

def main():
    """Run validation."""
    print("=" * 70)
    print("Phase 4 Python Files Validation Report")
    print("=" * 70)
    print()

    # 1. Syntax Validation
    print("### 1. Syntax Validation")
    print()
    all_syntax_valid = True

    for filepath_str in FILES_TO_TEST:
        filepath = Path(filepath_str)
        if not filepath.exists():
            print(f"{filepath.name:30s} ⚠️ FILE NOT FOUND")
            all_syntax_valid = False
            continue

        valid, message = validate_syntax(filepath)
        print(f"{filepath.name:30s} {message}")
        all_syntax_valid = all_syntax_valid and valid

    print()

    # 2. Import Analysis
    print("### 2. Import Analysis")
    print()

    for filepath_str in FILES_TO_TEST:
        filepath = Path(filepath_str)
        if not filepath.exists():
            continue

        success, imports = check_imports(filepath)
        if success:
            print(f"{filepath.name}:")
            print(f"  Imports: {', '.join(sorted(imports))}")
        else:
            print(f"{filepath.name}: {imports[0]}")
        print()

    # 3. Dependency Check
    print("### 3. Dependency Check")
    print()

    available, missing = check_dependency_availability()

    if available:
        print("✅ Available dependencies:")
        for dep in sorted(available):
            print(f"   - {dep}")
        print()

    if missing:
        print("📦 Missing dependencies:")
        for dep in sorted(missing):
            print(f"   - {dep}")
        print()
        print("🔧 Install with:")
        print(f"   pip install {' '.join(missing)}")
        print()
    else:
        print("✅ All required dependencies are available")
        print()

    # Summary
    print("=" * 70)
    print("### Summary")
    print()

    if all_syntax_valid:
        print("✅ All files passed syntax validation")
    else:
        print("❌ Some files failed syntax validation")

    if not missing:
        print("✅ All dependencies available")
    else:
        print(f"⚠️ {len(missing)} dependencies missing")

    print("=" * 70)

    return 0 if (all_syntax_valid and not missing) else 1

if __name__ == "__main__":
    sys.exit(main())
