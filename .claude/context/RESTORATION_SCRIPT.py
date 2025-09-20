#!/usr/bin/env python3
"""
LLM Ecosystem Context Restoration Script
Quickly restore project context and verify system state
"""

import subprocess
import sys
from pathlib import Path
import json
import os

# Project root
PROJECT_ROOT = Path("C:/codedev/llm")
CONTEXT_DIR = PROJECT_ROOT / ".claude" / "context"

def run_command(cmd, cwd=None):
    """Execute command and return success status"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd or PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_directories():
    """Verify critical directories exist"""
    dirs = [
        PROJECT_ROOT / "rag-redis",
        PROJECT_ROOT / "stats",
        PROJECT_ROOT / "gemma",
        CONTEXT_DIR
    ]

    print("📁 Checking directories...")
    for dir_path in dirs:
        if dir_path.exists():
            print(f"  ✅ {dir_path.relative_to(PROJECT_ROOT)}")
        else:
            print(f"  ❌ {dir_path.relative_to(PROJECT_ROOT)} - MISSING")
    print()

def check_python_env():
    """Verify Python environment and imports"""
    print("🐍 Checking Python environment...")

    # Check uv availability
    success, _, _ = run_command("uv --version")
    if success:
        print("  ✅ uv is installed")
    else:
        print("  ❌ uv not found - install with: pip install uv")

    # Check critical imports
    imports = [
        "import stats.tools",
        "import pydantic",
        "import redis",
    ]

    for import_stmt in imports:
        success, _, _ = run_command(f'uv run python -c "{import_stmt}"')
        module = import_stmt.split()[1]
        if success:
            print(f"  ✅ {module} importable")
        else:
            print(f"  ❌ {module} import failed")
    print()

def check_rust_env():
    """Verify Rust environment and build"""
    print("🦀 Checking Rust environment...")

    # Check cargo
    success, _, _ = run_command("cargo --version")
    if success:
        print("  ✅ cargo is installed")
    else:
        print("  ❌ cargo not found")
        return

    # Check rag-redis workspace
    rag_dir = PROJECT_ROOT / "rag-redis"
    if rag_dir.exists():
        success, stdout, _ = run_command("cargo check", cwd=rag_dir)
        if success:
            print("  ✅ rag-redis workspace valid")
        else:
            print("  ⚠️  rag-redis build check failed (expected - 90% complete)")
    else:
        print("  ❌ rag-redis directory missing")
    print()

def check_redis():
    """Check Redis availability"""
    print("🔴 Checking Redis...")

    success, _, _ = run_command("redis-cli -p 6380 ping")
    if success:
        print("  ✅ Redis responding on port 6380")
    else:
        print("  ⚠️  Redis not running on port 6380")
        print("     Start with: redis-server --port 6380")
    print()

def check_git_status():
    """Show Git repository status"""
    print("📊 Git Status...")

    success, stdout, _ = run_command("git status --short")
    if success:
        if stdout.strip():
            print("  Modified files:")
            for line in stdout.strip().split('\n'):
                print(f"    {line}")
        else:
            print("  ✅ Working directory clean")
    else:
        print("  ❌ Not a git repository")
    print()

def load_context_summary():
    """Load and display context summary"""
    print("📄 Context Summary...")

    context_file = CONTEXT_DIR / "PROJECT_CONTEXT.md"
    if context_file.exists():
        print("  ✅ Full context available at:")
        print(f"     {context_file}")

        # Extract key sections
        with open(context_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find current state section
        if "### Work in Progress" in content:
            start = content.index("### Work in Progress")
            end = content.index("### Known Issues", start)
            wip = content[start:end].strip()
            print("\n  Current Work in Progress:")
            for line in wip.split('\n')[1:]:  # Skip header
                if line.strip().startswith('🔄'):
                    print(f"    {line.strip()}")
    else:
        print("  ❌ Context file not found")
    print()

def show_quick_commands():
    """Display quick command reference"""
    print("🚀 Quick Commands:")
    print("  Build all:     uv pip install -r requirements.txt && cd rag-redis && cargo build")
    print("  Test Python:   uv run pytest stats/tests/ -v")
    print("  Test Rust:     cd rag-redis && cargo test")
    print("  Start Redis:   redis-server --port 6380")
    print("  Check memory:  wmic process get Name,WorkingSetSize")
    print()

def main():
    """Main restoration check"""
    print("=" * 60)
    print("LLM Ecosystem - Context Restoration Check")
    print("=" * 60)
    print()

    # Change to project root
    os.chdir(PROJECT_ROOT)

    # Run all checks
    check_directories()
    check_python_env()
    check_rust_env()
    check_redis()
    check_git_status()
    load_context_summary()
    show_quick_commands()

    print("=" * 60)
    print("✅ Context restoration check complete!")
    print("📚 Full context: .claude/context/PROJECT_CONTEXT.md")
    print("⚡ Quick ref:    .claude/context/QUICK_REFERENCE.md")
    print("=" * 60)

if __name__ == "__main__":
    main()