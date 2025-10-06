#!/usr/bin/env python3
"""
Build script for consolidated MCP Gemma server.
Handles both C++ and Python components.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run a command with proper error handling."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    if cwd:
        print(f"Working directory: {cwd}")

    result = subprocess.run(
        cmd,
        cwd=cwd,
        shell=isinstance(cmd, str),
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")

    return result


def build_cpp_server(build_dir="build", clean=False):
    """Build the C++ MCP server."""
    cpp_dir = Path(__file__).parent / "cpp-server"
    build_path = cpp_dir / build_dir

    print(f"Building C++ MCP server in {cpp_dir}")

    # Clean build directory if requested
    if clean and build_path.exists():
        import shutil
        print(f"Cleaning build directory: {build_path}")
        shutil.rmtree(build_path)

    # Create build directory
    build_path.mkdir(exist_ok=True)

    # Configure with CMake
    configure_cmd = [
        "cmake",
        "..",
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    # Platform-specific configuration
    if sys.platform == "win32":
        configure_cmd.extend(["-G", "Visual Studio 17 2022", "-A", "x64"])

    run_command(configure_cmd, cwd=build_path)

    # Build
    build_cmd = ["cmake", "--build", ".", "--config", "Release"]
    if sys.platform != "win32":
        # Use parallel build on Unix-like systems
        import multiprocessing
        build_cmd.extend(["-j", str(multiprocessing.cpu_count())])

    run_command(build_cmd, cwd=build_path)

    print(f"C++ MCP server built successfully in {build_path}")
    return build_path


def setup_python_environment():
    """Setup Python environment with required dependencies."""
    print("Setting up Python environment...")

    # Check if we're in a virtual environment or have uv
    if sys.prefix == sys.base_prefix:
        print("Warning: Not in a virtual environment. Consider using 'uv venv' or 'python -m venv'")

    # Install Python dependencies
    requirements = [
        "mcp>=1.0.0",
        "asyncio",
        "redis>=4.0.0",
        "aiohttp>=3.8.0",
        "websockets>=10.0",
        "pydantic>=2.0.0",
    ]

    # Try uv first, then pip
    try:
        cmd = ["uv", "pip", "install"] + requirements
        run_command(cmd)
        print("Python dependencies installed with uv")
    except (FileNotFoundError, RuntimeError):
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + requirements
            run_command(cmd)
            print("Python dependencies installed with pip")
        except RuntimeError as e:
            print(f"Failed to install Python dependencies: {e}")
            print("Please install manually:")
            print(f"  pip install {' '.join(requirements)}")
            return False

    return True


def test_installation():
    """Test the consolidated MCP server installation."""
    print("Testing installation...")

    cpp_dir = Path(__file__).parent / "cpp-server"
    server_dir = Path(__file__).parent / "server"

    # Test C++ build
    cpp_build_dir = cpp_dir / "build"
    if sys.platform == "win32":
        cpp_executable = cpp_build_dir / "Release" / "gemma_mcp_server.exe"
        stdio_executable = cpp_build_dir / "Release" / "gemma_mcp_stdio_server.exe"
    else:
        cpp_executable = cpp_build_dir / "gemma_mcp_server"
        stdio_executable = cpp_build_dir / "gemma_mcp_stdio_server"

    if cpp_executable.exists():
        print(f"✅ C++ MCP server built: {cpp_executable}")
    else:
        print(f"❌ C++ MCP server not found: {cpp_executable}")

    if stdio_executable.exists():
        print(f"✅ C++ stdio server built: {stdio_executable}")
    else:
        print(f"❌ C++ stdio server not found: {stdio_executable}")

    # Test Python server
    python_server = server_dir / "consolidated_server.py"
    if python_server.exists():
        print(f"✅ Python consolidated server: {python_server}")
        # Test import
        try:
            result = run_command([
                sys.executable, "-c",
                "import sys; sys.path.insert(0, '.'); "
                "from server.consolidated_server import ConsolidatedMCPServer; "
                "print('Python server imports successfully')"
            ], cwd=Path(__file__).parent)
            print("✅ Python server imports successfully")
        except RuntimeError:
            print("❌ Python server import failed")
    else:
        print(f"❌ Python server not found: {python_server}")

    # Test chat handler
    chat_handler = server_dir / "chat_handler.py"
    if chat_handler.exists():
        print(f"✅ Chat handler: {chat_handler}")
    else:
        print(f"❌ Chat handler not found: {chat_handler}")


def main():
    parser = argparse.ArgumentParser(description="Build consolidated MCP Gemma server")
    parser.add_argument("--cpp", action="store_true", help="Build C++ server")
    parser.add_argument("--python", action="store_true", help="Setup Python environment")
    parser.add_argument("--all", action="store_true", help="Build everything")
    parser.add_argument("--clean", action="store_true", help="Clean build directories")
    parser.add_argument("--test", action="store_true", help="Test installation")
    parser.add_argument("--build-dir", default="build", help="Build directory name")

    args = parser.parse_args()

    # Default to building everything if no specific options
    if not any([args.cpp, args.python, args.test]):
        args.all = True

    try:
        if args.all or args.python:
            if not setup_python_environment():
                sys.exit(1)

        if args.all or args.cpp:
            build_cpp_server(args.build_dir, args.clean)

        if args.test:
            test_installation()

        print("\n🎉 Build completed successfully!")
        print("\nUsage:")
        print("  # Python server (recommended)")
        print(f"  python {Path(__file__).parent}/server/consolidated_server.py --model /path/to/model.sbs")
        print("\n  # C++ server (high performance)")
        print(f"  {Path(__file__).parent}/cpp-server/build/gemma_mcp_server --model /path/to/model.sbs")

    except Exception as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()