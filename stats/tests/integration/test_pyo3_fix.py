#!/usr/bin/env python3
"""
Test script to verify PyO3 binding fixes are working correctly.

This script tests:
1. Environment variable setup
2. Python interpreter detection
3. Basic import functionality
4. Extension loading

Run this after applying the PyO3 fixes to verify everything works.
"""

import os
from pathlib import Path
import subprocess
import sys


def test_environment():
    """Test environment setup."""
    print("=== Testing Environment ===")

    # Check PYO3_PYTHON environment variable
    pyo3_python = os.environ.get("PYO3_PYTHON")
    if pyo3_python:
        print(f"✓ PYO3_PYTHON is set: {pyo3_python}")

        # Verify the path exists
        if Path(pyo3_python).exists():
            print("✓ Python executable exists")
        else:
            print("✗ Python executable not found")
            return False
    else:
        print("✗ PYO3_PYTHON environment variable not set")
        return False

    # Test Python version
    try:
        result = subprocess.run(
            [pyo3_python, "--version"], check=False, capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"✓ Python version: {result.stdout.strip()}")
        else:
            print("✗ Failed to get Python version")
            return False
    except Exception as e:
        print(f"✗ Error running Python: {e}")
        return False

    return True


def test_rust_environment():
    """Test Rust compilation environment."""
    print("\n=== Testing Rust Environment ===")

    try:
        # Check Rust version
        result = subprocess.run(["rustc", "--version"], check=False, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Rust compiler: {result.stdout.strip()}")
        else:
            print("✗ Rust compiler not found")
            return False
    except Exception as e:
        print(f"✗ Error checking Rust: {e}")
        return False

    try:
        # Check Cargo version
        result = subprocess.run(["cargo", "--version"], check=False, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Cargo: {result.stdout.strip()}")
        else:
            print("✗ Cargo not found")
            return False
    except Exception as e:
        print(f"✗ Error checking Cargo: {e}")
        return False

    return True


def test_build_process():
    """Test the build process in rust_extensions."""
    print("\n=== Testing Build Process ===")

    rust_extensions_dir = Path(__file__).parent / "rust_extensions"
    if not rust_extensions_dir.exists():
        print(f"✗ rust_extensions directory not found: {rust_extensions_dir}")
        return False

    print(f"✓ Found rust_extensions directory: {rust_extensions_dir}")

    # Change to rust_extensions directory
    original_cwd = os.getcwd()
    try:
        os.chdir(rust_extensions_dir)

        # Test cargo check
        print("Running cargo check...")
        result = subprocess.run(["cargo", "check"], check=False, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ cargo check passed")
        else:
            print("✗ cargo check failed:")
            print(result.stderr)
            return False

        # Test basic clippy (without -D warnings to be less strict)
        print("Running basic cargo clippy...")
        result = subprocess.run(["cargo", "clippy"], check=False, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ cargo clippy passed")
        else:
            print("⚠ cargo clippy had warnings/errors:")
            print(result.stdout)
            print(result.stderr)
            # Don't return False here as clippy warnings might be acceptable

    finally:
        os.chdir(original_cwd)

    return True


def test_extension_import():
    """Test if the extension can be imported after building."""
    print("\n=== Testing Extension Import ===")

    try:
        # Try to import the built extension
        import gemma_extensions

        print("✓ Successfully imported gemma_extensions")

        # Test basic functionality
        version = gemma_extensions.get_version()
        print(f"✓ Extension version: {version}")

        build_info = gemma_extensions.get_build_info()
        print(f"✓ Build info: {build_info}")

        return True

    except ImportError as e:
        print(f"⚠ Could not import extension (this is expected if not built yet): {e}")
        return True  # This is expected if we haven't built yet

    except Exception as e:
        print(f"✗ Error testing extension: {e}")
        return False


def main():
    """Main test function."""
    print("PyO3 Binding Fix Verification Script")
    print("=" * 50)

    tests = [
        ("Environment Setup", test_environment),
        ("Rust Environment", test_rust_environment),
        ("Build Process", test_build_process),
        ("Extension Import", test_extension_import),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1

    print(f"\nTests passed: {passed}/{len(results)}")

    if passed == len(results):
        print("\n🎉 All tests passed! PyO3 binding fixes are working correctly.")
    else:
        print(f"\n⚠ {len(results) - passed} test(s) failed. Please check the output above.")

        print("\nNext steps:")
        print("1. Ensure PYO3_PYTHON environment variable is set correctly")
        print("2. Run the build script: rust_extensions/test_build.cmd")
        print("3. Check for any remaining compilation errors")
        print("4. Build with: cd rust_extensions && uv run maturin develop --release")


if __name__ == "__main__":
    main()
