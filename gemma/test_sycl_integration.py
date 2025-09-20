#!/usr/bin/env python3
"""
Intel SYCL Backend Integration Test Script

This script tests the complete Intel SYCL backend integration for Gemma.cpp,
including compilation, device detection, and basic functionality.
"""

import os
import sys
import subprocess
import json
import platform
from pathlib import Path

def run_command(cmd, capture_output=True, check=True, cwd=None):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    try:
        if isinstance(cmd, str):
            cmd = cmd.split()
        
        result = subprocess.run(
            cmd, 
            capture_output=capture_output, 
            text=True, 
            check=check,
            cwd=cwd
        )
        
        if capture_output:
            print(f"Output: {result.stdout}")
            if result.stderr:
                print(f"Error: {result.stderr}")
        
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        if capture_output:
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
        return None

def check_intel_oneapi():
    """Check if Intel oneAPI is installed and accessible."""
    print("\n=== Checking Intel oneAPI Installation ===")
    
    # Check for Intel oneAPI installation
    if platform.system() == "Windows":
        oneapi_paths = [
            "C:/Program Files (x86)/Intel/oneAPI",
            "C:/Program Files/Intel/oneAPI"
        ]
        
        for path in oneapi_paths:
            if os.path.exists(path):
                print(f"✅ Intel oneAPI found at: {path}")
                
                # Check for SYCL compiler
                icpx_path = Path(path) / "compiler/latest/windows/bin/icpx.exe"
                if icpx_path.exists():
                    print(f"✅ Intel SYCL compiler found: {icpx_path}")
                    return str(icpx_path)
                else:
                    print(f"❌ Intel SYCL compiler not found at: {icpx_path}")
        
        print("❌ Intel oneAPI not found in standard locations")
        return None
    else:
        # Linux/macOS
        result = run_command("which icpx", check=False)
        if result and result.returncode == 0:
            print(f"✅ Intel SYCL compiler found: {result.stdout.strip()}")
            return result.stdout.strip()
        else:
            print("❌ Intel SYCL compiler not found in PATH")
            return None

def check_sycl_devices():
    """Check for available SYCL devices."""
    print("\n=== Checking SYCL Devices ===")
    
    # Try to run sycl-ls if available
    result = run_command("sycl-ls", check=False)
    if result and result.returncode == 0:
        print("Available SYCL devices:")
        print(result.stdout)
        return True
    else:
        print("sycl-ls not available, will check devices via our test")
        return False

def test_cmake_configuration():
    """Test CMake configuration for SYCL backend."""
    print("\n=== Testing CMake Configuration ===")
    
    gemma_root = Path(__file__).parent
    build_dir = gemma_root / "build-sycl-test"
    
    # Clean previous build
    if build_dir.exists():
        import shutil
        shutil.rmtree(build_dir)
    
    build_dir.mkdir(exist_ok=True)
    
    # Configure with SYCL backend enabled
    cmake_cmd = [
        "cmake",
        "-S", str(gemma_root),
        "-B", str(build_dir),
        "-DGEMMA_BUILD_SYCL_BACKEND=ON",
        "-DGEMMA_BUILD_ENHANCED_TESTS=ON",
        "-DCMAKE_BUILD_TYPE=Release"
    ]
    
    if platform.system() == "Windows":
        cmake_cmd.extend([
            "-G", "Visual Studio 17 2022",
            "-T", "v143"
        ])
    
    result = run_command(cmake_cmd, cwd=gemma_root)
    if not result:
        print("❌ CMake configuration failed")
        return False
    
    print("✅ CMake configuration successful")
    return True

def test_compilation():
    """Test compilation of SYCL backend."""
    print("\n=== Testing Compilation ===")
    
    gemma_root = Path(__file__).parent
    build_dir = gemma_root / "build-sycl-test"
    
    if not build_dir.exists():
        print("❌ Build directory not found, run configuration first")
        return False
    
    # Build the SYCL backend
    cmake_build_cmd = [
        "cmake",
        "--build", str(build_dir),
        "--config", "Release",
        "--target", "gemma_sycl_backend",
        "-j", "4"
    ]
    
    result = run_command(cmake_build_cmd, cwd=gemma_root)
    if not result:
        print("❌ SYCL backend compilation failed")
        return False
    
    print("✅ SYCL backend compilation successful")
    
    # Build the test if available
    cmake_test_cmd = [
        "cmake",
        "--build", str(build_dir),
        "--config", "Release",
        "--target", "test_sycl_backend",
        "-j", "4"
    ]
    
    result = run_command(cmake_test_cmd, cwd=gemma_root, check=False)
    if result and result.returncode == 0:
        print("✅ SYCL backend test compilation successful")
        return True
    else:
        print("⚠️  SYCL backend test compilation failed, but main backend built")
        return True  # Main backend built, so consider it a success

def test_backend_functionality():
    """Test SYCL backend functionality."""
    print("\n=== Testing Backend Functionality ===")
    
    gemma_root = Path(__file__).parent
    build_dir = gemma_root / "build-sycl-test"
    
    # Find the test executable
    if platform.system() == "Windows":
        test_exe = build_dir / "backends/sycl/Release/test_sycl_backend.exe"
        if not test_exe.exists():
            test_exe = build_dir / "backends/sycl/test_sycl_backend.exe"
    else:
        test_exe = build_dir / "backends/sycl/test_sycl_backend"
    
    if not test_exe.exists():
        print(f"❌ Test executable not found at: {test_exe}")
        print("Available files in SYCL build directory:")
        sycl_build_dir = build_dir / "backends/sycl"
        if sycl_build_dir.exists():
            for item in sycl_build_dir.rglob("*"):
                if item.is_file():
                    print(f"  {item}")
        return False
    
    # Set up environment for Intel GPU
    env = os.environ.copy()
    env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:*;opencl:*"
    
    # Run the test
    result = run_command([str(test_exe)], check=False, cwd=gemma_root)
    
    if result and result.returncode == 0:
        print("✅ SYCL backend functionality test passed")
        return True
    else:
        print("❌ SYCL backend functionality test failed")
        return False

def generate_report():
    """Generate a summary report."""
    print("\n" + "="*50)
    print("Intel SYCL Backend Integration Report")
    print("="*50)
    
    checks = [
        ("Intel oneAPI Installation", check_intel_oneapi() is not None),
        ("SYCL Device Detection", check_sycl_devices()),
        ("CMake Configuration", test_cmake_configuration()),
        ("SYCL Backend Compilation", test_compilation()),
        ("Backend Functionality", test_backend_functionality())
    ]
    
    passed = 0
    total = len(checks)
    
    for name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<30} {status}")
        if result:
            passed += 1
    
    print(f"\nSummary: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 Intel SYCL backend integration is fully functional!")
        return True
    elif passed >= 3:
        print("\n⚠️  Intel SYCL backend partially functional, some issues detected.")
        return False
    else:
        print("\n❌ Intel SYCL backend integration has significant issues.")
        return False

def main():
    """Main test function."""
    print("Intel SYCL Backend Integration Test")
    print("===================================")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    
    success = generate_report()
    
    if success:
        print("\nNext steps:")
        print("1. The SYCL backend is ready for use with Gemma.cpp")
        print("2. You can enable it in your build with -DGEMMA_BUILD_SYCL_BACKEND=ON")
        print("3. Run your Gemma models with Intel GPU acceleration")
        sys.exit(0)
    else:
        print("\nTroubleshooting:")
        print("1. Ensure Intel oneAPI toolkit 2024.1+ is installed")
        print("2. Source the Intel oneAPI environment: setvars.bat (Windows) or source setvars.sh (Linux)")
        print("3. Check that Intel GPU drivers are up to date")
        print("4. Verify SYCL device availability with 'sycl-ls'")
        sys.exit(1)

if __name__ == "__main__":
    main()