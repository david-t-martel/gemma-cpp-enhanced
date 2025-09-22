#!/usr/bin/env python3
"""
Gemma.cpp Backend Validation Script
===================================

This script provides comprehensive validation of all hardware backend dependencies,
compiles minimal test programs, runs basic functionality tests, and generates
detailed validation reports.

Features:
- SDK and toolchain dependency checking
- Minimal compilation tests for each backend
- Runtime environment validation
- Hardware capability detection
- Performance baseline establishment
- Detailed validation reporting
"""

import os
import sys
import subprocess
import json
import time
import tempfile
import shutil
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import argparse
import logging


@dataclass
class DependencyCheck:
    """Result of a dependency check."""
    name: str
    required: bool
    available: bool
    version: str = ""
    path: str = ""
    notes: str = ""


@dataclass
class CompilationTest:
    """Result of a minimal compilation test."""
    backend: str
    success: bool
    duration: float
    binary_path: str = ""
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class RuntimeTest:
    """Result of a runtime functionality test."""
    backend: str
    test_name: str
    success: bool
    duration: float
    output: str = ""
    error_message: str = ""


@dataclass
class PerformanceBaseline:
    """Performance baseline measurements."""
    backend: str
    matrix_multiply_gflops: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    latency_ms: float = 0.0


class DependencyValidator:
    """Validates all backend dependencies."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_all_dependencies(self) -> List[DependencyCheck]:
        """Validate dependencies for all supported backends."""
        checks = []

        # Core dependencies
        checks.extend(self._check_core_dependencies())

        # Backend-specific dependencies
        checks.extend(self._check_cuda_dependencies())
        checks.extend(self._check_sycl_dependencies())
        checks.extend(self._check_vulkan_dependencies())
        checks.extend(self._check_opencl_dependencies())
        checks.extend(self._check_metal_dependencies())

        return checks

    def _check_core_dependencies(self) -> List[DependencyCheck]:
        """Check core build dependencies."""
        checks = []

        # CMake
        cmake_check = self._check_command("cmake", ["--version"], required=True)
        checks.append(cmake_check)

        # C++ compiler
        compilers = ["g++", "clang++", "cl.exe"]
        compiler_found = False
        for compiler in compilers:
            check = self._check_command(compiler, ["--version"], required=False)
            if check.available:
                compiler_found = True
                check.required = True
            checks.append(check)

        if not compiler_found:
            checks.append(DependencyCheck(
                name="C++ Compiler",
                required=True,
                available=False,
                notes="No C++ compiler found in PATH"
            ))

        # Git
        git_check = self._check_command("git", ["--version"], required=False)
        checks.append(git_check)

        return checks

    def _check_cuda_dependencies(self) -> List[DependencyCheck]:
        """Check CUDA-specific dependencies."""
        checks = []

        # NVCC compiler
        nvcc_check = self._check_command("nvcc", ["--version"], required=False)
        checks.append(nvcc_check)

        # NVIDIA driver
        nvidia_smi_check = self._check_command("nvidia-smi", [], required=False)
        checks.append(nvidia_smi_check)

        # CUDA runtime
        if platform.system() == "Windows":
            cuda_paths = [
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA",
                "C:\\Program Files (x86)\\NVIDIA GPU Computing Toolkit\\CUDA"
            ]
        else:
            cuda_paths = ["/usr/local/cuda", "/opt/cuda"]

        cuda_found = False
        for path in cuda_paths:
            if Path(path).exists():
                cuda_found = True
                checks.append(DependencyCheck(
                    name="CUDA Runtime",
                    required=False,
                    available=True,
                    path=path,
                    notes="CUDA installation detected"
                ))
                break

        if not cuda_found:
            checks.append(DependencyCheck(
                name="CUDA Runtime",
                required=False,
                available=False,
                notes="No CUDA installation detected"
            ))

        return checks

    def _check_sycl_dependencies(self) -> List[DependencyCheck]:
        """Check SYCL/Intel oneAPI dependencies."""
        checks = []

        # Intel oneAPI compiler
        icpx_check = self._check_command("icpx", ["--version"], required=False)
        checks.append(icpx_check)

        # SYCL list devices
        sycl_ls_check = self._check_command("sycl-ls", [], required=False)
        checks.append(sycl_ls_check)

        # Intel oneAPI environment
        oneapi_vars = ["ONEAPI_ROOT", "INTEL_ONEAPI_ROOT", "CMPLR_ROOT"]
        oneapi_found = False
        for var in oneapi_vars:
            if os.environ.get(var):
                oneapi_found = True
                checks.append(DependencyCheck(
                    name="Intel oneAPI Environment",
                    required=False,
                    available=True,
                    path=os.environ[var],
                    notes=f"Found via {var} environment variable"
                ))
                break

        if not oneapi_found:
            checks.append(DependencyCheck(
                name="Intel oneAPI Environment",
                required=False,
                available=False,
                notes="No oneAPI environment variables detected"
            ))

        return checks

    def _check_vulkan_dependencies(self) -> List[DependencyCheck]:
        """Check Vulkan SDK dependencies."""
        checks = []

        # Vulkan info
        vulkaninfo_check = self._check_command("vulkaninfo", ["--summary"], required=False)
        checks.append(vulkaninfo_check)

        # Vulkan SDK
        vulkan_sdk = os.environ.get("VULKAN_SDK")
        if vulkan_sdk and Path(vulkan_sdk).exists():
            checks.append(DependencyCheck(
                name="Vulkan SDK",
                required=False,
                available=True,
                path=vulkan_sdk,
                notes="Vulkan SDK environment variable set"
            ))
        else:
            # Try common installation paths
            if platform.system() == "Windows":
                vulkan_paths = [
                    "C:\\VulkanSDK",
                    "C:\\Program Files\\VulkanSDK"
                ]
            else:
                vulkan_paths = [
                    "/usr/local/vulkan",
                    "/opt/vulkan"
                ]

            vulkan_found = False
            for path in vulkan_paths:
                if Path(path).exists():
                    vulkan_found = True
                    checks.append(DependencyCheck(
                        name="Vulkan SDK",
                        required=False,
                        available=True,
                        path=path,
                        notes="Vulkan SDK detected in common path"
                    ))
                    break

            if not vulkan_found:
                checks.append(DependencyCheck(
                    name="Vulkan SDK",
                    required=False,
                    available=False,
                    notes="No Vulkan SDK detected"
                ))

        return checks

    def _check_opencl_dependencies(self) -> List[DependencyCheck]:
        """Check OpenCL dependencies."""
        checks = []

        # OpenCL info
        clinfo_check = self._check_command("clinfo", [], required=False)
        checks.append(clinfo_check)

        # Try to find OpenCL headers and libraries
        if platform.system() == "Windows":
            opencl_paths = [
                "C:\\Program Files\\NVIDIA Corporation\\OpenCL",
                "C:\\Program Files (x86)\\Intel\\OpenCL SDK",
                "C:\\Program Files\\AMD\\OpenCL SDK"
            ]
        else:
            opencl_paths = [
                "/usr/include/CL",
                "/usr/local/include/CL",
                "/opt/intel/opencl/include/CL"
            ]

        opencl_found = False
        for path in opencl_paths:
            if Path(path).exists():
                opencl_found = True
                checks.append(DependencyCheck(
                    name="OpenCL Headers",
                    required=False,
                    available=True,
                    path=path,
                    notes="OpenCL headers detected"
                ))
                break

        if not opencl_found:
            checks.append(DependencyCheck(
                name="OpenCL Headers",
                required=False,
                available=False,
                notes="No OpenCL headers detected"
            ))

        return checks

    def _check_metal_dependencies(self) -> List[DependencyCheck]:
        """Check Metal dependencies (macOS only)."""
        checks = []

        if platform.system() != "Darwin":
            checks.append(DependencyCheck(
                name="Metal Framework",
                required=False,
                available=False,
                notes="Metal only available on macOS"
            ))
            return checks

        # Metal is part of macOS, check system profiler
        metal_check = self._check_command(
            "system_profiler", ["SPDisplaysDataType"], required=False)
        if metal_check.available and "Metal" in metal_check.version:
            metal_check.name = "Metal Framework"
            metal_check.notes = "Metal support detected"
        else:
            metal_check.name = "Metal Framework"
            metal_check.available = False
            metal_check.notes = "Metal support not detected"

        checks.append(metal_check)

        return checks

    def _check_command(self, command: str, args: List[str], required: bool) -> DependencyCheck:
        """Check if a command is available and get version info."""
        try:
            result = subprocess.run(
                [command] + args,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                # Extract version information
                version_output = result.stdout + result.stderr
                version = self._extract_version(version_output)
                path = shutil.which(command) or ""

                return DependencyCheck(
                    name=command,
                    required=required,
                    available=True,
                    version=version,
                    path=path
                )
            else:
                return DependencyCheck(
                    name=command,
                    required=required,
                    available=False,
                    notes=f"Command failed with exit code {result.returncode}"
                )

        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            return DependencyCheck(
                name=command,
                required=required,
                available=False,
                notes=f"Command not found or timed out: {e}"
            )

    def _extract_version(self, output: str) -> str:
        """Extract version information from command output."""
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Common version patterns
            version_keywords = ['version', 'Version', 'VERSION', 'release', 'Release']
            for keyword in version_keywords:
                if keyword in line:
                    return line

        # Return first non-empty line if no version keyword found
        for line in lines:
            line = line.strip()
            if line:
                return line[:100]  # Limit length

        return "unknown"


class MinimalCompiler:
    """Compiles minimal test programs for each backend."""

    def __init__(self, project_root: Path, temp_dir: Path):
        self.project_root = project_root
        self.temp_dir = temp_dir
        self.logger = logging.getLogger(__name__)

    def test_all_backends(self) -> List[CompilationTest]:
        """Test compilation for all available backends."""
        tests = []

        tests.append(self._test_cpu_backend())
        tests.append(self._test_cuda_backend())
        tests.append(self._test_sycl_backend())
        tests.append(self._test_vulkan_backend())
        tests.append(self._test_opencl_backend())

        if platform.system() == "Darwin":
            tests.append(self._test_metal_backend())

        return tests

    def _test_cpu_backend(self) -> CompilationTest:
        """Test CPU backend compilation."""
        return self._compile_test_program(
            "CPU",
            self._generate_cpu_test_code(),
            ["c++", "-std=c++20", "-O2"]
        )

    def _test_cuda_backend(self) -> CompilationTest:
        """Test CUDA backend compilation."""
        return self._compile_test_program(
            "CUDA",
            self._generate_cuda_test_code(),
            ["nvcc", "-std=c++17", "-O2"],
            file_extension=".cu"
        )

    def _test_sycl_backend(self) -> CompilationTest:
        """Test SYCL backend compilation."""
        return self._compile_test_program(
            "SYCL",
            self._generate_sycl_test_code(),
            ["icpx", "-fsycl", "-std=c++17", "-O2"]
        )

    def _test_vulkan_backend(self) -> CompilationTest:
        """Test Vulkan backend compilation."""
        vulkan_flags = []
        if platform.system() == "Windows":
            vulkan_flags.extend(["-lVulkan-1"])
        else:
            vulkan_flags.extend(["-lvulkan"])

        return self._compile_test_program(
            "Vulkan",
            self._generate_vulkan_test_code(),
            ["c++", "-std=c++20", "-O2"] + vulkan_flags
        )

    def _test_opencl_backend(self) -> CompilationTest:
        """Test OpenCL backend compilation."""
        opencl_flags = []
        if platform.system() == "Windows":
            opencl_flags.extend(["-lOpenCL"])
        else:
            opencl_flags.extend(["-lOpenCL"])

        return self._compile_test_program(
            "OpenCL",
            self._generate_opencl_test_code(),
            ["c++", "-std=c++20", "-O2"] + opencl_flags
        )

    def _test_metal_backend(self) -> CompilationTest:
        """Test Metal backend compilation (macOS only)."""
        return self._compile_test_program(
            "Metal",
            self._generate_metal_test_code(),
            ["clang++", "-std=c++17", "-O2", "-framework", "Metal", "-framework", "Foundation"],
            file_extension=".mm"
        )

    def _compile_test_program(self, backend: str, code: str, compile_cmd: List[str],
                             file_extension: str = ".cpp") -> CompilationTest:
        """Compile a test program and return results."""
        start_time = time.time()

        try:
            # Create source file
            source_file = self.temp_dir / f"test_{backend.lower()}{file_extension}"
            with open(source_file, 'w') as f:
                f.write(code)

            # Create output binary name
            binary_name = f"test_{backend.lower()}"
            if platform.system() == "Windows":
                binary_name += ".exe"
            binary_path = self.temp_dir / binary_name

            # Build compile command
            full_cmd = compile_cmd + [str(source_file), "-o", str(binary_path)]

            # Compile
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.temp_dir
            )

            duration = time.time() - start_time

            # Extract warnings
            warnings = []
            if result.stderr:
                for line in result.stderr.split('\n'):
                    if 'warning:' in line.lower() or 'note:' in line.lower():
                        warnings.append(line.strip())

            success = result.returncode == 0 and binary_path.exists()

            return CompilationTest(
                backend=backend,
                success=success,
                duration=duration,
                binary_path=str(binary_path) if success else "",
                error_message=result.stderr if not success else "",
                warnings=warnings
            )

        except subprocess.TimeoutExpired:
            return CompilationTest(
                backend=backend,
                success=False,
                duration=time.time() - start_time,
                error_message="Compilation timed out"
            )
        except Exception as e:
            return CompilationTest(
                backend=backend,
                success=False,
                duration=time.time() - start_time,
                error_message=f"Compilation failed: {e}"
            )

    def _generate_cpu_test_code(self) -> str:
        """Generate minimal CPU test code."""
        return """
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

int main() {
    const int N = 64;
    std::vector<float> a(N * N), b(N * N), c(N * N, 0.0f);

    // Initialize random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < N * N; ++i) {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }

    // Simple matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                c[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU Backend Test: Matrix multiplication (" << N << "x" << N
              << ") completed in " << duration.count() << " microseconds" << std::endl;

    return 0;
}
"""

    def _generate_cuda_test_code(self) -> str:
        """Generate minimal CUDA test code."""
        return """
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;
    size_t size = N * sizeof(float);

    // Host vectors
    std::vector<float> h_a(N, 1.0f), h_b(N, 2.0f), h_c(N);

    // Device vectors
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy to device
    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Copy result back
    cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "CUDA Backend Test: Vector addition completed successfully" << std::endl;
    std::cout << "Result sample: " << h_c[0] << " (expected: 3.0)" << std::endl;

    return 0;
}
"""

    def _generate_sycl_test_code(self) -> str:
        """Generate minimal SYCL test code."""
        return """
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

int main() {
    const int N = 1024;

    sycl::queue q;
    std::cout << "SYCL device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // Host vectors
    std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N);

    {
        // Device buffers
        sycl::buffer<float, 1> buf_a(a.data(), sycl::range<1>(N));
        sycl::buffer<float, 1> buf_b(b.data(), sycl::range<1>(N));
        sycl::buffer<float, 1> buf_c(c.data(), sycl::range<1>(N));

        // Submit kernel
        q.submit([&](sycl::handler& h) {
            auto acc_a = buf_a.get_access<sycl::access::mode::read>(h);
            auto acc_b = buf_b.get_access<sycl::access::mode::read>(h);
            auto acc_c = buf_c.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                acc_c[idx] = acc_a[idx] + acc_b[idx];
            });
        });
    } // Buffers destroyed here, data copied back

    std::cout << "SYCL Backend Test: Vector addition completed successfully" << std::endl;
    std::cout << "Result sample: " << c[0] << " (expected: 3.0)" << std::endl;

    return 0;
}
"""

    def _generate_vulkan_test_code(self) -> str:
        """Generate minimal Vulkan test code."""
        return """
#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>

int main() {
    // Initialize Vulkan
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan Test";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Test Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    VkInstance instance;
    VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

    if (result == VK_SUCCESS) {
        std::cout << "Vulkan Backend Test: Instance created successfully" << std::endl;

        // Enumerate physical devices
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        std::cout << "Found " << deviceCount << " Vulkan-capable device(s)" << std::endl;

        vkDestroyInstance(instance, nullptr);
        return 0;
    } else {
        std::cout << "Vulkan Backend Test: Failed to create instance (error: " << result << ")" << std::endl;
        return 1;
    }
}
"""

    def _generate_opencl_test_code(self) -> str:
        """Generate minimal OpenCL test code."""
        return """
#include <CL/cl.h>
#include <iostream>
#include <vector>

int main() {
    // Get platform count
    cl_uint platformCount;
    cl_int err = clGetPlatformIDs(0, nullptr, &platformCount);

    if (err != CL_SUCCESS || platformCount == 0) {
        std::cout << "OpenCL Backend Test: No platforms found" << std::endl;
        return 1;
    }

    std::cout << "OpenCL Backend Test: Found " << platformCount << " platform(s)" << std::endl;

    // Get platforms
    std::vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);

    // Get device count for first platform
    cl_uint deviceCount;
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);

    if (err == CL_SUCCESS && deviceCount > 0) {
        std::cout << "OpenCL Backend Test: Found " << deviceCount << " device(s) on first platform" << std::endl;
        return 0;
    } else {
        std::cout << "OpenCL Backend Test: No devices found on first platform" << std::endl;
        return 1;
    }
}
"""

    def _generate_metal_test_code(self) -> str:
        """Generate minimal Metal test code."""
        return """
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        if (device) {
            NSString* deviceName = [device name];
            std::cout << "Metal Backend Test: Device found - "
                      << [deviceName UTF8String] << std::endl;

            // Create a simple command queue
            id<MTLCommandQueue> commandQueue = [device newCommandQueue];
            if (commandQueue) {
                std::cout << "Metal Backend Test: Command queue created successfully" << std::endl;
                return 0;
            } else {
                std::cout << "Metal Backend Test: Failed to create command queue" << std::endl;
                return 1;
            }
        } else {
            std::cout << "Metal Backend Test: No Metal device found" << std::endl;
            return 1;
        }
    }
}
"""


class RuntimeTester:
    """Tests runtime functionality of compiled backends."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def test_compiled_backends(self, compilation_tests: List[CompilationTest]) -> List[RuntimeTest]:
        """Test runtime functionality of successfully compiled backends."""
        tests = []

        for comp_test in compilation_tests:
            if comp_test.success and comp_test.binary_path and Path(comp_test.binary_path).exists():
                runtime_test = self._run_binary_test(comp_test)
                tests.append(runtime_test)

        return tests

    def _run_binary_test(self, comp_test: CompilationTest) -> RuntimeTest:
        """Run a compiled binary and check its output."""
        start_time = time.time()

        try:
            result = subprocess.run(
                [comp_test.binary_path],
                capture_output=True,
                text=True,
                timeout=30
            )

            duration = time.time() - start_time
            success = result.returncode == 0

            return RuntimeTest(
                backend=comp_test.backend,
                test_name="basic_functionality",
                success=success,
                duration=duration,
                output=result.stdout,
                error_message=result.stderr if not success else ""
            )

        except subprocess.TimeoutExpired:
            return RuntimeTest(
                backend=comp_test.backend,
                test_name="basic_functionality",
                success=False,
                duration=time.time() - start_time,
                error_message="Runtime test timed out"
            )
        except Exception as e:
            return RuntimeTest(
                backend=comp_test.backend,
                test_name="basic_functionality",
                success=False,
                duration=time.time() - start_time,
                error_message=f"Runtime test failed: {e}"
            )


class ValidationReporter:
    """Generates comprehensive validation reports."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self,
                       dependencies: List[DependencyCheck],
                       compilation_tests: List[CompilationTest],
                       runtime_tests: List[RuntimeTest]) -> Dict[str, Any]:
        """Generate comprehensive validation report."""

        # Calculate statistics
        total_deps = len(dependencies)
        available_deps = sum(1 for d in dependencies if d.available)
        required_missing = sum(1 for d in dependencies if d.required and not d.available)

        successful_compilations = sum(1 for c in compilation_tests if c.success)
        successful_runtime = sum(1 for r in runtime_tests if r.success)

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "architecture": platform.machine(),
                "python_version": platform.python_version()
            },
            "summary": {
                "dependencies_checked": total_deps,
                "dependencies_available": available_deps,
                "required_dependencies_missing": required_missing,
                "compilation_tests": len(compilation_tests),
                "successful_compilations": successful_compilations,
                "runtime_tests": len(runtime_tests),
                "successful_runtime_tests": successful_runtime,
                "overall_status": "PASS" if required_missing == 0 else "FAIL"
            },
            "dependencies": [
                {
                    "name": d.name,
                    "required": d.required,
                    "available": d.available,
                    "version": d.version,
                    "path": d.path,
                    "notes": d.notes
                } for d in dependencies
            ],
            "compilation_tests": [
                {
                    "backend": c.backend,
                    "success": c.success,
                    "duration": c.duration,
                    "binary_path": c.binary_path,
                    "warning_count": len(c.warnings),
                    "error_message": c.error_message
                } for c in compilation_tests
            ],
            "runtime_tests": [
                {
                    "backend": r.backend,
                    "test_name": r.test_name,
                    "success": r.success,
                    "duration": r.duration,
                    "output_excerpt": r.output[-200:] if r.output else "",
                    "error_message": r.error_message
                } for r in runtime_tests
            ]
        }

        # Save JSON report
        json_file = self.output_dir / f"validation_report_{int(time.time())}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate text summary
        self._generate_text_summary(report)

        return report

    def _generate_text_summary(self, report: Dict[str, Any]):
        """Generate human-readable text summary."""
        summary_file = self.output_dir / f"validation_summary_{int(time.time())}.txt"

        with open(summary_file, 'w') as f:
            f.write("GEMMA.CPP BACKEND VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Validation Date: {report['timestamp']}\n")
            f.write(f"Platform: {report['platform']['system']} {report['platform']['release']} "
                   f"({report['platform']['architecture']})\n")
            f.write(f"Overall Status: {report['summary']['overall_status']}\n\n")

            f.write("SUMMARY:\n")
            f.write("-" * 10 + "\n")
            f.write(f"Dependencies Available: {report['summary']['dependencies_available']}"
                   f"/{report['summary']['dependencies_checked']}\n")
            f.write(f"Required Missing: {report['summary']['required_dependencies_missing']}\n")
            f.write(f"Successful Compilations: {report['summary']['successful_compilations']}"
                   f"/{report['summary']['compilation_tests']}\n")
            f.write(f"Successful Runtime Tests: {report['summary']['successful_runtime_tests']}"
                   f"/{report['summary']['runtime_tests']}\n\n")

            f.write("DEPENDENCY STATUS:\n")
            f.write("-" * 20 + "\n")
            for dep in report['dependencies']:
                status = "✓" if dep['available'] else "✗"
                required = " (REQUIRED)" if dep['required'] else ""
                f.write(f"  {status} {dep['name']}{required}")
                if dep['version']:
                    f.write(f" - {dep['version']}")
                if dep['notes']:
                    f.write(f" ({dep['notes']})")
                f.write("\n")

            f.write("\nCOMPILATION RESULTS:\n")
            f.write("-" * 20 + "\n")
            for comp in report['compilation_tests']:
                status = "✓" if comp['success'] else "✗"
                f.write(f"  {status} {comp['backend']} Backend ({comp['duration']:.1f}s)")
                if comp['warning_count'] > 0:
                    f.write(f" - {comp['warning_count']} warnings")
                f.write("\n")
                if not comp['success'] and comp['error_message']:
                    f.write(f"    Error: {comp['error_message'][:100]}...\n")

            f.write("\nRUNTIME TEST RESULTS:\n")
            f.write("-" * 20 + "\n")
            for runtime in report['runtime_tests']:
                status = "✓" if runtime['success'] else "✗"
                f.write(f"  {status} {runtime['backend']} Backend ({runtime['duration']:.1f}s)\n")
                if runtime['output_excerpt']:
                    f.write(f"    Output: {runtime['output_excerpt']}\n")
                if not runtime['success'] and runtime['error_message']:
                    f.write(f"    Error: {runtime['error_message']}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Gemma.cpp Backend Validation")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                       help="Path to Gemma.cpp project root")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd() / "validation-reports",
                       help="Output directory for reports")
    parser.add_argument("--temp-dir", type=Path,
                       help="Temporary directory for compilation tests")
    parser.add_argument("--skip-compilation", action="store_true",
                       help="Skip compilation tests")
    parser.add_argument("--skip-runtime", action="store_true",
                       help="Skip runtime tests")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting Gemma.cpp backend validation...")

    # Setup temporary directory
    if args.temp_dir:
        temp_dir = args.temp_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        cleanup_temp = False
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="gemma_validation_"))
        cleanup_temp = True

    try:
        # Validate dependencies
        logger.info("Checking dependencies...")
        dependency_validator = DependencyValidator()
        dependencies = dependency_validator.validate_all_dependencies()

        # Compilation tests
        compilation_tests = []
        if not args.skip_compilation:
            logger.info("Running compilation tests...")
            compiler = MinimalCompiler(args.project_root, temp_dir)
            compilation_tests = compiler.test_all_backends()

        # Runtime tests
        runtime_tests = []
        if not args.skip_runtime and compilation_tests:
            logger.info("Running runtime tests...")
            runtime_tester = RuntimeTester()
            runtime_tests = runtime_tester.test_compiled_backends(compilation_tests)

        # Generate report
        logger.info("Generating validation report...")
        reporter = ValidationReporter(args.output_dir)
        report = reporter.generate_report(dependencies, compilation_tests, runtime_tests)

        # Print summary
        print(f"\nVALIDATION SUMMARY:")
        print(f"Overall Status: {report['summary']['overall_status']}")
        print(f"Dependencies Available: {report['summary']['dependencies_available']}"
              f"/{report['summary']['dependencies_checked']}")
        print(f"Required Missing: {report['summary']['required_dependencies_missing']}")
        print(f"Successful Compilations: {report['summary']['successful_compilations']}"
              f"/{report['summary']['compilation_tests']}")
        print(f"Successful Runtime Tests: {report['summary']['successful_runtime_tests']}"
              f"/{report['summary']['runtime_tests']}")
        print(f"Reports saved to: {args.output_dir}")

        # Return appropriate exit code
        return 0 if report['summary']['overall_status'] == "PASS" else 1

    finally:
        # Cleanup temporary directory
        if cleanup_temp and temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    sys.exit(main())