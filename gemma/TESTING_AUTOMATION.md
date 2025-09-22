# Gemma.cpp Testing Automation Framework

This document describes the comprehensive testing automation framework for Gemma.cpp hardware backends, providing tools for compilation testing, functional validation, and performance benchmarking.

## Overview

The testing framework consists of four main components:

1. **Compilation Test Script** (`compile_test.py`) - Tests compilation across all backend configurations
2. **Backend Validation Script** (`validate_backends.py`) - Validates dependencies and basic functionality
3. **Functional Integration Tests** (`tests/functional/`) - End-to-end backend integration testing
4. **Comprehensive Test Runner** (`run_comprehensive_tests.py`) - Orchestrates the complete test suite

## Quick Start

### Basic Validation
```bash
# Validate all backend dependencies
python validate_backends.py

# Test compilation for all available backends
python compile_test.py

# Run comprehensive test suite
python run_comprehensive_tests.py
```

### Advanced Usage
```bash
# Comprehensive testing with custom configuration
python run_comprehensive_tests.py \
    --project-root /path/to/gemma \
    --build-dir ./build-test \
    --output-dir ./test-results \
    --parallel 4 \
    --verbose

# Test specific backends only
python compile_test.py --backends CUDA SYCL --compilers gcc clang

# Validate dependencies with detailed output
python validate_backends.py --verbose --output-dir ./validation-reports
```

## Component Details

### 1. Compilation Test Script (`compile_test.py`)

Tests compilation of all backend configurations with different compilers.

**Features:**
- Automatic toolchain detection (GCC, Clang, MSVC, Intel compilers)
- Hardware backend detection (CUDA, SYCL, Vulkan, OpenCL, Metal)
- Parallel compilation testing
- Detailed compilation reports with timing and error analysis
- Cross-platform support

**Usage:**
```bash
python compile_test.py [options]

Options:
  --project-root PATH      Path to Gemma.cpp project root
  --build-dir PATH         Build directory for tests
  --output-dir PATH        Output directory for reports
  --parallel N             Number of parallel build jobs
  --backends BACKENDS      Specific backends to test
  --compilers COMPILERS    Specific compilers to test
  --verbose               Enable verbose logging
  --clean                 Clean build directories before testing
```

**Output:**
- JSON compilation report with detailed metrics
- Text summary with pass/fail status
- Binary size analysis
- Warning and error categorization

### 2. Backend Validation Script (`validate_backends.py`)

Validates backend dependencies and compiles minimal test programs.

**Features:**
- SDK and toolchain dependency checking
- Minimal compilation tests for each backend
- Runtime environment validation
- Hardware capability detection
- Performance baseline establishment

**Usage:**
```bash
python validate_backends.py [options]

Options:
  --project-root PATH      Path to Gemma.cpp project root
  --output-dir PATH        Output directory for reports
  --temp-dir PATH          Temporary directory for compilation tests
  --skip-compilation       Skip compilation tests
  --skip-runtime          Skip runtime tests
  --verbose               Enable verbose logging
```

**Validation Checks:**
- **Core Dependencies**: CMake, C++ compilers, Git
- **CUDA**: NVCC, NVIDIA drivers, CUDA runtime
- **SYCL**: Intel oneAPI, SYCL runtime, device detection
- **Vulkan**: Vulkan SDK, vulkaninfo, driver support
- **OpenCL**: OpenCL headers, runtime libraries, device enumeration
- **Metal**: Metal framework (macOS only), device capabilities

### 3. Functional Integration Tests (`tests/functional/`)

End-to-end tests for backend integration and inference comparison.

**Test Files:**
- `test_backend_integration.cpp` - Backend registration, fallback, memory operations
- `test_inference_backends.cpp` - Inference result consistency, performance comparison

**Features:**
- Mock backend implementations for testing
- Backend registration and discovery tests
- Fallback mechanism validation
- Memory transfer operations testing
- Matrix operations verification
- Activation function testing
- Performance comparison between backends
- Numerical accuracy validation
- Error handling and recovery testing
- Concurrent operation testing

**CMake Integration:**
```bash
# Build and run functional tests
cmake --build . --target test_functional
ctest -L functional

# Run specific backend tests
ctest -L backend
ctest -L cuda
ctest -L sycl
```

### 4. Comprehensive Test Runner (`run_comprehensive_tests.py`)

Orchestrates the complete test suite with dependency resolution and parallel execution.

**Features:**
- Dependency-aware test execution
- Parallel test suite execution
- Comprehensive reporting
- Integration with existing CMake/CTest infrastructure
- Automatic failure analysis

**Test Suites:**
1. **Dependency Validation** - Validate backend dependencies
2. **Compilation Tests** - Test compilation across configurations
3. **CMake Configure** - Configure project with CMake
4. **CMake Build** - Build project
5. **Unit Tests** - Run unit tests
6. **Integration Tests** - Run integration tests
7. **Functional Tests** - Run functional backend tests
8. **Backend Tests** - Run backend-specific tests
9. **Performance Tests** - Run performance benchmarks
10. **Memory Tests** - Run memory tests (if valgrind available)

**Usage:**
```bash
python run_comprehensive_tests.py [options]

Options:
  --project-root PATH      Path to Gemma.cpp project root
  --build-dir PATH         Build directory for tests
  --output-dir PATH        Output directory for reports
  --parallel N             Maximum parallel test suites
  --clean                  Clean build directory before testing
  --verbose                Enable verbose logging
  --suite SUITE            Run only specific test suite(s)
```

## CMake Integration

The test framework integrates with the existing CMake build system:

### Test Targets
```bash
# Build all test executables
make test_functional

# Run specific test categories
make test_unit          # Unit tests
make test_integration   # Integration tests
make test_functional    # Functional tests
make test_backends      # Backend-specific tests
make test_performance   # Performance tests
make test_all          # All tests

# Run backend-specific tests
make run_backend_specific
```

### CTest Labels
Tests are organized with CTest labels for easy filtering:

```bash
# Run tests by category
ctest -L unit
ctest -L integration
ctest -L functional
ctest -L performance

# Run tests by backend
ctest -L backend
ctest -L cuda
ctest -L sycl
ctest -L vulkan
ctest -L opencl
ctest -L metal
ctest -L cpu

# Run tests by type
ctest -L comparison
ctest -L inference
```

## Report Generation

All test scripts generate comprehensive reports:

### JSON Reports
- Detailed test results with metrics
- Performance benchmarks
- Error analysis
- Environment information

### Text Summaries
- Human-readable test summaries
- Pass/fail status for each component
- Performance comparisons
- Failure diagnostics

### Report Locations
```
test-results/
├── validation/
│   ├── validation_report_*.json
│   └── validation_summary_*.txt
├── compilation/
│   ├── compilation_report_*.json
│   └── compilation_summary_*.txt
├── comprehensive_test_report_*.json
├── test_summary_*.txt
└── detailed/
    ├── dependency_validation/
    ├── compilation_tests/
    ├── functional_tests/
    └── ...
```

## Platform-Specific Notes

### Windows
- Supports MSVC, MinGW-w64, and Clang compilers
- CUDA backend requires NVIDIA CUDA Toolkit
- Vulkan backend requires Vulkan SDK
- Intel oneAPI for SYCL backend

### Linux
- Supports GCC, Clang, and Intel compilers
- Package managers for dependency installation
- Better support for OpenCL vendors
- Comprehensive hardware detection

### macOS
- Supports Clang and GCC (via Homebrew)
- Native Metal backend support
- Vulkan via MoltenVK
- Intel oneAPI available

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   python validate_backends.py --verbose
   ```
   Check the validation report for missing SDKs or libraries.

2. **Compilation Failures**
   ```bash
   python compile_test.py --verbose --backends CUDA
   ```
   Check compiler-specific error messages and ensure proper SDK installation.

3. **Test Failures**
   ```bash
   python run_comprehensive_tests.py --verbose
   ```
   Check detailed test logs in the output directory.

### Environment Setup

Ensure proper environment variables are set:

```bash
# CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# Intel oneAPI
source /opt/intel/oneapi/setvars.sh

# Vulkan
export VULKAN_SDK=/path/to/vulkan/sdk
export PATH=$VULKAN_SDK/bin:$PATH
```

## Integration with CI/CD

The test framework is designed for CI/CD integration:

### GitHub Actions Example
```yaml
name: Comprehensive Backend Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]

    steps:
    - uses: actions/checkout@v3

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip

    - name: Validate backends
      run: python validate_backends.py --output-dir ./reports

    - name: Run comprehensive tests
      run: python run_comprehensive_tests.py --output-dir ./test-results

    - name: Upload test reports
      uses: actions/upload-artifact@v3
      with:
        name: test-reports-${{ matrix.os }}
        path: |
          ./reports/
          ./test-results/
```

## Performance Benchmarking

The framework includes performance benchmarking capabilities:

### Metrics Collected
- Matrix multiplication GFLOPS
- Memory bandwidth (GB/s)
- Latency measurements
- Binary size analysis
- Compilation time tracking

### Baseline Establishment
```bash
# Run performance tests and establish baselines
python run_comprehensive_tests.py --suite performance_tests
```

### Cross-Backend Comparison
The functional tests include cross-backend result comparison to ensure numerical consistency while measuring performance differences.

## Contributing

When adding new backends or test cases:

1. Update `validate_backends.py` with new dependency checks
2. Add backend-specific test programs
3. Update CMake configuration in `tests/functional/CMakeLists.txt`
4. Add appropriate CTest labels
5. Update this documentation

## Conclusion

This testing automation framework provides comprehensive validation of the Gemma.cpp hardware backends, ensuring reliable cross-platform functionality and performance. The modular design allows for easy extension and maintenance as new backends are added.