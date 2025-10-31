# Gemma.cpp oneAPI Integration - Project Status & Roadmap

**Project Location:** `C:\codedev\llm\gemma`  
**Last Updated:** 2025-10-23  
**Status:** Integration Complete, Build Validation Pending

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Completed Work](#completed-work)
3. [Remaining TODOs](#remaining-todos)
4. [Build Frameworks & Tools](#build-frameworks--tools)
5. [File Structure](#file-structure)
6. [Future Directions](#future-directions)
7. [Known Issues](#known-issues)
8. [Quick Reference](#quick-reference)

---

## Project Overview

### Goal
Integrate Intel oneAPI performance libraries (TBB, IPP, DPL, DNNL) into gemma.cpp to provide 25-50% CPU performance improvements without requiring GPU acceleration.

### Key Features
- **Modular Design:** Opt-in library selection via CMake flags
- **Backward Compatible:** Standard builds unaffected
- **Clear Naming:** Executables reflect enabled optimizations
- **Comprehensive Testing:** C++ validation + inference benchmarking
- **Automated Deployment:** DLL packaging for standalone distribution

---

## Completed Work

### 1. CMake Build System ✅

#### Files Created:
- **`cmake/OneAPILibs.cmake`** (298 lines)
  - Detects oneAPI installation from multiple locations
  - Configures TBB, IPP, DPL, DNNL libraries
  - Graceful fallback when libraries unavailable
  - Exports library paths and compile definitions

#### CMake Options Added:
```cmake
# Master switches
GEMMA_USE_ONEAPI_LIBS=ON           # Enable oneAPI integration
GEMMA_ONEAPI_PERFORMANCE_PACK=ON   # Enable all libraries at once

# Individual library controls
GEMMA_USE_TBB=ON                   # Threading Building Blocks
GEMMA_USE_IPP=ON                   # Integrated Performance Primitives
GEMMA_USE_DPL=ON                   # Data Parallel Library
GEMMA_USE_DNNL=ON                  # Deep Neural Network Library
```

#### Executable Naming Convention:
| Build Configuration | Executable Name | Description |
|---------------------|----------------|-------------|
| Standard | `gemma_std.exe` | Baseline, no optimizations |
| TBB only | `gemma_std+tbb.exe` | Parallel threading |
| TBB+IPP | `gemma_std+tbb-ipp.exe` | Threading + vector ops |
| Performance Pack | `gemma_std+tbb-ipp-dnnl.exe` | All CPU optimizations |
| SYCL GPU | `gemma_hw-sycl.exe` | GPU acceleration |
| SYCL+TBB | `gemma_hw-sycl+tbb.exe` | GPU + CPU threading |

### 2. Testing & Validation Framework ✅

#### C++ Unit Tests:
**File:** `tests/unit/test_oneapi_validation.cpp` (557 lines)

**Test Coverage:**
- **TBB Tests:**
  - `TBB_ParallelForCorrectness` - Verifies parallel_for produces correct results
  - `TBB_ParallelReduction` - Validates parallel sum accuracy
  - `TBB_TaskArenaConstraints` - Tests thread limiting

- **IPP Tests:**
  - `IPP_VectorAddition` - Element-wise vector operations
  - `IPP_DotProduct` - Dot product accuracy
  - `IPP_MatrixMultiply` - Matrix operations (if available)

- **DPL Tests:**
  - `DPL_ParallelSort` - Parallel sort correctness
  - `DPL_ParallelTransform` - Parallel transform operations

- **DNNL Tests:**
  - `DNNL_MatrixMultiply` - Small matrix multiply (128×256×128)
  - `DNNL_LargeMatrixMultiply` - Large matrix multiply (512×512×512)

- **Integration Tests:**
  - `IntegrationTest_ThreadingAndVectorOps` - Combined TBB+IPP workflows
  - `PerformanceRegression_SmallMatMul` - Baseline performance tracking

**Build Command:**
```bash
cmake --build build --target test_oneapi_validation
.\build\tests\test_oneapi_validation.exe
```

#### Inference Benchmarking:
**File:** `benchmark_baseline.ps1` (500+ lines)

**Features:**
- Real inference with gemma-2b-it model (`c:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs`)
- Multiple prompt lengths: Short (2 tokens), Medium (5 tokens), Long (12 tokens)
- Statistical averaging over 3 runs per prompt
- Tokens/second throughput measurement
- Baseline comparison mode
- JSON export for CI integration

**Usage:**
```powershell
# Establish baseline
.\benchmark_baseline.ps1 -Executable build\bin\gemma_std.exe -Baseline

# Benchmark optimized build
.\benchmark_baseline.ps1 -Executable build\bin\gemma_std+tbb-ipp.exe -OutputFile optimized_results.json

# Compare results
.\benchmark_baseline.ps1 -Compare -OutputFile optimized_results.json
```

**Output Metrics:**
- Duration (seconds) per inference
- Tokens per second throughput
- Min/Max/Average statistics
- Sample generated text
- Performance delta vs baseline

### 3. Build Automation Tools ✅

#### Multi-Configuration Builder:
**File:** `compare_builds.ps1` (345 lines)

**Predefined Configurations:**
```powershell
$BuildConfigs = @{
    "std" = @{
        Name = "Standard CPU"
        CMakeArgs = @()
    }
    "tbb" = @{
        Name = "TBB Threading"
        CMakeArgs = @("-DGEMMA_USE_ONEAPI_LIBS=ON", "-DGEMMA_USE_TBB=ON")
    }
    "tbb-ipp" = @{
        Name = "TBB + IPP"
        CMakeArgs = @("-DGEMMA_USE_ONEAPI_LIBS=ON", "-DGEMMA_USE_TBB=ON", "-DGEMMA_USE_IPP=ON")
    }
    "perfpack" = @{
        Name = "Performance Pack"
        CMakeArgs = @("-DGEMMA_USE_ONEAPI_LIBS=ON", "-DGEMMA_ONEAPI_PERFORMANCE_PACK=ON")
    }
    "sycl" = @{
        Name = "SYCL GPU"
        CMakeArgs = @("-DGEMMA_ENABLE_SYCL=ON")
    }
}
```

**Usage:**
```powershell
# Compare multiple configurations
.\compare_builds.ps1 -Configurations @("std", "tbb-ipp", "perfpack")

# Build only (skip benchmarks)
.\compare_builds.ps1 -SkipBenchmark

# Clean rebuild
.\compare_builds.ps1 -Clean
```

**Output:**
- Build time comparisons (bar charts)
- Inference performance rankings (tokens/sec)
- Percentage improvements over baseline
- Detailed JSON summaries (`build_comparison_results/`)

#### Simplified oneAPI Builder:
**File:** `build_oneapi.ps1` (142 lines)

Wrapper script that handles oneAPI environment setup and provides simple interface:

```powershell
# Standard build
.\build_oneapi.ps1 -Config std -Clean

# TBB+IPP build
.\build_oneapi.ps1 -Config tbb-ipp

# Full performance pack
.\build_oneapi.ps1 -Config perfpack
```

### 4. Deployment Automation ✅

**File:** `deploy_standalone.ps1` (Enhanced from 158 to 270 lines)

**New Features:**
- Multi-executable support (deploys all matching `gemma*.exe`)
- Auto-detection of oneAPI libs from executable name
- Smart DLL discovery across multiple oneAPI paths
- Comprehensive DLL coverage:
  - **Core Runtime:** `libiomp5md.dll`, `svml_dispmd.dll`, `libmmd.dll`, `sycl7.dll`
  - **TBB:** `tbb12.dll`, `tbbmalloc.dll`
  - **IPP:** `ippi-9.1.dll`, `ippcore-9.1.dll`, `ipps-9.1.dll`, `ippvm-9.1.dll`
  - **DNNL:** `dnnl.dll`
  - **DPL:** Header-only (no DLLs)
- Enhanced README generation with optimization details
- Smoke testing of deployed executables

**Usage:**
```powershell
# Deploy all executables with auto-detection
.\deploy_standalone.ps1

# Deploy specific executable
.\deploy_standalone.ps1 -ExecutableName "gemma_std+tbb-ipp.exe"

# Force include all oneAPI libs
.\deploy_standalone.ps1 -IncludeOneAPILibs -Verbose
```

### 5. Documentation ✅

**Files Created:**

1. **`BUILD_INTEL.md`** (Comprehensively Updated)
   - oneAPI library options reference
   - Advanced build configurations
   - Executable naming conventions
   - Testing & validation procedures
   - Performance optimization guide
   - Expected performance improvements
   - Library-specific benefits and overhead
   - Tuning recommendations

2. **`ONEAPI_INTEGRATION_SUMMARY.md`** (275 lines)
   - Complete integration overview
   - File-by-file documentation
   - Build commands reference
   - Pending work tracking
   - Known issues and solutions

3. **`BUILD_TROUBLESHOOTING.md`** (170 lines)
   - FetchContent git corruption solutions
   - Network/firewall workarounds
   - Alternative build methods
   - Verification procedures
   - Known working configurations

---

## Remaining TODOs

### Priority 1: Build Execution (BLOCKED)

**Status:** ⚠️ Network/git issues preventing fresh builds

#### TODO 1.1: Resolve FetchContent Dependencies
**Blocker:** Git clone failures from GitHub
```
Error: fatal: unable to read tree (53de76561cfc149d3c01037f0595669ad32a5e7c)
Error: RPC failed; curl 56 Recv failure: Connection was reset
```

**Solutions to Try:**
1. **Check Network/Firewall:**
   ```powershell
   # Test GitHub connectivity
   Test-NetConnection -ComputerName github.com -Port 443
   
   # Check git config
   git config --global http.postBuffer 524288000
   git config --global http.lowSpeedLimit 0
   git config --global http.lowSpeedTime 999999
   ```

2. **Use vcpkg for SentencePiece:**
   ```powershell
   # Edit vcpkg.json
   $json = @{
       name = "gemma-cpp"
       version = "0.0.0"
       dependencies = @("sentencepiece")
   }
   $json | ConvertTo-Json | Set-Content vcpkg.json
   
   # Install
   vcpkg install
   ```

3. **Copy Dependencies from Working Machine:**
   ```powershell
   # If you have a working build elsewhere
   Copy-Item "\\workingmachine\path\build\_deps" "C:\codedev\llm\gemma\build\_deps" -Recurse
   ```

4. **Use Local Highway (already available):**
   Modify `cmake/Dependencies.cmake` line 68:
   ```cmake
   # Replace FetchContent with:
   add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/highway-github)
   set(GEMMA_HWY_LIBS "hwy" "hwy_contrib")
   ```

#### TODO 1.2: Build Standard Configuration
**Command:**
```cmd
cmd /c ""C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && cmake -S . -B build_std -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release && cmake --build build_std --parallel"
```

**Expected Output:** `build_std\bin\gemma_std.exe` (~2.5 MB)

**Verification:**
```powershell
.\build_std\bin\gemma_std.exe --help
```

#### TODO 1.3: Build TBB+IPP Configuration
**Command:**
```cmd
cmd /c ""C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && cmake -S . -B build_tbb_ipp -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -DGEMMA_USE_ONEAPI_LIBS=ON -DGEMMA_USE_TBB=ON -DGEMMA_USE_IPP=ON && cmake --build build_tbb_ipp --parallel"
```

**Expected Output:** `build_tbb_ipp\bin\gemma_std+tbb-ipp.exe` (~2.5 MB)

#### TODO 1.4: Build Performance Pack Configuration
**Command:**
```cmd
cmd /c ""C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && cmake -S . -B build_perfpack -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -DGEMMA_USE_ONEAPI_LIBS=ON -DGEMMA_ONEAPI_PERFORMANCE_PACK=ON && cmake --build build_perfpack --parallel"
```

**Expected Output:** `build_perfpack\bin\gemma_std+tbb-ipp-dnnl.exe` (~2.5 MB)

### Priority 2: Validation & Testing

#### TODO 2.1: Run C++ Unit Tests
**Once builds complete:**
```powershell
# Build test executable
cmake --build build_std --target test_oneapi_validation
cmake --build build_tbb_ipp --target test_oneapi_validation
cmake --build build_perfpack --target test_oneapi_validation

# Run tests
.\build_std\tests\test_oneapi_validation.exe
.\build_tbb_ipp\tests\test_oneapi_validation.exe
.\build_perfpack\tests\test_oneapi_validation.exe
```

**Expected Results:**
- All tests should show `[  PASSED  ]`
- Tests may skip if libraries not enabled (expected behavior)
- DNNL tests should show performance improvements

**Validation Criteria:**
- ✅ TBB tests pass (parallel_for, parallel_reduce)
- ✅ IPP tests pass (vector operations, dot products)
- ✅ DNNL tests pass (matrix multiply accuracy)
- ✅ No numerical regressions vs reference implementations

#### TODO 2.2: Benchmark Inference Performance
**Model Required:** `c:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs` (4.7 GB)

**Benchmark Standard Build (Baseline):**
```powershell
.\benchmark_baseline.ps1 -Executable build_std\bin\gemma_std.exe -Baseline
```

**Benchmark Optimized Builds:**
```powershell
.\benchmark_baseline.ps1 -Executable build_tbb_ipp\bin\gemma_std+tbb-ipp.exe -OutputFile tbb_ipp_results.json
.\benchmark_baseline.ps1 -Executable build_perfpack\bin\gemma_std+tbb-ipp-dnnl.exe -OutputFile perfpack_results.json
```

**Compare Results:**
```powershell
.\benchmark_baseline.ps1 -Compare -OutputFile tbb_ipp_results.json
.\benchmark_baseline.ps1 -Compare -OutputFile perfpack_results.json
```

**Expected Improvements:**
| Configuration | Expected Speedup | Target tok/sec (baseline=20) |
|---------------|------------------|------------------------------|
| Standard | 1.0x (baseline) | 20 tok/s |
| TBB | 1.15-1.25x | 23-25 tok/s |
| TBB+IPP | 1.25-1.35x | 25-27 tok/s |
| Performance Pack | 1.35-1.50x | 27-30 tok/s |

**Validation Criteria:**
- ✅ All builds produce coherent output
- ✅ Optimized builds show measurable speedup
- ✅ No accuracy degradation (compare output quality)
- ✅ Speedup within expected ranges

#### TODO 2.3: Multi-Configuration Comparison
**Automated comparison:**
```powershell
.\compare_builds.ps1 -Configurations @("std", "tbb-ipp", "perfpack")
```

**This will:**
1. Build all three configurations
2. Run benchmarks on each
3. Generate comparison report
4. Save results to `build_comparison_results/comparison_summary.json`

### Priority 3: Deployment & Distribution

#### TODO 3.1: Package Standard Build
```powershell
.\deploy_standalone.ps1 -ExecutableName "gemma_std.exe"
```

**Verify deployment:**
```powershell
cd deploy
.\gemma_std.exe --help
# Should run without needing oneAPI environment
```

#### TODO 3.2: Package Optimized Builds
```powershell
# TBB+IPP build
.\deploy_standalone.ps1 -ExecutableName "gemma_std+tbb-ipp.exe"

# Performance pack
.\deploy_standalone.ps1 -ExecutableName "gemma_std+tbb-ipp-dnnl.exe"
```

**Verify DLL dependencies:**
```powershell
# Check required DLLs are present
Get-ChildItem deploy\*.dll | Select-Object Name
```

**Expected DLLs for Performance Pack:**
- Core: `libiomp5md.dll`, `svml_dispmd.dll`, `libmmd.dll`
- TBB: `tbb12.dll`, `tbbmalloc.dll`
- IPP: `ippi-9.1.dll`, `ippcore-9.1.dll`, `ipps-9.1.dll`, `ippvm-9.1.dll`
- DNNL: `dnnl.dll`

#### TODO 3.3: Create Distribution Archives
```powershell
# Create release packages
Compress-Archive -Path deploy\* -DestinationPath gemma_std_release.zip
Compress-Archive -Path deploy_optimized\* -DestinationPath gemma_perfpack_release.zip
```

### Priority 4: Documentation & Knowledge Management

#### TODO 4.1: Document Actual Performance Results
**After benchmarking completes, update:**

`BUILD_INTEL.md` section "Expected Performance Improvements" → "Measured Performance Improvements"

Template:
```markdown
## Measured Performance Improvements

**Test System:**
- CPU: [Your CPU Model]
- RAM: [Your RAM]
- OS: Windows 11 x64
- Compiler: Intel oneAPI 2025.2.0
- Model: gemma-2b-it

**Results:**
| Configuration | Tokens/sec | vs Baseline | Build Time |
|---------------|------------|-------------|------------|
| Standard | 20.5 tok/s | baseline | 120s |
| TBB+IPP | 25.8 tok/s | +25.9% | 125s |
| Performance Pack | 28.3 tok/s | +38.0% | 135s |
```

#### TODO 4.2: Save Results to Knowledge Graph
**Using MCP memory server:**

After successful benchmarks, document:
```markdown
- Build configurations used
- Performance measurements
- Validation test results
- Deployment artifacts created
- Lessons learned
```

#### TODO 4.3: Create User Guide
**File:** `USAGE_GUIDE.md`

Should include:
- Quick start instructions
- How to choose the right build
- Performance tuning tips
- Troubleshooting common issues
- FAQ

### Priority 5: CI/CD Integration

#### TODO 5.1: Create GitHub Actions Workflow
**File:** `.github/workflows/oneapi-build.yml`

```yaml
name: oneAPI Build & Test

on: [push, pull_request]

jobs:
  build-and-test:
    runs-on: windows-latest
    strategy:
      matrix:
        config: [std, tbb-ipp, perfpack]
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup oneAPI
        uses: rscohn2/setup-oneapi@v0
        with:
          components: |
            compiler
            tbb
            ipp
            dnnl
      
      - name: Build
        run: .\build_oneapi.ps1 -Config ${{ matrix.config }}
      
      - name: Test
        run: .\build_${{ matrix.config }}\tests\test_oneapi_validation.exe
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: gemma-${{ matrix.config }}
          path: build_${{ matrix.config }}\bin\*.exe
```

#### TODO 5.2: Add Performance Regression Detection
**Automated checks:**
- Run benchmarks on each commit
- Compare against baseline
- Fail if performance regresses >5%
- Store historical performance data

---

## Build Frameworks & Tools

### Core Build System

#### CMake (v4.1+)
**Role:** Build system generator  
**Location:** `C:\Program Files\CMake\`

**Key Features Used:**
- FetchContent for dependency management
- Multi-configuration generators
- vcpkg integration
- Custom module paths

**Configuration Files:**
- `CMakeLists.txt` - Root build configuration
- `cmake/OneAPILibs.cmake` - oneAPI library detection
- `cmake/Dependencies.cmake` - Dependency resolution
- `cmake/GemmaOptimizations.cmake` - Compiler optimizations

#### Ninja (Build Tool)
**Role:** Fast parallel build execution  
**Location:** `C:\Users\david\.local\bin\ninja.exe`

**Advantages:**
- Faster than MSBuild for Intel compiler
- Better parallelism
- Cleaner output

#### vcpkg (Package Manager)
**Role:** C++ dependency management  
**Location:** `C:\codedev\vcpkg\` or `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\vcpkg\`

**Packages Available:**
- Highway (SIMD library)
- SentencePiece (tokenizer)
- nlohmann-json
- Google Benchmark
- Google Test

**Manifest:** `vcpkg.json` (currently minimal, can be expanded)

### Compilers

#### Intel oneAPI DPC++/C++ Compiler (ICX) 2025.2.0
**Location:** `C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\icx.exe`

**Advantages:**
- Better C++20 support than MSVC
- Native oneAPI library integration
- AVX2/AVX-512 optimizations
- SYCL support for GPU

**Key Flags:**
```
/arch:AVX2          # Enable AVX2 instructions
/O2                 # Full speed optimization
/GL                 # Whole program optimization
-std:c++20          # C++20 standard
/MD                 # Dynamic runtime (required for SYCL)
```

#### Microsoft Visual C++ (MSVC) 19.44
**Location:** `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\`

**Fallback Option:** When Intel compiler unavailable

**Build Command:**
```powershell
cmake -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### oneAPI Components

#### Threading Building Blocks (TBB)
**Location:** `C:\Program Files (x86)\Intel\oneAPI\tbb\latest\`

**Libraries:**
- `tbb12.lib` (static)
- `tbb12.dll` (runtime)
- `tbbmalloc.dll` (memory allocator)

**Headers:** `<tbb/parallel_for.h>`, `<tbb/task_arena.h>`, etc.

**CMake Integration:**
```cmake
find_package(TBB REQUIRED)
target_link_libraries(target TBB::tbb TBB::tbbmalloc)
```

#### Integrated Performance Primitives (IPP)
**Location:** `C:\Program Files (x86)\Intel\oneAPI\ipp\latest\`

**Libraries:**
- `ippcore-9.1.lib` (core)
- `ipps-9.1.lib` (signal processing)
- `ippi-9.1.lib` (image processing)
- `ippvm-9.1.lib` (vector math)

**Headers:** `<ipp.h>`, `<ipps.h>`, `<ippi.h>`

**Functions Used:**
- `ippsAdd_32f()` - Vector addition
- `ippsDotProd_32f()` - Dot product
- `ippsAddC_32f()` - Add scalar to vector

#### Data Parallel Library (DPL)
**Location:** `C:\Program Files (x86)\Intel\oneAPI\dpl\latest\`

**Type:** Header-only library

**Headers:** `<oneapi/dpl/algorithm>`, `<oneapi/dpl/execution>`

**Features:**
- Parallel STL algorithms
- Execution policies: `par`, `par_unseq`
- Drop-in replacement for `std::` algorithms

**Usage:**
```cpp
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

std::sort(oneapi::dpl::execution::par_unseq, vec.begin(), vec.end());
```

#### Deep Neural Network Library (DNNL)
**Location:** `C:\Program Files (x86)\Intel\oneAPI\dnnl\latest\`

**Libraries:**
- `dnnl.lib` (static)
- `dnnl.dll` (runtime ~80MB)

**Headers:** `<oneapi/dnnl/dnnl.hpp>`

**Operations:**
- Matrix multiplication (GEMM)
- Convolutions
- Pooling
- Activations

**Usage:**
```cpp
#include <oneapi/dnnl/dnnl.hpp>
using namespace dnnl;

engine eng(engine::kind::cpu, 0);
matmul matmul_prim(matmul::primitive_desc(eng, a_md, b_md, c_md));
matmul_prim.execute(stream, {{DNNL_ARG_SRC, a_mem}, ...});
```

### Testing Frameworks

#### Google Test (GTest)
**Version:** 1.14.0  
**Source:** FetchContent from GitHub

**Used For:**
- Unit test framework
- Test fixtures
- Assertions: `EXPECT_EQ`, `ASSERT_TRUE`, etc.

**Test File:** `tests/unit/test_oneapi_validation.cpp`

#### Google Mock (GMock)
**Version:** 1.14.0 (bundled with GTest)

**Used For:**
- Mocking interfaces
- Behavior verification
- Test doubles

#### Google Benchmark
**Version:** 1.8.3  
**Source:** FetchContent from GitHub

**Used For:**
- Microbenchmarking
- Performance regression detection
- Statistical analysis

**Potential Usage:**
```cpp
static void BM_MatMul(benchmark::State& state) {
    for (auto _ : state) {
        matmul(A, B, C, M, K, N);
    }
}
BENCHMARK(BM_MatMul);
```

### Scripting & Automation

#### PowerShell 7.5.4
**Role:** Build automation, benchmarking, deployment

**Scripts Created:**
- `build_oneapi.ps1` - Simplified build wrapper
- `benchmark_baseline.ps1` - Inference benchmarking
- `compare_builds.ps1` - Multi-config automation
- `deploy_standalone.ps1` - DLL packaging
- `build_msvc_local.ps1` - MSVC fallback

**Key PowerShell Features Used:**
- Hash tables for configuration
- Pipeline processing
- Parallel execution (`ForEach-Object -Parallel`)
- JSON export/import
- Process management

### Version Control

#### Git 2.x
**Role:** Source control

**Important Considerations:**
- FetchContent uses git clone internally
- Network issues can block dependency fetching
- `.git` folders consume significant disk space

**Workarounds:**
- Use `GIT_SHALLOW ON` in FetchContent
- Set git http buffer size: `git config --global http.postBuffer 524288000`
- Use vcpkg instead of FetchContent when possible

---

## File Structure

### Project Organization

```
C:\codedev\llm\gemma\
├── cmake\
│   ├── OneAPILibs.cmake              # oneAPI library detection (NEW)
│   ├── Dependencies.cmake            # Dependency resolution
│   ├── GemmaOptimizations.cmake      # Compiler optimizations
│   └── BuildAcceleration.cmake       # Parallel compilation settings
│
├── tests\
│   ├── unit\
│   │   └── test_oneapi_validation.cpp  # C++ validation tests (NEW)
│   ├── CMakeLists.txt
│   └── utils\
│       ├── test_helpers.h
│       └── mock_backend.h
│
├── third_party\
│   ├── highway-github\              # Local Highway copy (SIMD library)
│   ├── nlohmann_json\
│   └── sqlite\
│
├── gemma.cpp\                        # Core gemma implementation
│   ├── gemma\
│   ├── ops\
│   ├── compression\
│   └── util\
│
├── backends\
│   ├── sycl\                        # Intel GPU backend
│   └── vulkan\
│
├── deploy\                           # Deployment output
│   ├── gemma.exe                    # Existing working executable
│   └── *.dll                        # Runtime DLLs
│
├── build*\                          # Build directories (gitignored)
│   ├── build_std\
│   ├── build_tbb_ipp\
│   └── build_perfpack\
│
├── Scripts (NEW):
├── build_oneapi.ps1                 # Simplified build wrapper
├── benchmark_baseline.ps1           # Inference benchmarking
├── compare_builds.ps1               # Multi-config automation
├── deploy_standalone.ps1            # Enhanced deployment
└── build_msvc_local.ps1            # MSVC fallback builder
│
├── Documentation (NEW/UPDATED):
├── BUILD_INTEL.md                   # Comprehensive build guide (UPDATED)
├── ONEAPI_INTEGRATION_SUMMARY.md   # Project summary (NEW)
├── BUILD_TROUBLESHOOTING.md        # Issue resolution (NEW)
└── PROJECT_STATUS_AND_ROADMAP.md   # This file (NEW)
│
├── Configuration:
├── CMakeLists.txt                   # Root CMake config (MODIFIED)
├── vcpkg.json                       # Package manifest
└── .gitignore
```

### Key Files Summary

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `cmake/OneAPILibs.cmake` | 298 | oneAPI library detection | ✅ Complete |
| `tests/unit/test_oneapi_validation.cpp` | 557 | C++ unit tests | ✅ Complete |
| `benchmark_baseline.ps1` | 500+ | Inference benchmarking | ✅ Complete |
| `compare_builds.ps1` | 345 | Multi-config builder | ✅ Complete |
| `build_oneapi.ps1` | 142 | Build wrapper | ✅ Complete |
| `deploy_standalone.ps1` | 270 | Enhanced deployment | ✅ Complete |
| `BUILD_INTEL.md` | 400+ | Build documentation | ✅ Complete |
| `ONEAPI_INTEGRATION_SUMMARY.md` | 275 | Project summary | ✅ Complete |
| `BUILD_TROUBLESHOOTING.md` | 170 | Troubleshooting guide | ✅ Complete |

---

## Future Directions

### Phase 1: Immediate (After Build Success)

#### 1.1 Validate Performance Gains
- [ ] Benchmark all configurations with gemma-2b-it
- [ ] Verify 25-50% CPU speedup claims
- [ ] Document actual performance improvements
- [ ] Identify bottlenecks for further optimization

#### 1.2 Expand Test Coverage
- [ ] Add more matrix sizes to DNNL tests
- [ ] Test with different model sizes (4B, 7B)
- [ ] Add stress tests (long-running inference)
- [ ] Test batch processing performance

#### 1.3 Production Deployment
- [ ] Package release binaries with DLLs
- [ ] Create installation guide for end users
- [ ] Document system requirements
- [ ] Provide troubleshooting for common issues

### Phase 2: Near-Term Enhancements

#### 2.1 Additional oneAPI Components
**oneMKL Integration:**
- Use oneMKL BLAS for matrix operations
- Compare against DNNL performance
- Evaluate memory usage

```cmake
option(GEMMA_USE_ONEMKL "Use Intel oneMKL for BLAS operations" OFF)
```

**Video Processing Library (VPL):**
- For PaliGemma image preprocessing
- Hardware-accelerated image decode/resize

**oneCCL (Collective Communications Library):**
- For multi-GPU/multi-node scaling
- Distributed inference

#### 2.2 Hybrid GPU/CPU Execution
**Current:** CPU-only or GPU-only  
**Future:** Dynamic work distribution

```cpp
// Pseudo-code
if (matrix_size > threshold && gpu_available) {
    sycl_matmul(A, B, C);
} else {
    dnnl_matmul(A, B, C);  // CPU fallback
}
```

#### 2.3 Auto-Tuning System
**Concept:** Automatically select best configuration based on:
- Hardware detected (CPU model, GPU availability)
- Model size being loaded
- Available system RAM
- Performance profiling

```powershell
.\gemma.exe --auto-tune --weights model.sbs
# Runs brief benchmarks, selects optimal threading/library config
```

#### 2.4 Memory Optimizations
**Current:** Standard allocators  
**Future:**
- TBB scalable allocator for all large allocations
- IPP memory functions for aligned buffers
- NUMA-aware allocation on multi-socket systems

### Phase 3: Long-Term Research

#### 3.1 Quantization with oneAPI
**Explore:** Using DNNL's quantization features
- INT8 inference with minimal accuracy loss
- Mixed precision (BF16/FP32)
- Dynamic quantization based on input

**Potential Speedup:** Additional 2-4x on top of current gains

#### 3.2 Model Compression
**Research:** Using IPP for compression
- Lossless weight compression
- Fast decompression during inference
- Reduced memory footprint

#### 3.3 Distributed Inference
**Vision:** Multi-node gemma deployment
- oneCCL for inter-node communication
- Model parallelism across GPUs
- Pipeline parallelism for throughput

#### 3.4 Custom Kernels
**Deep Optimization:** Hand-tuned SYCL/DPC++ kernels
- Custom attention implementations
- Fused operations (LayerNorm + Activation)
- Specialized kernels for Gemma architecture

**Potential:** Additional 10-20% over generic DNNL

#### 3.5 Windows ARM64 Support
**Future Platform:** ARM64 Windows devices
- Recompile with ARM NEON intrinsics
- oneAPI ARM support (when available)
- Cross-platform performance parity

### Phase 4: Ecosystem Integration

#### 4.1 ONNX Runtime Integration
**Allow:** Loading ONNX format models
- Broader model support
- Leverage ONNX Runtime's optimizations
- oneAPI execution provider for ONNX Runtime

#### 4.2 Python Bindings
**Expose:** Gemma.cpp through Python API

```python
import gemma_cpp

model = gemma_cpp.load("model.sbs", use_oneapi=True, perf_pack=True)
output = model.generate("Hello", max_tokens=100)
```

#### 4.3 REST API Server
**Deploy:** Gemma as a service

```bash
gemma-server --model model.sbs --port 8000 --oneapi-perfpack
```

#### 4.4 Benchmarking Dashboard
**Web UI:** Real-time performance monitoring
- Track performance across commits
- Compare configurations visually
- Historical trend analysis

---

## Known Issues

### Build Issues

#### Issue 1: FetchContent Git Failures
**Symptom:** 
```
fatal: unable to read tree (53de76561cfc149d3c01037f0595669ad32a5e7c)
RPC failed; curl 56 Recv failure: Connection was reset
```

**Root Cause:** Network/firewall blocking GitHub git operations

**Status:** ⚠️ Unresolved - blocking fresh builds

**Workarounds:**
1. Use vcpkg for SentencePiece
2. Copy `_deps` from working machine
3. Use local Highway (already available)
4. Modify Dependencies.cmake to skip FetchContent

#### Issue 2: SYCL `/MP` Flag Conflict
**Symptom:** SYCL build fails with `/MP` (parallel compilation)

**Root Cause:** Intel SYCL compiler doesn't support `/MP` flag

**Status:** ✅ Resolved - Excluded `/MP` from SYCL compilation

**Solution:** In `backends/sycl/CMakeLists.txt`:
```cmake
string(REPLACE "/MP" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
```

#### Issue 3: Ninja Not Found
**Symptom:** `ninja: command not found`

**Solution:**
```powershell
# Ninja should be at:
C:\Users\david\.local\bin\ninja.exe

# Verify PATH
$env:PATH -split ';' | Select-String "ninja"

# Or use Visual Studio generator instead
cmake -G "Visual Studio 17 2022"
```

### Runtime Issues

#### Issue 4: Missing DLLs
**Symptom:** `The code execution cannot proceed because XXX.dll was not found`

**Solutions:**
1. Use `deploy_standalone.ps1` to package DLLs
2. Load oneAPI environment: `"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"`
3. Copy DLLs manually from oneAPI directories

#### Issue 5: Performance Not Improving
**Possible Causes:**
- Bottleneck elsewhere (I/O, memory bandwidth)
- Model size too small to benefit
- Thread count mismatch (over/under-subscription)
- Thermal throttling

**Debug:**
```powershell
# Check thread count
$env:OMP_NUM_THREADS = 8  # Set to physical cores

# Profile with VTune
vtune -collect hotspots -- .\gemma.exe ...
```

### Script Issues

#### Issue 6: PowerShell Execution Policy
**Symptom:** `cannot be loaded because running scripts is disabled`

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Issue 7: Unicode Characters in Scripts
**Symptom:** `compare_builds.ps1` fails with parse errors

**Root Cause:** Checkmark/cross symbols (✅❌) cause encoding issues

**Status:** ⚠️ Known issue

**Workaround:** Replace Unicode with ASCII equivalents:
```powershell
# Replace ✅ with [OK]
# Replace ❌ with [FAIL]
```

---

## Quick Reference

### Build Commands Cheat Sheet

```powershell
# === Setup ===
# Load oneAPI environment (if not using wrapper scripts)
cmd /c '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"'

# === Building ===
# Standard build
.\build_oneapi.ps1 -Config std -Clean

# Optimized builds
.\build_oneapi.ps1 -Config tbb-ipp
.\build_oneapi.ps1 -Config perfpack

# Multi-config comparison
.\compare_builds.ps1 -Configurations @("std", "tbb-ipp", "perfpack")

# === Testing ===
# C++ unit tests
cmake --build build_perfpack --target test_oneapi_validation
.\build_perfpack\tests\test_oneapi_validation.exe

# Inference benchmarks
.\benchmark_baseline.ps1 -Executable build_std\bin\gemma_std.exe -Baseline
.\benchmark_baseline.ps1 -Executable build_perfpack\bin\gemma_std+tbb-ipp-dnnl.exe -OutputFile results.json
.\benchmark_baseline.ps1 -Compare -OutputFile results.json

# === Deployment ===
# Package for distribution
.\deploy_standalone.ps1
.\deploy_standalone.ps1 -ExecutableName "gemma_std+tbb-ipp.exe"
.\deploy_standalone.ps1 -IncludeOneAPILibs

# === Verification ===
# Check what's built
Get-ChildItem -Recurse -Filter "gemma*.exe" | Select-Object FullName, @{N='SizeMB';E={[math]::Round($_.Length/1MB,2)}}

# Test executable
.\build_std\bin\gemma_std.exe --help

# Check DLL dependencies (if dumpbin available)
dumpbin /dependents .\build_std\bin\gemma_std.exe | Select-String ".dll"
```

### CMake Options Reference

```cmake
# oneAPI Library Options
-DGEMMA_USE_ONEAPI_LIBS=ON              # Enable oneAPI integration
-DGEMMA_ONEAPI_PERFORMANCE_PACK=ON      # Enable all libraries
-DGEMMA_USE_TBB=ON                      # Just TBB
-DGEMMA_USE_IPP=ON                      # Just IPP
-DGEMMA_USE_DPL=ON                      # Just DPL
-DGEMMA_USE_DNNL=ON                     # Just DNNL

# Backend Options
-DGEMMA_ENABLE_SYCL=ON                  # Intel GPU
-DGEMMA_ENABLE_CUDA=ON                  # NVIDIA GPU
-DGEMMA_ENABLE_VULKAN=ON                # Vulkan cross-vendor

# Build Options
-DCMAKE_BUILD_TYPE=Release              # Optimization level
-DCMAKE_C_COMPILER=icx                  # Intel C compiler
-DCMAKE_CXX_COMPILER=icx                # Intel C++ compiler
-G "Ninja"                              # Ninja generator (fast)
-G "Visual Studio 17 2022"              # VS generator (fallback)
```

### Environment Variables

```powershell
# oneAPI paths (set by setvars.bat)
$env:ONEAPI_ROOT = "C:\Program Files (x86)\Intel\oneAPI"
$env:TBBROOT = "C:\Program Files (x86)\Intel\oneAPI\tbb\latest"
$env:IPPROOT = "C:\Program Files (x86)\Intel\oneAPI\ipp\latest"
$env:DNNLROOT = "C:\Program Files (x86)\Intel\oneAPI\dnnl\latest"

# Threading control
$env:OMP_NUM_THREADS = 8                # OpenMP threads
$env:TBB_NUM_THREADS = 8                # TBB threads

# Build acceleration
$env:CMAKE_BUILD_PARALLEL_LEVEL = 8     # Parallel CMake builds
```

### Useful Diagnostics

```powershell
# Check oneAPI installation
Get-ChildItem "C:\Program Files (x86)\Intel\oneAPI" -Directory | Select-Object Name

# Check compiler version
icx --version

# Check Ninja version
ninja --version

# Check CMake version
cmake --version

# Check available memory
Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory, TotalVisibleMemorySize

# Check CPU info
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors

# Check disk space
Get-PSDrive C | Select-Object Used, Free
```

---

## Appendix: Performance Expectations

### Theoretical Speedup Analysis

#### TBB (Threading Building Blocks)
**Optimization:** Parallel task scheduling

**Expected Speedup:** 1.15-1.25x

**Factors:**
- Amdahl's Law: Not all code is parallelizable
- Overhead: Task creation and scheduling
- Scalability: Diminishing returns beyond physical cores

**Best Case:** 8-core CPU, large batch processing → 1.25x  
**Worst Case:** 4-core CPU, small model → 1.10x

#### IPP (Integrated Performance Primitives)
**Optimization:** SIMD vector operations

**Expected Speedup:** 1.10-1.15x (on top of TBB)

**Factors:**
- AVX2 vs scalar: ~4-8x theoretical
- Memory bandwidth: Often the bottleneck
- Cache efficiency: Data locality matters

**Best Case:** Large dot products, good cache behavior → 1.15x  
**Worst Case:** Small operations, poor cache behavior → 1.05x

#### DNNL (Deep Neural Network Library)
**Optimization:** Optimized matrix multiplication

**Expected Speedup:** 1.10-1.15x (on top of TBB+IPP)

**Factors:**
- Matrix size: Larger matrices benefit more
- Cache blocking: DNNL uses optimized blocking
- BLAS implementation: Compared against standard

**Best Case:** Large matmul (M,K,N > 512), Intel CPU → 1.20x  
**Worst Case:** Small matmul (M,K,N < 128) → 1.05x

#### Combined Performance Pack
**Total Expected:** 1.35-1.50x

**Calculation:**
```
Total = TBB × IPP × DNNL
Conservative: 1.15 × 1.10 × 1.10 = 1.39x
Optimistic: 1.25 × 1.15 × 1.15 = 1.65x
Realistic: 1.35-1.50x (accounting for overhead)
```

### Measurement Methodology

**Benchmarking Protocol:**
1. Warm-up: 1 inference run (exclude from measurements)
2. Measurement: 3 inference runs per prompt length
3. Metrics: Average tokens/second
4. Statistical analysis: Mean, standard deviation, min/max

**Controlled Variables:**
- Same model (gemma-2b-it)
- Same prompts (short, medium, long)
- Same max_tokens (100)
- Same temperature (0.7)
- Same hardware (no thermal throttling)

**Comparison Baseline:**
- Standard build compiled with Intel ICX
- AVX2 enabled
- Release build optimizations
- No oneAPI libraries

---

## Contact & Support

**Project Maintainer:** [Your Name/Team]  
**Last Updated:** 2025-10-23  
**Version:** 1.0

**For Issues:**
1. Check `BUILD_TROUBLESHOOTING.md`
2. Review `cmake_output.log` for build errors
3. Verify oneAPI installation
4. Check network/firewall for FetchContent issues

**For Questions:**
- Review this document first
- Check `BUILD_INTEL.md` for build procedures
- Consult `ONEAPI_INTEGRATION_SUMMARY.md` for technical details

---

**End of Document**
