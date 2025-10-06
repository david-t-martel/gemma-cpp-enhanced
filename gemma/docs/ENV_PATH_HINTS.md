# Backend Environment Path Hints

Centralized quick-reference for typical include/library paths required when enabling optional backends locally or in CI. Mirrors the hint messages printed when `GEMMA_PRINT_BACKEND_INCLUDE_HINTS=ON` (default in tests).

## CUDA

- Windows (default install): `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v<version>/include`
- Linux: `/usr/local/cuda/include`

Add to CMake with:

```bash
cmake -DCMAKE_CUDA_COMPILER:FILEPATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin/nvcc.exe" \
      -DCMAKE_PREFIX_PATH="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" ..
```

## Intel oneAPI (SYCL)

- Windows root (BaseKit + HPC): `C:/Program Files (x86)/Intel/oneAPI`
- Linux root: `/opt/intel/oneapi`

Environment setup examples:

```bash
# Windows PowerShell
& "C:/Program Files (x86)/Intel/oneAPI/setvars.bat" intel64 vs2022

# Linux
source /opt/intel/oneapi/setvars.sh
```

## Vulkan

- Windows: `C:/VulkanSDK/<version>/Include`
- Linux system: `/usr/include/vulkan`
- If using the LunarG SDK: set `VULKAN_SDK` to the specific version directory.

## CMake Diagnostic Tips

If a backend reports missing headers:

1. Confirm the toolkit root exists.
2. Ensure the compiler toolchain sees the include path (check `CMAKE_CXX_TARGET_INCLUDE_PATH` in `CMakeCache.txt`).
3. For CUDA + MSVC, verify cl.exe is first in PATH before older toolchains.
4. Delete the build directory and re-configure after environment changes.

## Quick Probe Scripts

- Windows: `scripts/setup_env_windows.ps1`
- Linux: `scripts/setup_env_linux.sh`

These scripts emit a `SUMMARY_BEGIN` / `SUMMARY_END` block that can be parsed by automation or LLM agents for adaptive build guidance.
