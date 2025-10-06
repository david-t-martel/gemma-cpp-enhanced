#!/usr/bin/env bash
set -euo pipefail

echo "=== Toolchain Detection ==="
command -v cmake  >/dev/null && echo "CMake: $(command -v cmake)" || echo "CMake: NOT FOUND"
command -v ninja  >/dev/null && echo "Ninja: $(command -v ninja)" || echo "Ninja: NOT FOUND"
command -v python >/dev/null && echo "Python: $(command -v python)" || echo "Python: NOT FOUND"

echo "=== oneAPI (SYCL) ==="
if [[ -d /opt/intel/oneapi ]]; then
  dpcpp=$(command -v dpcpp || true)
  echo "oneAPI root: /opt/intel/oneapi"
  [[ -n "$dpcpp" ]] && echo "dpcpp: $dpcpp"
else
  echo "oneAPI not detected (expected /opt/intel/oneapi)"
fi

echo "=== CUDA Toolkit ==="
if [[ -d /usr/local/cuda ]]; then
  echo "CUDA root: /usr/local/cuda"
  echo "Include: /usr/local/cuda/include"
else
  echo "CUDA not detected (expected /usr/local/cuda)"
fi

echo "=== Vulkan SDK ==="
if [[ -n "${VULKAN_SDK:-}" ]]; then
  echo "VULKAN_SDK=$VULKAN_SDK"
elif [[ -d /usr/include/vulkan ]]; then
  echo "System Vulkan headers present at /usr/include/vulkan"
else
  echo "Vulkan SDK not detected"
fi

echo "=== LLM Summary (machine-parse) ==="
echo "SUMMARY_BEGIN"
echo "CMAKE_PATH=$(command -v cmake 2>/dev/null || echo NONE)"
echo "NINJA_PATH=$(command -v ninja 2>/dev/null || echo NONE)"
echo "PYTHON_PATH=$(command -v python 2>/dev/null || echo NONE)"
echo "ONEAPI_ROOT=/opt/intel/oneapi"; [[ -d /opt/intel/oneapi ]] || echo "ONEAPI_ROOT=NONE"
echo "CUDA_ROOT=/usr/local/cuda"; [[ -d /usr/local/cuda ]] || echo "CUDA_ROOT=NONE"
echo "VULKAN_SDK=${VULKAN_SDK:-NONE}"
echo "SUMMARY_END"
