# Building Gemma Enhanced

This document supplements `README.md` with up-to-date build instructions for the enhanced CMake system (vcpkg-first, modular dependency summary, backend auto-detection, custom build types).

## Quick Start (Windows, MSVC)

```powershell
# (Optional) Set or confirm vcpkg root; auto-detected if VCPKG_ROOT is set.
$env:VCPKG_ROOT = "C:\vcpkg"          # if not already

# (Optional) Provide oneAPI root for SYCL explicitly (skip if on PATH / standard location)
$env:ONEAPI_ROOT = "C:\Program Files (x86)\Intel\oneAPI"

# Configure (RelWithSymbols custom build type: O2 + symbols)
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithSymbols -DGEMMA_ENABLE_AUTO_BACKENDS=ON -DGEMMA_ENABLE_SYCL=ON

# Build
cmake --build build --config RelWithSymbols
```

If you use the Visual Studio multi-config generator, omit `-DCMAKE_BUILD_TYPE` and choose the config with `--config` on build.

## Custom Build Types

Two additional build configurations are supported beyond the CMake defaults:

- FastDebug: `-O1 -g` (MSVC: `/Od /O1 /Zi`) – quicker iteration with moderate optimization.
- RelWithSymbols: `-O2 -g -DNDEBUG` (MSVC: `/O2 /Zi /DNDEBUG`) – optimized with symbols retained.

Use via `-DCMAKE_BUILD_TYPE=FastDebug` or `RelWithSymbols` (single-config) or select in the IDE for multi-config generators.

## Highway (SIMD) Dependency

Highway is resolved via system/vcpkg first; falls back to a pinned FetchContent revision otherwise.

- `GEMMA_REQUIRE_HWY_CONTRIB=ON` forces configuration to fail if the optional `hwy_contrib` target is not available (useful when you rely on contrib utilities).

## Dependency Provenance Summary

At the end of configuration, if `cmake/DependencySummary.cmake` is present (it is in this fork), a provenance block prints whether core third-party dependencies came from system/vcpkg (IMPORTED/built) or were fetched.

You can re-run just configuration to see the summary:

```powershell
cmake -S . -B build -LAH | Select-String "Gemma Dependency"
```

## SYCL / oneAPI Backend

SYCL backend detection order for candidate roots:

1. `-DGEMMA_ONEAPI_ROOT=...`
2. `-Doneapi_root=...`
3. Environment variables: `GEMMA_ONEAPI_ROOT`, `ONEAPI_ROOT`, `INTEL_ONEAPI_ROOT`
4. Conventional path: `C:/Program Files (x86)/Intel/oneAPI`

If a complete triplet (compiler, headers, library) is found, the backend is enabled and `GEMMA_ONEAPI_ROOT` is cached.

Disable auto-detection but keep explicit SYCL request:


```powershell
cmake -S . -B build -DGEMMA_ENABLE_AUTO_BACKENDS=OFF -DGEMMA_ENABLE_SYCL=ON -DGEMMA_ONEAPI_ROOT="C:/Program Files (x86)/Intel/oneAPI"
```

## CUDA / Vulkan / OpenCL

- CUDA, Vulkan, OpenCL are toggled similarly via `GEMMA_ENABLE_<NAME>`; auto-detection (if on) attempts to enable them if toolkits are found.
- Vulkan intentionally defaults OFF on auto to avoid unnecessary heavyweight dependency unless requested.

## Legacy Root Backend Path

The original root-driven backend add_subdirectory path is deprecated. If you need it for transitional scripts:

```powershell
cmake -S . -B build -DGEMMA_USE_LEGACY_BACKEND_BUILD=ON
```

This forwards legacy cache toggles to the subproject.

## Verbose Diagnostics

Set `-DGEMMA_PREFER_SYSTEM_DEPS=OFF` to force fetch of the pinned revisions (useful for hermetic builds). Compiler flags for each configuration are echoed in the gemma.cpp summary.

Additional flags:

- `-DGEMMA_VERBOSE_BACKEND_DIAGNOSTICS=ON` prints detailed SYCL/oneAPI candidate root scanning and environment snapshots.
- `-DGEMMA_REQUIRE_HWY_CONTRIB=ON` fails configuration if Highway's contrib target is absent.
- Automatic environment validation runs at root configure (CMAKE_ROOT, oneAPI roots, vcpkg status). Disable by removing the `EnvValidation` include if customizing.

## Troubleshooting

- CMake cannot find modules (`CMAKE_ROOT` error): verify a valid CMake install on PATH; run `cmake --version`.
- SYCL not detected: ensure `icx`/`icpx` exist under the root you passed; check the summary block for which component (compiler/include/lib) was missing.
- `hwy_contrib` required but missing: install highway via vcpkg (`vcpkg install highway`) or build from source ensuring contrib targets are generated.

## Minimal Fetch-Only Build

```powershell
cmake -S . -B build -DGEMMA_PREFER_SYSTEM_DEPS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 4
```

## Installing

Artifacts install (by default) under `build/install`:

```powershell
cmake --install build --config RelWithSymbols
```

Customize with `-DCMAKE_INSTALL_PREFIX=...`.

## Generating Dependency Graph (Optional)

You can generate a DOT graph of targets:

```powershell
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_GRAPHVIZ_OUTPUT=graph
```

---
Last updated: 2025-09-22
