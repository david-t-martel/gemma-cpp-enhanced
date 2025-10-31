# Gemma.cpp Build Troubleshooting Guide

## Current Build Issue: FetchContent Git Corruption

### Symptom
CMake configuration fails with:
```
fatal: unable to read tree (53de76561cfc149d3c01037f0595669ad32a5e7c)
CMake Error: Failed to checkout tag
```

### Root Cause
The FetchContent cache (`build/_deps/`) contains corrupted git repositories for:
- Highway
- SentencePiece

This occurs when:
1. Git clone was interrupted
2. Disk errors during FetchContent population
3. Antivirus interference with git operations

### Solution 1: Clean FetchContent Cache (Recommended)

```powershell
# Remove all build directories and FetchContent cache
Remove-Item -Recurse -Force build*, .cmake -ErrorAction SilentlyContinue

# Fresh configuration
cmd /c '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && cmake -S . -B build_fresh -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release'

# Build
cmd /c '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && cmake --build build_fresh --config Release -j'
```

### Solution 2: Use Local Dependencies

If you have a working gemma.cpp installation elsewhere, copy the `_deps` folder:

```powershell
# From a working build
Copy-Item "path\to\working\build\_deps" "C:\codedev\llm\gemma\build\_deps" -Recurse

# Then configure without re-fetching
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release
```

### Solution 3: Manual Dependency Installation

#### Install SentencePiece via vcpkg
```powershell
# Edit vcpkg.json to add sentencepiece
$json = Get-Content vcpkg.json | ConvertFrom-Json
$json.dependencies = @("sentencepiece")
$json | ConvertTo-Json | Set-Content vcpkg.json

# Let vcpkg install it
vcpkg install
```

#### Use Local Highway
The project includes Highway in `third_party/highway-github`. Modify `cmake/Dependencies.cmake` to use it:

```cmake
# Around line 68, replace FetchContent with:
set(GEMMA_HWY_LIBS "hwy" "hwy_contrib")
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/highway-github)
```

### Solution 4: Bypass CMake, Use Pre-built Binary

If you have a previously working `gemma.exe`:

```powershell
# Copy from existing build
Copy-Item "path\to\working\gemma.exe" "C:\codedev\llm\gemma\gemma.exe"

# Test it
.\gemma.exe --help
```

## Quick Status Check

Check if you have a working executable:

```powershell
Get-ChildItem -Recurse -Filter "gemma*.exe" | Select-Object FullName, @{N='SizeMB';E={[math]::Round($_.Length/1MB,2)}}
```

## Alternative: Use MSVC Instead of Intel Compiler

If oneAPI builds are problematic, fall back to MSVC:

```powershell
cmake -S . -B build_msvc -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build build_msvc --config Release -j
```

## oneAPI-Specific Build Script

Use the simplified wrapper:

```powershell
# Standard build
.\build_oneapi.ps1 -Config std -Clean

# With TBB+IPP
.\build_oneapi.ps1 -Config tbb-ipp -Clean

# Full performance pack
.\build_oneapi.ps1 -Config perfpack -Clean
```

## Verification Steps After Successful Build

### 1. Check Executable Exists
```powershell
Test-Path "build_*/bin/gemma*.exe"
```

### 2. Verify oneAPI Lib Naming
```powershell
Get-ChildItem "build_*/bin/gemma*.exe" | Select-Object Name
# Should show: gemma_std.exe, gemma_std+tbb-ipp.exe, etc.
```

### 3. Run Smoke Test
```powershell
& "build_std\bin\gemma_std.exe" --help
```

### 4. Check DLL Dependencies
```powershell
dumpbin /dependents "build_std\bin\gemma_std.exe" | Select-String ".dll"
```

## Next Steps After Build Success

1. **Run Validation Tests**:
   ```powershell
   cmake --build build_std --target test_oneapi_validation
   .\build_std\tests\test_oneapi_validation.exe
   ```

2. **Benchmark Performance**:
   ```powershell
   .\benchmark_baseline.ps1 -Executable build_std\bin\gemma_std.exe -Baseline
   ```

3. **Deploy Standalone Package**:
   ```powershell
   .\deploy_standalone.ps1
   ```

## Contact & Support

If issues persist:
1. Check `cmake_output.log` for detailed errors
2. Verify oneAPI installation: `"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"`
3. Ensure Ninja is in PATH: `ninja --version`
4. Check disk space: `Get-PSDrive C | Select-Object Used,Free`

## Known Working Configuration

- **OS**: Windows 11 x64
- **Compiler**: Intel oneAPI 2025.2.0
- **CMake**: 4.1+
- **Generator**: Ninja
- **oneAPI Components**: compiler, tbb, ipp, dnnl, dpl

Last successful build: [Document when you get a successful build]
