@echo off
REM =====================================================================
REM Enhanced Gemma.cpp Master Build Script
REM Production-ready build system with all backends and comprehensive testing
REM =====================================================================

setlocal EnableDelayedExpansion

echo.
echo =====================================================================
echo Enhanced Gemma.cpp Master Build System
echo =====================================================================
echo.

REM Set build configuration (can be overridden via command line)
set BUILD_TYPE=%~1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

REM Build directories
set PROJECT_ROOT=%~dp0
set BUILD_DIR=%PROJECT_ROOT%build-all
set INSTALL_DIR=%PROJECT_ROOT%install

echo Project Root: %PROJECT_ROOT%
echo Build Directory: %BUILD_DIR%
echo Install Directory: %INSTALL_DIR%
echo Build Type: %BUILD_TYPE%
echo.

REM =====================================================================
REM Environment Setup
REM =====================================================================

echo [SETUP] Configuring build environment...

REM CMake Configuration
set CMAKE_ROOT=C:\Program Files\CMake
set PATH=%CMAKE_ROOT%\bin;%PATH%

REM Check CMake availability
cmake --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] CMake not found. Please ensure CMake is installed at: %CMAKE_ROOT%
    exit /b 1
)

REM Intel oneAPI Setup (for SYCL backend)
set ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI
if exist "%ONEAPI_ROOT%\setvars.bat" (
    echo [SETUP] Initializing Intel oneAPI environment...
    call "%ONEAPI_ROOT%\setvars.bat" --config=quiet
    set GEMMA_SYCL_AVAILABLE=1
) else (
    echo [WARNING] Intel oneAPI not found at %ONEAPI_ROOT%
    set GEMMA_SYCL_AVAILABLE=0
)

REM CUDA Setup (for CUDA backend)
set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
if exist "%CUDA_ROOT%\bin\nvcc.exe" (
    echo [SETUP] CUDA Toolkit detected at %CUDA_ROOT%
    set PATH=%CUDA_ROOT%\bin;%PATH%
    set CUDA_PATH=%CUDA_ROOT%
    set GEMMA_CUDA_AVAILABLE=1
) else (
    echo [WARNING] CUDA Toolkit not found at %CUDA_ROOT%
    set GEMMA_CUDA_AVAILABLE=0
)

REM vcpkg Setup
set VCPKG_ROOT=C:\codedev\vcpkg
if exist "%VCPKG_ROOT%\vcpkg.exe" (
    echo [SETUP] vcpkg detected at %VCPKG_ROOT%
    set CMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake
    set GEMMA_VCPKG_AVAILABLE=1
) else (
    echo [WARNING] vcpkg not found at %VCPKG_ROOT%
    set GEMMA_VCPKG_AVAILABLE=0
)

REM Visual Studio Detection
call :detect_visual_studio
if !VS_FOUND!==0 (
    echo [ERROR] Visual Studio 2022 not found. Please install Visual Studio 2022 with C++ tools.
    exit /b 1
)

REM =====================================================================
REM Build Configuration Summary
REM =====================================================================

echo.
echo [CONFIG] Build Configuration Summary:
echo   - Build Type: %BUILD_TYPE%
echo   - Visual Studio: !VS_VERSION! (!VS_ARCH!)
echo   - CMake: Available
echo   - Intel oneAPI/SYCL: !GEMMA_SYCL_AVAILABLE!
echo   - NVIDIA CUDA: !GEMMA_CUDA_AVAILABLE!
echo   - vcpkg: !GEMMA_VCPKG_AVAILABLE!
echo.

REM =====================================================================
REM Clean Previous Builds
REM =====================================================================

echo [CLEAN] Cleaning previous builds...
if exist "%BUILD_DIR%" (
    rmdir /s /q "%BUILD_DIR%" 2>nul
)
if exist "%INSTALL_DIR%" (
    rmdir /s /q "%INSTALL_DIR%" 2>nul
)

mkdir "%BUILD_DIR%" 2>nul
mkdir "%INSTALL_DIR%" 2>nul

REM =====================================================================
REM CMake Configuration Phase
REM =====================================================================

echo.
echo [CMAKE] Configuring build with all available backends...

cd /d "%BUILD_DIR%"

REM Build CMake command with all options
set CMAKE_CMD=cmake
set CMAKE_CMD=!CMAKE_CMD! -G "Visual Studio 17 2022"
set CMAKE_CMD=!CMAKE_CMD! -A x64
set CMAKE_CMD=!CMAKE_CMD! -T v143

REM Build type
set CMAKE_CMD=!CMAKE_CMD! -DCMAKE_BUILD_TYPE=%BUILD_TYPE%

REM Installation
set CMAKE_CMD=!CMAKE_CMD! -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"

REM Enhanced components
set CMAKE_CMD=!CMAKE_CMD! -DGEMMA_BUILD_MCP_SERVER=ON
set CMAKE_CMD=!CMAKE_CMD! -DGEMMA_BUILD_BACKENDS=ON
set CMAKE_CMD=!CMAKE_CMD! -DGEMMA_BUILD_ENHANCED_TESTS=ON
set CMAKE_CMD=!CMAKE_CMD! -DGEMMA_BUILD_BACKEND_TESTS=ON
set CMAKE_CMD=!CMAKE_CMD! -DGEMMA_BUILD_BENCHMARKS=ON

REM Auto-detection
set CMAKE_CMD=!CMAKE_CMD! -DGEMMA_AUTO_DETECT_BACKENDS=ON

REM Backend-specific options (will be auto-detected but explicitly set for clarity)
if !GEMMA_SYCL_AVAILABLE!==1 (
    set CMAKE_CMD=!CMAKE_CMD! -DGEMMA_BUILD_SYCL_BACKEND=ON
)
if !GEMMA_CUDA_AVAILABLE!==1 (
    set CMAKE_CMD=!CMAKE_CMD! -DGEMMA_BUILD_CUDA_BACKEND=ON
)

REM Always try Vulkan and OpenCL (will be auto-detected)
set CMAKE_CMD=!CMAKE_CMD! -DGEMMA_BUILD_VULKAN_BACKEND=ON
set CMAKE_CMD=!CMAKE_CMD! -DGEMMA_BUILD_OPENCL_BACKEND=ON

REM vcpkg integration
if !GEMMA_VCPKG_AVAILABLE!==1 (
    set CMAKE_CMD=!CMAKE_CMD! -DCMAKE_TOOLCHAIN_FILE="%CMAKE_TOOLCHAIN_FILE%"
)

REM Optimization flags
set CMAKE_CMD=!CMAKE_CMD! -DCMAKE_CXX_FLAGS_RELEASE="/O2 /DNDEBUG /arch:AVX2"
set CMAKE_CMD=!CMAKE_CMD! -DCMAKE_C_FLAGS_RELEASE="/O2 /DNDEBUG /arch:AVX2"

REM Point to source directory
set CMAKE_CMD=!CMAKE_CMD! "%PROJECT_ROOT%"

echo [CMAKE] Executing: !CMAKE_CMD!
echo.

!CMAKE_CMD!
if errorlevel 1 (
    echo [ERROR] CMake configuration failed!
    exit /b 1
)

REM =====================================================================
REM Build Phase
REM =====================================================================

echo.
echo [BUILD] Building all targets...

REM Build with maximum parallelism
set CPU_COUNT=%NUMBER_OF_PROCESSORS%
if !CPU_COUNT! GTR 8 set CPU_COUNT=8

echo [BUILD] Using !CPU_COUNT! parallel jobs
echo.

cmake --build . --config %BUILD_TYPE% --parallel !CPU_COUNT!
if errorlevel 1 (
    echo [ERROR] Build failed!
    exit /b 1
)

REM =====================================================================
REM Installation Phase
REM =====================================================================

echo.
echo [INSTALL] Installing built components...

cmake --install . --config %BUILD_TYPE%
if errorlevel 1 (
    echo [ERROR] Installation failed!
    exit /b 1
)

REM =====================================================================
REM Build Verification
REM =====================================================================

echo.
echo [VERIFY] Verifying build artifacts...

set VERIFICATION_FAILED=0

REM Check core executable
if not exist "%BUILD_DIR%\%BUILD_TYPE%\gemma.exe" (
    echo [ERROR] Core gemma.exe not found!
    set VERIFICATION_FAILED=1
)

REM Check MCP server
if not exist "%BUILD_DIR%\%BUILD_TYPE%\gemma_mcp_stdio_server.exe" (
    echo [WARNING] MCP server not built
) else (
    echo [OK] MCP server built successfully
)

REM Check backend libraries
if exist "%BUILD_DIR%\backends\sycl\%BUILD_TYPE%\gemma_sycl_backend.lib" (
    echo [OK] SYCL backend built successfully
)
if exist "%BUILD_DIR%\backends\cuda\%BUILD_TYPE%\gemma_cuda_backend.lib" (
    echo [OK] CUDA backend built successfully
)
if exist "%BUILD_DIR%\backends\vulkan\%BUILD_TYPE%\gemma_vulkan_backend.lib" (
    echo [OK] Vulkan backend built successfully
)

if !VERIFICATION_FAILED!==1 (
    echo [ERROR] Build verification failed!
    exit /b 1
)

REM =====================================================================
REM Quick Test Run
REM =====================================================================

echo.
echo [TEST] Running quick validation tests...

if exist "%BUILD_DIR%\tests\unit\%BUILD_TYPE%\test_unit.exe" (
    echo [TEST] Running unit tests...
    "%BUILD_DIR%\tests\unit\%BUILD_TYPE%\test_unit.exe" --gtest_brief=1
    if errorlevel 1 (
        echo [WARNING] Some unit tests failed
    ) else (
        echo [OK] Unit tests passed
    )
)

REM =====================================================================
REM Build Summary
REM =====================================================================

echo.
echo =====================================================================
echo BUILD COMPLETED SUCCESSFULLY
echo =====================================================================
echo.
echo Build artifacts location: %BUILD_DIR%\%BUILD_TYPE%\
echo Installation directory: %INSTALL_DIR%
echo.
echo Available executables:
if exist "%BUILD_DIR%\%BUILD_TYPE%\gemma.exe" echo   - gemma.exe (Core inference engine)
if exist "%BUILD_DIR%\%BUILD_TYPE%\gemma_mcp_stdio_server.exe" echo   - gemma_mcp_stdio_server.exe (MCP server)
if exist "%BUILD_DIR%\%BUILD_TYPE%\benchmarks.exe" echo   - benchmarks.exe (Performance benchmarks)
echo.
echo Available backends:
if exist "%BUILD_DIR%\backends\sycl\%BUILD_TYPE%\gemma_sycl_backend.lib" echo   - SYCL (Intel oneAPI)
if exist "%BUILD_DIR%\backends\cuda\%BUILD_TYPE%\gemma_cuda_backend.lib" echo   - CUDA (NVIDIA)
if exist "%BUILD_DIR%\backends\vulkan\%BUILD_TYPE%\gemma_vulkan_backend.lib" echo   - Vulkan (Cross-platform)
if exist "%BUILD_DIR%\backends\opencl\%BUILD_TYPE%\gemma_opencl_backend.lib" echo   - OpenCL (Cross-platform)
echo.
echo Next steps:
echo   1. Run quick start: quick_start.bat
echo   2. Run comprehensive tests: test_all.bat
echo   3. Deploy: deploy_windows.bat
echo.
echo =====================================================================

cd /d "%PROJECT_ROOT%"
goto :eof

REM =====================================================================
REM Helper Functions
REM =====================================================================

:detect_visual_studio
set VS_FOUND=0
set VS_VERSION=Unknown
set VS_ARCH=x64

REM Try to find Visual Studio 2022
if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    set VS_FOUND=1
    set VS_VERSION=2022 Professional
    goto :vs_found
)

if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    set VS_FOUND=1
    set VS_VERSION=2022 Community
    goto :vs_found
)

if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
    set VS_FOUND=1
    set VS_VERSION=2022 Enterprise
    goto :vs_found
)

:vs_found
goto :eof

:show_help
echo.
echo Usage: build_all.bat [BUILD_TYPE]
echo.
echo BUILD_TYPE:
echo   Release     (default) - Optimized release build
echo   Debug       - Debug build with symbols
echo   RelWithDebInfo - Release with debug symbols
echo.
echo Examples:
echo   build_all.bat           - Build Release configuration
echo   build_all.bat Debug     - Build Debug configuration
echo.
goto :eof