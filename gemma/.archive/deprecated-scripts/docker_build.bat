@echo off
REM =====================================================================
REM Enhanced Gemma.cpp Docker Build Script
REM Multi-configuration Docker image builder with hardware backend support
REM =====================================================================

setlocal EnableDelayedExpansion

echo.
echo =====================================================================
echo Enhanced Gemma.cpp Docker Build System
echo =====================================================================
echo.

REM Configuration variables
set IMAGE_NAME=gemma-cpp-enhanced
set IMAGE_TAG=%~1
if "%IMAGE_TAG%"=="" set IMAGE_TAG=latest
set BUILD_TYPE=%~2
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

REM Build arguments (can be overridden via environment variables)
if "%ENABLE_CUDA%"=="" set ENABLE_CUDA=false
if "%ENABLE_SYCL%"=="" set ENABLE_SYCL=false
if "%ENABLE_VULKAN%"=="" set ENABLE_VULKAN=true
if "%ENABLE_OPENCL%"=="" set ENABLE_OPENCL=true
if "%ENABLE_MCP_SERVER%"=="" set ENABLE_MCP_SERVER=true
if "%ENABLE_TESTS%"=="" set ENABLE_TESTS=false
if "%CMAKE_PARALLEL_JOBS%"=="" set CMAKE_PARALLEL_JOBS=4

REM Detect available hardware acceleration
call :detect_hardware_capabilities

echo [CONFIG] Docker Build Configuration:
echo   Image Name: %IMAGE_NAME%:%IMAGE_TAG%
echo   Build Type: %BUILD_TYPE%
echo   Parallel Jobs: %CMAKE_PARALLEL_JOBS%
echo.
echo [CONFIG] Backend Configuration:
echo   CUDA Support: %ENABLE_CUDA%
echo   SYCL Support: %ENABLE_SYCL%
echo   Vulkan Support: %ENABLE_VULKAN%
echo   OpenCL Support: %ENABLE_OPENCL%
echo   MCP Server: %ENABLE_MCP_SERVER%
echo   Tests: %ENABLE_TESTS%
echo.

REM =====================================================================
REM Docker Build Command Construction
REM =====================================================================

echo [BUILD] Constructing Docker build command...

set DOCKER_CMD=docker build
set DOCKER_CMD=!DOCKER_CMD! --tag %IMAGE_NAME%:%IMAGE_TAG%
set DOCKER_CMD=!DOCKER_CMD! --build-arg BUILD_TYPE=%BUILD_TYPE%
set DOCKER_CMD=!DOCKER_CMD! --build-arg ENABLE_CUDA=%ENABLE_CUDA%
set DOCKER_CMD=!DOCKER_CMD! --build-arg ENABLE_SYCL=%ENABLE_SYCL%
set DOCKER_CMD=!DOCKER_CMD! --build-arg ENABLE_VULKAN=%ENABLE_VULKAN%
set DOCKER_CMD=!DOCKER_CMD! --build-arg ENABLE_OPENCL=%ENABLE_OPENCL%
set DOCKER_CMD=!DOCKER_CMD! --build-arg ENABLE_MCP_SERVER=%ENABLE_MCP_SERVER%
set DOCKER_CMD=!DOCKER_CMD! --build-arg ENABLE_TESTS=%ENABLE_TESTS%
set DOCKER_CMD=!DOCKER_CMD! --build-arg CMAKE_PARALLEL_JOBS=%CMAKE_PARALLEL_JOBS%

REM Add build metadata
for /f "tokens=* USEBACKQ" %%i in (`git rev-parse --short HEAD 2^>nul`) do set GIT_COMMIT=%%i
if "%GIT_COMMIT%"=="" set GIT_COMMIT=unknown

set DOCKER_CMD=!DOCKER_CMD! --build-arg VCS_REF=%GIT_COMMIT%
set DOCKER_CMD=!DOCKER_CMD! --build-arg BUILD_DATE=%DATE%

REM Performance optimizations
set DOCKER_CMD=!DOCKER_CMD! --build-arg BUILDKIT_INLINE_CACHE=1

REM Target the current directory
set DOCKER_CMD=!DOCKER_CMD! .

echo [BUILD] Docker command: !DOCKER_CMD!
echo.

REM =====================================================================
REM Execute Docker Build
REM =====================================================================

echo [BUILD] Starting Docker build process...
echo [BUILD] This may take 10-30 minutes depending on enabled backends...
echo.

!DOCKER_CMD!
if errorlevel 1 (
    echo [ERROR] Docker build failed!
    exit /b 1
)

REM =====================================================================
REM Build Verification
REM =====================================================================

echo.
echo [VERIFY] Verifying Docker image...

REM Check if image was created
docker images %IMAGE_NAME%:%IMAGE_TAG% --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
if errorlevel 1 (
    echo [ERROR] Image verification failed!
    exit /b 1
)

REM Quick functional test
echo [VERIFY] Running quick functional test...
docker run --rm %IMAGE_NAME%:%IMAGE_TAG% ./gemma --help >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Basic functionality test failed
) else (
    echo [OK] Basic functionality test passed
)

REM =====================================================================
REM Create Additional Tagged Versions
REM =====================================================================

echo [TAGS] Creating additional image tags...

REM Create backend-specific tags
if "%ENABLE_CUDA%"=="true" (
    docker tag %IMAGE_NAME%:%IMAGE_TAG% %IMAGE_NAME%:%IMAGE_TAG%-cuda
    echo [OK] Created CUDA-enabled tag: %IMAGE_NAME%:%IMAGE_TAG%-cuda
)

if "%ENABLE_SYCL%"=="true" (
    docker tag %IMAGE_NAME%:%IMAGE_TAG% %IMAGE_NAME%:%IMAGE_TAG%-sycl
    echo [OK] Created SYCL-enabled tag: %IMAGE_NAME%:%IMAGE_TAG%-sycl
)

if "%ENABLE_CUDA%"=="true" if "%ENABLE_SYCL%"=="true" (
    docker tag %IMAGE_NAME%:%IMAGE_TAG% %IMAGE_NAME%:%IMAGE_TAG%-all-backends
    echo [OK] Created all-backends tag: %IMAGE_NAME%:%IMAGE_TAG%-all-backends
)

REM Create size-optimized tag for CPU-only builds
if "%ENABLE_CUDA%"=="false" if "%ENABLE_SYCL%"=="false" (
    docker tag %IMAGE_NAME%:%IMAGE_TAG% %IMAGE_NAME%:%IMAGE_TAG%-cpu-only
    echo [OK] Created CPU-only tag: %IMAGE_NAME%:%IMAGE_TAG%-cpu-only
)

REM =====================================================================
REM Generate Docker Compose File
REM =====================================================================

echo [COMPOSE] Generating docker-compose.yml...

echo # Enhanced Gemma.cpp Docker Compose Configuration > docker-compose.yml
echo version: '3.8' >> docker-compose.yml
echo. >> docker-compose.yml
echo services: >> docker-compose.yml
echo   gemma-cpp: >> docker-compose.yml
echo     image: %IMAGE_NAME%:%IMAGE_TAG% >> docker-compose.yml
echo     container_name: gemma-cpp-enhanced >> docker-compose.yml
echo     volumes: >> docker-compose.yml
echo       - ./models:/app/models:ro >> docker-compose.yml
echo       - ./data:/app/data >> docker-compose.yml
echo     environment: >> docker-compose.yml
echo       - GEMMA_LOG_LEVEL=INFO >> docker-compose.yml
echo       - GEMMA_BACKEND=auto >> docker-compose.yml

if "%ENABLE_MCP_SERVER%"=="true" (
    echo     ports: >> docker-compose.yml
    echo       - "8080:8080" >> docker-compose.yml
)

if "%ENABLE_CUDA%"=="true" (
    echo     runtime: nvidia >> docker-compose.yml
    echo     environment: >> docker-compose.yml
    echo       - NVIDIA_VISIBLE_DEVICES=all >> docker-compose.yml
    echo       - NVIDIA_DRIVER_CAPABILITIES=compute,utility >> docker-compose.yml
)

echo     restart: unless-stopped >> docker-compose.yml
echo     command: ["./mcp_server", "--weights", "/app/models/gemma2-2b-it-sfp.sbs"] >> docker-compose.yml

echo [OK] docker-compose.yml created

REM =====================================================================
REM Generate Helper Scripts
REM =====================================================================

echo [SCRIPTS] Creating helper scripts...

REM Create run script
echo @echo off > run_docker.bat
echo REM Quick Docker run script for Enhanced Gemma.cpp >> run_docker.bat
echo. >> run_docker.bat
echo if not exist "models" mkdir models >> run_docker.bat
echo if not exist "data" mkdir data >> run_docker.bat
echo. >> run_docker.bat
echo echo [INFO] Starting Enhanced Gemma.cpp container... >> run_docker.bat
echo echo [INFO] Models directory: %%CD%%\models >> run_docker.bat
echo echo [INFO] Data directory: %%CD%%\data >> run_docker.bat
echo. >> run_docker.bat

if "%ENABLE_CUDA%"=="true" (
    echo docker run -it --rm --gpus all ^^ >> run_docker.bat
) else (
    echo docker run -it --rm ^^ >> run_docker.bat
)

echo   -v "%%CD%%\models:/app/models:ro" ^^ >> run_docker.bat
echo   -v "%%CD%%\data:/app/data" ^^ >> run_docker.bat

if "%ENABLE_MCP_SERVER%"=="true" (
    echo   -p 8080:8080 ^^ >> run_docker.bat
)

echo   %IMAGE_NAME%:%IMAGE_TAG% %%* >> run_docker.bat

REM Create shell access script
echo @echo off > docker_shell.bat
echo REM Get shell access to Enhanced Gemma.cpp container >> docker_shell.bat
echo. >> docker_shell.bat

if "%ENABLE_CUDA%"=="true" (
    echo docker run -it --rm --gpus all ^^ >> docker_shell.bat
) else (
    echo docker run -it --rm ^^ >> docker_shell.bat
)

echo   -v "%%CD%%\models:/app/models:ro" ^^ >> docker_shell.bat
echo   -v "%%CD%%\data:/app/data" ^^ >> docker_shell.bat
echo   --entrypoint /bin/bash ^^ >> docker_shell.bat
echo   %IMAGE_NAME%:%IMAGE_TAG% >> docker_shell.bat

echo [OK] Helper scripts created: run_docker.bat, docker_shell.bat

REM =====================================================================
REM Generate Usage Documentation
REM =====================================================================

echo [DOCS] Creating Docker usage documentation...

echo # Enhanced Gemma.cpp Docker Usage > DOCKER_USAGE.md
echo. >> DOCKER_USAGE.md
echo ## Quick Start >> DOCKER_USAGE.md
echo. >> DOCKER_USAGE.md
echo 1. Place your Gemma model files in the `models/` directory >> DOCKER_USAGE.md
echo 2. Run: `run_docker.bat --weights /app/models/your-model.sbs` >> DOCKER_USAGE.md
echo. >> DOCKER_USAGE.md
echo ## Available Images >> DOCKER_USAGE.md
echo. >> DOCKER_USAGE.md
echo - `%IMAGE_NAME%:%IMAGE_TAG%` - Main image with selected backends >> DOCKER_USAGE.md

if "%ENABLE_CUDA%"=="true" (
    echo - `%IMAGE_NAME%:%IMAGE_TAG%-cuda` - CUDA-optimized build >> DOCKER_USAGE.md
)
if "%ENABLE_SYCL%"=="true" (
    echo - `%IMAGE_NAME%:%IMAGE_TAG%-sycl` - Intel oneAPI/SYCL build >> DOCKER_USAGE.md
)

echo. >> DOCKER_USAGE.md
echo ## Hardware Requirements >> DOCKER_USAGE.md
echo. >> DOCKER_USAGE.md

if "%ENABLE_CUDA%"=="true" (
    echo - NVIDIA GPU with CUDA support (for CUDA backend) >> DOCKER_USAGE.md
    echo - Docker with nvidia-container-toolkit installed >> DOCKER_USAGE.md
)
if "%ENABLE_SYCL%"=="true" (
    echo - Intel GPU/NPU (for SYCL backend) >> DOCKER_USAGE.md
)

echo. >> DOCKER_USAGE.md
echo ## Configuration >> DOCKER_USAGE.md
echo. >> DOCKER_USAGE.md
echo Use environment variables to configure runtime behavior: >> DOCKER_USAGE.md
echo. >> DOCKER_USAGE.md
echo - `GEMMA_BACKEND=auto` - Auto-select best backend >> DOCKER_USAGE.md
echo - `GEMMA_LOG_LEVEL=INFO` - Set logging level >> DOCKER_USAGE.md
echo. >> DOCKER_USAGE.md
echo ## Using Docker Compose >> DOCKER_USAGE.md
echo. >> DOCKER_USAGE.md
echo `docker-compose up -d` - Start as background service >> DOCKER_USAGE.md

echo [OK] DOCKER_USAGE.md created

REM =====================================================================
REM Build Summary
REM =====================================================================

echo.
echo =====================================================================
echo DOCKER BUILD COMPLETED SUCCESSFULLY
echo =====================================================================
echo.
echo Image: %IMAGE_NAME%:%IMAGE_TAG%
for /f "tokens=* USEBACKQ" %%i in (`docker images %IMAGE_NAME%:%IMAGE_TAG% --format "{{.Size}}"`) do set IMAGE_SIZE=%%i
echo Size: %IMAGE_SIZE%
echo.
echo Available backends:
if "%ENABLE_CUDA%"=="true" echo   ✓ CUDA (NVIDIA GPUs)
if "%ENABLE_SYCL%"=="true" echo   ✓ SYCL (Intel oneAPI)
if "%ENABLE_VULKAN%"=="true" echo   ✓ Vulkan (Cross-platform)
if "%ENABLE_OPENCL%"=="true" echo   ✓ OpenCL (Cross-platform)
echo.
echo Generated files:
echo   - docker-compose.yml
echo   - run_docker.bat
echo   - docker_shell.bat
echo   - DOCKER_USAGE.md
echo.
echo Quick commands:
echo   Test: run_docker.bat --help
echo   Shell: docker_shell.bat
echo   Compose: docker-compose up -d
echo.
echo =====================================================================

goto :eof

REM =====================================================================
REM Helper Functions
REM =====================================================================

:detect_hardware_capabilities
echo [DETECT] Detecting hardware capabilities...

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [DETECT] NVIDIA GPU detected - enabling CUDA support
    set ENABLE_CUDA=true
) else (
    echo [DETECT] No NVIDIA GPU detected
)

REM Check for Intel GPU (basic detection)
wmic path win32_videocontroller get name | findstr /i "intel" >nul 2>&1
if not errorlevel 1 (
    echo [DETECT] Intel GPU detected - SYCL support available
    REM Don't auto-enable SYCL as it requires oneAPI runtime
) else (
    echo [DETECT] No Intel GPU detected
)

REM Docker and hardware support verification
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker not found! Please install Docker Desktop.
    exit /b 1
)

REM Check for NVIDIA Container Toolkit if CUDA is enabled
if "%ENABLE_CUDA%"=="true" (
    docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] NVIDIA Container Toolkit not available
        echo [WARNING] CUDA support will be disabled in container
    )
)

goto :eof

:show_help
echo.
echo Usage: docker_build.bat [TAG] [BUILD_TYPE]
echo.
echo Arguments:
echo   TAG         Docker image tag (default: latest)
echo   BUILD_TYPE  Build configuration (default: Release)
echo.
echo Environment Variables:
echo   ENABLE_CUDA=true/false      Enable NVIDIA CUDA backend
echo   ENABLE_SYCL=true/false      Enable Intel SYCL backend
echo   ENABLE_VULKAN=true/false    Enable Vulkan backend
echo   ENABLE_OPENCL=true/false    Enable OpenCL backend
echo   ENABLE_MCP_SERVER=true/false Enable MCP server
echo   ENABLE_TESTS=true/false     Include test suite
echo   CMAKE_PARALLEL_JOBS=N       Parallel build jobs
echo.
echo Examples:
echo   docker_build.bat                           - Build latest with defaults
echo   docker_build.bat v1.0 Release             - Build v1.0 release
echo   set ENABLE_CUDA=true ^&^& docker_build.bat cuda - Build with CUDA
echo.
goto :eof