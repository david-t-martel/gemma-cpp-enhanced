@echo off
echo Starting memory-optimized build for RAG-Redis system...
echo.

:: Set environment variables for memory-optimized build
set CARGO_TARGET_DIR=target\memory-optimized
set CARGO_BUILD_JOBS=2
set CARGO_NET_GIT_FETCH_WITH_CLI=true

echo [1/5] Cleaning previous builds...
cargo clean
if exist target\memory-optimized rmdir /s /q target\memory-optimized

echo [2/5] Building main workspace with minimal features...
cargo build --profile release-memory-optimized --workspace --no-default-features --features minimal
if errorlevel 1 (
    echo ERROR: Main workspace build failed
    exit /b 1
)

echo [3/5] Building PyO3 extensions separately...
cd rust-rag-extensions
cargo build --profile release-python --no-default-features --features python-bindings
if errorlevel 1 (
    echo ERROR: PyO3 extensions build failed
    cd ..
    exit /b 1
)
cd ..

echo [4/5] Building with Redis backend...
cargo build --profile release-memory-optimized --package rag-redis-system --no-default-features --features default
if errorlevel 1 (
    echo ERROR: Redis backend build failed
    exit /b 1
)

echo [5/5] Verifying build artifacts...
if exist "target\memory-optimized\release-memory-optimized\rag-redis-system.exe" (
    echo SUCCESS: rag-redis-system.exe built successfully
) else (
    echo WARNING: rag-redis-system.exe not found
)

if exist "target\memory-optimized\release-python\rag_extensions.dll" (
    echo SUCCESS: rag_extensions.dll built successfully
) else (
    echo WARNING: rag_extensions.dll not found
)

echo.
echo Build completed! Check target\memory-optimized\ for artifacts.
echo To run: .\target\memory-optimized\release-memory-optimized\rag-redis-system.exe