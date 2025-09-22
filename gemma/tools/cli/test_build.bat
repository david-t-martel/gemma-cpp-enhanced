@echo off
REM Test build script for Enhanced Gemma CLI

echo Testing Enhanced Gemma CLI build...

REM Check if we're in the correct directory
if not exist "main.cpp" (
    echo Error: main.cpp not found. Please run from tools/cli directory.
    exit /b 1
)

REM Check if parent gemma.cpp exists
if not exist "..\..\gemma.cpp" (
    echo Error: gemma.cpp directory not found. Please ensure project structure is correct.
    exit /b 1
)

REM Create build directory
if not exist "build" mkdir build
cd build

echo Configuring CMake...
cmake .. -G "Visual Studio 17 2022" -T v143
if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed!
    exit /b 1
)

echo Building project...
cmake --build . --config Release
if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

echo Build successful!
echo.
echo To test the CLI:
echo 1. Ensure you have model files in C:\codedev\llm\.models\
echo 2. Run: build\Release\gemma_cli.exe --help
echo.

cd ..