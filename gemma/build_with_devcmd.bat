@echo off
echo === Building Gemma.cpp with Developer Command Prompt ===

:: Initialize VS Developer Command Prompt
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64

:: Clean
if exist build (
    echo Cleaning build directory...
    rmdir /s /q build
)

:: Configure
echo Configuring with CMake...
cmake -B build -G "Visual Studio 17 2022" -T v143 -A x64 -DCMAKE_BUILD_TYPE=Release

if %ERRORLEVEL% neq 0 (
    echo Configuration failed!
    exit /b 1
)

:: Build
echo Building Release configuration...
cmake --build build --config Release -j 10

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

:: Check output
echo.
echo === Build Complete ===
if exist "build\bin\Release\gemma.exe" (
    echo Binary found: build\bin\Release\gemma.exe
    dir "build\bin\Release\gemma.exe"

    echo.
    echo Testing --help flag...
    "build\bin\Release\gemma.exe" --help

    echo.
    if not exist deploy mkdir deploy
    copy /Y "build\bin\Release\gemma.exe" "deploy\gemma.exe"
    echo Binary copied to deploy\gemma.exe
) else (
    echo ERROR: Binary not found!
    exit /b 1
)
