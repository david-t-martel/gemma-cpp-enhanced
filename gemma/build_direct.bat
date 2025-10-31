@echo off
echo Initializing Visual Studio 2022 environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64

echo.
echo Configuring CMake...
"C:\Program Files\CMake\bin\cmake.exe" -B build_direct -G "Visual Studio 17 2022" -A x64 -T v143 -DCMAKE_TOOLCHAIN_FILE=""

if errorlevel 1 (
    echo Configuration failed!
    exit /b 1
)

echo.
echo Building gemma.exe...
"C:\Program Files\CMake\bin\cmake.exe" --build build_direct --config Release -j 10 --target gemma

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo.
echo Build complete!
echo Binary: build_direct\Release\gemma.exe
