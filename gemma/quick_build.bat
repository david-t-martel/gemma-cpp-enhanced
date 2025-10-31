@echo off
echo Starting Gemma build...

:: Setup Visual Studio environment for Ninja
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

:: Configure with Ninja (faster and more reliable)
echo Configuring with Ninja...
"C:\Program Files\CMake\bin\cmake.exe" -S . -B build_simple -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON -DGEMMA_PREFER_SYSTEM_DEPS=OFF
if %ERRORLEVEL% NEQ 0 (
    echo Configuration failed!
    exit /b 1
)

:: Build
echo Building with Ninja...
"C:\Program Files\CMake\bin\cmake.exe" --build build_simple --parallel 10
if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    exit /b 1
)

echo Build complete!
dir build_simple\gemma.exe 2>nul || dir build_simple\bin\gemma.exe
