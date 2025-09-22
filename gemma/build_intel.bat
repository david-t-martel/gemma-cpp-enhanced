@echo off
setlocal enabledelayedexpansion

echo ===================================
echo Intel oneAPI Gemma Build Script
echo ===================================

:: Set up Intel oneAPI environment first
echo Setting up Intel oneAPI environment...
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 vs2022

:: Verify Intel compiler is available
echo Testing Intel compiler...
icx --version
if !errorlevel! neq 0 (
    echo ERROR: Intel C++ compiler not found!
    echo Make sure Intel oneAPI is installed and setvars.bat ran successfully.
    pause
    exit /b 1
)

echo Intel compiler detected successfully!

:: Clean up old build directory
if exist "build-intel" (
    echo Cleaning old Intel build directory...
    rmdir /s /q "build-intel"
)

:: Create fresh build directory
mkdir "build-intel"
cd "build-intel"

echo Configuring with Intel C++ compiler...

:: Configure CMake with Intel compiler
cmake .. ^
  -G "Ninja" ^
  -DCMAKE_C_COMPILER=icx ^
  -DCMAKE_CXX_COMPILER=icx ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGEMMA_BUILD_SYCL_BACKEND=ON ^
  -DGEMMA_AUTO_DETECT_BACKENDS=ON ^
  -DCMAKE_CXX_FLAGS_RELEASE="/O3 /DNDEBUG /fp:fast /QxHost" ^
  -DCMAKE_C_FLAGS_RELEASE="/O3 /DNDEBUG /fp:fast /QxHost"

if !errorlevel! neq 0 (
    echo ERROR: CMake configuration failed!
    cd ..
    pause
    exit /b 1
)

echo Configuration successful! Building...

:: Build the project
cmake --build . --config Release -j 4

if !errorlevel! neq 0 (
    echo ERROR: Build failed!
    cd ..
    pause
    exit /b 1
)

echo Build completed successfully!
echo Binary location: %cd%\Release\gemma.exe

cd ..
echo ===================================
echo Intel oneAPI Build Complete!
echo ===================================
pause