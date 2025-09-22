@echo off
setlocal

echo ===================================
echo Fixed MSVC Gemma Build Script
echo ===================================

:: First, let's apply the source code fixes for the known Highway SIMD and lambda issues
echo Applying source code fixes for Highway SIMD and C++20 compatibility...

:: Fix 1: Replace .empty() with .size() == 0 in Highway Span usage
echo Fixing Highway Span .empty() calls...
powershell -Command "(Get-Content 'gemma.cpp\ops\ops-inl.h') -replace '\.empty\(\)', '.size() == 0' | Set-Content 'gemma.cpp\ops\ops-inl.h'"

:: Fix 2: Fix lambda capture issue in gemma.cc
echo Fixing lambda capture syntax...
powershell -Command "(Get-Content 'gemma.cpp\gemma\gemma.cc') -replace '\[&, &recent_tokens\]', '[&]' | Set-Content 'gemma.cpp\gemma\gemma.cc'"

echo Source code fixes applied!

:: Clean up old build directory
if exist "build-intel" (
    echo Cleaning old build directory...
    rmdir /s /q "build-intel"
)

:: Create fresh build directory
mkdir "build-intel"
cd "build-intel"

echo Configuring with MSVC and optimized flags...

:: Configure CMake with MSVC but highly optimized flags
"C:\Program Files\CMake\bin\cmake.exe" .. ^
  -G "Visual Studio 17 2022" ^
  -T v143 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGEMMA_BUILD_SYCL_BACKEND=OFF ^
  -DGEMMA_AUTO_DETECT_BACKENDS=ON ^
  -DGEMMA_BUILD_ENHANCED_TESTS=OFF ^
  -DGEMMA_BUILD_BACKEND_TESTS=OFF ^
  -DGEMMA_BUILD_BENCHMARKS=ON ^
  -DCMAKE_CXX_FLAGS_RELEASE="/O2 /Ob2 /DNDEBUG /arch:AVX2 /fp:fast /GL /std:c++20" ^
  -DCMAKE_C_FLAGS_RELEASE="/O2 /Ob2 /DNDEBUG /arch:AVX2 /fp:fast /GL" ^
  -DCMAKE_EXE_LINKER_FLAGS_RELEASE="/LTCG /OPT:REF /OPT:ICF"

if %errorlevel% neq 0 (
    echo ERROR: CMake configuration failed!
    cd ..
    pause
    exit /b 1
)

echo Configuration successful! Building...

:: Build the project
"C:\Program Files\CMake\bin\cmake.exe" --build . --config Release --parallel 4

if %errorlevel% neq 0 (
    echo ERROR: Build failed!
    echo Checking what was built...
    dir /s *.exe *.lib
    cd ..
    pause
    exit /b 1
)

echo Build completed successfully!

:: Check for the main binary
if exist "gemma.cpp\Release\gemma.exe" (
    echo SUCCESS: gemma.exe binary created at: %cd%\gemma.cpp\Release\gemma.exe
    
    :: Copy to a convenient location
    copy "gemma.cpp\Release\gemma.exe" "..\gemma-fixed.exe"
    echo Binary copied to: %cd%\..\gemma-fixed.exe
    
    :: Test the binary
    echo Testing the binary...
    "..\gemma-fixed.exe" --help
    
) else (
    echo Searching for gemma binaries...
    dir /s gemma*.exe
)

cd ..
echo ===================================
echo Fixed MSVC Build Complete!
echo ===================================
pause