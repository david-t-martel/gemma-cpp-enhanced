@echo off
echo Setting up Visual Studio 2022 Environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

echo Building with CMake...
mkdir build_vs 2>nul
cd build_vs

"C:\Program Files\CMake\bin\cmake.exe" -G "Visual Studio 17 2022" -A x64 ..
if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed
    pause
    exit /b 1
)

"C:\Program Files\CMake\bin\cmake.exe" --build . --config Release
if %ERRORLEVEL% neq 0 (
    echo Build failed
    pause
    exit /b 1
)

echo Build completed successfully!
pause