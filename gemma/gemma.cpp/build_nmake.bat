@echo off
echo Setting up Visual Studio 2022 Environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

echo Cleaning previous builds...
if exist build_nmake rmdir /s /q build_nmake
mkdir build_nmake
cd build_nmake

echo Building with NMake Makefiles...
"C:\Program Files\CMake\bin\cmake.exe" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed
    pause
    exit /b 1
)

echo Starting build...
nmake
if %ERRORLEVEL% neq 0 (
    echo Build failed
    pause
    exit /b 1
)

echo Build completed successfully!
echo Executable should be at: %CD%\gemma.exe
dir gemma.exe
pause