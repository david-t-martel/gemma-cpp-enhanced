@echo off
setlocal

echo ============================================
echo PyO3 Binding Warnings Fix - Test Script
echo ============================================

REM Set PYO3_PYTHON environment variable
set "PYO3_PYTHON=C:\Users\david\AppData\Roaming\uv\python\cpython-3.13.3-windows-x86_64-none\python.exe"
echo Setting PYO3_PYTHON to: %PYO3_PYTHON%

REM Verify Python exists
if not exist "%PYO3_PYTHON%" (
    echo ERROR: Python executable not found at %PYO3_PYTHON%
    exit /b 1
)

echo Python executable verified: OK

REM Change to rust_extensions directory
cd /d "C:\codedev\llm\stats\rust_extensions"
if %errorlevel% neq 0 (
    echo ERROR: Could not change to rust_extensions directory
    exit /b 1
)

echo Current directory: %CD%

echo.
echo ============================================
echo Step 1: Running cargo check
echo ============================================
cargo check 2>&1
set CHECK_RESULT=%errorlevel%
if %CHECK_RESULT% equ 0 (
    echo SUCCESS: cargo check passed
) else (
    echo WARNING: cargo check failed with exit code %CHECK_RESULT%
    echo This may be due to missing dependencies or environment issues
)

echo.
echo ============================================
echo Step 2: Running cargo clippy with -D warnings
echo ============================================
cargo clippy --all-targets --all-features -- -D warnings 2>&1
set CLIPPY_RESULT=%errorlevel%
if %CLIPPY_RESULT% equ 0 (
    echo SUCCESS: cargo clippy passed with no warnings
) else (
    echo WARNING: cargo clippy failed with exit code %CLIPPY_RESULT%
    echo There may be warnings that need to be addressed
)

echo.
echo ============================================
echo Step 3: Testing maturin develop
echo ============================================
uv run maturin develop --release 2>&1
set MATURIN_RESULT=%errorlevel%
if %MATURIN_RESULT% equ 0 (
    echo SUCCESS: maturin develop completed successfully
) else (
    echo ERROR: maturin develop failed with exit code %MATURIN_RESULT%
)

echo.
echo ============================================
echo SUMMARY
echo ============================================
echo cargo check result: %CHECK_RESULT%
echo cargo clippy result: %CLIPPY_RESULT%
echo maturin develop result: %MATURIN_RESULT%

if %CHECK_RESULT% equ 0 if %CLIPPY_RESULT% equ 0 if %MATURIN_RESULT% equ 0 (
    echo.
    echo *** ALL TESTS PASSED ***
    echo PyO3 binding warnings have been successfully fixed!
) else (
    echo.
    echo *** SOME TESTS FAILED ***
    echo Please check the output above for specific error messages.
    echo.
    echo Common solutions:
    echo 1. Ensure UV Python is correctly installed
    echo 2. Check that all Rust dependencies are available
    echo 3. Verify that the workspace is properly configured
    echo 4. Try running 'cargo clean' and rebuilding
)

echo.
echo ============================================
echo Build Information
echo ============================================
echo PYO3_PYTHON: %PYO3_PYTHON%
echo Working Directory: %CD%
echo Rust Version:
rustc --version
echo Cargo Version:
cargo --version
echo UV Version:
uv --version

pause
