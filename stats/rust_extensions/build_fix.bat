@echo off
rem Script to fix PyO3 binding warnings in rust_extensions

rem Set the PYO3_PYTHON environment variable
set PYO3_PYTHON=C:\Users\david\AppData\Roaming\uv\python\cpython-3.13.3-windows-x86_64-none\python.exe

echo Setting PYO3_PYTHON to: %PYO3_PYTHON%

rem Change to the rust_extensions directory
cd /d "C:\codedev\llm\stats\rust_extensions"

echo Current directory: %CD%

rem Verify Python executable exists
if exist "%PYO3_PYTHON%" (
    echo ✓ Python executable found
    "%PYO3_PYTHON%" --version
) else (
    echo ✗ Python executable not found at: %PYO3_PYTHON%
    exit /b 1
)

echo.
echo === Running cargo check ===
cargo check
if %ERRORLEVEL% equ 0 (
    echo ✓ cargo check passed
) else (
    echo ✗ cargo check failed with exit code: %ERRORLEVEL%
)

echo.
echo === Running cargo clippy with warnings as errors ===
cargo clippy --all-targets --all-features -- -D warnings
if %ERRORLEVEL% equ 0 (
    echo ✓ cargo clippy passed
) else (
    echo ✗ cargo clippy failed with exit code: %ERRORLEVEL%
)

echo.
echo === Building with maturin ===
uv run maturin develop --release
if %ERRORLEVEL% equ 0 (
    echo ✓ maturin develop passed
) else (
    echo ✗ maturin develop failed with exit code: %ERRORLEVEL%
)

echo.
echo Script completed.
pause
