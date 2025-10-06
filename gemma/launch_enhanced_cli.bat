@echo off
REM Enhanced Gemma CLI Launcher Script
REM This script launches the enhanced Gemma CLI with default settings

set GEMMA_MODEL=C:\codedev\llm\.models\gemma2-2b-it-sfp.sbs
set GEMMA_TOKENIZER=C:\codedev\llm\.models\tokenizer.spm
set GEMMA_EXE=C:\codedev\llm\gemma\build-avx2-sycl\bin\RELEASE\gemma.exe

echo Enhanced Gemma CLI with RAG-Redis Integration
echo ==============================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not available in PATH
    echo Please install Python or add it to your PATH
    pause
    exit /b 1
)

REM Check if model file exists
if not exist "%GEMMA_MODEL%" (
    echo ERROR: Model file not found: %GEMMA_MODEL%
    echo Please update GEMMA_MODEL path in this script
    pause
    exit /b 1
)

REM Check if gemma.exe exists
if not exist "%GEMMA_EXE%" (
    echo ERROR: Gemma executable not found: %GEMMA_EXE%
    echo Please build gemma.exe or update GEMMA_EXE path in this script
    pause
    exit /b 1
)

echo Found model: %GEMMA_MODEL%
echo Found executable: %GEMMA_EXE%

REM Check if Redis is available for RAG features
echo.
echo Checking Redis availability for RAG features...
redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo WARNING: Redis is not available - RAG features will be disabled
    echo To enable RAG: Start Redis server with: redis-server
    echo.
    echo Starting Gemma CLI without RAG...
    python gemma-cli.py --model "%GEMMA_MODEL%" --tokenizer "%GEMMA_TOKENIZER%" --debug
) else (
    echo Redis is available - RAG features enabled!
    echo.
    echo Starting Enhanced Gemma CLI with RAG...
    python gemma-cli.py --model "%GEMMA_MODEL%" --tokenizer "%GEMMA_TOKENIZER%" --enable-rag --debug
)

pause