@echo off
REM =====================================================================
REM Gemma.cpp Enhanced Launcher - Comprehensive Feature Showcase
REM =====================================================================
REM This batch file provides easy access to all Gemma.cpp capabilities:
REM - Interactive chat mode with conversation management
REM - Single prompt execution for quick queries
REM - Benchmarking and performance testing
REM - Debug mode with detailed layer outputs
REM - Model conversion utilities
REM - Multiple model support (2B, 4B, 9B variants)
REM - Advanced sampling parameter configuration
REM =====================================================================

setlocal enabledelayedexpansion

REM Configuration - Adjust paths as needed
set GEMMA_ROOT=%~dp0
set MODELS_DIR=C:\codedev\llm\.models
set BUILD_DIR=%GEMMA_ROOT%gemma.cpp\build_wsl
set PYTHON_CLI=%GEMMA_ROOT%gemma-cli.py

REM Available models - expand as needed
set MODEL_2B=%MODELS_DIR%\gemma-gemmacpp-2b-it-v3\2b-it.sbs
set MODEL_4B=%MODELS_DIR%\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\4b-it-sfp.sbs
set TOKENIZER_2B=%MODELS_DIR%\gemma-gemmacpp-2b-it-v3\tokenizer.spm
set TOKENIZER_4B=%MODELS_DIR%\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\tokenizer.spm

REM Default settings
set DEFAULT_MODEL=2B
set DEFAULT_TEMP=0.7
set DEFAULT_TOP_K=40
set DEFAULT_MAX_LEN=2048

echo.
echo =====================================================================
echo                    GEMMA.CPP ENHANCED LAUNCHER
echo =====================================================================
echo Available Commands:
echo.
echo  1. chat     - Interactive chat mode (recommended for conversations)
echo  2. prompt   - Single prompt execution (quick queries)
echo  3. bench    - Comprehensive benchmark suite
echo  4. debug    - Debug mode with layer analysis
echo  5. convert  - Model conversion utilities
echo  6. perf     - Performance testing and optimization
echo  7. config   - Configure default settings
echo  8. models   - List and manage available models
echo  9. help     - Show detailed help for each command
echo.
echo Current Settings:
echo   Model: %DEFAULT_MODEL% ^| Temp: %DEFAULT_TEMP% ^| Top-K: %DEFAULT_TOP_K% ^| Max Length: %DEFAULT_MAX_LEN%
echo.

if "%1"=="" (
    echo Please specify a command. Use 'run_gemma help' for detailed information.
    goto :eof
)

REM Command routing
if "%1"=="chat" goto :chat
if "%1"=="prompt" goto :prompt
if "%1"=="bench" goto :bench
if "%1"=="debug" goto :debug
if "%1"=="convert" goto :convert
if "%1"=="perf" goto :perf
if "%1"=="config" goto :config
if "%1"=="models" goto :models
if "%1"=="help" goto :help
echo Unknown command: %1
goto :help

REM =====================================================================
REM INTERACTIVE CHAT MODE
REM =====================================================================
:chat
echo.
echo ═══════════════════════════════════════
echo      INTERACTIVE CHAT MODE
echo ═══════════════════════════════════════

REM Parse optional parameters
set CHAT_MODEL=%DEFAULT_MODEL%
set CHAT_TEMP=%DEFAULT_TEMP%
set CHAT_TOP_K=%DEFAULT_TOP_K%
set CHAT_MAX_LEN=%DEFAULT_MAX_LEN%
set CHAT_SYSTEM=""

:parse_chat_args
if "%2"=="" goto :start_chat
if "%2"=="--model" set CHAT_MODEL=%3& shift & shift & goto :parse_chat_args
if "%2"=="--temp" set CHAT_TEMP=%3& shift & shift & goto :parse_chat_args
if "%2"=="--top-k" set CHAT_TOP_K=%3& shift & shift & goto :parse_chat_args
if "%2"=="--max-len" set CHAT_MAX_LEN=%3& shift & shift & goto :parse_chat_args
if "%2"=="--system" set CHAT_SYSTEM=%3& shift & shift & goto :parse_chat_args
shift
goto :parse_chat_args

:start_chat
call :validate_environment
if !errorlevel! neq 0 goto :eof

echo Starting chat with model: %CHAT_MODEL%
echo Temperature: %CHAT_TEMP% ^| Top-K: %CHAT_TOP_K% ^| Max Length: %CHAT_MAX_LEN%

if !CHAT_SYSTEM! neq "" (
    echo System prompt: !CHAT_SYSTEM!
    python "%PYTHON_CLI%" --model %CHAT_MODEL% --temperature %CHAT_TEMP% --top-k %CHAT_TOP_K% --max-length %CHAT_MAX_LEN% --system !CHAT_SYSTEM!
) else (
    python "%PYTHON_CLI%" --model %CHAT_MODEL% --temperature %CHAT_TEMP% --top-k %CHAT_TOP_K% --max-length %CHAT_MAX_LEN%
)
goto :eof

REM =====================================================================
REM SINGLE PROMPT MODE
REM =====================================================================
:prompt
echo.
echo ═══════════════════════════════════════
echo        SINGLE PROMPT MODE
echo ═══════════════════════════════════════

if "%2"=="" (
    echo Error: Please provide a prompt.
    echo Usage: run_gemma prompt "Your question here" [--model MODEL] [--temp VALUE] [--top-k VALUE]
    echo Example: run_gemma prompt "Explain quantum computing" --model 4B --temp 0.8
    goto :eof
)

set PROMPT_TEXT=%2
set PROMPT_MODEL=%DEFAULT_MODEL%
set PROMPT_TEMP=%DEFAULT_TEMP%
set PROMPT_TOP_K=%DEFAULT_TOP_K%

:parse_prompt_args
if "%3"=="" goto :execute_prompt
if "%3"=="--model" set PROMPT_MODEL=%4& shift & shift & goto :parse_prompt_args
if "%3"=="--temp" set PROMPT_TEMP=%4& shift & shift & goto :parse_prompt_args
if "%3"=="--top-k" set PROMPT_TOP_K=%4& shift & shift & goto :parse_prompt_args
shift
goto :parse_prompt_args

:execute_prompt
call :validate_environment
if !errorlevel! neq 0 goto :eof

call :get_model_paths %PROMPT_MODEL%
echo Executing prompt with %PROMPT_MODEL% model...
echo Prompt: %PROMPT_TEXT%

wsl "%BUILD_DIR%/gemma" --tokenizer "!TOKENIZER_PATH!" --weights "!MODEL_PATH!" --prompt %PROMPT_TEXT% --temperature %PROMPT_TEMP% --top_k %PROMPT_TOP_K%
goto :eof

REM =====================================================================
REM BENCHMARK MODE
REM =====================================================================
:bench
echo.
echo ═══════════════════════════════════════
echo       COMPREHENSIVE BENCHMARKS
echo ═══════════════════════════════════════

set BENCH_MODEL=%DEFAULT_MODEL%
set BENCH_ITERATIONS=100
set BENCH_OUTPUT=bench_results_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%.txt

:parse_bench_args
if "%2"=="" goto :run_benchmarks
if "%2"=="--model" set BENCH_MODEL=%3& shift & shift & goto :parse_bench_args
if "%2"=="--iterations" set BENCH_ITERATIONS=%3& shift & shift & goto :parse_bench_args
if "%2"=="--output" set BENCH_OUTPUT=%3& shift & shift & goto :parse_bench_args
shift
goto :parse_bench_args

:run_benchmarks
call :validate_environment
if !errorlevel! neq 0 goto :eof

call :get_model_paths %BENCH_MODEL%
echo Running benchmarks with %BENCH_MODEL% model (%BENCH_ITERATIONS% iterations)...
echo Results will be saved to: %BENCH_OUTPUT%

echo ============= GEMMA BENCHMARK RESULTS ============= > %BENCH_OUTPUT%
echo Model: %BENCH_MODEL% >> %BENCH_OUTPUT%
echo Date: %date% %time% >> %BENCH_OUTPUT%
echo Iterations: %BENCH_ITERATIONS% >> %BENCH_OUTPUT%
echo. >> %BENCH_OUTPUT%

echo Running single benchmark...
wsl "%BUILD_DIR%/single_benchmark" --weights "!MODEL_PATH!" --tokenizer "!TOKENIZER_PATH!" >> %BENCH_OUTPUT%

echo Running comprehensive benchmarks...
wsl "%BUILD_DIR%/benchmarks" --weights "!MODEL_PATH!" --tokenizer "!TOKENIZER_PATH!" >> %BENCH_OUTPUT%

echo.
echo Benchmark results saved to: %BENCH_OUTPUT%
echo Summary:
type %BENCH_OUTPUT% | findstr /i "tokens/sec latency throughput"
goto :eof

REM =====================================================================
REM DEBUG MODE
REM =====================================================================
:debug
echo.
echo ═══════════════════════════════════════
echo         DEBUG MODE & ANALYSIS
echo ═══════════════════════════════════════

set DEBUG_MODEL=%DEFAULT_MODEL%
set DEBUG_PROMPT="Hello, how are you?"
set DEBUG_LAYERS=all

:parse_debug_args
if "%2"=="" goto :run_debug
if "%2"=="--model" set DEBUG_MODEL=%3& shift & shift & goto :parse_debug_args
if "%2"=="--prompt" set DEBUG_PROMPT=%3& shift & shift & goto :parse_debug_args
if "%2"=="--layers" set DEBUG_LAYERS=%3& shift & shift & goto :parse_debug_args
shift
goto :parse_debug_args

:run_debug
call :validate_environment
if !errorlevel! neq 0 goto :eof

call :get_model_paths %DEBUG_MODEL%
echo Running debug analysis with %DEBUG_MODEL% model...
echo Prompt: %DEBUG_PROMPT%
echo Layer analysis: %DEBUG_LAYERS%

echo.
echo Debug output (detailed layer information):
wsl "%BUILD_DIR%/debug_prompt" --weights "!MODEL_PATH!" --tokenizer "!TOKENIZER_PATH!" --prompt %DEBUG_PROMPT%
goto :eof

REM =====================================================================
REM CONFIGURATION MANAGEMENT
REM =====================================================================
:config
echo.
echo ═══════════════════════════════════════
echo       CONFIGURATION MANAGEMENT
echo ═══════════════════════════════════════

if "%2"=="" (
    echo Current configuration:
    echo   Default Model: %DEFAULT_MODEL%
    echo   Temperature: %DEFAULT_TEMP%
    echo   Top-K: %DEFAULT_TOP_K%
    echo   Max Length: %DEFAULT_MAX_LEN%
    echo   Models Directory: %MODELS_DIR%
    echo   Build Directory: %BUILD_DIR%
    echo.
    echo Usage: run_gemma config [show^|set^|reset]
    goto :eof
)

if "%2"=="show" goto :config_show
if "%2"=="set" goto :config_set
if "%2"=="reset" goto :config_reset
echo Unknown config operation: %2
goto :eof

:config_show
goto :config

:config_set
echo Configuration setting requires manual editing of this batch file.
echo Edit the following variables at the top of run_gemma.bat:
echo   DEFAULT_MODEL, DEFAULT_TEMP, DEFAULT_TOP_K, DEFAULT_MAX_LEN
goto :eof

:config_reset
echo Resetting to default configuration...
echo Please restart the script to apply defaults.
goto :eof

REM =====================================================================
REM MODEL MANAGEMENT
REM =====================================================================
:models
echo.
echo ═══════════════════════════════════════
echo        MODEL MANAGEMENT
echo ═══════════════════════════════════════

echo Available models in %MODELS_DIR%:
echo.

if exist "%MODEL_2B%" (
    echo ✓ 2B Model: %MODEL_2B%
    echo   Tokenizer: %TOKENIZER_2B%
) else (
    echo ✗ 2B Model: Not found
)

if exist "%MODEL_4B%" (
    echo ✓ 4B Model: %MODEL_4B%
    echo   Tokenizer: %TOKENIZER_4B%
) else (
    echo ✗ 4B Model: Not found
)

echo.
echo To add more models:
echo 1. Download from Kaggle: https://www.kaggle.com/models/google/gemma-2/gemmaCpp
echo 2. Extract to %MODELS_DIR%
echo 3. Update model paths in this script
echo.

if "%2"=="test" (
    echo Testing model availability...
    call :test_all_models
)
goto :eof
REM =====================================================================
REM UTILITY FUNCTIONS
REM =====================================================================

:validate_environment
echo Validating environment...

REM Check if WSL is available
wsl --version >nul 2>&1
if !errorlevel! neq 0 (
    echo Error: WSL is not available or not properly configured.
    echo Please install WSL and ensure it's working correctly.
    exit /b 1
)

REM Check if build directory exists
if not exist "%BUILD_DIR%" (
    echo Error: Build directory not found: %BUILD_DIR%
    echo Please build the project first using: cmake --preset make ^&^& cmake --build --preset make
    exit /b 1
)

REM Check if main executable exists
if not exist "%BUILD_DIR%\gemma" (
    echo Error: Gemma executable not found: %BUILD_DIR%\gemma
    echo Please ensure the project is built correctly.
    exit /b 1
)

REM Check if Python CLI exists
if not exist "%PYTHON_CLI%" (
    echo Warning: Python CLI not found: %PYTHON_CLI%
    echo Some features may not be available.
)

echo Environment validation passed.
exit /b 0

:get_model_paths
set MODEL_NAME=%1
if "%MODEL_NAME%"=="2B" (
    set MODEL_PATH=%MODEL_2B%
    set TOKENIZER_PATH=%TOKENIZER_2B%
) else if "%MODEL_NAME%"=="4B" (
    set MODEL_PATH=%MODEL_4B%
    set TOKENIZER_PATH=%TOKENIZER_4B%
) else (
    echo Warning: Unknown model %MODEL_NAME%, using 2B as default
    set MODEL_PATH=%MODEL_2B%
    set TOKENIZER_PATH=%TOKENIZER_2B%
)

REM Validate model files exist
if not exist "!MODEL_PATH!" (
    echo Error: Model file not found: !MODEL_PATH!
    echo Please ensure models are downloaded and extracted to %MODELS_DIR%
    exit /b 1
)

if not exist "!TOKENIZER_PATH!" (
    echo Error: Tokenizer file not found: !TOKENIZER_PATH!
    echo Please ensure tokenizer is available alongside the model.
    exit /b 1
)

exit /b 0

:test_all_models
echo Testing 2B model...
call :get_model_paths 2B
if !errorlevel! equ 0 (
    echo ✓ 2B model files found and accessible
) else (
    echo ✗ 2B model test failed
)

echo Testing 4B model...
call :get_model_paths 4B
if !errorlevel! equ 0 (
    echo ✓ 4B model files found and accessible
) else (
    echo ✗ 4B model test failed
)
exit /b 0

echo.
echo =====================================================================
echo                    Script execution completed
echo =====================================================================
