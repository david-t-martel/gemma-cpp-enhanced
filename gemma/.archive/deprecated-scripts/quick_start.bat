@echo off
REM =====================================================================
REM Enhanced Gemma.cpp Quick Start Script
REM Interactive setup and first-run experience for new users
REM =====================================================================

setlocal EnableDelayedExpansion

echo.
echo =====================================================================
echo Enhanced Gemma.cpp Quick Start Guide
echo =====================================================================
echo.
echo Welcome to Enhanced Gemma.cpp! This script will help you get started
echo with your first inference run using Google's Gemma models.
echo.

REM Configuration
set PROJECT_ROOT=%~dp0
set BUILD_DIR=%PROJECT_ROOT%build-all\Release
set MODELS_DIR=%PROJECT_ROOT%models
set EXAMPLES_DIR=%PROJECT_ROOT%examples

echo [WELCOME] Project Directory: %PROJECT_ROOT%
echo [WELCOME] Looking for builds in: %BUILD_DIR%
echo.

REM =====================================================================
REM System Requirements Check
REM =====================================================================

echo [CHECK] Checking system requirements...

REM Check if build exists
if not exist "%BUILD_DIR%\gemma.exe" (
    echo [ERROR] Gemma.cpp not built yet!
    echo.
    echo Please run one of these commands first:
    echo   1. build_all.bat          - Full build with all backends
    echo   2. build_all.bat Debug    - Debug build for development
    echo.
    echo After building, run this script again.
    pause
    exit /b 1
)

echo [OK] Gemma.cpp executable found
echo [OK] Build directory: %BUILD_DIR%

REM Check available backends
call :detect_system_capabilities

echo.
echo [CHECK] System capabilities detected:
echo   CPU: Always available
if %GPU_CUDA_AVAILABLE%==1 echo   NVIDIA GPU: Available (CUDA)
if %GPU_INTEL_AVAILABLE%==1 echo   Intel GPU: Available (SYCL)
if %GPU_VULKAN_AVAILABLE%==1 echo   Vulkan: Available (Cross-platform GPU)
if %GPU_OPENCL_AVAILABLE%==1 echo   OpenCL: Available (Parallel computing)
echo.

REM =====================================================================
REM Model Files Setup
REM =====================================================================

echo [SETUP] Checking for model files...

if not exist "%MODELS_DIR%" (
    mkdir "%MODELS_DIR%"
    echo [SETUP] Created models directory: %MODELS_DIR%
)

REM Check for common model files
set MODEL_FOUND=0
set TOKENIZER_FOUND=0

REM Look for model files
for %%f in ("%MODELS_DIR%\*.sbs") do (
    set MODEL_FILE=%%f
    set MODEL_FOUND=1
    goto :model_search_done
)
:model_search_done

REM Look for tokenizer
if exist "%MODELS_DIR%\tokenizer.spm" (
    set TOKENIZER_FILE=%MODELS_DIR%\tokenizer.spm
    set TOKENIZER_FOUND=1
) else (
    for %%f in ("%MODELS_DIR%\*.spm") do (
        set TOKENIZER_FILE=%%f
        set TOKENIZER_FOUND=1
        goto :tokenizer_search_done
    )
)
:tokenizer_search_done

if %MODEL_FOUND%==1 if %TOKENIZER_FOUND%==1 (
    echo [OK] Model files found:
    echo   Model: %MODEL_FILE%
    echo   Tokenizer: %TOKENIZER_FILE%
    echo.
    goto :run_inference
) else (
    echo [SETUP] Model files not found. Let's set them up!
    call :setup_model_files
    if errorlevel 1 exit /b 1
)

REM =====================================================================
REM First Inference Run
REM =====================================================================

:run_inference
echo [RUN] Ready to run your first inference!
echo.

set /p RUN_DEMO="Would you like to run a demo inference? (y/n): "
if /i not "%RUN_DEMO%"=="y" goto :advanced_options

echo.
echo [DEMO] Running demo inference...
echo [DEMO] This may take a moment to load the model...
echo.

REM Choose an appropriate prompt
set DEMO_PROMPT="Hello! Can you tell me a short story about AI?"

echo [DEMO] Prompt: %DEMO_PROMPT%
echo [DEMO] Starting inference...
echo.

REM Run the inference
"%BUILD_DIR%\gemma.exe" --weights "%MODEL_FILE%" --tokenizer "%TOKENIZER_FILE%" --prompt %DEMO_PROMPT%

if errorlevel 1 (
    echo.
    echo [ERROR] Demo inference failed!
    echo This might be due to:
    echo   - Incompatible model files
    echo   - Insufficient memory
    echo   - Missing dependencies
    echo.
    echo Please check the error message above.
    pause
    exit /b 1
) else (
    echo.
    echo [SUCCESS] Demo inference completed successfully!
    echo.
)

REM =====================================================================
REM Advanced Options
REM =====================================================================

:advanced_options
echo =====================================================================
echo Advanced Options and Next Steps
echo =====================================================================
echo.

echo What would you like to do next?
echo.
echo 1. Run interactive chat mode
echo 2. Test MCP server
echo 3. Run performance benchmarks
echo 4. Test hardware acceleration backends
echo 5. View example scripts
echo 6. Exit
echo.

set /p CHOICE="Enter your choice (1-6): "

if "%CHOICE%"=="1" goto :interactive_chat
if "%CHOICE%"=="2" goto :test_mcp
if "%CHOICE%"=="3" goto :run_benchmarks
if "%CHOICE%"=="4" goto :test_backends
if "%CHOICE%"=="5" goto :show_examples
if "%CHOICE%"=="6" goto :exit_script

echo [ERROR] Invalid choice. Please enter 1-6.
goto :advanced_options

REM =====================================================================
REM Interactive Chat Mode
REM =====================================================================

:interactive_chat
echo.
echo [CHAT] Starting interactive chat mode...
echo [CHAT] Type 'quit' or 'exit' to end the session.
echo.

REM Create a simple chat script
echo @echo off > "%PROJECT_ROOT%\temp_chat.bat"
echo :chat_loop >> "%PROJECT_ROOT%\temp_chat.bat"
echo set /p USER_INPUT="You: " >> "%PROJECT_ROOT%\temp_chat.bat"
echo if /i "%%USER_INPUT%%"=="quit" goto :end_chat >> "%PROJECT_ROOT%\temp_chat.bat"
echo if /i "%%USER_INPUT%%"=="exit" goto :end_chat >> "%PROJECT_ROOT%\temp_chat.bat"
echo echo. >> "%PROJECT_ROOT%\temp_chat.bat"
echo echo Gemma: >> "%PROJECT_ROOT%\temp_chat.bat"
echo "%BUILD_DIR%\gemma.exe" --weights "%MODEL_FILE%" --tokenizer "%TOKENIZER_FILE%" --prompt "%%USER_INPUT%%" >> "%PROJECT_ROOT%\temp_chat.bat"
echo echo. >> "%PROJECT_ROOT%\temp_chat.bat"
echo goto :chat_loop >> "%PROJECT_ROOT%\temp_chat.bat"
echo :end_chat >> "%PROJECT_ROOT%\temp_chat.bat"
echo echo Thank you for using Enhanced Gemma.cpp! >> "%PROJECT_ROOT%\temp_chat.bat"

call "%PROJECT_ROOT%\temp_chat.bat"
del "%PROJECT_ROOT%\temp_chat.bat" 2>nul

goto :advanced_options

REM =====================================================================
REM Test MCP Server
REM =====================================================================

:test_mcp
echo.
echo [MCP] Testing MCP server functionality...

if exist "%BUILD_DIR%\gemma_mcp_stdio_server.exe" (
    echo [MCP] MCP server found. Testing basic functionality...
    
    echo [MCP] Starting MCP server (stdio mode)...
    echo [MCP] This will show the server capabilities and then exit.
    echo.
    
    "%BUILD_DIR%\gemma_mcp_stdio_server.exe" --help
    
    echo.
    echo [MCP] For full MCP integration, you can:
    echo   1. Use it with MCP-compatible clients
    echo   2. Integrate with development tools
    echo   3. Use the stdio transport for tool calling
    echo.
) else (
    echo [MCP] MCP server not available in this build.
    echo To enable MCP server, rebuild with:
    echo   build_all.bat (MCP is enabled by default)
    echo.
)

pause
goto :advanced_options

REM =====================================================================
REM Performance Benchmarks
REM =====================================================================

:run_benchmarks
echo.
echo [BENCH] Running performance benchmarks...

if exist "%BUILD_DIR%\benchmarks.exe" (
    echo [BENCH] Running comprehensive benchmarks...
    echo [BENCH] This will test inference speed and memory usage.
    echo.
    
    "%BUILD_DIR%\benchmarks.exe" --benchmark_min_time=2
    
    echo.
    echo [BENCH] Benchmark completed!
    echo Check the results above for performance metrics.
    echo.
) else (
    echo [BENCH] Benchmark executable not found.
    echo To enable benchmarks, rebuild with:
    echo   build_all.bat (benchmarks are enabled by default)
    echo.
)

pause
goto :advanced_options

REM =====================================================================
REM Test Hardware Backends
REM =====================================================================

:test_backends
echo.
echo [BACKEND] Testing available hardware acceleration backends...

echo [BACKEND] Available backends on your system:
echo.

REM CPU (always available)
echo   ✓ CPU Backend - Always available (default)

REM Test each backend if available
if %GPU_CUDA_AVAILABLE%==1 (
    echo   ✓ CUDA Backend - NVIDIA GPU acceleration
    if exist "%BUILD_DIR%\..\backends\cuda\Release\gemma_cuda_backend.lib" (
        echo     Status: Built and ready
    ) else (
        echo     Status: Not built (rebuild with CUDA enabled)
    )
)

if %GPU_INTEL_AVAILABLE%==1 (
    echo   ✓ SYCL Backend - Intel GPU/NPU acceleration
    if exist "%BUILD_DIR%\..\backends\sycl\Release\gemma_sycl_backend.lib" (
        echo     Status: Built and ready
    ) else (
        echo     Status: Not built (rebuild with SYCL enabled)
    )
)

if %GPU_VULKAN_AVAILABLE%==1 (
    echo   ✓ Vulkan Backend - Cross-platform GPU acceleration
    if exist "%BUILD_DIR%\..\backends\vulkan\Release\gemma_vulkan_backend.lib" (
        echo     Status: Built and ready
    ) else (
        echo     Status: Not built (rebuild with Vulkan enabled)
    )
)

if %GPU_OPENCL_AVAILABLE%==1 (
    echo   ✓ OpenCL Backend - Cross-platform parallel computing
    if exist "%BUILD_DIR%\..\backends\opencl\Release\gemma_opencl_backend.lib" (
        echo     Status: Built and ready
    ) else (
        echo     Status: Not built (rebuild with OpenCL enabled)
    )
)

echo.
echo [BACKEND] To rebuild with specific backends:
echo   set ENABLE_CUDA=true ^&^& build_all.bat
echo   set ENABLE_SYCL=true ^&^& build_all.bat
echo.

pause
goto :advanced_options

REM =====================================================================
REM Show Examples
REM =====================================================================

:show_examples
echo.
echo [EXAMPLES] Available example scripts and usage patterns:
echo.

if not exist "%EXAMPLES_DIR%" mkdir "%EXAMPLES_DIR%"

REM Create example scripts if they don't exist
if not exist "%EXAMPLES_DIR%\basic_inference.bat" (
    call :create_example_scripts
)

echo Available examples:
echo.
echo 1. Basic Inference:
echo    %EXAMPLES_DIR%\basic_inference.bat
echo    Simple text generation with custom prompts
echo.
echo 2. Batch Processing:
echo    %EXAMPLES_DIR%\batch_processing.bat
echo    Process multiple prompts from a file
echo.
echo 3. MCP Server Usage:
echo    %EXAMPLES_DIR%\mcp_example.bat
echo    Demonstrate MCP server capabilities
echo.
echo 4. Backend Comparison:
echo    %EXAMPLES_DIR%\backend_comparison.bat
echo    Compare performance across different backends
echo.

set /p RUN_EXAMPLE="Run an example? (1-4, or 'n' to skip): "
if "%RUN_EXAMPLE%"=="1" call "%EXAMPLES_DIR%\basic_inference.bat"
if "%RUN_EXAMPLE%"=="2" call "%EXAMPLES_DIR%\batch_processing.bat"
if "%RUN_EXAMPLE%"=="3" call "%EXAMPLES_DIR%\mcp_example.bat"
if "%RUN_EXAMPLE%"=="4" call "%EXAMPLES_DIR%\backend_comparison.bat"

goto :advanced_options

REM =====================================================================
REM Exit Script
REM =====================================================================

:exit_script
echo.
echo =====================================================================
echo Thank you for using Enhanced Gemma.cpp!
echo =====================================================================
echo.
echo Quick reference for future use:
echo.
echo Basic inference:
echo   gemma.exe --weights "%MODEL_FILE%" --tokenizer "%TOKENIZER_FILE%" --prompt "Your prompt here"
echo.
echo Available scripts:
echo   build_all.bat     - Build system
echo   test_all.bat      - Comprehensive testing
echo   deploy_windows.bat - Create deployment package
echo   docker_build.bat  - Create Docker images
echo.
echo Documentation:
echo   README.md         - General information
echo   DOCKER_USAGE.md   - Docker-specific usage
echo   CLAUDE.md         - Development guidelines
echo.
echo For support and more information:
echo   https://github.com/google/gemma.cpp
echo.

pause
exit /b 0

REM =====================================================================
REM Helper Functions
REM =====================================================================

:detect_system_capabilities
REM Detect NVIDIA GPU
set GPU_CUDA_AVAILABLE=0
nvidia-smi > nul 2>&1
if not errorlevel 1 set GPU_CUDA_AVAILABLE=1

REM Detect Intel GPU
set GPU_INTEL_AVAILABLE=0
if exist "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\windows\bin\icpx.exe" (
    set GPU_INTEL_AVAILABLE=1
)

REM Detect Vulkan
set GPU_VULKAN_AVAILABLE=0
where /q vulkaninfo > nul 2>&1
if not errorlevel 1 (
    set GPU_VULKAN_AVAILABLE=1
) else (
    if exist "C:\VulkanSDK\*\Bin\vulkaninfo.exe" set GPU_VULKAN_AVAILABLE=1
)

REM Detect OpenCL
set GPU_OPENCL_AVAILABLE=0
if exist "%SYSTEMROOT%\System32\OpenCL.dll" set GPU_OPENCL_AVAILABLE=1
goto :eof

:setup_model_files
echo.
echo [SETUP] Model files are required to run Enhanced Gemma.cpp.
echo.
echo You need to download Gemma model files from one of these sources:
echo.
echo 1. Kaggle (Recommended):
echo    https://www.kaggle.com/models/google/gemma-2/gemmaCpp
echo.
echo 2. Hugging Face:
echo    https://huggingface.co/google/gemma-2b
echo.
echo After downloading, place the files in: %MODELS_DIR%
echo.
echo Required files:
echo   - gemma2-2b-it-sfp.sbs (or similar .sbs model file)
echo   - tokenizer.spm (tokenizer model)
echo.

set /p MANUAL_SETUP="Have you already downloaded the model files? (y/n): "
if /i "%MANUAL_SETUP%"=="y" (
    echo.
    echo [SETUP] Please place your model files in: %MODELS_DIR%
    echo Then run this script again.
    echo.
    pause
    exit /b 1
)

echo.
echo [SETUP] Model download is not automated due to licensing requirements.
echo Please:
echo.
echo 1. Visit: https://www.kaggle.com/models/google/gemma-2/gemmaCpp
echo 2. Accept the license agreement
echo 3. Download the model files
echo 4. Extract to: %MODELS_DIR%
echo 5. Run this script again
echo.

explorer "%MODELS_DIR%"

echo [SETUP] Opening models directory in Explorer...
echo Place your downloaded model files there and run quick_start.bat again.
echo.
pause
exit /b 1

:create_example_scripts
echo [EXAMPLES] Creating example scripts...

REM Basic inference example
echo @echo off > "%EXAMPLES_DIR%\basic_inference.bat"
echo echo Running basic inference example... >> "%EXAMPLES_DIR%\basic_inference.bat"
echo echo. >> "%EXAMPLES_DIR%\basic_inference.bat"
echo set /p PROMPT="Enter your prompt: " >> "%EXAMPLES_DIR%\basic_inference.bat"
echo echo. >> "%EXAMPLES_DIR%\basic_inference.bat"
echo "%BUILD_DIR%\gemma.exe" --weights "%MODEL_FILE%" --tokenizer "%TOKENIZER_FILE%" --prompt "%%PROMPT%%" >> "%EXAMPLES_DIR%\basic_inference.bat"
echo pause >> "%EXAMPLES_DIR%\basic_inference.bat"

REM Batch processing example
echo @echo off > "%EXAMPLES_DIR%\batch_processing.bat"
echo echo Batch processing example... >> "%EXAMPLES_DIR%\batch_processing.bat"
echo echo Creating sample prompts file... >> "%EXAMPLES_DIR%\batch_processing.bat"
echo echo Tell me about artificial intelligence. > prompts.txt >> "%EXAMPLES_DIR%\batch_processing.bat"
echo echo What is machine learning? >> prompts.txt >> "%EXAMPLES_DIR%\batch_processing.bat"
echo echo Explain neural networks simply. >> prompts.txt >> "%EXAMPLES_DIR%\batch_processing.bat"
echo echo Processing prompts from prompts.txt... >> "%EXAMPLES_DIR%\batch_processing.bat"
echo for /f "delims=" %%%%i in (prompts.txt) do ( >> "%EXAMPLES_DIR%\batch_processing.bat"
echo   echo Prompt: %%%%i >> "%EXAMPLES_DIR%\batch_processing.bat"
echo   "%BUILD_DIR%\gemma.exe" --weights "%MODEL_FILE%" --tokenizer "%TOKENIZER_FILE%" --prompt "%%%%i" >> "%EXAMPLES_DIR%\batch_processing.bat"
echo   echo. >> "%EXAMPLES_DIR%\batch_processing.bat"
echo ^) >> "%EXAMPLES_DIR%\batch_processing.bat"
echo pause >> "%EXAMPLES_DIR%\batch_processing.bat"

REM MCP example
echo @echo off > "%EXAMPLES_DIR%\mcp_example.bat"
echo echo MCP Server example... >> "%EXAMPLES_DIR%\mcp_example.bat"
echo if exist "%BUILD_DIR%\gemma_mcp_stdio_server.exe" ( >> "%EXAMPLES_DIR%\mcp_example.bat"
echo   echo Starting MCP server... >> "%EXAMPLES_DIR%\mcp_example.bat"
echo   "%BUILD_DIR%\gemma_mcp_stdio_server.exe" --weights "%MODEL_FILE%" --tokenizer "%TOKENIZER_FILE%" >> "%EXAMPLES_DIR%\mcp_example.bat"
echo ^) else ( >> "%EXAMPLES_DIR%\mcp_example.bat"
echo   echo MCP server not available >> "%EXAMPLES_DIR%\mcp_example.bat"
echo ^) >> "%EXAMPLES_DIR%\mcp_example.bat"
echo pause >> "%EXAMPLES_DIR%\mcp_example.bat"

REM Backend comparison
echo @echo off > "%EXAMPLES_DIR%\backend_comparison.bat"
echo echo Backend performance comparison... >> "%EXAMPLES_DIR%\backend_comparison.bat"
echo echo This would compare different hardware backends if available. >> "%EXAMPLES_DIR%\backend_comparison.bat"
echo echo (Implementation depends on backend availability) >> "%EXAMPLES_DIR%\backend_comparison.bat"
echo pause >> "%EXAMPLES_DIR%\backend_comparison.bat"

echo [EXAMPLES] Example scripts created in %EXAMPLES_DIR%
goto :eof