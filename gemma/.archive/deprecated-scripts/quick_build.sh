#!/bin/bash
# Quick Build Script for Gemma.cpp
# This script provides multiple build options for different use cases

echo "============================================"
echo "Gemma.cpp Quick Build Script"
echo "============================================"
echo

# Set CMake path for Windows/WSL
export PATH="/c/Program Files/CMake/bin:$PATH"

# Check if CMake is available
if ! command -v cmake &> /dev/null; then
    echo "ERROR: CMake not found"
    echo "Please install CMake 4.1.1 or update PATH"
    exit 1
fi

echo "CMake version: $(cmake --version | head -n1)"
echo

echo "Select build option:"
echo "1. Original Gemma.cpp (Basic - Recommended for first time)"
echo "2. Enhanced Project without backends (MCP server, tests)"
echo "3. Enhanced Project with auto-detected backends (requires SDKs)"
echo
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo
        echo "============================================"
        echo "Building Original Gemma.cpp (Basic)"
        echo "============================================"
        cd gemma.cpp || exit 1
        cmake -B build-quick \
          -G "Visual Studio 17 2022" \
          -T v143 \
          -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
          -DCMAKE_BUILD_TYPE=Release
        if [ $? -ne 0 ]; then
            echo "Configuration failed!"
            exit 1
        fi
        echo
        echo "Configuration complete. Building..."
        cmake --build build-quick --config Release -j 4
        if [ $? -eq 0 ]; then
            echo
            echo "============================================"
            echo "Build Complete!"
            echo "Executable: gemma.cpp/build-quick/Release/gemma.exe"
            echo "============================================"
        else
            echo "Build failed!"
            exit 1
        fi
        ;;
    2)
        echo
        echo "============================================"
        echo "Building Enhanced Project (No Backends)"
        echo "============================================"
        cmake -B build-enhanced-quick \
          -G "Visual Studio 17 2022" \
          -T v143 \
          -DGEMMA_BUILD_BACKENDS=OFF \
          -DGEMMA_BUILD_MCP_SERVER=ON \
          -DGEMMA_BUILD_ENHANCED_TESTS=ON \
          -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        if [ $? -ne 0 ]; then
            echo "Configuration failed!"
            exit 1
        fi
        echo
        echo "Configuration complete. Building..."
        cmake --build build-enhanced-quick --config Release -j 4
        if [ $? -eq 0 ]; then
            echo
            echo "============================================"
            echo "Build Complete!"
            echo "Core Executable: build-enhanced-quick/gemma.cpp/Release/gemma.exe"
            echo "============================================"
        else
            echo "Build failed!"
            exit 1
        fi
        ;;
    3)
        echo
        echo "============================================"
        echo "Building Enhanced Project (With Backends)"
        echo "============================================"
        echo "WARNING: This requires hardware SDKs to be installed:"
        echo "- Intel oneAPI (for SYCL)"
        echo "- CUDA Toolkit (for NVIDIA GPUs)"
        echo "- Vulkan SDK (for cross-platform GPU)"
        echo
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        cmake -B build-accelerated-quick \
          -G "Visual Studio 17 2022" \
          -T v143 \
          -DGEMMA_BUILD_BACKENDS=ON \
          -DGEMMA_AUTO_DETECT_BACKENDS=ON \
          -DGEMMA_BUILD_MCP_SERVER=ON \
          -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        if [ $? -ne 0 ]; then
            echo "Configuration failed!"
            exit 1
        fi
        echo
        echo "Configuration complete. Building..."
        cmake --build build-accelerated-quick --config Release -j 4
        if [ $? -eq 0 ]; then
            echo
            echo "============================================"
            echo "Build Complete!"
            echo "Core Executable: build-accelerated-quick/gemma.cpp/Release/gemma.exe"
            echo "============================================"
        else
            echo "Build failed!"
            exit 1
        fi
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo
echo "Next steps:"
echo "1. Copy model files to /c/codedev/llm/.models/"
echo "2. Test with: ./gemma.exe --weights /c/codedev/llm/.models/model.sbs --prompt \"Hello\""
echo "3. See BUILD_DEPLOY_SOLUTION.md for detailed usage instructions"
echo