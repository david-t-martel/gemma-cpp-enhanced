#!/bin/bash
# Test build script for Enhanced Gemma CLI (Linux/WSL)

set -e

echo "Testing Enhanced Gemma CLI build..."

# Check if we're in the correct directory
if [ ! -f "main.cpp" ]; then
    echo "Error: main.cpp not found. Please run from tools/cli directory."
    exit 1
fi

# Check if parent gemma.cpp exists
if [ ! -d "../../gemma.cpp" ]; then
    echo "Error: gemma.cpp directory not found. Please ensure project structure is correct."
    exit 1
fi

# Create build directory
mkdir -p build
cd build

echo "Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo "Building project..."
make -j$(nproc)

echo "Build successful!"
echo
echo "To test the CLI:"
echo "1. Ensure you have model files in /c/codedev/llm/.models/"
echo "2. Run: ./gemma_cli --help"
echo

cd ..