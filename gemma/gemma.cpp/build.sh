#!/bin/bash
# Build script for gemma.cpp on Windows/WSL

cd "$(dirname "$0")"

# Clean build directory
rm -rf build_clean
mkdir -p build_clean
cd build_clean

echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc

echo "Building gemma..."
make -j$(nproc) gemma

if [ -f gemma ]; then
    echo "Build successful! Binary at: $(pwd)/gemma"
else
    echo "Build failed!"
    exit 1
fi