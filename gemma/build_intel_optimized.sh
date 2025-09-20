#!/bin/bash

# Intel oneAPI Optimized Build Script for Gemma.cpp
# This script configures and builds Gemma.cpp with Intel compiler optimizations

set -e  # Exit on any error

echo "==================================================================="
echo "Intel oneAPI Optimized Build for Gemma.cpp"
echo "==================================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Intel oneAPI availability
print_status "Checking Intel oneAPI availability..."

if [[ -f "/c/Program Files (x86)/Intel/oneAPI/setvars.bat" ]]; then
    print_success "Intel oneAPI found at standard Windows location"
    INTEL_ONEAPI_ROOT="/c/Program Files (x86)/Intel/oneAPI"
elif [[ -f "/opt/intel/oneapi/setvars.sh" ]]; then
    print_success "Intel oneAPI found at standard Linux location"
    INTEL_ONEAPI_ROOT="/opt/intel/oneapi"
    source "$INTEL_ONEAPI_ROOT/setvars.sh" intel64
else
    print_warning "Intel oneAPI not found at standard locations"
    print_status "Using Intel wrapper scripts..."
fi

# Set compiler paths
if [[ -f "C:/users/david/.local/bin/intel-icx.cmd" ]]; then
    export CC="C:/users/david/.local/bin/intel-icx.cmd"
    export CXX="C:/users/david/.local/bin/intel-icpx.cmd"
    print_success "Using Intel wrapper scripts"
elif command -v icx >/dev/null 2>&1; then
    export CC="icx"
    export CXX="icpx"
    print_success "Using Intel compilers from PATH"
else
    print_error "Intel compilers not found!"
    print_error "Please install Intel oneAPI toolkit or check installation"
    exit 1
fi

print_status "Compiler configuration:"
echo "  CC=$CC"
echo "  CXX=$CXX"
echo ""

# Create clean build directory
print_status "Setting up build directory..."
if [[ -d "build-intel-optimized" ]]; then
    rm -rf build-intel-optimized
fi
mkdir -p build-intel-optimized
cd build-intel-optimized

# Intel-specific optimization flags
INTEL_OPT_FLAGS="-O3 -xHost -march=native -mtune=native -ffast-math -fopenmp"
INTEL_MATH_FLAGS="-mkl=parallel -ipp=parallel"
INTEL_SIMD_FLAGS="-msse4.2 -mavx2 -mfma"

print_status "Intel optimization flags:"
echo "  Base: $INTEL_OPT_FLAGS"
echo "  Math: $INTEL_MATH_FLAGS"
echo "  SIMD: $INTEL_SIMD_FLAGS"
echo ""

# Configure with CMake
print_status "Configuring Intel optimized build..."
echo "==================================================================="

CMAKE_CMD="/c/Program\ Files/CMake/bin/cmake"
if [[ ! -f "$CMAKE_CMD" ]]; then
    CMAKE_CMD="cmake"
fi

$CMAKE_CMD \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_C_FLAGS="$INTEL_OPT_FLAGS $INTEL_SIMD_FLAGS" \
    -DCMAKE_CXX_FLAGS="$INTEL_OPT_FLAGS $INTEL_SIMD_FLAGS" \
    -DGEMMA_BUILD_SYCL_BACKEND=ON \
    -DGEMMA_BUILD_CUDA_BACKEND=ON \
    -DGEMMA_BUILD_BENCHMARKS=ON \
    -DGEMMA_ENABLE_LTO=ON \
    -DGEMMA_ENABLE_PCH=ON \
    -DGEMMA_ENABLE_UNITY_BUILDS=OFF \
    ..

if [[ $? -ne 0 ]]; then
    print_error "CMake configuration failed!"
    exit 1
fi

print_success "Configuration completed successfully"
echo ""

# Build the project
print_status "Building Intel optimized Gemma.cpp..."
echo "==================================================================="

NUM_CORES=$(nproc 2>/dev/null || echo 4)
print_status "Using $NUM_CORES parallel jobs"

$CMAKE_CMD --build . --config Release --parallel $NUM_CORES

if [[ $? -ne 0 ]]; then
    print_error "Build failed!"
    exit 1
fi

print_success "Build completed successfully!"
echo ""

# List built executables
print_status "Built executables:"
if [[ -d "Release" ]]; then
    ls -la Release/*.exe 2>/dev/null || echo "  No executables found in Release/"
else
    find . -name "*.exe" -o -name "gemma" -o -name "*benchmark*" | head -10
fi

echo ""
print_success "Intel optimized build complete!"
echo ""
print_status "To run benchmarks on 2B model:"
echo "  ./Release/single_benchmark.exe --weights '/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs' --tokenizer '/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/tokenizer.spm'"
echo ""
print_status "To run benchmarks on 4B model:"
echo "  ./Release/single_benchmark.exe --weights '/c/codedev/llm/.models/gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/4b-it-sfp.sbs' --tokenizer '/c/codedev/llm/.models/gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/tokenizer.spm'"
echo ""

# Check for Intel MKL and IPP availability
if [[ -n "$MKLROOT" ]]; then
    print_success "Intel MKL detected: $MKLROOT"
else
    print_warning "Intel MKL not detected in environment"
fi

if [[ -n "$IPPROOT" ]]; then
    print_success "Intel IPP detected: $IPPROOT"
else
    print_warning "Intel IPP not detected in environment"
fi

print_status "Build artifacts available in: $(pwd)"