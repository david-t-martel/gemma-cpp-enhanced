#!/bin/bash
# Comprehensive WSL Build Script for gemma.cpp
# Designed to work around Windows native build issues with Highway SIMD scalar fallbacks

set -e  # Exit on any error

# Color output functions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Path conversion between Windows and WSL
WINDOWS_PROJECT_ROOT="/mnt/c/codedev/llm/gemma"
WSL_PROJECT_ROOT="/mnt/c/codedev/llm/gemma"
GEMMA_CPP_DIR="${WSL_PROJECT_ROOT}/gemma.cpp"
WSL_BUILD_DIR="${GEMMA_CPP_DIR}/build_wsl_clean"
MODELS_DIR="/mnt/c/codedev/llm/.models"

log_info "Starting WSL build for gemma.cpp"
log_info "Project root: ${WSL_PROJECT_ROOT}"
log_info "Gemma.cpp source: ${GEMMA_CPP_DIR}"
log_info "Build directory: ${WSL_BUILD_DIR}"

# Check if we're in WSL
if [ ! -f /proc/version ] || ! grep -q "WSL\|Microsoft" /proc/version; then
    log_error "This script must be run in WSL (Windows Subsystem for Linux)"
    exit 1
fi

# Verify project directory exists
if [ ! -d "${GEMMA_CPP_DIR}" ]; then
    log_error "gemma.cpp directory not found: ${GEMMA_CPP_DIR}"
    exit 1
fi

log_success "WSL environment detected and project directory found"

# Check required tools
log_info "Checking required build tools..."

check_tool() {
    if command -v "$1" &> /dev/null; then
        log_success "$1 is available ($(which $1))"
        return 0
    else
        log_error "$1 is not available"
        return 1
    fi
}

MISSING_TOOLS=0

# Essential build tools
check_tool "cmake" || MISSING_TOOLS=$((MISSING_TOOLS + 1))
check_tool "make" || MISSING_TOOLS=$((MISSING_TOOLS + 1))
check_tool "g++" || MISSING_TOOLS=$((MISSING_TOOLS + 1))
check_tool "git" || MISSING_TOOLS=$((MISSING_TOOLS + 1))

if [ $MISSING_TOOLS -gt 0 ]; then
    log_warning "Missing $MISSING_TOOLS required tools. Installing..."
    sudo apt update
    sudo apt install -y build-essential cmake git ninja-build
    log_success "Build tools installed"
fi

# Check compiler versions
log_info "Compiler information:"
gcc --version | head -1
g++ --version | head -1
cmake --version | head -1

# Clean previous build if requested or if cache conflicts exist
if [ "$1" = "--clean" ] || [ "$1" = "-c" ]; then
    log_info "Cleaning previous build directory..."
    rm -rf "${WSL_BUILD_DIR}"
    log_success "Build directory cleaned"
fi

# Also clean any Windows build artifacts that might conflict
if [ -d "${GEMMA_CPP_DIR}/build" ]; then
    log_warning "Removing conflicting Windows build directory..."
    rm -rf "${GEMMA_CPP_DIR}/build"
fi

# Create build directory
log_info "Creating build directory: ${WSL_BUILD_DIR}"
mkdir -p "${WSL_BUILD_DIR}"
cd "${WSL_BUILD_DIR}"

# Configure CMake for Linux build
log_info "Configuring CMake for Linux build..."

# Always use manual configuration to avoid preset conflicts between Windows/WSL
log_info "Using manual CMake configuration (avoiding Windows/WSL path conflicts)..."
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    -G "Unix Makefiles" \
    "${GEMMA_CPP_DIR}" 2>&1 | tee cmake_configure.log

if [ $? -ne 0 ]; then
    log_error "CMake configuration failed. Check cmake_configure.log for details"
    exit 1
fi

log_success "CMake configuration completed"

# Build the project
log_info "Building gemma.cpp (this may take several minutes)..."

# Determine optimal number of jobs
NPROC=$(nproc)
OPTIMAL_JOBS=$((NPROC > 8 ? 8 : NPROC))
log_info "Using ${OPTIMAL_JOBS} parallel jobs (${NPROC} CPUs available)"

make -j${OPTIMAL_JOBS} 2>&1 | tee build.log

if [ $? -ne 0 ]; then
    log_error "Build failed. Check build.log for details"
    exit 1
fi

log_success "Build completed successfully!"

# Verify built executables
log_info "Verifying built executables..."

EXECUTABLES=("gemma" "single_benchmark" "debug_prompt" "migrate_weights" "benchmarks")
BUILT_EXECUTABLES=0

for exe in "${EXECUTABLES[@]}"; do
    if [ -f "${WSL_BUILD_DIR}/${exe}" ]; then
        SIZE=$(stat -c%s "${WSL_BUILD_DIR}/${exe}")
        SIZE_MB=$((SIZE / 1024 / 1024))
        log_success "${exe} built successfully (${SIZE_MB} MB)"
        BUILT_EXECUTABLES=$((BUILT_EXECUTABLES + 1))
    else
        log_warning "${exe} not found"
    fi
done

log_info "Built ${BUILT_EXECUTABLES}/${#EXECUTABLES[@]} executables"

# Test basic functionality
log_info "Testing basic functionality..."

# Check if models directory exists
if [ ! -d "${MODELS_DIR}" ]; then
    log_warning "Models directory not found: ${MODELS_DIR}"
    log_info "Download models to test inference"
else
    log_success "Models directory found: ${MODELS_DIR}"

    # List available models
    log_info "Available models:"
    ls -la "${MODELS_DIR}"/*.sbs 2>/dev/null || log_warning "No .sbs model files found"
    ls -la "${MODELS_DIR}"/*.spm 2>/dev/null || log_warning "No .spm tokenizer files found"
fi

# Test gemma executable
if [ -f "${WSL_BUILD_DIR}/gemma" ]; then
    log_info "Testing gemma executable..."
    "${WSL_BUILD_DIR}/gemma" --help &> /dev/null && log_success "gemma --help works" || log_warning "gemma --help failed"
fi

# Create convenience scripts
log_info "Creating convenience scripts..."

# Path translation helper
cat > "${WSL_BUILD_DIR}/run_gemma.sh" << 'EOF'
#!/bin/bash
# Convenience script to run gemma with proper paths

MODELS_DIR="/mnt/c/codedev/llm/.models"
GEMMA_BIN="$(dirname "$0")/gemma"

if [ ! -f "${GEMMA_BIN}" ]; then
    echo "Error: gemma binary not found: ${GEMMA_BIN}"
    exit 1
fi

if [ ! -d "${MODELS_DIR}" ]; then
    echo "Error: Models directory not found: ${MODELS_DIR}"
    echo "Please download models to ${MODELS_DIR}"
    exit 1
fi

# Default model files
DEFAULT_TOKENIZER="${MODELS_DIR}/tokenizer.spm"
DEFAULT_WEIGHTS="${MODELS_DIR}/gemma2-2b-it-sfp.sbs"

# Use single-file format if available
if [ -f "${MODELS_DIR}/gemma2-2b-it-sfp-single.sbs" ]; then
    echo "Using single-file format model..."
    exec "${GEMMA_BIN}" --weights "${MODELS_DIR}/gemma2-2b-it-sfp-single.sbs" "$@"
elif [ -f "${DEFAULT_WEIGHTS}" ] && [ -f "${DEFAULT_TOKENIZER}" ]; then
    echo "Using separate tokenizer and weights..."
    exec "${GEMMA_BIN}" --tokenizer "${DEFAULT_TOKENIZER}" --weights "${DEFAULT_WEIGHTS}" "$@"
else
    echo "Error: No suitable model files found in ${MODELS_DIR}"
    echo "Available files:"
    ls -la "${MODELS_DIR}" 2>/dev/null || echo "Directory does not exist"
    echo ""
    echo "Please download models from Kaggle or use the Python download script:"
    echo "cd /mnt/c/codedev/llm/stats && uv run python -m src.gcp.gemma_download --auto"
    exit 1
fi
EOF

chmod +x "${WSL_BUILD_DIR}/run_gemma.sh"

# Benchmark runner
cat > "${WSL_BUILD_DIR}/run_benchmark.sh" << 'EOF'
#!/bin/bash
# Convenience script to run benchmarks

MODELS_DIR="/mnt/c/codedev/llm/.models"
BENCHMARK_BIN="$(dirname "$0")/single_benchmark"

if [ ! -f "${BENCHMARK_BIN}" ]; then
    echo "Error: benchmark binary not found: ${BENCHMARK_BIN}"
    exit 1
fi

DEFAULT_TOKENIZER="${MODELS_DIR}/tokenizer.spm"
DEFAULT_WEIGHTS="${MODELS_DIR}/gemma2-2b-it-sfp.sbs"

if [ -f "${DEFAULT_WEIGHTS}" ] && [ -f "${DEFAULT_TOKENIZER}" ]; then
    echo "Running benchmark with gemma2-2b-it-sfp model..."
    exec "${BENCHMARK_BIN}" --weights "${DEFAULT_WEIGHTS}" --tokenizer "${DEFAULT_TOKENIZER}" "$@"
else
    echo "Error: Model files not found"
    echo "Expected: ${DEFAULT_WEIGHTS} and ${DEFAULT_TOKENIZER}"
    exit 1
fi
EOF

chmod +x "${WSL_BUILD_DIR}/run_benchmark.sh"

log_success "Convenience scripts created:"
log_info "  - ${WSL_BUILD_DIR}/run_gemma.sh"
log_info "  - ${WSL_BUILD_DIR}/run_benchmark.sh"

# Create Windows batch wrapper
cat > "${WSL_PROJECT_ROOT}/run_gemma_wsl.bat" << 'EOF'
@echo off
echo Starting gemma.cpp in WSL...
wsl -d Ubuntu bash -c "cd /mnt/c/codedev/llm/gemma/gemma.cpp/build_wsl_clean && ./run_gemma.sh %*"
EOF

log_success "Windows wrapper created: ${WSL_PROJECT_ROOT}/run_gemma_wsl.bat"

# Final summary
echo ""
log_success "=== WSL Build Complete ==="
log_info "Build directory: ${WSL_BUILD_DIR}"
log_info "Built executables: ${BUILT_EXECUTABLES}/${#EXECUTABLES[@]}"
echo ""
log_info "To run gemma.cpp from WSL:"
log_info "  cd ${WSL_BUILD_DIR}"
log_info "  ./run_gemma.sh"
echo ""
log_info "To run from Windows:"
log_info "  ${WSL_PROJECT_ROOT}/run_gemma_wsl.bat"
echo ""
log_info "To run benchmarks:"
log_info "  cd ${WSL_BUILD_DIR}"
log_info "  ./run_benchmark.sh"
echo ""

if [ ! -d "${MODELS_DIR}" ] || [ ! -f "${MODELS_DIR}/gemma2-2b-it-sfp.sbs" ]; then
    log_warning "No models found. Download models using:"
    log_info "  cd /mnt/c/codedev/llm/stats"
    log_info "  uv run python -m src.gcp.gemma_download --auto"
fi

log_success "WSL build setup complete!"