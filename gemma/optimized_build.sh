#!/bin/bash
# Optimized Build Script for Gemma.cpp
# Incorporates ccache, precompiled headers, and intelligent build strategies

set -e  # Exit on any error

echo "============================================="
echo "Gemma.cpp Optimized Build System"
echo "============================================="
echo

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
BUILD_JOBS=$(nproc 2>/dev/null || echo 4)
CMAKE_GENERATOR="Visual Studio 17 2022"
CMAKE_TOOLSET="v143"

# Set CMake path for Windows/WSL
export PATH="/c/Program Files/CMake/bin:$PATH"

# ccache configuration
if command -v ccache &> /dev/null; then
    echo "✓ Configuring ccache for optimal performance..."
    export CCACHE_DIR="$HOME/.ccache"
    export CCACHE_MAXSIZE="5G"
    export CCACHE_COMPRESS="true"
    export CCACHE_COMPRESSLEVEL="6"
    export CCACHE_SLOPPINESS="pch_defines,time_macros"
    export CCACHE_NOHASHDIR="true"
    export CCACHE_BASEDIR="$PROJECT_ROOT"

    # Create ccache directory
    mkdir -p "$CCACHE_DIR"

    # Zero stats for this build
    ccache --zero-stats >/dev/null 2>&1 || true

    echo "  • Cache directory: $CCACHE_DIR"
    echo "  • Max size: 5GB"
    echo "  • Compression: enabled (level 6)"
    echo
else
    echo "⚠ ccache not found. Install for ~50% faster rebuilds:"
    echo "  Windows: scoop install ccache"
    echo "  WSL/Linux: sudo apt install ccache"
    echo
fi

# Check if CMake is available
if ! command -v cmake &> /dev/null; then
    echo "❌ ERROR: CMake not found"
    echo "Please install CMake 3.16+ or update PATH"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | sed 's/cmake version //')
echo "✓ CMake version: $CMAKE_VERSION"

# Check for required CMake version
CMAKE_MAJOR=$(echo "$CMAKE_VERSION" | cut -d. -f1)
CMAKE_MINOR=$(echo "$CMAKE_VERSION" | cut -d. -f2)
if [ "$CMAKE_MAJOR" -lt 3 ] || ([ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 16 ]); then
    echo "⚠ WARNING: CMake 3.16+ recommended for optimal performance"
fi

echo

# Build type selection
if [ -z "$1" ]; then
    echo "Select build configuration:"
    echo "1. FastDebug    - Quick debug builds with minimal optimization (O1)"
    echo "2. Debug        - Full debug builds with symbols (O0)"
    echo "3. RelWithSymbols - Optimized builds with debug symbols (O2 + symbols)"
    echo "4. Release      - Full optimization, production builds (O3)"
    echo "5. Basic        - Original gemma.cpp without enhancements"
    echo
    read -p "Enter choice (1-5): " BUILD_CHOICE
else
    BUILD_CHOICE="$1"
fi

case $BUILD_CHOICE in
    1)
        BUILD_TYPE="FastDebug"
        BUILD_DESC="Fast Debug (O1 + symbols)"
        CMAKE_OPTS="-DGEMMA_BUILD_BACKENDS=OFF -DGEMMA_ENABLE_UNITY_BUILDS=ON"
        ;;
    2)
        BUILD_TYPE="Debug"
        BUILD_DESC="Full Debug (O0 + symbols)"
        CMAKE_OPTS="-DGEMMA_BUILD_BACKENDS=OFF -DGEMMA_ENABLE_UNITY_BUILDS=ON"
        ;;
    3)
        BUILD_TYPE="RelWithSymbols"
        BUILD_DESC="Optimized with Symbols (O2 + symbols)"
        CMAKE_OPTS="-DGEMMA_BUILD_BACKENDS=ON -DGEMMA_AUTO_DETECT_BACKENDS=ON"
        ;;
    4)
        BUILD_TYPE="Release"
        BUILD_DESC="Production Release (O3)"
        CMAKE_OPTS="-DGEMMA_BUILD_BACKENDS=ON -DGEMMA_AUTO_DETECT_BACKENDS=ON -DGEMMA_ENABLE_LTO=ON"
        ;;
    5)
        BUILD_TYPE="Release"
        BUILD_DESC="Basic Original Gemma.cpp"
        CMAKE_OPTS=""
        PROJECT_ROOT="$PROJECT_ROOT/gemma.cpp"
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "============================================="
echo "Building: $BUILD_DESC"
echo "Build Type: $BUILD_TYPE"
echo "Project Root: $PROJECT_ROOT"
echo "Parallel Jobs: $BUILD_JOBS"
echo "============================================="
echo

# Change to project directory
cd "$PROJECT_ROOT"

# Build directory
BUILD_DIR="build-optimized-$(echo $BUILD_TYPE | tr '[:upper:]' '[:lower:]')"

# Check for incremental build
if [ -d "$BUILD_DIR" ]; then
    echo "🔄 Found existing build directory: $BUILD_DIR"
    read -p "Use incremental build? (Y/n): " USE_INCREMENTAL
    if [[ "$USE_INCREMENTAL" =~ ^[Nn] ]]; then
        echo "🗑️  Cleaning build directory..."
        rm -rf "$BUILD_DIR"
    else
        echo "✓ Using incremental build"
    fi
fi

# Configuration phase
echo
echo "⚙️  Configuring build..."
echo "CMAKE_OPTS: $CMAKE_OPTS"

# Time the configuration
CONFIG_START=$(date +%s)

cmake -B "$BUILD_DIR" \
    -G "$CMAKE_GENERATOR" \
    -T "$CMAKE_TOOLSET" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DGEMMA_ENABLE_PCH=ON \
    -DGEMMA_ENABLE_ENHANCED_TESTS=ON \
    $CMAKE_OPTS \
    -DCMAKE_POLICY_DEFAULT_CMP0077=NEW \
    -DCMAKE_POLICY_DEFAULT_CMP0048=NEW

CONFIG_END=$(date +%s)
CONFIG_TIME=$((CONFIG_END - CONFIG_START))

if [ $? -ne 0 ]; then
    echo "❌ Configuration failed!"
    exit 1
fi

echo "✓ Configuration complete in ${CONFIG_TIME}s"
echo

# Build phase
echo "🔨 Building..."
echo "Using $BUILD_JOBS parallel jobs"

BUILD_START=$(date +%s)

cmake --build "$BUILD_DIR" \
    --config "$BUILD_TYPE" \
    --parallel "$BUILD_JOBS" \
    --verbose

BUILD_END=$(date +%s)
BUILD_TIME=$((BUILD_END - BUILD_START))

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"

    # Show ccache stats if available
    if command -v ccache &> /dev/null; then
        echo
        echo "📊 ccache statistics:"
        ccache --show-stats || true
    fi

    exit 1
fi

echo
echo "✅ Build completed successfully!"
echo "⏱️  Build time: ${BUILD_TIME}s (Configuration: ${CONFIG_TIME}s)"

# Show ccache statistics
if command -v ccache &> /dev/null; then
    echo
    echo "📊 ccache statistics:"
    ccache --show-stats
fi

echo
echo "============================================="
echo "Build Summary"
echo "============================================="
echo "Build Type: $BUILD_DESC"
echo "Build Directory: $BUILD_DIR"
echo "Total Time: $((BUILD_TIME + CONFIG_TIME))s"

# Find and show executable paths
GEMMA_EXE=""
if [ "$BUILD_CHOICE" = "5" ]; then
    # Original gemma.cpp build
    GEMMA_EXE="$BUILD_DIR/$BUILD_TYPE/gemma.exe"
else
    # Enhanced build
    GEMMA_EXE="$BUILD_DIR/gemma.cpp/$BUILD_TYPE/gemma.exe"
fi

if [ -f "$GEMMA_EXE" ]; then
    echo "Executable: $GEMMA_EXE"

    # Show file size
    if command -v stat &> /dev/null; then
        FILE_SIZE=$(stat -c%s "$GEMMA_EXE" 2>/dev/null || stat -f%z "$GEMMA_EXE" 2>/dev/null || echo "unknown")
        if [ "$FILE_SIZE" != "unknown" ]; then
            FILE_SIZE_MB=$((FILE_SIZE / 1024 / 1024))
            echo "Size: ${FILE_SIZE_MB}MB"
        fi
    fi
else
    echo "⚠ Executable not found at expected location"
    echo "Check build directory: $BUILD_DIR"
fi

# Show other useful executables
if [ "$BUILD_CHOICE" != "5" ]; then
    echo
    echo "Additional executables (if built):"
    for exe in single_benchmark benchmarks debug_prompt; do
        EXE_PATH="$BUILD_DIR/gemma.cpp/$BUILD_TYPE/${exe}.exe"
        if [ -f "$EXE_PATH" ]; then
            echo "  • $exe: $EXE_PATH"
        fi
    done
fi

echo "============================================="
echo

# Next steps
echo "🚀 Next steps:"
echo "1. Copy model files to /c/codedev/llm/.models/"
echo "2. Test with: \"$GEMMA_EXE\" --weights /c/codedev/llm/.models/model.sbs --prompt \"Hello\""

if [ "$BUILD_CHOICE" != "5" ]; then
    echo "3. Run tests: cd \"$BUILD_DIR\" && ctest --output-on-failure"
    echo "4. See enhanced features in BUILD_DEPLOY_SOLUTION.md"
fi

echo

# Optimization tips
if [ "$BUILD_TIME" -gt 300 ]; then  # 5 minutes
    echo "💡 Build optimization tips:"
    echo "• Enable unity builds: -DGEMMA_ENABLE_UNITY_BUILDS=ON"
    echo "• Install ccache for faster rebuilds"
    echo "• Use Release builds for production"
    echo "• Consider selective backend building"
    echo
fi

# Save build info for future incremental builds
BUILD_INFO_FILE="$BUILD_DIR/build_info.txt"
cat > "$BUILD_INFO_FILE" << EOF
Build Type: $BUILD_TYPE
Build Description: $BUILD_DESC
Configuration Time: ${CONFIG_TIME}s
Build Time: ${BUILD_TIME}s
Total Time: $((BUILD_TIME + CONFIG_TIME))s
Built On: $(date)
CMAKE Options: $CMAKE_OPTS
Parallel Jobs: $BUILD_JOBS
EOF

echo "Build information saved to: $BUILD_INFO_FILE"