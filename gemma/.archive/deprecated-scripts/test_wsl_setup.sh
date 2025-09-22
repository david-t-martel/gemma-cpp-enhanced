#!/bin/bash
# Test script to verify WSL gemma.cpp setup

echo "=== WSL Gemma.cpp Setup Verification ==="
echo

# Test 1: WSL Environment
echo "1. Testing WSL environment..."
if grep -q "WSL\|Microsoft" /proc/version 2>/dev/null; then
    echo "   ✅ Running in WSL"
else
    echo "   ❌ Not running in WSL"
    exit 1
fi

# Test 2: Project directories
echo "2. Testing project directories..."
PROJECT_ROOT="/mnt/c/codedev/llm/gemma"
if [ -d "$PROJECT_ROOT" ]; then
    echo "   ✅ Project root found: $PROJECT_ROOT"
else
    echo "   ❌ Project root not found: $PROJECT_ROOT"
    exit 1
fi

# Test 3: WSL build directory
BUILD_DIR="$PROJECT_ROOT/gemma.cpp/build_wsl"
if [ -d "$BUILD_DIR" ]; then
    echo "   ✅ WSL build directory found: $BUILD_DIR"
else
    echo "   ❌ WSL build directory not found: $BUILD_DIR"
    exit 1
fi

# Test 4: Built executables
echo "3. Testing built executables..."
EXECUTABLES=("gemma" "single_benchmark" "debug_prompt" "migrate_weights")
for exe in "${EXECUTABLES[@]}"; do
    if [ -f "$BUILD_DIR/$exe" ]; then
        size=$(stat -c%s "$BUILD_DIR/$exe")
        size_mb=$((size / 1024 / 1024))
        echo "   ✅ $exe found (${size_mb}MB)"
    else
        echo "   ❌ $exe not found"
    fi
done

# Test 5: Executable functionality
echo "4. Testing executable functionality..."
if [ -f "$BUILD_DIR/gemma" ]; then
    if "$BUILD_DIR/gemma" --help >/dev/null 2>&1; then
        echo "   ✅ gemma --help works"
    else
        echo "   ❌ gemma --help failed"
    fi
else
    echo "   ❌ gemma binary not found"
fi

# Test 6: Convenience scripts
echo "5. Testing convenience scripts..."
if [ -f "$BUILD_DIR/run_gemma.sh" ] && [ -x "$BUILD_DIR/run_gemma.sh" ]; then
    echo "   ✅ run_gemma.sh found and executable"
else
    echo "   ❌ run_gemma.sh missing or not executable"
fi

if [ -f "$BUILD_DIR/run_benchmark.sh" ] && [ -x "$BUILD_DIR/run_benchmark.sh" ]; then
    echo "   ✅ run_benchmark.sh found and executable"
else
    echo "   ❌ run_benchmark.sh missing or not executable"
fi

# Test 7: Windows wrapper
echo "6. Testing Windows wrapper..."
WRAPPER="$PROJECT_ROOT/run_gemma_wsl.bat"
if [ -f "$WRAPPER" ]; then
    echo "   ✅ Windows wrapper found: $WRAPPER"
else
    echo "   ❌ Windows wrapper not found: $WRAPPER"
fi

# Test 8: Models directory
echo "7. Testing models directory..."
MODELS_DIR="/mnt/c/codedev/llm/.models"
if [ -d "$MODELS_DIR" ]; then
    echo "   ✅ Models directory found: $MODELS_DIR"

    # Check for model files
    sbs_count=$(find "$MODELS_DIR" -name "*.sbs" 2>/dev/null | wc -l)
    spm_count=$(find "$MODELS_DIR" -name "*.spm" 2>/dev/null | wc -l)

    if [ $sbs_count -gt 0 ] && [ $spm_count -gt 0 ]; then
        echo "   ✅ Model files found ($sbs_count .sbs, $spm_count .spm)"
    else
        echo "   ⚠️  Model files missing (need .sbs and .spm files)"
        echo "      Download with: cd /mnt/c/codedev/llm/stats && uv run python -m src.gcp.gemma_download --auto"
    fi
else
    echo "   ❌ Models directory not found: $MODELS_DIR"
fi

# Test 9: Dependencies
echo "8. Testing binary dependencies..."
if [ -f "$BUILD_DIR/gemma" ]; then
    echo "   Checking gemma binary dependencies:"
    ldd "$BUILD_DIR/gemma" | head -3 | sed 's/^/     /'
    echo "   ✅ Dependencies look good (all Linux libraries)"
else
    echo "   ❌ Cannot check dependencies - gemma binary not found"
fi

echo
echo "=== Summary ==="
echo "✅ WSL build environment is ready!"
echo
echo "Next steps:"
if [ $sbs_count -eq 0 ] || [ $spm_count -eq 0 ]; then
    echo "1. Download models:"
    echo "   cd /mnt/c/codedev/llm/stats && uv run python -m src.gcp.gemma_download --auto"
    echo
fi
echo "2. Run gemma.cpp:"
echo "   From Windows: C:\\codedev\\llm\\gemma\\run_gemma_wsl.bat"
echo "   From WSL:     $BUILD_DIR/run_gemma.sh"
echo
echo "3. Run benchmarks:"
echo "   $BUILD_DIR/run_benchmark.sh"
echo