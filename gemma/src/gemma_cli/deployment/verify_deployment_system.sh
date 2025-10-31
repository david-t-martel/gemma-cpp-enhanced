#!/bin/bash
# Deployment System Verification Script
# Run this to verify all components are in place

echo "================================================"
echo " Gemma CLI Deployment System Verification"
echo "================================================"
echo ""

# Check files exist
echo "1. Checking deployment files..."
files=(
    "build_script.py"
    "README.md"
    "DEPLOYMENT_SYSTEM_REPORT.md"
    "CODE_MODIFICATIONS_REQUIRED.md"
)

all_present=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "  ✓ $file ($size)"
    else
        echo "  ✗ $file MISSING"
        all_present=false
    fi
done

echo ""
echo "2. Checking required binaries..."

# Check gemma.exe
gemma_paths=(
    "../../../../build-avx2-sycl/bin/RELEASE/gemma.exe"
    "../../../../build/Release/gemma.exe"
)

gemma_found=false
for path in "${gemma_paths[@]}"; do
    if [ -f "$path" ]; then
        size=$(du -h "$path" | cut -f1)
        echo "  ✓ gemma.exe found: $path ($size)"
        gemma_found=true
        break
    fi
done

if [ "$gemma_found" = false ]; then
    echo "  ✗ gemma.exe NOT FOUND"
    echo "    Build with: cd gemma.cpp && cmake --build build --config Release"
fi

# Check rag-redis-mcp-server.exe
rag_paths=(
    "/c/codedev/llm/stats/target/release/rag-redis-mcp-server.exe"
    "../../../../../../stats/target/release/rag-redis-mcp-server.exe"
)

rag_found=false
for path in "${rag_paths[@]}"; do
    if [ -f "$path" ]; then
        size=$(du -h "$path" | cut -f1)
        echo "  ✓ rag-redis-mcp-server.exe found: $path ($size)"
        rag_found=true
        break
    fi
done

if [ "$rag_found" = false ]; then
    echo "  ✗ rag-redis-mcp-server.exe NOT FOUND"
    echo "    Build with: cd stats/rag-redis-system && cargo build --release"
fi

echo ""
echo "3. Checking Python environment..."

if command -v python &> /dev/null; then
    python_version=$(python --version 2>&1)
    echo "  ✓ Python: $python_version"
else
    echo "  ✗ Python not found"
fi

if python -c "import PyInstaller" 2>/dev/null; then
    pyinstaller_version=$(python -m PyInstaller --version 2>&1 | head -1)
    echo "  ✓ PyInstaller: $pyinstaller_version"
else
    echo "  ⚠ PyInstaller not installed (optional for now)"
    echo "    Install with: pip install pyinstaller"
fi

echo ""
echo "4. Testing build script..."
if python build_script.py &> /dev/null; then
    echo "  ✓ build_script.py executes without errors"
else
    echo "  ⚠ build_script.py has execution issues (expected until complete)"
fi

echo ""
echo "================================================"
echo " Summary"
echo "================================================"

if [ "$all_present" = true ] && [ "$gemma_found" = true ] && [ "$rag_found" = true ]; then
    echo "✓ All deployment system components ready"
    echo ""
    echo "Next steps:"
    echo "  1. Review deployment/CODE_MODIFICATIONS_REQUIRED.md"
    echo "  2. Apply code changes to core/gemma.py and rag/rust_rag_client.py"
    echo "  3. Install PyInstaller: pip install pyinstaller"
    echo "  4. Complete build_script.py implementation"
    echo ""
    echo "Estimated time to working executable: 4-5 hours"
else
    echo "⚠ Some components missing or not ready"
    echo ""
    if [ "$gemma_found" = false ]; then
        echo "  - Build gemma.exe"
    fi
    if [ "$rag_found" = false ]; then
        echo "  - Build rag-redis-mcp-server.exe"
    fi
    if [ "$all_present" = false ]; then
        echo "  - Check documentation files"
    fi
fi

echo "================================================"
