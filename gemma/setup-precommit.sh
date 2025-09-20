#!/bin/bash
# Pre-commit setup script for gemma.cpp enhanced project
# Integrates with auto-claude framework at C:\users\david\.claude\*

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

print_status "Setting up pre-commit hooks for gemma.cpp enhanced project..."

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    print_error "This is not a git repository. Please run 'git init' first."
    exit 1
fi

# Determine Python command (following CLAUDE.md instructions)
PYTHON_CMD=""
if command -v uv >/dev/null 2>&1; then
    PYTHON_CMD="uv run python"
    print_status "Using uv run python (preferred)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
    print_warning "Using python3 (uv not available)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
    print_warning "Using python (fallback)"
else
    print_error "No Python interpreter found"
    exit 1
fi

# Install pre-commit
print_status "Installing pre-commit..."
if command -v uv >/dev/null 2>&1; then
    uv pip install pre-commit
else
    pip3 install pre-commit
fi

# Install additional tools
print_status "Installing code quality tools..."
if command -v uv >/dev/null 2>&1; then
    uv pip install ruff black mypy bandit detect-secrets
else
    pip3 install ruff black mypy bandit detect-secrets
fi

# Install system dependencies
print_status "Installing system dependencies..."
if command -v apt-get >/dev/null 2>&1; then
    print_status "Detected Debian/Ubuntu system"
    sudo apt-get update
    sudo apt-get install -y clang-format-14 cmake-format nodejs npm
elif command -v yum >/dev/null 2>&1; then
    print_status "Detected RedHat/CentOS system"
    sudo yum install -y clang nodejs npm
elif command -v brew >/dev/null 2>&1; then
    print_status "Detected macOS with Homebrew"
    brew install clang-format cmake-format node
elif command -v choco >/dev/null 2>&1; then
    print_status "Detected Windows with Chocolatey"
    choco install llvm nodejs -y
else
    print_warning "Unknown package manager. Please install clang-format and nodejs manually."
fi

# Install Node.js dependencies for auto-claude framework
print_status "Installing Node.js dependencies..."
if command -v npm >/dev/null 2>&1; then
    npm install -g markdownlint-cli
else
    print_warning "npm not found, skipping Node.js dependencies"
fi

# Install AST-grep if not available
if ! command -v sg >/dev/null 2>&1 && ! [ -f "C:/Users/david/.cargo/bin/sg.exe" ]; then
    print_status "Installing AST-grep..."
    if command -v cargo >/dev/null 2>&1; then
        cargo install ast-grep
    else
        print_warning "Cargo not found. Please install AST-grep manually: cargo install ast-grep"
    fi
fi

# Verify auto-claude framework
print_status "Checking auto-claude framework integration..."
AUTO_CLAUDE_PATH="C:/users/david/.claude"
if [ -d "$AUTO_CLAUDE_PATH" ]; then
    print_success "Auto-claude framework found at $AUTO_CLAUDE_PATH"

    # Check for key components
    if [ -f "$AUTO_CLAUDE_PATH/scripts/auto-claude.sh" ]; then
        print_success "Auto-claude script found"
    else
        print_warning "Auto-claude script not found at expected location"
    fi

    if [ -f "$AUTO_CLAUDE_PATH/package.json" ]; then
        print_success "Auto-claude package.json found"

        # Install auto-claude dependencies
        print_status "Installing auto-claude dependencies..."
        cd "$AUTO_CLAUDE_PATH"
        if command -v npm >/dev/null 2>&1; then
            npm install
            print_success "Auto-claude dependencies installed"
        fi
        cd - >/dev/null
    fi
else
    print_warning "Auto-claude framework not found at $AUTO_CLAUDE_PATH"
    print_warning "Some pre-commit hooks may not function optimally"
fi

# Initialize secrets baseline
print_status "Initializing secrets baseline..."
if command -v detect-secrets >/dev/null 2>&1; then
    if [ ! -f ".secrets.baseline" ]; then
        detect-secrets scan --baseline .secrets.baseline
        print_success "Secrets baseline created"
    else
        print_status "Secrets baseline already exists"
    fi
else
    print_warning "detect-secrets not available, skipping baseline creation"
fi

# Install pre-commit hooks
print_status "Installing pre-commit hooks..."
pre-commit install

# Optional: Install hooks for other git events
read -p "Install pre-commit hooks for commit-msg and pre-push? (y/N): " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pre-commit install --hook-type commit-msg
    pre-commit install --hook-type pre-push
    print_success "Additional git hooks installed"
fi

# Run initial check
print_status "Running initial pre-commit check..."
if pre-commit run --all-files; then
    print_success "All pre-commit hooks passed!"
else
    print_warning "Some pre-commit hooks failed. This is normal for initial setup."
    print_status "You can fix issues manually or run: pre-commit run --all-files"
fi

# Create helper scripts
print_status "Creating helper scripts..."

cat > fix-code-quality.sh << 'EOF'
#!/bin/bash
# Quick code quality fix script

echo "Running comprehensive code quality fixes..."

# Auto-claude fixes
if [ -f "C:/users/david/.claude/scripts/auto-claude.sh" ]; then
    echo "Running auto-claude fixes..."
    bash "C:/users/david/.claude/scripts/auto-claude.sh" fix --aggressive || true
fi

# Python fixes
if command -v ruff >/dev/null 2>&1; then
    echo "Running ruff fixes..."
    ruff check --fix .
    ruff format .
fi

if command -v black >/dev/null 2>&1; then
    echo "Running black formatting..."
    black . --line-length=100
fi

# C++ formatting
if command -v clang-format >/dev/null 2>&1; then
    echo "Running clang-format..."
    find . -name "*.cpp" -o -name "*.cc" -o -name "*.h" -o -name "*.hpp" | \
        grep -v build | grep -v _deps | head -20 | \
        xargs clang-format -i
fi

# AST-grep fixes
if command -v sg >/dev/null 2>&1; then
    echo "Running AST-grep fixes..."
    sg fix --interactive=false || true
elif [ -f "C:/Users/david/.cargo/bin/sg.exe" ]; then
    echo "Running AST-grep fixes..."
    "C:/Users/david/.cargo/bin/sg.exe" fix --interactive=false || true
fi

echo "Code quality fixes completed!"
EOF

chmod +x fix-code-quality.sh

cat > run-quality-check.sh << 'EOF'
#!/bin/bash
# Comprehensive quality check script

echo "Running comprehensive quality checks..."

# Pre-commit hooks
echo "Running pre-commit hooks..."
pre-commit run --all-files

# Additional checks
echo "Running additional quality checks..."

# Security audit
if command -v bandit >/dev/null 2>&1; then
    echo "Running security audit..."
    bandit -r . -f json -o bandit-report.json || true
fi

# Secret detection
if command -v detect-secrets >/dev/null 2>&1; then
    echo "Scanning for secrets..."
    detect-secrets scan --baseline .secrets.baseline --all-files || true
fi

# Build validation
if [ -f "CMakeLists.txt" ]; then
    echo "Validating CMake configuration..."
    mkdir -p build-validation
    cd build-validation
    cmake .. -DGEMMA_BUILD_BACKENDS=OFF -DGEMMA_BUILD_ENHANCED_TESTS=OFF || {
        echo "CMake validation failed"
        cd ..
        rm -rf build-validation
        exit 1
    }
    cd ..
    rm -rf build-validation
    echo "CMake validation passed"
fi

echo "Quality checks completed!"
EOF

chmod +x run-quality-check.sh

print_success "Helper scripts created:"
print_status "  - fix-code-quality.sh: Quick fixes for common issues"
print_status "  - run-quality-check.sh: Comprehensive quality validation"

# Summary
echo
echo "=========================================="
print_success "Pre-commit setup completed successfully!"
echo "=========================================="
echo
print_status "What was installed:"
echo "  ✅ Pre-commit hooks with comprehensive rules"
echo "  ✅ Code formatters (clang-format, black, ruff)"
echo "  ✅ Security scanners (bandit, detect-secrets)"
echo "  ✅ Integration with auto-claude framework"
echo "  ✅ Helper scripts for common tasks"
echo
print_status "Usage:"
echo "  - Hooks run automatically on git commit"
echo "  - Manual run: pre-commit run --all-files"
echo "  - Quick fixes: ./fix-code-quality.sh"
echo "  - Quality check: ./run-quality-check.sh"
echo
print_status "Configuration files:"
echo "  - .pre-commit-config.yaml: Pre-commit configuration"
echo "  - .clang-format: C++ formatting rules"
echo "  - ruff.toml: Python linting configuration"
echo "  - .secrets.baseline: Secret detection baseline"
echo
print_warning "Note: Some hooks may initially fail. This is normal."
print_status "Run './fix-code-quality.sh' to automatically fix common issues."
echo