#!/bin/bash
# Setup script for terminal UI test automation

set -e

echo "Setting up Terminal UI Test Automation..."

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r playwright_requirements.txt

# Optional: Install asciinema
if ! command -v asciinema &> /dev/null; then
    echo ""
    echo "⚠ asciinema not found (optional for terminal recording)"
    echo "  Install with:"
    echo "    Linux: sudo apt install asciinema"
    echo "    macOS: brew install asciinema"
    echo "    Windows: scoop install asciinema"
fi

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p screenshots videos recordings snapshots
echo "✓ Directories created"

# Test installation
echo ""
echo "Testing installation..."
python -c "import pytest; import pyte; import rich; print('✓ All core dependencies available')"

echo ""
echo "✓ Setup complete!"
echo ""
echo "Run tests with:"
echo "  pytest tests/playwright/ -v"
echo "  python run_tests.py"
echo ""
echo "See README.md for more usage examples."
