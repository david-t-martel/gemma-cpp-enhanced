#!/bin/bash
# Shell script to run Gemma developer tool demos
# Requires Ollama to be installed

echo "========================================"
echo "Gemma Developer Tools Demo Runner"
echo "========================================"
echo

# Check if Ollama is available
if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama is not installed or not in PATH"
    echo "Please install from https://ollama.ai"
    echo
    echo "Installation:"
    echo "  curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# Check if Gemma model is available
if ! ollama list | grep -qi "gemma"; then
    echo "Gemma model not found. Pulling gemma2:2b..."
    ollama pull gemma2:2b
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to pull Gemma model"
        exit 1
    fi
fi

# Function to run a demo
run_demo() {
    local script=$1
    local name=$2
    echo
    echo "Running $name Demo..."
    echo "----------------------------------------"
    python3 "$script"
    echo
}

# Menu
while true; do
    echo
    echo "Select a demo to run:"
    echo "1. Code Review Automation"
    echo "2. Documentation Generator"
    echo "3. Test Case Generator"
    echo "4. Bug Triage and Analysis"
    echo "5. Refactoring Suggestions"
    echo "6. Run All Demos"
    echo "0. Exit"
    echo
    read -p "Enter your choice (0-6): " choice

    case $choice in
        1)
            run_demo "1_code_review.py" "Code Review"
            ;;
        2)
            run_demo "2_doc_generator.py" "Documentation Generator"
            ;;
        3)
            run_demo "3_test_generator.py" "Test Generator"
            ;;
        4)
            run_demo "4_bug_triage.py" "Bug Triage"
            ;;
        5)
            run_demo "5_refactoring.py" "Refactoring Analysis"
            ;;
        6)
            echo
            echo "Running All Demos..."
            echo "========================================"
            run_demo "1_code_review.py" "[1/5] Code Review"
            run_demo "2_doc_generator.py" "[2/5] Documentation Generator"
            run_demo "3_test_generator.py" "[3/5] Test Generator"
            run_demo "4_bug_triage.py" "[4/5] Bug Triage"
            run_demo "5_refactoring.py" "[5/5] Refactoring Analysis"
            echo "All demos completed!"
            ;;
        0)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice. Please select 0-6."
            ;;
    esac

    echo
    read -p "Press Enter to continue..."
done