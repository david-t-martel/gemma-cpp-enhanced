#!/usr/bin/env python3
"""
Code Review Automation with Gemma via Ollama

This script demonstrates using Gemma to automatically review C++ code
for common issues, performance problems, and style violations.
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

def run_ollama(prompt: str, model: str = "gemma2:2b") -> str:
    """Run Ollama with the specified prompt and model."""
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama: {e}")
        return ""
    except FileNotFoundError:
        print("Ollama is not installed. Please install from https://ollama.ai")
        sys.exit(1)

def review_cpp_code(code: str) -> Dict[str, List[str]]:
    """Review C++ code for various issues."""

    review_prompt = f"""You are a senior C++ developer reviewing code. Analyze this C++ code and provide:
1. Critical issues (bugs, memory leaks, undefined behavior)
2. Performance concerns
3. Style and readability improvements
4. Security vulnerabilities
5. Best practices violations

Format your response as JSON with these categories.

Code to review:
```cpp
{code}
```

Respond ONLY with valid JSON in this format:
{{
    "critical_issues": ["issue1", "issue2"],
    "performance": ["concern1", "concern2"],
    "style": ["improvement1", "improvement2"],
    "security": ["vulnerability1"],
    "best_practices": ["violation1"]
}}"""

    response = run_ollama(review_prompt)

    # Parse the JSON response
    try:
        # Extract JSON from response (sometimes models add text around it)
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            json_str = response[json_start:json_end]
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: parse text response into categories
    return parse_text_review(response)

def parse_text_review(text: str) -> Dict[str, List[str]]:
    """Parse text review into structured format."""
    result = {
        "critical_issues": [],
        "performance": [],
        "style": [],
        "security": [],
        "best_practices": []
    }

    lines = text.split('\n')
    current_category = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect category headers
        lower_line = line.lower()
        if 'critical' in lower_line or 'bug' in lower_line or 'issue' in lower_line:
            current_category = 'critical_issues'
        elif 'performance' in lower_line:
            current_category = 'performance'
        elif 'style' in lower_line or 'readability' in lower_line:
            current_category = 'style'
        elif 'security' in lower_line:
            current_category = 'security'
        elif 'best practice' in lower_line:
            current_category = 'best_practices'
        elif current_category and (line.startswith('-') or line.startswith('*') or line.startswith('•')):
            # Add item to current category
            item = line.lstrip('-*• ').strip()
            if item:
                result[current_category].append(item)

    return result

def format_review_output(review: Dict[str, List[str]], filename: str) -> str:
    """Format the review results for display."""
    output = []
    output.append(f"\n{'='*60}")
    output.append(f"Code Review Report for: {filename}")
    output.append(f"{'='*60}\n")

    severity_emoji = {
        "critical_issues": "🔴 CRITICAL",
        "security": "🔐 SECURITY",
        "performance": "⚡ PERFORMANCE",
        "best_practices": "📋 BEST PRACTICES",
        "style": "✨ STYLE"
    }

    has_issues = False
    for category, issues in review.items():
        if issues:
            has_issues = True
            output.append(f"\n{severity_emoji.get(category, category.upper())}:")
            for i, issue in enumerate(issues, 1):
                output.append(f"  {i}. {issue}")

    if not has_issues:
        output.append("✅ No issues found - code looks good!")

    output.append(f"\n{'='*60}\n")
    return '\n'.join(output)

def review_file(filepath: Path) -> None:
    """Review a C++ file."""
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return

    code = filepath.read_text(encoding='utf-8')
    print(f"Reviewing {filepath.name}...")

    review = review_cpp_code(code)
    print(format_review_output(review, filepath.name))

def main():
    """Main function with example usage."""

    # Example 1: Review sample problematic code
    sample_code = """
#include <iostream>
#include <vector>

class DataProcessor {
private:
    int* data;
    int size;

public:
    DataProcessor(int n) {
        size = n;
        data = new int[size];
    }

    void processData() {
        for(int i = 0; i <= size; i++) {  // Bug: buffer overflow
            data[i] = i * 2;
        }
    }

    int* getData() { return data; }  // Exposing internal pointer

    // Missing destructor - memory leak!
};

void unsafeFunction(char* input) {
    char buffer[10];
    strcpy(buffer, input);  // No bounds checking
    printf(buffer);         // Format string vulnerability
}

int main() {
    DataProcessor processor(100);
    processor.processData();

    int* ptr = processor.getData();
    delete[] ptr;  // Wrong: deleting what we don't own

    return 0;
}
"""

    # Create a temporary file for demonstration
    temp_file = Path("C:/codedev/llm/gemma/examples/temp_review.cpp")
    temp_file.write_text(sample_code)

    print("Gemma C++ Code Review Automation Demo")
    print("=" * 60)
    print("\nReviewing sample C++ code with known issues...\n")

    review = review_cpp_code(sample_code)
    print(format_review_output(review, "sample_code.cpp"))

    # Clean up
    temp_file.unlink(missing_ok=True)

    # Example 2: Review actual gemma.cpp files if they exist
    gemma_path = Path("C:/codedev/llm/gemma")
    if gemma_path.exists():
        print("\nWould you like to review a specific gemma.cpp file?")
        print("Example files to review:")
        cpp_files = list(gemma_path.glob("**/*.cc"))[:5]
        for i, f in enumerate(cpp_files, 1):
            print(f"  {i}. {f.relative_to(gemma_path)}")

if __name__ == "__main__":
    main()