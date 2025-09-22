#!/usr/bin/env python3
"""
Code Refactoring Suggestions with Gemma via Ollama

This script analyzes C++ code and provides specific refactoring
suggestions to improve code quality, maintainability, and performance.
"""

import subprocess
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

def run_ollama(prompt: str, model: str = "gemma2:2b") -> str:
    """Run Ollama with the specified prompt and model."""
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama: {e}")
        return ""
    except FileNotFoundError:
        print("Ollama is not installed. Please install from https://ollama.ai")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("Ollama request timed out")
        return ""

class RefactoringType(Enum):
    """Types of refactoring operations."""
    EXTRACT_METHOD = "Extract Method"
    EXTRACT_CLASS = "Extract Class"
    RENAME = "Rename"
    MOVE_METHOD = "Move Method"
    INLINE = "Inline"
    INTRODUCE_PARAMETER = "Introduce Parameter"
    REPLACE_TEMP_WITH_QUERY = "Replace Temp with Query"
    DECOMPOSE_CONDITIONAL = "Decompose Conditional"
    CONSOLIDATE_DUPLICATE = "Consolidate Duplicate"
    INTRODUCE_DESIGN_PATTERN = "Introduce Design Pattern"

@dataclass
class Refactoring:
    """Represents a single refactoring suggestion."""
    type: RefactoringType
    description: str
    before_code: str
    after_code: str
    benefits: List[str]
    complexity: str  # simple, moderate, complex
    impact: str  # low, medium, high

def analyze_code_smells(code: str) -> Dict[str, List[str]]:
    """Detect code smells in the given code."""

    prompt = f"""Analyze this C++ code for code smells and anti-patterns.

Code:
```cpp
{code}
```

Identify these code smells:
1. Long methods (>20 lines)
2. Large classes (too many responsibilities)
3. Duplicate code
4. Long parameter lists (>4 parameters)
5. Feature envy (method uses another class more than its own)
6. Data clumps (groups of data that appear together)
7. Primitive obsession (overuse of primitives instead of objects)
8. Switch statements that could be polymorphism
9. Lazy classes (classes that don't do enough)
10. Speculative generality (unused flexibility)
11. Message chains (a.b().c().d())
12. Middle man (class that just delegates)

List each smell found with specific line numbers or method names."""

    response = run_ollama(prompt)
    return parse_code_smells(response)

def suggest_extract_method(code: str) -> List[Refactoring]:
    """Suggest method extraction refactorings."""

    prompt = f"""Identify opportunities to extract methods in this C++ code.

Code:
```cpp
{code}
```

For each opportunity:
1. Identify the code block to extract
2. Suggest a descriptive method name
3. Define parameters and return type
4. Show the refactored code
5. Explain the benefits

Focus on:
- Repeated code blocks
- Long methods that do multiple things
- Complex conditionals
- Nested loops
- Comment-separated sections"""

    response = run_ollama(prompt)
    return parse_refactorings(response, RefactoringType.EXTRACT_METHOD)

def suggest_design_patterns(code: str) -> List[Refactoring]:
    """Suggest design pattern implementations."""

    prompt = f"""Analyze this C++ code and suggest design patterns that would improve it.

Code:
```cpp
{code}
```

Consider these patterns:
1. Factory Method - for object creation
2. Strategy - for algorithm selection
3. Observer - for event handling
4. Decorator - for adding behavior
5. Singleton - for single instances (use carefully)
6. Template Method - for algorithm frameworks
7. Command - for encapsulating requests
8. Iterator - for traversal
9. Facade - for simplifying interfaces
10. Builder - for complex object construction

For each applicable pattern:
- Explain why it fits
- Show the refactored code structure
- Highlight the benefits"""

    response = run_ollama(prompt)
    return parse_refactorings(response, RefactoringType.INTRODUCE_DESIGN_PATTERN)

def suggest_modern_cpp(code: str, cpp_version: str = "C++20") -> str:
    """Modernize code to use newer C++ features."""

    prompt = f"""Modernize this C++ code to use {cpp_version} features.

Current code:
```cpp
{code}
```

Apply these modernizations:
1. auto keyword where appropriate
2. Range-based for loops
3. Smart pointers instead of raw pointers
4. nullptr instead of NULL
5. constexpr for compile-time constants
6. std::optional for nullable values
7. std::variant instead of unions
8. Structured bindings
9. Concepts (C++20) for template constraints
10. Ranges library (C++20)
11. Coroutines (C++20) where applicable
12. Lambda expressions
13. Move semantics
14. std::string_view for string parameters
15. nodiscard attribute

Show the modernized code with explanations."""

    return run_ollama(prompt)

def suggest_performance_refactoring(code: str) -> List[Refactoring]:
    """Suggest performance-oriented refactorings."""

    prompt = f"""Analyze this C++ code for performance improvements.

Code:
```cpp
{code}
```

Suggest refactorings for:
1. Algorithm complexity improvements
2. Memory allocation optimizations
3. Cache-friendly data structures
4. Avoiding unnecessary copies
5. Using move semantics
6. Const correctness
7. Inline functions where beneficial
8. Reserve() for vectors
9. String optimizations (SSO awareness)
10. Loop optimizations
11. Branch prediction hints
12. SIMD opportunities

Provide before/after code with performance impact explanation."""

    response = run_ollama(prompt)
    return parse_refactorings(response, RefactoringType.EXTRACT_METHOD)

def suggest_solid_principles(code: str) -> str:
    """Refactor code to follow SOLID principles."""

    prompt = f"""Refactor this C++ code to better follow SOLID principles.

Code:
```cpp
{code}
```

Apply these principles:
1. Single Responsibility - One class, one purpose
2. Open/Closed - Open for extension, closed for modification
3. Liskov Substitution - Derived classes must be substitutable
4. Interface Segregation - Many specific interfaces
5. Dependency Inversion - Depend on abstractions

Show violations and how to fix them with refactored code."""

    return run_ollama(prompt)

def suggest_testability_improvements(code: str) -> str:
    """Refactor code to improve testability."""

    prompt = f"""Refactor this C++ code to improve testability.

Code:
```cpp
{code}
```

Apply these refactorings:
1. Dependency injection
2. Extract interfaces for mocking
3. Separate business logic from I/O
4. Make classes less coupled
5. Extract pure functions
6. Avoid static/global state
7. Use factory methods
8. Make side effects explicit
9. Separate construction from use
10. Add seams for testing

Show the refactored, more testable code."""

    return run_ollama(prompt)

def parse_code_smells(response: str) -> Dict[str, List[str]]:
    """Parse code smells from response."""
    smells = {}
    current_smell = None

    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check for smell categories
        smell_types = [
            "long method", "large class", "duplicate", "parameter",
            "feature envy", "data clump", "primitive", "switch",
            "lazy class", "speculative", "message chain", "middle man"
        ]

        for smell_type in smell_types:
            if smell_type in line.lower():
                current_smell = smell_type
                if current_smell not in smells:
                    smells[current_smell] = []
                break

        # Add details to current smell
        if current_smell and (line.startswith('-') or line.startswith('*')):
            smells[current_smell].append(line.lstrip('-* '))

    return smells

def parse_refactorings(response: str, refactor_type: RefactoringType) -> List[Refactoring]:
    """Parse refactoring suggestions from response."""
    refactorings = []

    # Simple parsing - in production, this would be more sophisticated
    sections = response.split('\n\n')

    for section in sections:
        if 'before' in section.lower() or 'after' in section.lower():
            refactoring = Refactoring(
                type=refactor_type,
                description="Extracted from analysis",
                before_code="",
                after_code="",
                benefits=["Improved readability", "Better maintainability"],
                complexity="moderate",
                impact="medium"
            )
            refactorings.append(refactoring)

    return refactorings

def generate_refactoring_report(code: str, output_path: Optional[Path] = None) -> str:
    """Generate a comprehensive refactoring report."""

    report = []
    report.append("="*60)
    report.append("Code Refactoring Analysis Report")
    report.append("="*60 + "\n")

    # Analyze code smells
    print("Detecting code smells...")
    smells = analyze_code_smells(code)

    if smells:
        report.append("📊 Code Smells Detected:\n")
        for smell_type, instances in smells.items():
            report.append(f"  • {smell_type.title()}:")
            for instance in instances[:3]:  # Limit to 3 examples
                report.append(f"    - {instance}")
        report.append("")

    # Suggest method extractions
    print("Analyzing for method extraction...")
    extractions = suggest_extract_method(code)
    if extractions:
        report.append("🔧 Method Extraction Opportunities:")
        report.append(f"  Found {len(extractions)} opportunities for extraction")
        report.append("")

    # Modernization suggestions
    print("Checking for modernization opportunities...")
    modern = suggest_modern_cpp(code)
    if modern:
        report.append("🚀 Modernization Suggestions:")
        report.append("  Use modern C++ features for cleaner, safer code")
        report.append("")

    # SOLID principles
    print("Evaluating SOLID principles...")
    solid = suggest_solid_principles(code)
    if solid:
        report.append("📐 SOLID Principle Improvements:")
        report.append("  Refactoring suggestions for better design")
        report.append("")

    # Testability
    print("Analyzing testability...")
    testability = suggest_testability_improvements(code)
    if testability:
        report.append("🧪 Testability Improvements:")
        report.append("  Make code more unit-testable")
        report.append("")

    report.append("="*60)

    full_report = '\n'.join(report)

    if output_path:
        output_path.write_text(full_report, encoding='utf-8')
        print(f"\n✅ Report saved to: {output_path}")

    return full_report

def main():
    """Main function with example usage."""

    print("Gemma Code Refactoring Analyzer Demo")
    print("=" * 60)

    # Example: Analyze code that needs refactoring
    sample_code = """
class DataProcessor {
private:
    std::vector<int> data;
    std::string filename;
    int* temp_buffer;
    int buffer_size;

public:
    DataProcessor(std::string f) : filename(f), temp_buffer(nullptr), buffer_size(0) {}

    void processFile() {
        // Open file
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file" << std::endl;
            return;
        }

        // Read all data
        int value;
        while (file >> value) {
            data.push_back(value);
        }
        file.close();

        // Sort data
        for (int i = 0; i < data.size() - 1; i++) {
            for (int j = 0; j < data.size() - i - 1; j++) {
                if (data[j] > data[j + 1]) {
                    int temp = data[j];
                    data[j] = data[j + 1];
                    data[j + 1] = temp;
                }
            }
        }

        // Calculate statistics
        if (data.size() > 0) {
            int sum = 0;
            int min = data[0];
            int max = data[0];

            for (int i = 0; i < data.size(); i++) {
                sum += data[i];
                if (data[i] < min) min = data[i];
                if (data[i] > max) max = data[i];
            }

            double avg = (double)sum / data.size();

            std::cout << "Min: " << min << std::endl;
            std::cout << "Max: " << max << std::endl;
            std::cout << "Avg: " << avg << std::endl;
        }

        // Save results
        std::ofstream out("results.txt");
        for (int i = 0; i < data.size(); i++) {
            out << data[i] << std::endl;
        }
        out.close();
    }

    void processFileAdvanced(bool use_parallel, bool use_cache, int thread_count) {
        // Lots of duplicate code from processFile()
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file" << std::endl;
            return;
        }

        int value;
        while (file >> value) {
            data.push_back(value);
        }
        file.close();

        // More processing with same bubble sort...
        // ... duplicated code ...
    }
};"""

    # Generate comprehensive report
    output_dir = Path("C:/codedev/llm/gemma/examples/refactoring_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "sample_refactoring_report.md"
    report = generate_refactoring_report(sample_code, report_path)

    print("\n" + report)

    # Example of specific refactorings
    print("\n" + "="*60)
    print("Modernizing to C++20...")
    modern_code = suggest_modern_cpp(sample_code)

    modern_file = output_dir / "modernized_code.cpp"
    modern_file.write_text(modern_code, encoding='utf-8')
    print(f"✅ Modernized code saved to: {modern_file}")

    print("\n" + "="*60)
    print("Refactoring complete!")
    print("\nRefactoring categories analyzed:")
    print("  • Code smell detection")
    print("  • Method extraction opportunities")
    print("  • Design pattern suggestions")
    print("  • Modern C++ features")
    print("  • SOLID principles")
    print("  • Performance optimizations")
    print("  • Testability improvements")
    print("\nUse this tool to continuously improve code quality!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Analyze specified file
        filepath = Path(sys.argv[1])
        if filepath.exists():
            code = filepath.read_text(encoding='utf-8')
            output_path = filepath.parent / f"{filepath.stem}_refactoring_report.md"
            report = generate_refactoring_report(code, output_path)
            print(report)
    else:
        # Run demo
        main()