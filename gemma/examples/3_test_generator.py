#!/usr/bin/env python3
"""
Test Case Generation with Gemma via Ollama

This script automatically generates comprehensive test cases
from specifications, function signatures, or existing code.
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

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

@dataclass
class TestCase:
    """Represents a single test case."""
    name: str
    description: str
    setup: str
    test_code: str
    assertions: List[str]
    teardown: str = ""
    tags: List[str] = None

def generate_unit_tests(code: str, framework: str = "googletest") -> str:
    """Generate unit tests for given C++ code."""

    framework_instructions = {
        "googletest": "Use Google Test (gtest) framework with TEST() macros",
        "catch2": "Use Catch2 framework with TEST_CASE() and SECTION()",
        "doctest": "Use doctest framework with TEST_CASE() and SUBCASE()",
        "boost": "Use Boost.Test framework"
    }

    prompt = f"""Generate comprehensive unit tests for this C++ code.

Instructions:
1. {framework_instructions.get(framework, framework_instructions['googletest'])}
2. Test all public methods and functions
3. Include edge cases and error conditions
4. Test boundary values
5. Add negative test cases
6. Include performance tests where relevant
7. Add comments explaining what each test verifies

Code to test:
```cpp
{code}
```

Generate complete, compilable test code with all necessary includes and fixtures."""

    return run_ollama(prompt)

def generate_integration_tests(specification: str) -> str:
    """Generate integration tests from specifications."""

    prompt = f"""Generate integration tests based on this specification.

Specification:
{specification}

Generate tests that:
1. Verify component interactions
2. Test data flow between modules
3. Validate system behavior under load
4. Check error propagation
5. Test configuration changes
6. Verify API contracts

Format as complete C++ test code with setup, execution, and verification phases."""

    return run_ollama(prompt)

def generate_parameterized_tests(function_signature: str, constraints: str = "") -> str:
    """Generate parameterized tests for a function."""

    prompt = f"""Generate parameterized tests for this C++ function.

Function signature:
{function_signature}

Constraints/Requirements:
{constraints if constraints else "None specified"}

Create a comprehensive test suite that:
1. Uses value-parameterized tests for different input combinations
2. Tests all equivalence classes
3. Includes boundary value analysis
4. Tests invalid inputs
5. Uses type-parameterized tests if the function is templated

Use Google Test's TEST_P() or similar parameterized test features."""

    return run_ollama(prompt)

def generate_property_tests(code: str) -> str:
    """Generate property-based tests."""

    prompt = f"""Generate property-based tests for this C++ code.

Code:
```cpp
{code}
```

Create tests that verify invariants and properties:
1. Identify mathematical properties (associativity, commutativity, etc.)
2. Test invariants that should always hold
3. Use random input generation
4. Verify relationships between operations
5. Test for no unexpected side effects

Include generator functions for random valid inputs."""

    return run_ollama(prompt)

def generate_benchmark_tests(code: str) -> str:
    """Generate performance benchmark tests."""

    prompt = f"""Generate performance benchmark tests for this C++ code.

Code:
```cpp
{code}
```

Create benchmarks that:
1. Measure function execution time
2. Test with different input sizes
3. Compare algorithm complexity
4. Measure memory usage
5. Test cache performance
6. Include warm-up iterations

Use Google Benchmark or similar framework."""

    return run_ollama(prompt)

def generate_fuzz_tests(code: str) -> str:
    """Generate fuzz tests for robustness testing."""

    prompt = f"""Generate fuzz tests for this C++ code to find crashes and undefined behavior.

Code:
```cpp
{code}
```

Create fuzz tests that:
1. Generate random malformed inputs
2. Test with extreme values
3. Use mutation-based fuzzing strategies
4. Test concurrency issues
5. Check for memory leaks and buffer overflows

Format for libFuzzer or AFL++ compatibility."""

    return run_ollama(prompt)

def generate_test_fixtures(class_definition: str) -> str:
    """Generate test fixtures for class testing."""

    prompt = f"""Generate comprehensive test fixtures for this C++ class.

Class definition:
```cpp
{class_definition}
```

Create fixtures that:
1. Set up common test data
2. Provide mock dependencies
3. Handle resource initialization and cleanup
4. Support different test scenarios
5. Include helper methods for assertions

Use Google Test fixture classes with SetUp() and TearDown()."""

    return run_ollama(prompt)

def save_test_file(tests: str, output_path: Path, test_type: str = "unit") -> None:
    """Save generated tests to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add header comments
    header = f"""// Auto-generated {test_type} tests using Gemma via Ollama
// Generated on: {Path.ctime(Path.cwd())}
// Review and modify as needed before use

"""

    full_content = header + tests
    output_path.write_text(full_content, encoding='utf-8')
    print(f"✅ Tests saved to: {output_path}")

def main():
    """Main function with example usage."""

    print("Gemma Test Case Generator Demo")
    print("=" * 60)

    # Example 1: Generate unit tests for a sample class
    sample_code = """
class Calculator {
private:
    double memory_;
    bool error_state_;

public:
    Calculator() : memory_(0.0), error_state_(false) {}

    double add(double a, double b) {
        if (error_state_) return 0;
        return a + b;
    }

    double divide(double a, double b) {
        if (b == 0) {
            error_state_ = true;
            return 0;
        }
        return a / b;
    }

    double sqrt(double x) {
        if (x < 0) {
            error_state_ = true;
            return 0;
        }
        return std::sqrt(x);
    }

    void store(double value) { memory_ = value; }
    double recall() const { return memory_; }
    void clear() { memory_ = 0; error_state_ = false; }
    bool hasError() const { return error_state_; }
};"""

    print("\n1. Generating Unit Tests...")
    unit_tests = generate_unit_tests(sample_code)

    # Example 2: Generate parameterized tests
    function_sig = "template<typename T> T clamp(T value, T min, T max)"
    constraints = "min must be less than or equal to max"

    print("\n2. Generating Parameterized Tests...")
    param_tests = generate_parameterized_tests(function_sig, constraints)

    # Example 3: Generate integration tests from specification
    spec = """
    The KVCache system must:
    - Store key-value pairs for multiple layers
    - Support sequential and random access
    - Handle cache overflow gracefully
    - Maintain consistency across parallel operations
    - Provide efficient retrieval with O(1) complexity
    """

    print("\n3. Generating Integration Tests from Specification...")
    integration_tests = generate_integration_tests(spec)

    # Example 4: Generate property-based tests
    property_code = """
    template<typename T>
    std::vector<T> merge_sorted(const std::vector<T>& a, const std::vector<T>& b) {
        std::vector<T> result;
        result.reserve(a.size() + b.size());
        std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(result));
        return result;
    }"""

    print("\n4. Generating Property-Based Tests...")
    property_tests = generate_property_tests(property_code)

    # Save all generated tests
    output_dir = Path("C:/codedev/llm/gemma/examples/generated_tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_test_file(unit_tests, output_dir / "calculator_test.cc", "unit")
    save_test_file(param_tests, output_dir / "clamp_parameterized_test.cc", "parameterized")
    save_test_file(integration_tests, output_dir / "kvcache_integration_test.cc", "integration")
    save_test_file(property_tests, output_dir / "merge_property_test.cc", "property")

    # Example 5: Generate benchmark tests
    print("\n5. Generating Benchmark Tests...")
    benchmark_tests = generate_benchmark_tests(sample_code)
    save_test_file(benchmark_tests, output_dir / "calculator_benchmark.cc", "benchmark")

    print("\n" + "="*60)
    print("Test generation complete!")
    print(f"\nGenerated tests saved to: {output_dir}")
    print("\nTest types generated:")
    print("  • Unit tests")
    print("  • Parameterized tests")
    print("  • Integration tests")
    print("  • Property-based tests")
    print("  • Benchmark tests")
    print("\nNext steps:")
    print("  1. Review generated tests")
    print("  2. Adjust for your specific testing framework")
    print("  3. Add to your CMakeLists.txt or build system")
    print("  4. Run tests with your test runner")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Process specified file
        filepath = Path(sys.argv[1])
        if filepath.exists():
            code = filepath.read_text(encoding='utf-8')
            tests = generate_unit_tests(code)
            output_path = filepath.parent / f"{filepath.stem}_test.cc"
            save_test_file(tests, output_path)
    else:
        # Run demo
        main()