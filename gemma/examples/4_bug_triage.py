#!/usr/bin/env python3
"""
Bug Triage and Error Analysis with Gemma via Ollama

This script analyzes error messages, stack traces, and build failures
to provide actionable fixes and debugging suggestions.
"""

import subprocess
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

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

@dataclass
class BugAnalysis:
    """Structured bug analysis result."""
    severity: str  # critical, high, medium, low
    category: str  # compilation, runtime, logic, performance, memory
    root_cause: str
    suggested_fixes: List[str]
    code_changes: List[Dict[str, str]]
    prevention_tips: List[str]
    related_issues: List[str]
    confidence: float  # 0.0 to 1.0

def analyze_compilation_error(error_message: str, code_context: str = "") -> BugAnalysis:
    """Analyze C++ compilation errors and suggest fixes."""

    prompt = f"""You are a C++ expert debugging a compilation error.

Error message:
```
{error_message}
```

{f"Code context:" if code_context else ""}
{f"```cpp\n{code_context}\n```" if code_context else ""}

Analyze this error and provide:
1. Root cause of the error
2. Specific fixes with code examples
3. Common mistakes that lead to this error
4. How to prevent this in the future

Format your response as JSON:
{{
    "severity": "critical|high|medium|low",
    "category": "compilation",
    "root_cause": "explanation",
    "suggested_fixes": ["fix1", "fix2"],
    "code_changes": [{{"file": "path", "change": "code"}}],
    "prevention_tips": ["tip1", "tip2"],
    "confidence": 0.0-1.0
}}"""

    response = run_ollama(prompt)
    return parse_bug_analysis(response, "compilation")

def analyze_runtime_error(stack_trace: str, error_message: str = "", code: str = "") -> BugAnalysis:
    """Analyze runtime errors and crashes."""

    prompt = f"""Debug this runtime error in C++.

Error: {error_message if error_message else "Program crash"}

Stack trace:
```
{stack_trace}
```

{f"Related code:" if code else ""}
{f"```cpp\n{code}\n```" if code else ""}

Provide:
1. What caused this crash/error
2. Exact location and reason for failure
3. Fix with code examples
4. How to debug this type of issue
5. Similar bugs to watch for

Respond with specific, actionable solutions."""

    response = run_ollama(prompt)
    return parse_bug_analysis(response, "runtime")

def analyze_memory_issue(valgrind_output: str = "", asan_output: str = "", code: str = "") -> BugAnalysis:
    """Analyze memory-related issues from sanitizers."""

    output = valgrind_output or asan_output or "Memory issue detected"

    prompt = f"""Analyze this memory issue in C++.

Memory checker output:
```
{output}
```

{f"Code:" if code else ""}
{f"```cpp\n{code}\n```" if code else ""}

Identify:
1. Type of memory issue (leak, overflow, use-after-free, etc.)
2. Exact cause and location
3. Fix with proper memory management
4. Best practices to avoid this
5. Tools to detect similar issues

Provide specific fixes, not generic advice."""

    response = run_ollama(prompt)
    return parse_bug_analysis(response, "memory")

def analyze_performance_issue(profile_data: str, code: str = "") -> BugAnalysis:
    """Analyze performance bottlenecks and suggest optimizations."""

    prompt = f"""Analyze this C++ performance issue.

Performance profile:
```
{profile_data}
```

{f"Code:" if code else ""}
{f"```cpp\n{code}\n```" if code else ""}

Provide:
1. Main performance bottleneck
2. Specific optimizations with code
3. Algorithm improvements
4. Data structure optimizations
5. Compiler optimization flags
6. Cache and memory access patterns

Focus on practical, measurable improvements."""

    response = run_ollama(prompt)
    return parse_bug_analysis(response, "performance")

def analyze_build_failure(build_log: str) -> BugAnalysis:
    """Analyze CMake/Make build failures."""

    prompt = f"""Debug this C++ build failure.

Build log:
```
{build_log}
```

Identify:
1. Root cause of build failure
2. Missing dependencies or configuration
3. Exact commands to fix
4. CMakeLists.txt or Makefile changes needed
5. Environment setup issues

Provide specific commands and file changes."""

    response = run_ollama(prompt)
    return parse_bug_analysis(response, "build")

def analyze_test_failure(test_output: str, test_code: str = "") -> BugAnalysis:
    """Analyze failing unit tests."""

    prompt = f"""Analyze this failing C++ test.

Test output:
```
{test_output}
```

{f"Test code:" if test_code else ""}
{f"```cpp\n{test_code}\n```" if test_code else ""}

Determine:
1. Why the test is failing
2. Is it a test bug or implementation bug?
3. Exact fix needed
4. Additional test cases needed
5. How to make the test more robust

Provide specific code fixes."""

    response = run_ollama(prompt)
    return parse_bug_analysis(response, "test")

def suggest_debugging_strategy(error_description: str) -> str:
    """Suggest a debugging strategy for a described problem."""

    prompt = f"""Create a debugging strategy for this C++ issue:

Issue description:
{error_description}

Provide a step-by-step debugging plan:
1. Initial investigation steps
2. Tools to use (debugger commands, sanitizers, etc.)
3. What to look for
4. Common causes to check
5. How to isolate the problem
6. Verification steps after fixing

Be specific with tool commands and techniques."""

    return run_ollama(prompt)

def parse_bug_analysis(response: str, default_category: str) -> BugAnalysis:
    """Parse LLM response into structured BugAnalysis."""

    # Try to parse as JSON first
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            return BugAnalysis(
                severity=data.get("severity", "medium"),
                category=data.get("category", default_category),
                root_cause=data.get("root_cause", ""),
                suggested_fixes=data.get("suggested_fixes", []),
                code_changes=data.get("code_changes", []),
                prevention_tips=data.get("prevention_tips", []),
                related_issues=data.get("related_issues", []),
                confidence=float(data.get("confidence", 0.7))
            )
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: parse text response
    lines = response.split('\n')
    fixes = []
    tips = []
    root_cause = ""

    for i, line in enumerate(lines):
        line = line.strip()
        if 'cause' in line.lower() or 'reason' in line.lower():
            root_cause = line
        elif line.startswith('-') or line.startswith('*') or line.startswith('•'):
            if 'fix' in lines[max(0, i-2):i][0].lower() if i >= 2 else False:
                fixes.append(line.lstrip('-*• '))
            else:
                tips.append(line.lstrip('-*• '))

    return BugAnalysis(
        severity="medium",
        category=default_category,
        root_cause=root_cause or "Analysis needed",
        suggested_fixes=fixes or ["Manual review required"],
        code_changes=[],
        prevention_tips=tips,
        related_issues=[],
        confidence=0.5
    )

def generate_fix_report(analysis: BugAnalysis, original_error: str) -> str:
    """Generate a formatted fix report."""

    report = []
    report.append("\n" + "="*60)
    report.append("🔍 Bug Analysis Report")
    report.append("="*60 + "\n")

    # Severity indicator
    severity_emoji = {
        "critical": "🔴",
        "high": "🟠",
        "medium": "🟡",
        "low": "🟢"
    }

    report.append(f"{severity_emoji.get(analysis.severity, '⚪')} Severity: {analysis.severity.upper()}")
    report.append(f"📁 Category: {analysis.category}")
    report.append(f"🎯 Confidence: {analysis.confidence:.0%}\n")

    report.append("📋 Root Cause:")
    report.append(f"  {analysis.root_cause}\n")

    if analysis.suggested_fixes:
        report.append("🔧 Suggested Fixes:")
        for i, fix in enumerate(analysis.suggested_fixes, 1):
            report.append(f"  {i}. {fix}")
        report.append("")

    if analysis.code_changes:
        report.append("💻 Code Changes:")
        for change in analysis.code_changes:
            report.append(f"  File: {change.get('file', 'unknown')}")
            report.append(f"  ```cpp")
            report.append(f"  {change.get('change', '')}")
            report.append(f"  ```")
        report.append("")

    if analysis.prevention_tips:
        report.append("🛡️ Prevention Tips:")
        for tip in analysis.prevention_tips:
            report.append(f"  • {tip}")
        report.append("")

    report.append("="*60 + "\n")

    return '\n'.join(report)

def main():
    """Main function with example usage."""

    print("Gemma Bug Triage and Error Analysis Demo")
    print("=" * 60)

    # Example 1: Compilation Error
    compilation_error = """
error: no matching function for call to 'std::vector<int>::push_back(const char [6])'
   42 |     numbers.push_back("hello");
      |     ~~~~~~~~~~~~~~~~~~^~~~~~~~~
note: candidate: 'void std::vector<_Tp>::push_back(const value_type&)'
note:   no known conversion for argument 1 from 'const char [6]' to 'const int&'
"""

    code_context = """
std::vector<int> numbers;
numbers.push_back(42);
numbers.push_back("hello");  // Error here
"""

    print("\n1. Analyzing Compilation Error...")
    analysis = analyze_compilation_error(compilation_error, code_context)
    print(generate_fix_report(analysis, compilation_error))

    # Example 2: Segmentation Fault
    stack_trace = """
Program received signal SIGSEGV, Segmentation fault.
0x0000555555555249 in MyClass::processData (this=0x0, data=0x7fffffffe410) at main.cpp:25
25          return data_->size();
#0  0x0000555555555249 in MyClass::processData (this=0x0, data=0x7fffffffe410) at main.cpp:25
#1  0x00005555555552a8 in main () at main.cpp:42
"""

    print("\n2. Analyzing Segmentation Fault...")
    analysis = analyze_runtime_error(stack_trace, "Segmentation fault (core dumped)")
    print(generate_fix_report(analysis, stack_trace))

    # Example 3: Memory Leak (AddressSanitizer output)
    asan_output = """
=================================================================
==12345==ERROR: AddressSanitizer: heap-use-after-free on address 0x60400000dff0
READ of size 4 at 0x60400000dff0 thread T0
    #0 0x5555555551a9 in processArray main.cpp:15
    #1 0x555555555234 in main main.cpp:30

0x60400000dff0 is located 0 bytes inside of 40-byte region [0x60400000dff0,0x60400000e018)
freed by thread T0 here:
    #0 0x7ffff7681b6f in operator delete[]
    #1 0x555555555189 in cleanupArray main.cpp:10
"""

    print("\n3. Analyzing Memory Issue...")
    analysis = analyze_memory_issue(asan_output=asan_output)
    print(generate_fix_report(analysis, asan_output))

    # Example 4: Build Failure
    build_log = """
CMake Error at CMakeLists.txt:15 (find_package):
  Could not find a package configuration file provided by "Boost" with any
  of the following names:

    BoostConfig.cmake
    boost-config.cmake

CMake Error at src/CMakeLists.txt:8 (target_link_libraries):
  Cannot specify link libraries for target "myapp" which is not built by
  this project.

make: *** [Makefile:123: all] Error 2
"""

    print("\n4. Analyzing Build Failure...")
    analysis = analyze_build_failure(build_log)
    print(generate_fix_report(analysis, build_log))

    # Example 5: Interactive debugging strategy
    print("\n5. Generating Debugging Strategy...")
    problem = "My C++ application randomly crashes after running for 10-15 minutes under load"
    strategy = suggest_debugging_strategy(problem)
    print("\n📚 Debugging Strategy:")
    print(strategy)

    print("\n" + "="*60)
    print("Bug triage demo complete!")
    print("\nUsage examples:")
    print("  • Paste compilation errors to get fixes")
    print("  • Analyze core dumps and stack traces")
    print("  • Debug memory issues from sanitizers")
    print("  • Troubleshoot build failures")
    print("  • Get debugging strategies for complex issues")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Analyze error from file
        error_file = Path(sys.argv[1])
        if error_file.exists():
            error_content = error_file.read_text(encoding='utf-8')

            # Detect error type and analyze
            if "error:" in error_content.lower() or "warning:" in error_content.lower():
                analysis = analyze_compilation_error(error_content)
            elif "segmentation fault" in error_content.lower() or "sigsegv" in error_content.lower():
                analysis = analyze_runtime_error(error_content)
            elif "addresssanitizer" in error_content.lower() or "leak" in error_content.lower():
                analysis = analyze_memory_issue(asan_output=error_content)
            elif "cmake" in error_content.lower() or "make" in error_content.lower():
                analysis = analyze_build_failure(error_content)
            else:
                # Generic analysis
                analysis = analyze_compilation_error(error_content)

            print(generate_fix_report(analysis, error_content))
    else:
        main()