# Gemma LLM Developer Tools via Ollama

This directory contains practical Python scripts that demonstrate how to use Gemma (or any Ollama-supported model) for automating common development tasks. These tools show real value for C++ developers by automating grunt work and improving code quality.

## Prerequisites

1. **Install Ollama**: Download from [https://ollama.ai](https://ollama.ai)
2. **Pull Gemma model**:
   ```bash
   ollama pull gemma2:2b  # Recommended: Fast and efficient
   # or
   ollama pull gemma2:9b  # Better quality, slower
   ```
3. **Python 3.8+** with standard library (no additional packages required)

## Available Tools

### 1. Code Review Automation (`1_code_review.py`)
Automatically reviews C++ code for bugs, performance issues, and best practices.

**Features:**
- Detects memory leaks and buffer overflows
- Identifies performance bottlenecks
- Checks style and readability issues
- Flags security vulnerabilities
- Suggests best practice improvements

**Usage:**
```bash
python 1_code_review.py                    # Run demo with sample code
python 1_code_review.py path/to/file.cpp   # Review specific file
```

**Example Output:**
```
🔴 CRITICAL:
  1. Memory leak: Missing destructor for allocated memory
  2. Buffer overflow in loop at line 15

⚡ PERFORMANCE:
  1. Use std::move for large object transfers
  2. Reserve vector capacity before loop insertions
```

### 2. Documentation Generator (`2_doc_generator.py`)
Generates comprehensive documentation from C++ code automatically.

**Features:**
- API documentation with parameter descriptions
- Class and method documentation
- Usage examples generation
- README sections for modules
- Doxygen-style inline comments

**Usage:**
```bash
python 2_doc_generator.py                  # Run demo
python 2_doc_generator.py path/to/file.h   # Document specific header
```

**Outputs:**
- Markdown API documentation
- README sections
- Inline documented code
- HTML documentation (basic)

### 3. Test Case Generator (`3_test_generator.py`)
Creates comprehensive test suites from specifications or code.

**Features:**
- Unit tests for classes and functions
- Parameterized tests for edge cases
- Integration tests from specifications
- Property-based tests
- Performance benchmarks
- Fuzz tests for robustness

**Usage:**
```bash
python 3_test_generator.py                  # Run demo with examples
python 3_test_generator.py path/to/file.cpp # Generate tests for file
```

**Generates:**
- Google Test compatible unit tests
- Parameterized test cases
- Benchmark tests
- Test fixtures with setup/teardown

### 4. Bug Triage and Analysis (`4_bug_triage.py`)
Analyzes error messages and suggests fixes with root cause analysis.

**Features:**
- Compilation error analysis
- Runtime crash debugging
- Memory issue detection (ASAN/Valgrind)
- Build failure troubleshooting
- Test failure analysis
- Step-by-step debugging strategies

**Usage:**
```bash
python 4_bug_triage.py                     # Run demo with examples
python 4_bug_triage.py error_log.txt       # Analyze error log file
```

**Provides:**
- Root cause analysis
- Specific code fixes
- Prevention strategies
- Debugging commands

### 5. Refactoring Suggestions (`5_refactoring.py`)
Analyzes code quality and suggests specific improvements.

**Features:**
- Code smell detection
- Method extraction opportunities
- Design pattern suggestions
- Modern C++ feature adoption
- SOLID principle compliance
- Performance optimizations
- Testability improvements

**Usage:**
```bash
python 5_refactoring.py                    # Run demo
python 5_refactoring.py path/to/file.cpp   # Analyze and refactor file
```

**Reports:**
- Code smell inventory
- Refactoring opportunities
- Modernization suggestions
- Design improvements

## Configuration

### Using Different Models

All scripts default to `gemma2:2b` but can use any Ollama model:

```python
# In any script, change the model parameter:
def run_ollama(prompt: str, model: str = "gemma2:9b")  # Use larger model
def run_ollama(prompt: str, model: str = "llama3")     # Use different model
```

### Available Gemma Models
- `gemma2:2b` - Fast, good for quick iterations (default)
- `gemma2:9b` - Better quality, more thorough analysis
- `gemma2:27b` - Highest quality, requires more resources

## Real-World Integration

### CI/CD Pipeline Integration
```yaml
# .github/workflows/code-review.yml
- name: AI Code Review
  run: |
    python examples/1_code_review.py src/*.cpp > review_report.md
    python examples/3_test_generator.py src/*.cpp
```

### Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit
python examples/1_code_review.py $(git diff --cached --name-only --diff-filter=ACM | grep '\.cpp$')
```

### VS Code Task
```json
{
  "label": "AI Review Current File",
  "type": "shell",
  "command": "python",
  "args": ["${workspaceFolder}/examples/1_code_review.py", "${file}"],
  "problemMatcher": []
}
```

## Performance Tips

1. **Model Selection**:
   - Use `gemma2:2b` for quick feedback during development
   - Use `gemma2:9b` for thorough reviews before commits

2. **Caching**: Ollama caches models in memory, subsequent calls are faster

3. **Batch Processing**: Process multiple files in one session for efficiency

4. **Parallel Execution**: Run different analysis tools in parallel

## Extending the Tools

Each script is designed to be easily extended:

```python
# Add custom analysis to code review
def check_custom_rules(code: str) -> List[str]:
    prompt = f"Check this code for our team's specific guidelines: {code}"
    return run_ollama(prompt)

# Add new test generation strategy
def generate_mutation_tests(code: str) -> str:
    prompt = f"Generate mutation tests for: {code}"
    return run_ollama(prompt)
```

## Troubleshooting

### Ollama Not Found
```bash
# Windows: Add Ollama to PATH or use full path
C:\Users\{username}\AppData\Local\Programs\Ollama\ollama.exe

# Linux/Mac: Install via curl
curl -fsSL https://ollama.ai/install.sh | sh
```

### Model Not Available
```bash
# List available models
ollama list

# Pull required model
ollama pull gemma2:2b
```

### Timeout Issues
Increase timeout in scripts:
```python
result = subprocess.run(..., timeout=60)  # Increase from 30 to 60 seconds
```

## Examples Output Directory Structure

After running the demos, you'll find generated files in:
```
examples/
├── generated_docs/       # Documentation outputs
│   ├── kvcache_api.md
│   ├── kvcache_readme.md
│   └── kvcache_documented.h
├── generated_tests/      # Test file outputs
│   ├── calculator_test.cc
│   ├── clamp_parameterized_test.cc
│   └── calculator_benchmark.cc
└── refactoring_reports/  # Refactoring analysis
    ├── sample_refactoring_report.md
    └── modernized_code.cpp
```

## Best Practices

1. **Review Generated Output**: Always review AI-generated code before using
2. **Iterative Refinement**: Run tools multiple times with refined prompts
3. **Combine Tools**: Use multiple tools together (review → fix → test → document)
4. **Custom Prompts**: Modify prompts in scripts for your specific needs
5. **Version Control**: Track changes suggested by tools in Git

## Support

These tools are examples of LLM-powered development automation. For production use:
- Add error handling for network issues
- Implement result caching
- Add configuration files for team standards
- Integrate with your existing toolchain

## License

These example scripts are provided as-is for demonstration purposes. Modify and extend them for your specific needs.