# Gemma.cpp Comprehensive Testing Framework

This directory contains a comprehensive testing framework for Gemma.cpp, designed to ensure code quality, performance, and reliability across different platforms and configurations. The framework includes unit tests, integration tests, performance benchmarks, and CI/CD automation.

## Test Structure

```
tests/
├── unit/                          # Unit tests
│   ├── test_model_loading.cc      # Model configuration and loading tests
│   ├── test_tokenization.cc       # Tokenizer functionality tests
│   ├── test_memory_management.cc  # Memory allocation and cleanup tests
│   └── test_error_handling.cc     # Error condition and edge case tests
├── integration/                   # Integration tests
│   └── test_inference.cc          # End-to-end inference tests
├── benchmarks/                    # Performance benchmarks
│   └── test_performance.cc        # Throughput and latency benchmarks
├── fixtures/                      # Test data and fixtures
├── CMakeLists.txt                 # Build configuration
├── run_tests.sh                   # Test runner script
└── README.md                      # This documentation
```

## Prerequisites

### System Requirements
- CMake 3.11 or later
- C++20 compatible compiler (GCC 9+, Clang 10+, MSVC 2019+)
- At least 8GB RAM (16GB recommended for full test suite)
- 10GB free disk space

### Dependencies
The test suite automatically fetches and builds the following dependencies:
- **Google Test** (gtest) - Unit testing framework
- **Google Benchmark** - Performance benchmarking
- **Highway** - SIMD optimization library
- **SentencePiece** - Tokenization library

### Model Files (Optional but Recommended)
For integration tests, place the following files in `/c/codedev/llm/.models/`:
- `tokenizer.spm` - SentencePiece tokenizer model
- `gemma2-2b-it-sfp.sbs` - Gemma 2B model weights (SFP format)

**Note**: Integration tests will be skipped if model files are not available.

## Quick Start

### Run All Tests
```bash
cd /c/codedev/llm/gemma/tests
./run_tests.sh
```

### Run Specific Test Categories
```bash
# Unit tests only
./run_tests.sh unit

# Integration tests only
./run_tests.sh integration

# Performance benchmarks
./run_tests.sh benchmarks
```

### Manual Build and Test
```bash
# Build tests
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run individual test executables
./tests/test_model_loading
./tests/test_tokenization
./tests/test_inference

# Run with CTest
ctest --output-on-failure
```

## Test Categories

### Unit Tests

#### Model Loading Tests (`test_model_loading.cc`)
- **Configuration Validation**: Tests model configuration creation and validation
- **File Handling**: Validates weights and tokenizer file loading
- **Memory Estimation**: Verifies memory requirement calculations
- **Multi-Model Support**: Tests loading different model variants

**Key Test Cases**:
- Valid model configuration loading
- Invalid model enum handling
- File existence validation
- Memory allocation estimation
- Multiple model configurations

#### Tokenization Tests (`test_tokenization.cc`)
- **Basic Operations**: Encoding and decoding functionality
- **Round-trip Consistency**: Encode→decode→encode stability
- **Special Tokens**: BOS, EOS, UNK token handling
- **Edge Cases**: Empty strings, whitespace, Unicode
- **Performance**: Tokenization speed benchmarks

**Key Test Cases**:
- Text encoding/decoding accuracy
- Unicode character support
- Special token recognition
- Long text handling
- Performance baselines

#### Memory Management Tests (`test_memory_management.cc`)
- **Allocation Patterns**: Sequential and concurrent allocations
- **Resource Cleanup**: Proper destruction and deallocation
- **Thread Safety**: Multi-threaded allocation scenarios
- **Stress Testing**: Resource exhaustion handling
- **Fragmentation**: Memory fragmentation resistance

**Key Test Cases**:
- KV cache allocation/deallocation
- Threading context lifecycle
- Large allocation handling
- Concurrent access patterns
- Resource leak detection

#### Error Handling Tests (`test_error_handling.cc`)
- **Invalid Inputs**: Malformed configurations and parameters
- **File Errors**: Missing, corrupted, or inaccessible files
- **Resource Constraints**: Out-of-memory conditions
- **Runtime Errors**: Generation failures and exceptions
- **Recovery**: System state after error conditions

**Key Test Cases**:
- Invalid configuration handling
- File system error scenarios
- Memory exhaustion recovery
- Generation parameter validation
- Exception safety guarantees

### Integration Tests

#### Inference Tests (`test_inference.cc`)
- **End-to-End Generation**: Complete inference pipeline testing
- **Parameter Sensitivity**: Temperature and top-k variations
- **Context Handling**: Multi-turn conversations
- **Performance Validation**: Timing and throughput metrics
- **Quality Assurance**: Output consistency and coherence

**Key Test Cases**:
- Basic text generation
- Parameter variation effects
- Long sequence generation
- Conversational context
- Performance characteristics

### Performance Benchmarks

#### Performance Tests (`test_performance.cc`)
- **Tokenization Speed**: Encoding/decoding throughput
- **Generation Latency**: Time-to-first-token and tokens-per-second
- **Memory Efficiency**: Allocation patterns and peak usage
- **Batch Processing**: Multi-query performance
- **Scalability**: Performance across different configurations

**Key Benchmarks**:
- Tokenization throughput (tokens/second)
- Generation latency (milliseconds)
- Memory allocation overhead
- KV cache performance
- Temperature sensitivity impact

## Running Tests

### Basic Usage

```bash
# Run all tests with default settings
./run_tests.sh

# Run with verbose output
./run_tests.sh --verbose

# Use specific number of build jobs
./run_tests.sh --jobs 8

# Debug build with coverage
./run_tests.sh --debug coverage
```

### Available Commands

| Command | Description |
|---------|-------------|
| `all` | Run all tests (default) |
| `unit` | Run only unit tests |
| `integration` | Run only integration tests |
| `benchmarks` | Run performance benchmarks |
| `ctest` | Run tests using CTest |
| `coverage` | Generate coverage report |
| `clean` | Clean build artifacts |
| `build` | Build tests only |
| `help` | Show help message |

### Test Runner Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output |
| `-j, --jobs N` | Use N parallel jobs for building |
| `--debug` | Build with debug information |
| `--no-check` | Skip prerequisite checks |

## Interpreting Results

### Test Output
- **GREEN**: Test passed successfully
- **RED**: Test failed
- **YELLOW**: Test skipped (usually due to missing dependencies)
- **BLUE**: Informational messages

### Performance Metrics
Benchmark results include:
- **Time**: Execution time per operation
- **CPU**: CPU time usage
- **Iterations**: Number of test iterations
- **Bytes Processed**: Data throughput metrics
- **Items Processed**: Operation throughput

### Coverage Reports
When running with coverage enabled:
```bash
./run_tests.sh --debug coverage
```
- HTML report generated in `build/coverage/`
- Line coverage, function coverage, and branch coverage
- Identifies untested code paths

## Continuous Integration

### GitHub Actions
The test suite is designed to work with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Gemma Tests
  run: |
    cd tests
    ./run_tests.sh unit integration
```

### Test Selection for CI
- **Fast Tests**: Unit tests (< 5 minutes)
- **Medium Tests**: Integration tests with small models (< 15 minutes)
- **Full Tests**: All tests including benchmarks (< 30 minutes)

## Troubleshooting

### Common Issues

#### Missing Model Files
```
[WARNING] Some model files are missing from /c/codedev/llm/.models/
```
**Solution**: Download model files or run unit tests only:
```bash
./run_tests.sh unit
```

#### Compilation Errors
```
error: 'std::filesystem' is not a member of 'std'
```
**Solution**: Ensure C++20 support:
```bash
cmake .. -DCMAKE_CXX_STANDARD=20
```

#### Memory Issues
```
std::bad_alloc: out of memory
```
**Solution**: Reduce test concurrency or use smaller test data:
```bash
./run_tests.sh --jobs 2 unit
```

#### Missing Dependencies
```
CMake Error: Could not find package X
```
**Solution**: Install system dependencies or let CMake fetch them automatically.

### Debug Mode
For detailed debugging:
```bash
# Build with debug symbols
./run_tests.sh --debug --verbose

# Run specific test with debugger
gdb ./build/tests/test_model_loading
```

### Performance Issues
If tests run slowly:
1. Check available RAM (need 8GB+)
2. Use SSD storage for model files
3. Reduce parallel jobs: `--jobs 2`
4. Run unit tests only for faster feedback

## Test Development

### Adding New Tests

1. **Create test file** in appropriate directory (`unit/`, `integration/`, `benchmarks/`)
2. **Follow naming convention**: `test_<component>.cc`
3. **Use appropriate framework**: GoogleTest for unit/integration, GoogleBenchmark for performance
4. **Update CMakeLists.txt** to include new test executable
5. **Add to test runner** script if needed

### Test Best Practices

- **Isolation**: Each test should be independent
- **Determinism**: Tests should produce consistent results
- **Speed**: Unit tests should complete quickly (< 1s each)
- **Coverage**: Test both success and failure paths
- **Documentation**: Clear test names and descriptions

### Example Test Structure

```cpp
#include <gtest/gtest.h>
#include "gemma/gemma.h"

namespace gcpp {
namespace {

class ComponentTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test setup
    }

    void TearDown() override {
        // Test cleanup
    }
};

TEST_F(ComponentTest, SpecificFunctionality) {
    // Arrange
    // Act
    // Assert
}

} // namespace
} // namespace gcpp
```

## Performance Baselines

### Expected Performance (Gemma 2B on modern hardware)
- **Tokenization**: > 10,000 tokens/second
- **Generation**: > 50 tokens/second
- **Memory**: < 4GB peak usage
- **Startup**: < 5 seconds to first token

### Regression Detection
The benchmark suite establishes performance baselines to detect regressions:
- **5% slowdown**: Warning threshold
- **15% slowdown**: Failure threshold
- **Memory increase**: > 20% increase fails

## Contributing

### Code Review Checklist
- [ ] Tests compile without warnings
- [ ] All tests pass on target platforms
- [ ] Performance impact assessed
- [ ] Edge cases covered
- [ ] Documentation updated
- [ ] CI integration works

### Test Coverage Goals
- **Unit Tests**: > 90% line coverage
- **Integration Tests**: Critical paths covered
- **Error Handling**: All error conditions tested
- **Performance**: Key metrics benchmarked

## Enhanced Testing Framework

### New Features

The testing framework has been enhanced with several advanced features:

#### Comprehensive Test Categories
- **Unit Tests**: Fast, isolated tests with mock implementations
  - Sampling algorithms (Softmax, TopK, temperature effects)
  - MCP protocol testing with mock server
  - Memory management and error handling
- **Integration Tests**: Component interaction testing
  - Backend compatibility across SIMD instruction sets
  - End-to-end model loading pipeline
  - Full inference workflow validation
- **Performance Tests**: Automated benchmarking and regression detection
  - Core operation benchmarks (matrix multiplication, attention)
  - Memory allocation and bandwidth testing
  - Automated baseline comparison

#### Enhanced Build System
```bash
# Enable comprehensive testing framework
cmake --preset make -DGEMMA_ENABLE_ENHANCED_TESTS=ON -DGEMMA_ENABLE_TESTS=ON

# Build specific test categories
cmake --build --preset make --target test_gemma_core --parallel 4
cmake --build --preset make --target test_backends_integration --parallel 4
cmake --build --preset make --target benchmark_inference --parallel 4
```

#### Test Runner Script
```bash
# Using the enhanced test runner
python ../run_tests.py build           # Build all tests
python ../run_tests.py unit            # Run unit tests only
python ../run_tests.py integration     # Run integration tests
python ../run_tests.py performance     # Run performance benchmarks
python ../run_tests.py quick           # Fast feedback tests
python ../run_tests.py run test_name   # Run specific test
```

#### CI/CD Integration
- **PR Validation**: Fast build and essential tests (5 minutes)
- **Comprehensive Testing**: Full test suite with coverage analysis
- **Performance Monitoring**: Daily benchmark execution and regression detection

#### Performance Analysis
```bash
# Generate performance baseline
./build/benchmark_inference --benchmark_out=baseline.json --benchmark_out_format=json

# Compare current performance to baseline
python tests/performance/compare_benchmarks.py baseline.json current.json --threshold 0.05
```

### Test Utilities and Fixtures

The framework provides comprehensive test utilities:

```cpp
// Base test class with common setup
class GemmaTestBase : public ::testing::Test {
    // Provides allocator, threading context, test data generator
};

// Parameterized tests for different model types
template<typename T>
class ModelTypeTest : public GemmaTestBase {
    // T can be Model::GEMMA_2B, Model::GEMMA_7B, etc.
};

// Test data generation
class TestDataGenerator {
    std::vector<float> GenerateRandomFloats(size_t count);
    std::vector<int> GenerateRandomTokens(size_t count, int vocab_size);
    std::vector<float> GenerateNormalizedProbabilities(size_t count);
};
```

### Mock Implementations

- **MockModelConfig**: Configurable model settings for testing
- **MockGemmaMCPServer**: Complete MCP protocol implementation
- **TestAllocator**: Instrumented allocator for memory testing

For questions or issues with the test suite, please refer to the main project documentation or open an issue in the repository.