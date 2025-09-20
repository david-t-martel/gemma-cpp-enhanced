# Gemma.cpp Testing Guide

## Overview

A comprehensive test suite has been created for gemma.cpp that validates all critical functionality including model loading, tokenization, inference, performance, memory management, and error handling.

## Test Suite Components

### 📁 Directory Structure
```
/c/codedev/llm/gemma/tests/
├── unit/                          # Unit tests (6 files)
│   ├── test_model_loading.cc      # Model initialization & config tests
│   ├── test_tokenization.cc       # SentencePiece tokenizer tests
│   ├── test_memory_management.cc  # Memory allocation & threading tests
│   └── test_error_handling.cc     # Error conditions & edge cases
├── integration/                   # Integration tests (1 file)
│   └── test_inference.cc          # End-to-end inference pipeline
├── benchmarks/                    # Performance tests (1 file)
│   └── test_performance.cc        # Throughput & latency benchmarks
├── fixtures/                      # Test data
│   └── test_prompts.txt           # Sample prompts for testing
├── CMakeLists.txt                 # Build configuration
├── run_tests.sh                   # Automated test runner
├── README.md                      # Detailed documentation
└── TESTING_GUIDE.md              # This guide
```

### 🧪 Test Categories

#### Unit Tests
- **Model Loading** (test_model_loading.cc)
  - Configuration validation for Gemma 2B/3B models
  - Weight file loading and validation
  - Memory requirement calculations
  - Multi-model support testing

- **Tokenization** (test_tokenization.cc)
  - Encoding/decoding accuracy
  - Round-trip consistency
  - Unicode and special character handling
  - Performance baselines

- **Memory Management** (test_memory_management.cc)
  - KV cache allocation patterns
  - Threading context lifecycle
  - Resource cleanup validation
  - Stress testing and fragmentation resistance

- **Error Handling** (test_error_handling.cc)
  - Invalid configuration handling
  - File system error scenarios
  - Resource exhaustion recovery
  - Exception safety guarantees

#### Integration Tests
- **Inference** (test_inference.cc)
  - Complete generation pipeline
  - Parameter sensitivity (temperature, top-k)
  - Long sequence generation
  - Conversational context handling
  - Performance timing validation

#### Performance Benchmarks
- **Performance** (test_performance.cc)
  - Tokenization throughput
  - Generation latency measurements
  - Memory allocation overhead
  - Batch processing efficiency
  - Temperature sensitivity impact

## 🚀 Quick Start

### Prerequisites
1. **Build Tools**: CMake 3.11+, C++20 compiler, Make
2. **Model Files** (optional): Place in `/c/codedev/llm/.models/`
   - `tokenizer.spm`
   - `gemma2-2b-it-sfp.sbs`

### Running Tests

#### Simple Commands
```bash
cd /c/codedev/llm/gemma/tests

# Run all tests
./run_tests.sh

# Run specific categories
./run_tests.sh unit           # Unit tests only
./run_tests.sh integration    # Integration tests only
./run_tests.sh benchmarks     # Performance benchmarks
```

#### Advanced Usage
```bash
# Debug build with coverage
./run_tests.sh --debug coverage

# Verbose output with custom job count
./run_tests.sh --verbose --jobs 4 all

# Clean and rebuild
./run_tests.sh clean build unit
```

### Manual Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run individual tests
./tests/test_model_loading
./tests/test_tokenization
./tests/test_inference

# Use CTest for automated runs
ctest --output-on-failure
```

## 📊 Expected Results

### Unit Tests (< 5 minutes)
- **test_model_loading**: Configuration and file validation
- **test_tokenization**: Text processing accuracy
- **test_memory_management**: Resource handling
- **test_error_handling**: Edge case robustness

### Integration Tests (5-15 minutes)
- **test_inference**: End-to-end generation with model files
  - Requires actual Gemma model weights
  - Tests multiple generation scenarios
  - Validates output quality and timing

### Benchmarks (2-10 minutes)
- **test_performance**: Establishes performance baselines
  - Tokenization: >10K tokens/sec expected
  - Generation: >50 tokens/sec expected
  - Memory: <4GB peak usage
  - Latency: <100ms per token

## 🔧 Test Framework Features

### Automatic Dependency Management
- **Google Test**: Unit testing framework
- **Google Benchmark**: Performance measurement
- **Highway**: SIMD operations
- **SentencePiece**: Tokenization library

### Smart Test Selection
- **Fast Tests**: Unit tests for rapid feedback
- **Model-Dependent**: Integration tests skip if models unavailable
- **Resource-Aware**: Benchmarks adjust to system capabilities

### Comprehensive Coverage
- **Success Paths**: Normal operation validation
- **Error Paths**: Exception and failure handling
- **Edge Cases**: Boundary conditions and limits
- **Performance**: Regression detection

## 🐛 Troubleshooting

### Common Issues

#### Missing Dependencies
```bash
# Install CMake on Ubuntu/WSL
sudo apt update && sudo apt install cmake build-essential

# Install on Windows (use vcpkg or VS installer)
```

#### Missing Model Files
```
[WARNING] Some model files are missing from /c/codedev/llm/.models/
```
**Solution**: Download from Kaggle/HuggingFace or run unit tests only:
```bash
./run_tests.sh unit
```

#### Memory Issues
```
std::bad_alloc: out of memory
```
**Solutions**:
- Ensure 8GB+ RAM available
- Reduce parallelism: `--jobs 2`
- Close other applications
- Run unit tests only

#### Compilation Errors
```
error: 'std::filesystem' is not a member of 'std'
```
**Solution**: Ensure C++20 support:
```bash
cmake .. -DCMAKE_CXX_STANDARD=20
```

### Debug Mode
```bash
# Build with debug symbols and run with verbose output
./run_tests.sh --debug --verbose unit

# Run specific test with debugger
gdb ./build/tests/test_model_loading

# Generate coverage report
./run_tests.sh --debug coverage
```

## 📈 Performance Monitoring

### Baseline Metrics (Gemma 2B on modern hardware)
- **Tokenization**: 10,000+ tokens/second
- **Generation**: 50+ tokens/second (varies by prompt)
- **Memory**: <4GB peak usage
- **Startup**: <5 seconds to first token

### Regression Detection
- **5% slowdown**: Warning threshold
- **15% slowdown**: Failure threshold
- **Memory increase**: >20% fails

### Benchmark Output
```
BM_TokenizeShortText        1000000    1.23 us    10.2M tokens/sec
BM_ShortGeneration             100   12.5 ms      80 tokens/sec
BM_MemoryFootprint             500    2.3 ms     4.2GB peak
```

## 🔄 Continuous Integration

### CI/CD Integration
```yaml
# Example GitHub Actions
- name: Run Gemma Tests
  run: |
    cd tests
    ./run_tests.sh unit integration

- name: Performance Regression Check
  run: |
    cd tests
    ./run_tests.sh benchmarks > results.txt
    # Compare with baseline
```

### Test Tiers
- **Smoke Tests**: Basic functionality (unit tests)
- **Integration Tests**: Core features with models
- **Full Suite**: All tests including benchmarks

## 📝 Adding New Tests

### File Template
```cpp
#include <gtest/gtest.h>
#include "gemma/gemma.h"

namespace gcpp {
namespace {

class NewFeatureTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test setup
    }

    void TearDown() override {
        // Test cleanup
    }
};

TEST_F(NewFeatureTest, BasicFunctionality) {
    // Arrange - Set up test data

    // Act - Execute the code under test

    // Assert - Verify results
    EXPECT_EQ(expected, actual);
}

} // namespace
} // namespace gcpp
```

### Update Build Configuration
1. Add test file to appropriate directory
2. Update `CMakeLists.txt` with new executable
3. Add to test runner script if needed
4. Update documentation

## 📚 Test Coverage Goals

- **Unit Tests**: >90% line coverage
- **Integration Tests**: Critical paths covered
- **Error Handling**: All error conditions tested
- **Performance**: Key metrics benchmarked

## 🤝 Best Practices

### Test Design
- **Independence**: Each test is isolated
- **Determinism**: Consistent results across runs
- **Speed**: Unit tests complete quickly (<1s each)
- **Clarity**: Descriptive names and assertions

### Maintenance
- **Regular Updates**: Keep tests current with code changes
- **Baseline Updates**: Refresh performance baselines periodically
- **Documentation**: Update guides when adding features
- **CI Integration**: Ensure tests run in automated pipelines

## 📞 Support

For issues with the test suite:
1. Check this guide and README.md
2. Review test output for specific error messages
3. Try simplified test runs (unit tests only)
4. Check system requirements and dependencies
5. Open an issue in the main repository

The test suite is designed to be comprehensive yet practical, providing confidence in code quality while maintaining reasonable execution times for development workflows.