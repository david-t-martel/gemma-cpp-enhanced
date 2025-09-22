# Gemma.cpp Enhanced

Enhanced C++ inference engine for Google's Gemma models with MCP integration, hardware acceleration, and comprehensive testing framework.

## Features

- **Multi-platform support**: Linux, Windows, macOS
- **Hardware acceleration**: CUDA, SYCL, Vulkan backends
- **MCP integration**: Model Context Protocol server
- **Comprehensive testing**: Unit, integration, performance tests
- **CI/CD pipeline**: Automated builds and deployments

## Getting Started

### Prerequisites

- CMake 3.24+
- C++20 compatible compiler
- Git with submodules support

### Quick Build

```bash
git clone --recursive https://github.com/david-t-martel/gemma-cpp-enhanced.git
cd gemma-cpp-enhanced
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## License

This project follows the original Gemma.cpp licensing terms.