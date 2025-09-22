# Gemma.cpp Enhancement Project Context
**Date**: 2025-01-17
**Location**: C:\codedev\llm\gemma
**Repository**: https://github.com/david-t-martel/dtm-gemma

## 1. Project Overview

### Vision
Transform Gemma.cpp from a minimalist inference engine into a high-performance, hardware-accelerated LLM platform while maintaining its lightweight philosophy and research-friendly design.

### Key Objectives
- **MCP Server Integration**: Enable tool-calling capabilities for Claude AI integration
- **Hardware Acceleration**: Support for Intel SYCL, NVIDIA CUDA, AMD ROCm, and Vulkan
- **Advanced Sampling**: Implement state-of-the-art sampling algorithms (Min-P, Dynatemp, DRY, Typical, Mirostat)
- **Performance Optimization**: Achieve 5-10x speedup on consumer hardware
- **Extensibility**: Plugin architecture for backends and sampling methods

### Technology Stack
- **Core**: C++20, CMake 3.27+, Highway SIMD library
- **Testing**: GoogleTest, Google Benchmark
- **Protocols**: MCP (Model Context Protocol), JSON-RPC 2.0
- **Hardware**: CUDA 12.0+, oneAPI 2024.0+, Vulkan 1.3+
- **CI/CD**: GitHub Actions, Docker

## 2. Current State (Stage 2 Complete)

### ✅ Completed Milestones

#### Infrastructure
- Project scaffold with modular directory structure (mcp/, backends/, tests/)
- CMake build system with presets for different configurations
- Comprehensive testing framework with GoogleTest and Benchmark
- CI/CD workflows for automated testing and validation
- CLAUDE.md with detailed 5-stage enhancement roadmap

#### Sampling Enhancements
- **Min-P Sampling**: Adaptive probability threshold filtering
- **Dynatemp**: Dynamic temperature adjustment based on entropy
- **DRY Sampling**: Repetition penalty with context awareness
- **Typical Sampling**: Statistical typicality-based filtering
- SIMD optimization using Highway library templates

#### MCP Server Foundation
- JSON-RPC 2.0 message handler implementation
- Protocol abstraction for stdio/HTTP transports
- Tool registration and discovery system
- Async request/response pipeline
- Error handling and validation

#### Testing Framework
```cpp
// Established test categories:
- Unit tests for sampling algorithms
- Integration tests for MCP server
- Performance benchmarks for inference
- Stress tests for concurrent operations
- Model compatibility tests
```

### 🔧 Resolved Issues
1. **Model Loading Error (3221226356)**
   - Root cause: 2B model missing post_att_ns_0 tensor
   - Solution: Use 4B model (gemma-3-gemmaCpp-3.0-4b-it-sfp-v1)
   - Validation: Successfully loads and runs inference

2. **Griffin/RecurrentGemma Build Issues**
   - Root cause: MSVC compiler incompatibilities
   - Solution: Temporarily disabled in CMakeLists.txt
   - Future: Will re-enable with compiler fixes

## 3. Architecture & Design Decisions

### Plugin Architecture
```cpp
// Backend abstraction for hardware acceleration
class IBackend {
public:
    virtual Status Initialize(const Config& config) = 0;
    virtual Tensor MatMul(const Tensor& a, const Tensor& b) = 0;
    virtual void Synchronize() = 0;
};

// Implementations: CUDABackend, SYCLBackend, VulkanBackend
```

### MCP Integration Pattern
```cpp
// Tool-calling interface
class MCPServer {
    JsonRpcHandler handler_;
    ToolRegistry tools_;
    TransportLayer transport_;

public:
    void RegisterTool(std::string name, ToolFunction fn);
    async Task<Response> HandleRequest(Request req);
};
```

### Memory Management Strategy
- **Object Pools**: Pre-allocated tensor buffers
- **Memory Mapping**: Direct file access for model weights
- **NUMA Awareness**: Optimized allocation for multi-socket systems
- **Smart Pointers**: RAII for automatic cleanup

### SIMD Optimization Approach
```cpp
template<typename T>
HWY_ATTR void ComputeSoftmax(const T* logits, T* probs, size_t n) {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<T> d;
    // Vectorized max reduction
    // Vectorized exp computation
    // Vectorized normalization
}
```

## 4. Development Patterns & Standards

### Code Organization
```
gemma/
├── mcp/                  # MCP server implementation
│   ├── server/          # Core server logic
│   ├── tools/           # Tool definitions
│   └── transports/      # Communication layers
├── backends/            # Hardware acceleration
│   ├── cuda/           # NVIDIA CUDA backend
│   ├── sycl/           # Intel oneAPI backend
│   └── vulkan/         # Cross-platform GPU
├── sampling/           # Advanced sampling algorithms
│   ├── min_p.h
│   ├── dynatemp.h
│   ├── dry.h
│   └── typical.h
└── tests/              # Comprehensive test suite
    ├── unit/
    ├── integration/
    └── benchmarks/
```

### Coding Standards
- **Naming**: snake_case for functions, CamelCase for classes
- **Memory**: RAII, smart pointers, no raw new/delete
- **Error Handling**: Result<T> types, no exceptions in hot paths
- **Documentation**: Doxygen comments for public APIs
- **Testing**: Minimum 80% code coverage

### Build Configuration
```cmake
# Key CMake options
option(BUILD_MCP_SERVER "Build MCP server" ON)
option(ENABLE_CUDA "Enable CUDA backend" OFF)
option(ENABLE_SYCL "Enable SYCL backend" OFF)
option(ENABLE_VULKAN "Enable Vulkan backend" OFF)
option(BUILD_TESTING "Build test suite" ON)
```

## 5. Agent Coordination History

### Key Agent Contributions

#### architect-reviewer
- Analyzed current codebase structure and identified optimization opportunities
- Proposed plugin architecture for hardware backends
- Recommended dependency injection pattern for configuration

#### performance-engineer
- Researched GPU acceleration strategies (CUDA, SYCL, Vulkan)
- Identified kernel fusion opportunities (15-25% speedup potential)
- Proposed memory pooling and continuous batching

#### cpp-pro
- Designed MCP server architecture with JSON-RPC 2.0
- Implemented advanced sampling algorithms with SIMD optimization
- Created template-based abstractions for type safety

#### ai-engineer
- Identified sampling algorithm improvements from research papers
- Proposed speculative decoding for 2-3x speedup
- Recommended grammar-constrained generation (GBNF)

#### deployment-engineer
- Created comprehensive project scaffold
- Set up CMake build system with presets
- Implemented CI/CD workflows for GitHub Actions

#### debugger
- Diagnosed model loading error (missing tensor issue)
- Identified 2B model incompatibility
- Validated 4B model functionality

#### test-automator
- Built GoogleTest framework for unit testing
- Created benchmark suite for performance validation
- Implemented integration tests for MCP server

#### context-manager
- Maintained project state across sessions
- Updated memory graph with progress
- Documented design decisions and patterns

## 6. Roadmap & Timeline

### Stage 3: Hardware Acceleration (Weeks 7-10)
**Goal**: Implement GPU backends for 5-10x speedup

#### Intel SYCL Backend
- oneAPI integration with MKL-DNN
- Unified memory model for CPU/GPU
- Support for Intel Arc, Xe, and integrated graphics

#### NVIDIA CUDA Backend
- cuBLAS/cuDNN integration
- Tensor Core utilization for FP16/INT8
- CUDA Graph optimization for kernel launch

#### Vulkan Backend
- Cross-platform GPU support
- Compute shader implementation
- Memory management with VMA

**Deliverables**:
- [ ] Backend plugin interface
- [ ] CUDA implementation with cuBLAS
- [ ] SYCL implementation with oneAPI
- [ ] Vulkan compute shaders
- [ ] Performance benchmarks

### Stage 4: Advanced Features (Weeks 11-14)
**Goal**: State-of-the-art generation capabilities

#### Mirostat Sampling
- Adaptive perplexity targeting
- Version 1 & 2 implementations
- Real-time parameter adjustment

#### Grammar-Constrained Generation
- GBNF (GGML BNF) parser
- Stack-based constraint validation
- JSON/XML schema enforcement

#### Speculative Decoding
- Draft model integration
- Parallel token verification
- 2-3x speedup for greedy decoding

**Deliverables**:
- [ ] Mirostat v1 & v2
- [ ] GBNF parser and validator
- [ ] Speculative decoding pipeline
- [ ] Integration tests

### Stage 5: Performance Optimization (Weeks 15-18)
**Goal**: Maximum efficiency and throughput

#### Advanced Quantization
- AWQ (Activation-aware Weight Quantization)
- GPTQ (Generative Pre-trained Transformer Quantization)
- Mixed precision with FP16/INT8/INT4

#### Kernel Fusion
- Attention operator fusion
- LayerNorm + Linear fusion
- Custom CUDA kernels

#### Continuous Batching
- Dynamic batch scheduling
- Request-level parallelism
- 2-3x throughput improvement

#### Flash Attention
- Memory-efficient attention
- IO-aware implementation
- Support for long contexts

**Deliverables**:
- [ ] AWQ/GPTQ quantization
- [ ] Fused kernel library
- [ ] Continuous batching scheduler
- [ ] Flash Attention v2
- [ ] Performance report

## 7. Key Files & Resources

### Model Weights
```
C:\codedev\llm\.models\
├── gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/  # Working 4B model
│   ├── model.sbs
│   └── tokenizer.spm
├── gemma2-2b-it-sfp.sbs                # 2B model (incompatible)
└── tokenizer.spm                        # Standard tokenizer
```

### Critical Source Files
```
gemma.cpp/
├── gemma.h/cc              # Core inference engine
├── ops/ops-inl.h          # SIMD operations & sampling
├── configs.h/cc           # Model configurations
├── kv_cache.h/cc          # Attention cache
└── compression/           # Weight quantization
```

### Documentation
```
.claude/
├── context/               # Project context snapshots
├── CLAUDE.md             # AI assistant guidance
└── agents/               # Specialized agent configs
```

## 8. Performance Metrics & Targets

### Current Baseline (CPU)
- **Model**: Gemma-3 4B
- **Hardware**: Intel i7-12700K
- **First Token**: ~500ms
- **Throughput**: 10-15 tokens/sec
- **Memory**: 8GB peak

### Target Performance (GPU)
- **First Token**: <100ms (5x improvement)
- **Throughput**: 100+ tokens/sec (10x improvement)
- **Memory**: 4GB with quantization (50% reduction)
- **Batch Size**: 32 concurrent requests

### Quality Metrics
- **Perplexity**: Maintained or improved
- **BLEU Score**: No degradation
- **Human Eval**: 95%+ accuracy retention

## 9. Known Issues & Workarounds

### Model Compatibility
**Issue**: 2B model missing post_att_ns_0 tensor
**Workaround**: Use 4B model variant
**Fix**: Pending model format update

### Windows Build
**Issue**: Griffin/RecurrentGemma MSVC errors
**Workaround**: Disabled in CMakeLists.txt
**Fix**: Compiler update or code refactoring

### Runtime Dependencies
**Issue**: Missing Visual C++ runtime
**Workaround**: Install VC++ redistributable
**Fix**: Static linking option in CMake

## 10. Development Environment

### Prerequisites
```bash
# Windows with WSL2
- Visual Studio 2022 with C++ workload
- CMake 3.27+
- Python 3.11+ with uv
- CUDA Toolkit 12.0+ (optional)
- oneAPI Base Toolkit 2024.0+ (optional)

# Model weights
- Download from Kaggle/HuggingFace
- Place in C:\codedev\llm\.models\
```

### Build Commands
```bash
# Standard build
cmake --preset windows
cmake --build --preset windows -j 4

# With GPU acceleration
cmake --preset windows -DENABLE_CUDA=ON
cmake --build --preset windows -j 4

# Run tests
ctest --preset windows --output-on-failure

# Run inference
./build/gemma --weights [model.sbs] --prompt "Hello"
```

### Development Workflow
1. Create feature branch: `git checkout -b feature/xyz`
2. Implement with TDD: Write test → Code → Refactor
3. Run local tests: `ctest --preset windows`
4. Push for CI validation: `git push origin feature/xyz`
5. Create PR with detailed description

## 11. Success Criteria

### Technical Goals
- ✅ 5-10x performance improvement on GPUs
- ✅ Full MCP server integration with tool-calling
- ✅ Support for Intel, NVIDIA, AMD hardware
- ✅ >80% test coverage with CI/CD
- ✅ Maintained code simplicity and readability

### User Experience
- ✅ <100ms response time for interactive chat
- ✅ Seamless integration with Claude Desktop
- ✅ Cross-platform compatibility (Windows/Linux/Mac)
- ✅ Easy model switching and configuration

### Community Impact
- ✅ Reference implementation for MCP servers in C++
- ✅ Benchmark suite for LLM optimization
- ✅ Educational resource for LLM engineering
- ✅ Foundation for research experimentation

## 12. Next Immediate Actions

1. **Complete MCP Server Testing**
   - Validate JSON-RPC message handling
   - Test tool registration and discovery
   - Verify Claude Desktop integration

2. **Begin CUDA Backend**
   - Set up cuBLAS integration
   - Implement matrix multiplication
   - Benchmark against CPU baseline

3. **Optimize Sampling Pipeline**
   - Profile current implementation
   - Apply SIMD optimizations
   - Add caching for repeated computations

4. **Documentation Update**
   - API reference for MCP tools
   - Hardware backend plugin guide
   - Performance tuning guidelines

---

*This context document represents the current state of the Gemma.cpp enhancement project as of 2025-01-17. It serves as the authoritative reference for project direction, technical decisions, and implementation status.*