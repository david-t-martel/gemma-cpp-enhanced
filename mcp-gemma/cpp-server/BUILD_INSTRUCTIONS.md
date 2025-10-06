# MCP-Gemma WebSocket Server - Build Instructions

This document provides instructions for building the MCP-Gemma WebSocket server implementation.

## Prerequisites

### Required Dependencies

1. **C++17 compatible compiler**
   - MSVC 2019+ (Windows)
   - GCC 8+ (Linux)
   - Clang 7+ (macOS)

2. **CMake 3.16+**

3. **WebSocket++ (header-only)**
   ```bash
   # Via vcpkg
   vcpkg install websocketpp

   # Or clone directly
   git clone https://github.com/zaphoyd/websocketpp.git third_party/websocketpp
   ```

4. **nlohmann/json (header-only)**
   ```bash
   # Via vcpkg
   vcpkg install nlohmann-json

   # Via package manager
   # Ubuntu/Debian: sudo apt-get install nlohmann-json3-dev
   # Fedora: sudo dnf install json-devel
   # macOS: brew install nlohmann-json
   ```

5. **Boost Libraries (system component)**
   ```bash
   # Via vcpkg
   vcpkg install boost-system

   # Via package manager
   # Ubuntu/Debian: sudo apt-get install libboost-system-dev
   # Fedora: sudo dnf install boost-devel
   # macOS: brew install boost
   ```

6. **Gemma.cpp** (must be built separately)
   - Clone and build the Gemma.cpp repository
   - Set `GEMMA_CPP_ROOT` to the path during CMake configuration

### Optional Dependencies

- **Threading library** (usually provided by system)
- **Network libraries** (ws2_32, wsock32 on Windows)

## Build Instructions

### Using vcpkg (Recommended)

1. **Install vcpkg** (if not already installed):
   ```bash
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg
   ./bootstrap-vcpkg.sh  # Linux/macOS
   # or
   .\bootstrap-vcpkg.bat  # Windows
   ```

2. **Install dependencies**:
   ```bash
   vcpkg install websocketpp nlohmann-json boost-system
   ```

3. **Configure and build**:
   ```bash
   cd mcp-gemma/cpp-server
   mkdir build
   cd build

   # Configure with vcpkg
   cmake .. -DCMAKE_TOOLCHAIN_FILE=[path-to-vcpkg]/scripts/buildsystems/vcpkg.cmake \
            -DGEMMA_CPP_ROOT=[path-to-gemma-cpp]

   # Build
   cmake --build . --config Release
   ```

### Manual Build (Without vcpkg)

1. **Install dependencies manually** using your system package manager

2. **Configure and build**:
   ```bash
   cd mcp-gemma/cpp-server
   mkdir build
   cd build

   # Configure
   cmake .. -DGEMMA_CPP_ROOT=[path-to-gemma-cpp] \
            -DWEBSOCKETPP_INCLUDE_DIR=[path-to-websocketpp] \
            -DNLOHMANN_JSON_INCLUDE_DIR=[path-to-nlohmann-json]

   # Build
   cmake --build . --config Release
   ```

### Windows-specific Build

```cmd
# Using Visual Studio 2022
cd mcp-gemma\cpp-server
mkdir build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64 ^
         -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake ^
         -DGEMMA_CPP_ROOT=C:\path\to\gemma

cmake --build . --config Release
```

## Configuration

### CMake Options

- `GEMMA_CPP_ROOT`: Path to Gemma.cpp repository (required)
- `WEBSOCKETPP_INCLUDE_DIR`: Path to WebSocket++ headers
- `NLOHMANN_JSON_INCLUDE_DIR`: Path to nlohmann/json headers
- `CMAKE_BUILD_TYPE`: Debug or Release (default: Release)

### Example Configuration

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGEMMA_CPP_ROOT=/home/user/gemma.cpp \
  -DWEBSOCKETPP_INCLUDE_DIR=/usr/include/websocketpp \
  -DNLOHMANN_JSON_INCLUDE_DIR=/usr/include
```

## Running the Server

After successful build, run:

```bash
./mcp_gemma_server --host localhost --port 8080 --model /path/to/model.sbs --tokenizer /path/to/tokenizer.spm
```

### Command Line Options

- `--host`: Server host (default: localhost)
- `--port`: Server port (default: 8080)
- `--model`: Path to Gemma model file
- `--tokenizer`: Path to tokenizer file
- `--max-connections`: Maximum WebSocket connections (default: 100)
- `--log-level`: Logging level (DEBUG, INFO, WARN, ERROR)

## Testing the Server

### Basic Connection Test

```javascript
const ws = new WebSocket('ws://localhost:8080', ['mcp']);

ws.onopen = function() {
    console.log('Connected to MCP server');

    // Send MCP initialize request
    ws.send(JSON.stringify({
        jsonrpc: '2.0',
        id: 1,
        method: 'initialize',
        params: {
            protocolVersion: '2024-11-05',
            clientInfo: {
                name: 'test-client',
                version: '1.0.0'
            }
        }
    }));
};

ws.onmessage = function(event) {
    console.log('Received:', JSON.parse(event.data));
};
```

### Using curl for HTTP tests

```bash
# Test server status (if HTTP endpoint is available)
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

## Troubleshooting

### Common Issues

1. **WebSocket++ not found**
   - Install via vcpkg or download headers manually
   - Set `WEBSOCKETPP_INCLUDE_DIR` correctly

2. **nlohmann/json not found**
   - Install development package or header-only library
   - Set `NLOHMANN_JSON_INCLUDE_DIR` correctly

3. **Boost not found**
   - Install boost-system development package
   - On Windows, use vcpkg for easier management

4. **Gemma.cpp integration issues**
   - Ensure Gemma.cpp is built and `GEMMA_CPP_ROOT` is correct
   - Check that all Gemma.cpp headers are accessible

5. **Linking errors on Windows**
   - Ensure ws2_32.lib and wsock32.lib are available
   - Use Visual Studio Command Prompt for building

### Debug Build

For debugging, use:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .
```

This enables additional logging and debug symbols.

## Performance Notes

- Use Release build for production deployment
- Consider system-specific optimizations (CPU flags, etc.)
- Monitor memory usage with large models
- Use appropriate connection limits based on hardware

## Integration with Gemma.cpp

Make sure to:

1. Build Gemma.cpp separately
2. Have model weights in `.sbs` format
3. Have compatible tokenizer files
4. Set appropriate memory limits for your hardware

For model conversion and preparation, refer to the Gemma.cpp documentation.