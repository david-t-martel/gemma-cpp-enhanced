# Enhanced Gemma.cpp Multi-Stage Docker Build
# Supports multiple hardware backends and optimized deployment

# ==============================================================================
# Build Stage 1: Base Development Environment
# ==============================================================================
FROM ubuntu:22.04 AS base-dev

LABEL maintainer="Enhanced Gemma.cpp Team"
LABEL description="Enhanced Gemma.cpp with Hardware Acceleration Backends"
LABEL version="1.0.0"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Update system and install essential build dependencies
RUN apt-get update && apt-get install -y \
    # Core build tools
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    unzip \
    pkg-config \
    # Compilers and toolchains
    gcc-11 \
    g++-11 \
    clang-14 \
    # Libraries and development headers
    libc6-dev \
    linux-headers-generic \
    # Python for build scripts
    python3 \
    python3-pip \
    python3-venv \
    # Additional utilities
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set default compiler versions
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-14 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-14 100

# ==============================================================================
# Build Stage 2: CUDA Support (Optional)
# ==============================================================================
FROM base-dev AS cuda-dev
ARG ENABLE_CUDA=false

# Install NVIDIA CUDA toolkit if enabled
RUN if [ "$ENABLE_CUDA" = "true" ]; then \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-0 && \
    rm -rf /var/lib/apt/lists/* cuda-keyring_1.0-1_all.deb; \
    fi

ENV CUDA_PATH=/usr/local/cuda
ENV PATH=${CUDA_PATH}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}

# ==============================================================================
# Build Stage 3: Intel oneAPI Support (Optional)
# ==============================================================================
FROM cuda-dev AS oneapi-dev
ARG ENABLE_SYCL=false

# Install Intel oneAPI if enabled
RUN if [ "$ENABLE_SYCL" = "true" ]; then \
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y intel-oneapi-dpcpp-cpp-compiler intel-oneapi-mkl-devel && \
    rm -rf /var/lib/apt/lists/*; \
    fi

ENV ONEAPI_ROOT=/opt/intel/oneapi
ENV PATH=${ONEAPI_ROOT}/compiler/latest/linux/bin:${PATH}

# ==============================================================================
# Build Stage 4: Vulkan SDK
# ==============================================================================
FROM oneapi-dev AS vulkan-dev
ARG ENABLE_VULKAN=true

# Install Vulkan SDK
RUN if [ "$ENABLE_VULKAN" = "true" ]; then \
    wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list && \
    apt-get update && \
    apt-get install -y vulkan-sdk && \
    rm -rf /var/lib/apt/lists/*; \
    fi

ENV VULKAN_SDK=/usr
ENV PATH=${VULKAN_SDK}/bin:${PATH}

# ==============================================================================
# Build Stage 5: OpenCL Support
# ==============================================================================
FROM vulkan-dev AS opencl-dev
ARG ENABLE_OPENCL=true

# Install OpenCL development files
RUN if [ "$ENABLE_OPENCL" = "true" ]; then \
    apt-get update && \
    apt-get install -y opencl-headers ocl-icd-opencl-dev && \
    rm -rf /var/lib/apt/lists/*; \
    fi

# ==============================================================================
# Build Stage 6: Source and Dependencies
# ==============================================================================
FROM opencl-dev AS build-stage

# Set working directory
WORKDIR /workspace

# Copy source code
COPY . .

# Create build directory
RUN mkdir -p build

# ==============================================================================
# Build Stage 7: CMake Configuration and Build
# ==============================================================================
FROM build-stage AS compile-stage

ARG BUILD_TYPE=Release
ARG ENABLE_CUDA=false
ARG ENABLE_SYCL=false
ARG ENABLE_VULKAN=true
ARG ENABLE_OPENCL=true
ARG ENABLE_MCP_SERVER=true
ARG ENABLE_TESTS=false
ARG CMAKE_PARALLEL_JOBS=4

# Configure build environment
ENV CC=gcc
ENV CXX=g++

# Initialize oneAPI environment if enabled
RUN if [ "$ENABLE_SYCL" = "true" ] && [ -f "${ONEAPI_ROOT}/setvars.sh" ]; then \
    /bin/bash -c "source ${ONEAPI_ROOT}/setvars.sh"; \
    fi

# Configure CMake
RUN cd build && \
    cmake .. \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DGEMMA_BUILD_MCP_SERVER=${ENABLE_MCP_SERVER} \
    -DGEMMA_BUILD_BACKENDS=ON \
    -DGEMMA_BUILD_ENHANCED_TESTS=${ENABLE_TESTS} \
    -DGEMMA_BUILD_BACKEND_TESTS=${ENABLE_TESTS} \
    -DGEMMA_BUILD_BENCHMARKS=ON \
    -DGEMMA_AUTO_DETECT_BACKENDS=ON \
    -DGEMMA_BUILD_SYCL_BACKEND=${ENABLE_SYCL} \
    -DGEMMA_BUILD_CUDA_BACKEND=${ENABLE_CUDA} \
    -DGEMMA_BUILD_VULKAN_BACKEND=${ENABLE_VULKAN} \
    -DGEMMA_BUILD_OPENCL_BACKEND=${ENABLE_OPENCL} \
    -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -DNDEBUG" \
    -DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -DNDEBUG"

# Build the project
RUN cd build && \
    ninja -j ${CMAKE_PARALLEL_JOBS}

# Install built artifacts
RUN cd build && \
    ninja install

# ==============================================================================
# Runtime Stage: Minimal Production Image
# ==============================================================================
FROM ubuntu:22.04 AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    # Core runtime libraries
    libstdc++6 \
    libgomp1 \
    # GPU runtime libraries (will be no-op if hardware not available)
    mesa-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

# Copy built executables and libraries
COPY --from=compile-stage /usr/local/bin/ /usr/local/bin/
COPY --from=compile-stage /usr/local/lib/ /usr/local/lib/

# Copy CUDA runtime libraries if built with CUDA
ARG ENABLE_CUDA=false
RUN if [ "$ENABLE_CUDA" = "true" ]; then \
    mkdir -p /usr/local/cuda/lib64; \
    fi
COPY --from=compile-stage /usr/local/cuda/lib64/libcudart*.so* /usr/local/cuda/lib64/ 2>/dev/null || true
COPY --from=compile-stage /usr/local/cuda/lib64/libcublas*.so* /usr/local/cuda/lib64/ 2>/dev/null || true

# Copy Intel oneAPI runtime if built with SYCL
ARG ENABLE_SYCL=false
COPY --from=compile-stage /opt/intel/oneapi/compiler/latest/linux/lib/libsycl*.so* /usr/local/lib/ 2>/dev/null || true

# Set up library paths
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Create non-root user for security
RUN groupadd -r gemma && useradd -r -g gemma -d /app -s /bin/bash gemma

# Set up application directory
WORKDIR /app
RUN chown gemma:gemma /app

# Create models directory with proper permissions
RUN mkdir -p /app/models && chown gemma:gemma /app/models

# Switch to non-root user
USER gemma

# Create convenience scripts
RUN echo '#!/bin/bash\nexec /usr/local/bin/gemma "$@"' > /app/gemma && chmod +x /app/gemma
RUN echo '#!/bin/bash\nexec /usr/local/bin/gemma_mcp_stdio_server "$@"' > /app/mcp_server && chmod +x /app/mcp_server
RUN echo '#!/bin/bash\nexec /usr/local/bin/benchmarks "$@"' > /app/benchmark && chmod +x /app/benchmark

# Expose MCP server port (if using WebSocket transport in future)
EXPOSE 8080

# Default command
CMD ["./gemma", "--help"]

# ==============================================================================
# Build Metadata and Labels
# ==============================================================================
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=1.0.0

LABEL org.label-schema.build-date=${BUILD_DATE}
LABEL org.label-schema.name="Enhanced Gemma.cpp"
LABEL org.label-schema.description="Gemma.cpp with hardware acceleration backends"
LABEL org.label-schema.url="https://github.com/google/gemma.cpp"
LABEL org.label-schema.vcs-ref=${VCS_REF}
LABEL org.label-schema.vcs-url="https://github.com/google/gemma.cpp"
LABEL org.label-schema.vendor="Enhanced Gemma.cpp Team"
LABEL org.label-schema.version=${VERSION}
LABEL org.label-schema.schema-version="1.0"

# Hardware acceleration capabilities
LABEL gemma.backends.cuda=${ENABLE_CUDA}
LABEL gemma.backends.sycl=${ENABLE_SYCL}
LABEL gemma.backends.vulkan=${ENABLE_VULKAN}
LABEL gemma.backends.opencl=${ENABLE_OPENCL}
LABEL gemma.mcp.enabled=${ENABLE_MCP_SERVER}