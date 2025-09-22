# Multi-stage build for hybrid Python-Rust LLM chatbot
FROM rust:1.75 AS rust-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy Rust workspace files
COPY Cargo.toml Cargo.lock ./
COPY rust_extensions/ ./rust_extensions/
COPY rag-redis-system/ ./rag-redis-system/

# Build Rust extensions
RUN cargo build --release --workspace

# Python stage
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    cmake \
    redis-server \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy Python requirements
COPY pyproject.toml uv.lock ./
COPY requirements.txt ./

# Install Python dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv pip install -r requirements.txt && \
    uv pip install -e .

# Copy Rust build artifacts from builder
COPY --from=rust-builder /build/target/release/*.so /app/rust_extensions/
COPY --from=rust-builder /build/target/release/*.dll /app/rust_extensions/
COPY --from=rust-builder /build/rag-redis-system/target/release/rag-redis-* /app/bin/

# Copy application code
COPY src/ ./src/
COPY main.py ./
COPY .env.template ./.env

# Copy test and example files
COPY tests/ ./tests/
COPY examples/ ./examples/

# Create necessary directories
RUN mkdir -p models models_cache .cache logs

# Environment variables
ENV PYTHONPATH=/app
ENV GEMMA_CACHE_DIR=/app/models
ENV RUST_LOG=info

# Redis configuration with container-optimized defaults
ENV DOCKER_CONTAINER=true
ENV REDIS_URL=redis://redis:6379
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV REDIS_DB=0
ENV REDIS_MAX_CONNECTIONS=20
ENV REDIS_CONNECTION_TIMEOUT=5
ENV REDIS_SOCKET_TIMEOUT=10
ENV REDIS_ENABLE_FALLBACK=false

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command (can be overridden)
CMD ["sh", "-c", "redis-server --daemonize yes && uv run python main.py --lightweight"]
