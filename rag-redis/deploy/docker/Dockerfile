# Multi-stage Dockerfile for RAG-Redis MCP Server
# Optimized for production deployment with minimal image size

# Build stage
FROM rust:1.75.0-slim-bullseye AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /build

# Copy workspace configuration
COPY Cargo.toml Cargo.lock ./
COPY rag-redis-system ./rag-redis-system
COPY rag-binaries ./rag-binaries
COPY rust-rag-extensions ./rust-rag-extensions

# Create dummy main files to cache dependencies
RUN mkdir -p rag-redis-system/mcp-server/src && \
    echo "fn main() {}" > rag-redis-system/mcp-server/src/main.rs && \
    mkdir -p rag-binaries/src/bin && \
    echo "fn main() {}" > rag-binaries/src/bin/cli.rs && \
    echo "fn main() {}" > rag-binaries/src/bin/server.rs

# Build dependencies (cached layer)
RUN cargo build --release --workspace

# Copy actual source code
COPY . .

# Build the actual binaries
RUN cargo build --release --bin mcp-server && \
    cargo build --release --bin cli && \
    cargo build --release --bin server

# Create health check script
RUN echo '#!/bin/bash\n\
timeout 5 bash -c "</dev/tcp/localhost/8080" 2>/dev/null\n\
exit_code=$?\n\
if [ $exit_code -eq 0 ]; then\n\
    echo "Health check passed"\n\
    exit 0\n\
else\n\
    echo "Health check failed"\n\
    exit 1\n\
fi' > /build/health-check && chmod +x /build/health-check

# Runtime stage
FROM debian:bullseye-slim AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r raguser \
    && useradd -r -g raguser -d /app -s /bin/bash raguser

# Create application directories
RUN mkdir -p /app/bin /app/data /app/cache /app/logs /app/models && \
    chown -R raguser:raguser /app

# Copy binaries from builder
COPY --from=builder /build/target/release/mcp-server /app/bin/
COPY --from=builder /build/target/release/cli /app/bin/
COPY --from=builder /build/target/release/server /app/bin/
COPY --from=builder /build/health-check /app/bin/

# Copy models directory if exists
COPY --chown=raguser:raguser models/ /app/models/

# Set executable permissions
RUN chmod +x /app/bin/* && \
    chown raguser:raguser /app/bin/*

# Switch to non-root user
USER raguser
WORKDIR /app

# Environment variables
ENV RUST_LOG=info \
    REDIS_URL=redis://redis:6379 \
    MCP_SERVER_HOST=0.0.0.0 \
    MCP_SERVER_PORT=8080 \
    RAG_DATA_DIR=/app/data \
    EMBEDDING_CACHE_DIR=/app/cache \
    LOG_DIR=/app/logs

# Expose ports
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/app/bin/health-check"]

# Default command
CMD ["/app/bin/mcp-server"]

# Development stage (for debugging)
FROM builder AS development

RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    strace \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install cargo tools for development
RUN cargo install cargo-watch cargo-audit cargo-outdated

WORKDIR /build

# Development command with hot reload
CMD ["cargo", "watch", "-x", "run --bin mcp-server"]

# Testing stage
FROM builder AS testing

# Install test dependencies
RUN apt-get update && apt-get install -y \
    redis-tools \
    netcat \
    && rm -rf /var/lib/apt/lists/*

# Run tests
RUN cargo test --workspace --all-features

# Copy test scripts
COPY deploy/scripts/test-container.sh /test-container.sh
RUN chmod +x /test-container.sh

CMD ["/test-container.sh"]