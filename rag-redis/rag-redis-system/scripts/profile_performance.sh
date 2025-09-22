#!/bin/bash

# RAG-Redis Performance Profiling Script
# This script uses cargo-flamegraph to identify hotspots and performance bottlenecks

set -e

echo "=== RAG-Redis Performance Profiling ==="
echo "This script will profile the RAG-Redis system using flame graphs"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"

    # Check if cargo-flamegraph is installed
    if ! command -v cargo-flamegraph &> /dev/null; then
        echo -e "${RED}cargo-flamegraph not found. Installing...${NC}"
        cargo install flamegraph
    fi

    # Check if perf is available (Linux)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if ! command -v perf &> /dev/null; then
            echo -e "${RED}perf not found. Please install linux-tools-generic${NC}"
            exit 1
        fi
    fi

    echo -e "${GREEN}Prerequisites satisfied${NC}"
}

# Create output directory
setup_output_dir() {
    OUTPUT_DIR="performance_profiles"
    mkdir -p "$OUTPUT_DIR"
    echo "Output directory: $OUTPUT_DIR"
}

# Profile baseline vector operations
profile_vector_operations() {
    echo -e "${YELLOW}Profiling vector operations (baseline)...${NC}"

    cargo flamegraph --test performance_test --output="$OUTPUT_DIR/vector_operations_baseline.svg" -- \
        --exact performance_test::tests::test_baseline_performance_medium

    echo -e "${GREEN}Vector operations profile saved to $OUTPUT_DIR/vector_operations_baseline.svg${NC}"
}

# Profile SIMD optimizations
profile_simd_operations() {
    echo -e "${YELLOW}Profiling SIMD distance calculations...${NC}"

    cargo flamegraph --test performance_test --output="$OUTPUT_DIR/simd_operations.svg" -- \
        --exact performance_test::tests::test_distance_metric_comparison

    echo -e "${GREEN}SIMD operations profile saved to $OUTPUT_DIR/simd_operations.svg${NC}"
}

# Profile search operations
profile_search_operations() {
    echo -e "${YELLOW}Profiling search operations...${NC}"

    cargo flamegraph --test vector_store --output="$OUTPUT_DIR/search_operations.svg" -- \
        --exact vector_store::tests::test_search_top_k

    echo -e "${GREEN}Search operations profile saved to $OUTPUT_DIR/search_operations.svg${NC}"
}

# Profile memory allocations
profile_memory_operations() {
    echo -e "${YELLOW}Profiling memory operations...${NC}"

    # Use CARGO_PROFILE_RELEASE_DEBUG=true to get better symbols
    CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --test vector_store \
        --output="$OUTPUT_DIR/memory_operations.svg" -- \
        --exact vector_store::tests::test_add_vector

    echo -e "${GREEN}Memory operations profile saved to $OUTPUT_DIR/memory_operations.svg${NC}"
}

# Create performance benchmark with profiling
run_comprehensive_benchmark() {
    echo -e "${YELLOW}Running comprehensive performance benchmark...${NC}"

    # Create a custom benchmark that runs longer for better profiling
    cat > benches/comprehensive_profile.rs << 'EOF'
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rag_redis_system::{Config, RagSystem};
use std::time::Duration;

async fn benchmark_full_pipeline() {
    let config = Config::default();
    let system = RagSystem::new(config).await.unwrap();

    // Simulate document ingestion and search
    for i in 0..100 {
        let content = format!("This is test document number {} with some meaningful content", i);
        let metadata = serde_json::json!({"id": i, "type": "test"});

        let doc_id = system.ingest_document(&content, metadata).await.unwrap();

        // Perform searches
        let results = system.search("test document", 5).await.unwrap();
        black_box(results);
    }
}

fn comprehensive_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("full_pipeline", |b| {
        b.to_async(&rt).iter(|| async {
            benchmark_full_pipeline().await;
        });
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(60))
        .sample_size(10);
    targets = comprehensive_benchmark
}
criterion_main!(benches);
EOF

    # Profile the comprehensive benchmark
    cargo flamegraph --bench comprehensive_profile \
        --output="$OUTPUT_DIR/comprehensive_benchmark.svg"

    echo -e "${GREEN}Comprehensive benchmark profile saved to $OUTPUT_DIR/comprehensive_benchmark.svg${NC}"
}

# Generate performance report
generate_report() {
    echo -e "${YELLOW}Generating performance report...${NC}"

    cat > "$OUTPUT_DIR/PROFILING_REPORT.md" << 'EOF'
# RAG-Redis Performance Profiling Report

This report contains flame graph profiles for different components of the RAG-Redis system.

## Flame Graphs Generated

### 1. Vector Operations Baseline (`vector_operations_baseline.svg`)
- **Purpose**: Profile baseline vector addition and distance calculations
- **Key metrics**: Function call distribution, CPU usage patterns
- **Look for**: Hot paths in distance calculations, memory allocation overhead

### 2. SIMD Operations (`simd_operations.svg`)
- **Purpose**: Profile SIMD-optimized distance calculations
- **Key metrics**: SIMD instruction usage, vectorization effectiveness
- **Look for**: SIMD intrinsic calls, performance improvements over scalar code

### 3. Search Operations (`search_operations.svg`)
- **Purpose**: Profile vector search and ranking operations
- **Key metrics**: Search algorithm efficiency, sorting overhead
- **Look for**: Linear vs logarithmic scaling, cache misses

### 4. Memory Operations (`memory_operations.svg`)
- **Purpose**: Profile memory allocation and deallocation patterns
- **Key metrics**: Allocator usage, memory pool effectiveness
- **Look for**: Allocation hotspots, fragmentation issues

### 5. Comprehensive Benchmark (`comprehensive_benchmark.svg`)
- **Purpose**: Profile the complete RAG pipeline under load
- **Key metrics**: End-to-end performance, bottleneck identification
- **Look for**: I/O vs CPU bound operations, async task scheduling

## How to Read Flame Graphs

1. **Width = Time**: Wider boxes represent functions that consume more CPU time
2. **Height = Call Stack**: Taller stacks show deeper function call chains
3. **Color = Random**: Colors are random and don't represent anything meaningful
4. **Click to Zoom**: Interactive SVGs allow zooming into specific functions

## Analysis Guidelines

### Performance Bottlenecks
- Look for wide boxes at the top level - these are your main bottlenecks
- Red/orange boxes often indicate system calls or allocations
- Deep call stacks may indicate inefficient algorithms

### SIMD Effectiveness
- Compare `simd_operations.svg` with `vector_operations_baseline.svg`
- Look for presence of SIMD intrinsic functions (`__mm256_*`, `vfma*`)
- Measure relative width of computation vs overhead

### Memory Efficiency
- In `memory_operations.svg`, look for allocator calls (`malloc`, `free`)
- Wide allocator boxes indicate memory allocation overhead
- Look for patterns of frequent allocation/deallocation

### I/O Performance
- In `comprehensive_benchmark.svg`, identify Redis/network operations
- Compare CPU computation time with I/O wait time
- Look for opportunities to batch operations

## Optimization Recommendations

Based on the flame graphs, prioritize optimizations for:
1. **Widest boxes**: Functions consuming the most CPU time
2. **Frequent calls**: Functions called many times (tall, narrow boxes)
3. **System calls**: Minimize expensive system calls
4. **Memory allocations**: Reduce allocation frequency

## Next Steps

1. **Analyze each flame graph** for the patterns described above
2. **Compare optimized vs baseline** profiles to measure improvements
3. **Identify remaining bottlenecks** for future optimization
4. **Validate optimizations** don't introduce new performance issues

EOF

    echo -e "${GREEN}Performance report generated: $OUTPUT_DIR/PROFILING_REPORT.md${NC}"
}

# Main execution
main() {
    echo "Starting performance profiling for RAG-Redis system..."

    check_prerequisites
    setup_output_dir

    # Create benches directory if it doesn't exist
    mkdir -p benches

    echo "Running profiling tests..."

    # Profile individual components
    if profile_vector_operations; then
        echo -e "${GREEN}✓ Vector operations profiled${NC}"
    else
        echo -e "${RED}✗ Vector operations profiling failed${NC}"
    fi

    if profile_simd_operations; then
        echo -e "${GREEN}✓ SIMD operations profiled${NC}"
    else
        echo -e "${RED}✗ SIMD operations profiling failed${NC}"
    fi

    if profile_search_operations; then
        echo -e "${GREEN}✓ Search operations profiled${NC}"
    else
        echo -e "${RED}✗ Search operations profiling failed${NC}"
    fi

    if profile_memory_operations; then
        echo -e "${GREEN}✓ Memory operations profiled${NC}"
    else
        echo -e "${RED}✗ Memory operations profiling failed${NC}"
    fi

    # Comprehensive benchmark (optional due to complexity)
    echo "Would you like to run the comprehensive benchmark? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        if run_comprehensive_benchmark; then
            echo -e "${GREEN}✓ Comprehensive benchmark profiled${NC}"
        else
            echo -e "${RED}✗ Comprehensive benchmark profiling failed${NC}"
        fi
    fi

    generate_report

    echo ""
    echo -e "${GREEN}=== Performance Profiling Complete ===${NC}"
    echo "Flame graphs and report generated in: $OUTPUT_DIR/"
    echo ""
    echo "To view flame graphs:"
    echo "  - Open .svg files in a web browser"
    echo "  - Click on functions to zoom in"
    echo "  - Use browser back button to zoom out"
    echo ""
    echo "Next steps:"
    echo "1. Analyze flame graphs for performance bottlenecks"
    echo "2. Compare baseline vs optimized implementations"
    echo "3. Focus optimization efforts on widest boxes"
    echo "4. Re-profile after implementing optimizations"
}

# Run main function
main "$@"
