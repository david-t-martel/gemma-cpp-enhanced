// Copyright 2025 Gemma.cpp Contributors
// SPDX-License-Identifier: Apache-2.0
//
// oneAPI Library Integration Validation Tests
// Tests numerical accuracy and threading correctness with TBB, IPP, DPL, DNNL

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <thread>
#include <vector>

// Conditional includes for oneAPI components
#ifdef GEMMA_USE_TBB
#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#endif

#ifdef GEMMA_USE_IPP
#include <ipp.h>
#endif

#ifdef GEMMA_USE_DPL
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#endif

#ifdef GEMMA_USE_DNNL
#include <oneapi/dnnl/dnnl.hpp>
#endif

namespace gcpp {
namespace test {

// ============================================================================
// Test Fixtures
// ============================================================================

class OneAPIValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random generator with fixed seed for reproducibility
        rng_.seed(42);
        
        #ifdef GEMMA_USE_TBB
        // Initialize TBB with detected parallelism
        tbb_max_threads_ = std::thread::hardware_concurrency();
        #endif
        
        #ifdef GEMMA_USE_IPP
        // Initialize IPP
        ipp_status_ = ippInit();
        ASSERT_EQ(ippStsNoErr, ipp_status_) << "IPP initialization failed";
        
        // Get IPP version
        const IppLibraryVersion* lib = ippGetLibVersion();
        std::cout << "IPP Version: " << lib->Version << "\n";
        #endif
    }
    
    void TearDown() override {
        // Cleanup if needed
    }
    
    // Generate random matrix
    std::vector<float> GenerateRandomMatrix(size_t rows, size_t cols, 
                                           float min = -1.0f, float max = 1.0f) {
        std::vector<float> matrix(rows * cols);
        std::uniform_real_distribution<float> dist(min, max);
        
        for (size_t i = 0; i < matrix.size(); ++i) {
            matrix[i] = dist(rng_);
        }
        
        return matrix;
    }
    
    // Reference CPU matrix multiplication (naive implementation)
    void MatMulReference(const float* A, const float* B, float* C,
                        size_t M, size_t K, size_t N) {
        // C(M x N) = A(M x K) * B(K x N)
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    sum += A[m * K + k] * B[k * N + n];
                }
                C[m * N + n] = sum;
            }
        }
    }
    
    // Compare two matrices with tolerance
    bool MatricesEqual(const float* A, const float* B, size_t size,
                      float rel_tol = 1e-4f, float abs_tol = 1e-5f) {
        for (size_t i = 0; i < size; ++i) {
            float diff = std::abs(A[i] - B[i]);
            float max_val = std::max(std::abs(A[i]), std::abs(B[i]));
            
            if (diff > abs_tol && diff > rel_tol * max_val) {
                std::cerr << "Mismatch at index " << i 
                         << ": expected " << A[i] 
                         << ", got " << B[i] 
                         << ", diff " << diff << "\n";
                return false;
            }
        }
        return true;
    }
    
    std::mt19937 rng_;
    
    #ifdef GEMMA_USE_TBB
    int tbb_max_threads_ = 0;
    #endif
    
    #ifdef GEMMA_USE_IPP
    IppStatus ipp_status_;
    #endif
};

// ============================================================================
// TBB Threading Validation Tests
// ============================================================================

#ifdef GEMMA_USE_TBB

TEST_F(OneAPIValidationTest, TBB_ParallelForCorrectness) {
    constexpr size_t N = 100000;
    std::vector<float> data(N, 0.0f);
    
    // Parallel fill with index values
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                data[i] = static_cast<float>(i);
            }
        });
    
    // Verify correctness
    for (size_t i = 0; i < N; ++i) {
        ASSERT_FLOAT_EQ(static_cast<float>(i), data[i])
            << "TBB parallel_for produced incorrect result at index " << i;
    }
}

TEST_F(OneAPIValidationTest, TBB_ParallelReduction) {
    constexpr size_t N = 1000000;
    std::vector<float> data = GenerateRandomMatrix(1, N);
    
    // Parallel sum
    float parallel_sum = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, N),
        0.0f,
        [&](const tbb::blocked_range<size_t>& r, float init) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                init += data[i];
            }
            return init;
        },
        std::plus<float>());
    
    // Reference sum
    float reference_sum = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        reference_sum += data[i];
    }
    
    // Compare with relative tolerance (accumulation order may differ)
    float rel_error = std::abs(parallel_sum - reference_sum) / 
                     std::max(std::abs(reference_sum), 1.0f);
    EXPECT_LT(rel_error, 1e-3f) << "TBB parallel reduction incorrect: "
                                << "parallel=" << parallel_sum
                                << " reference=" << reference_sum;
}

TEST_F(OneAPIValidationTest, TBB_TaskArenaConstraints) {
    // Test thread limiting
    constexpr int limited_threads = 2;
    tbb::task_arena arena(limited_threads);
    
    std::atomic<int> max_observed_threads{0};
    std::atomic<int> counter{0};
    
    arena.execute([&] {
        tbb::parallel_for(0, 100, [&](int) {
            int current = ++counter;
            int prev = max_observed_threads.load();
            while (current > prev && 
                   !max_observed_threads.compare_exchange_weak(prev, current)) {}
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            --counter;
        });
    });
    
    // Should never exceed requested thread count
    EXPECT_LE(max_observed_threads.load(), limited_threads)
        << "TBB task arena violated thread limit";
}

#else
TEST_F(OneAPIValidationTest, TBB_NotEnabled) {
    GTEST_SKIP() << "TBB not enabled in build (GEMMA_USE_TBB not defined)";
}
#endif // GEMMA_USE_TBB

// ============================================================================
// IPP Vector Operations Validation
// ============================================================================

#ifdef GEMMA_USE_IPP

TEST_F(OneAPIValidationTest, IPP_VectorAddition) {
    constexpr int N = 1024;
    std::vector<float> a = GenerateRandomMatrix(1, N);
    std::vector<float> b = GenerateRandomMatrix(1, N);
    std::vector<float> c_ipp(N);
    std::vector<float> c_ref(N);
    
    // IPP vector addition
    IppStatus status = ippsAdd_32f(a.data(), b.data(), c_ipp.data(), N);
    ASSERT_EQ(ippStsNoErr, status) << "IPP vector addition failed";
    
    // Reference addition
    for (int i = 0; i < N; ++i) {
        c_ref[i] = a[i] + b[i];
    }
    
    // Verify
    EXPECT_TRUE(MatricesEqual(c_ref.data(), c_ipp.data(), N))
        << "IPP vector addition produced incorrect results";
}

TEST_F(OneAPIValidationTest, IPP_DotProduct) {
    constexpr int N = 1024;
    std::vector<float> a = GenerateRandomMatrix(1, N);
    std::vector<float> b = GenerateRandomMatrix(1, N);
    
    // IPP dot product
    Ipp32f dot_ipp = 0.0f;
    IppStatus status = ippsDotProd_32f(a.data(), b.data(), N, &dot_ipp);
    ASSERT_EQ(ippStsNoErr, status) << "IPP dot product failed";
    
    // Reference dot product
    float dot_ref = 0.0f;
    for (int i = 0; i < N; ++i) {
        dot_ref += a[i] * b[i];
    }
    
    // Verify with relative tolerance
    float rel_error = std::abs(dot_ipp - dot_ref) / 
                     std::max(std::abs(dot_ref), 1.0f);
    EXPECT_LT(rel_error, 1e-5f) << "IPP dot product incorrect: "
                                << "IPP=" << dot_ipp 
                                << " reference=" << dot_ref;
}

TEST_F(OneAPIValidationTest, IPP_MatrixMultiply) {
    constexpr int M = 32;
    constexpr int K = 64;
    constexpr int N = 32;
    
    std::vector<float> A = GenerateRandomMatrix(M, K);
    std::vector<float> B = GenerateRandomMatrix(K, N);
    std::vector<float> C_ipp(M * N, 0.0f);
    std::vector<float> C_ref(M * N, 0.0f);
    
    // IPP matrix multiply (if available in your IPP version)
    // Note: Basic IPP may not have GEMM, this is a placeholder
    // For actual test, you'd use ippmMul_mm_32f or similar
    
    // For now, test element-wise operations that feed into matmul
    MatMulReference(A.data(), B.data(), C_ref.data(), M, K, N);
    
    // Placeholder: If IPP GEMM available, test here
    // IppStatus status = ippmMul_mm_32f(...);
    
    GTEST_SKIP() << "IPP GEMM test requires specific IPP library version with matrix ops";
}

#else
TEST_F(OneAPIValidationTest, IPP_NotEnabled) {
    GTEST_SKIP() << "IPP not enabled in build (GEMMA_USE_IPP not defined)";
}
#endif // GEMMA_USE_IPP

// ============================================================================
// DPL Algorithm Validation
// ============================================================================

#ifdef GEMMA_USE_DPL

TEST_F(OneAPIValidationTest, DPL_ParallelSort) {
    constexpr size_t N = 10000;
    std::vector<float> data = GenerateRandomMatrix(1, N);
    std::vector<float> data_ref = data;
    
    // DPL parallel sort
    std::sort(oneapi::dpl::execution::par_unseq, 
              data.begin(), data.end());
    
    // Reference sort
    std::sort(data_ref.begin(), data_ref.end());
    
    // Verify
    EXPECT_EQ(data, data_ref) << "DPL parallel sort produced incorrect results";
}

TEST_F(OneAPIValidationTest, DPL_ParallelTransform) {
    constexpr size_t N = 10000;
    std::vector<float> input = GenerateRandomMatrix(1, N);
    std::vector<float> output_dpl(N);
    std::vector<float> output_ref(N);
    
    auto transform_func = [](float x) { return x * 2.0f + 1.0f; };
    
    // DPL parallel transform
    std::transform(oneapi::dpl::execution::par_unseq,
                  input.begin(), input.end(), 
                  output_dpl.begin(), 
                  transform_func);
    
    // Reference transform
    std::transform(input.begin(), input.end(), 
                  output_ref.begin(), 
                  transform_func);
    
    // Verify
    EXPECT_TRUE(MatricesEqual(output_ref.data(), output_dpl.data(), N))
        << "DPL parallel transform produced incorrect results";
}

#else
TEST_F(OneAPIValidationTest, DPL_NotEnabled) {
    GTEST_SKIP() << "DPL not enabled in build (GEMMA_USE_DPL not defined)";
}
#endif // GEMMA_USE_DPL

// ============================================================================
// DNNL Matrix Operations Validation
// ============================================================================

#ifdef GEMMA_USE_DNNL

TEST_F(OneAPIValidationTest, DNNL_MatrixMultiply) {
    using namespace dnnl;
    
    constexpr int M = 128;
    constexpr int K = 256;
    constexpr int N = 128;
    
    std::vector<float> A = GenerateRandomMatrix(M, K);
    std::vector<float> B = GenerateRandomMatrix(K, N);
    std::vector<float> C_dnnl(M * N, 0.0f);
    std::vector<float> C_ref(M * N, 0.0f);
    
    try {
        // Create DNNL engine
        engine eng(engine::kind::cpu, 0);
        stream s(eng);
        
        // Create memory descriptors
        memory::desc a_md({M, K}, memory::data_type::f32, memory::format_tag::ab);
        memory::desc b_md({K, N}, memory::data_type::f32, memory::format_tag::ab);
        memory::desc c_md({M, N}, memory::data_type::f32, memory::format_tag::ab);
        
        // Create memory objects
        memory a_mem(a_md, eng, A.data());
        memory b_mem(b_md, eng, B.data());
        memory c_mem(c_md, eng, C_dnnl.data());
        
        // Create matmul primitive descriptor
        matmul::primitive_desc matmul_pd(eng, a_md, b_md, c_md);
        
        // Create matmul primitive
        matmul matmul_prim(matmul_pd);
        
        // Execute
        matmul_prim.execute(s, {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem}
        });
        
        s.wait();
        
    } catch (const dnnl::error& e) {
        FAIL() << "DNNL error: " << e.what() << " (status: " << e.status << ")";
    }
    
    // Reference matmul
    MatMulReference(A.data(), B.data(), C_ref.data(), M, K, N);
    
    // Verify with appropriate tolerance for accumulated errors
    EXPECT_TRUE(MatricesEqual(C_ref.data(), C_dnnl.data(), M * N, 1e-3f, 1e-4f))
        << "DNNL matrix multiply produced incorrect results";
}

TEST_F(OneAPIValidationTest, DNNL_LargeMatrixMultiply) {
    using namespace dnnl;
    
    constexpr int M = 512;
    constexpr int K = 512;
    constexpr int N = 512;
    
    std::vector<float> A = GenerateRandomMatrix(M, K, -0.5f, 0.5f);
    std::vector<float> B = GenerateRandomMatrix(K, N, -0.5f, 0.5f);
    std::vector<float> C_dnnl(M * N, 0.0f);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        engine eng(engine::kind::cpu, 0);
        stream s(eng);
        
        memory::desc a_md({M, K}, memory::data_type::f32, memory::format_tag::ab);
        memory::desc b_md({K, N}, memory::data_type::f32, memory::format_tag::ab);
        memory::desc c_md({M, N}, memory::data_type::f32, memory::format_tag::ab);
        
        memory a_mem(a_md, eng, A.data());
        memory b_mem(b_md, eng, B.data());
        memory c_mem(c_md, eng, C_dnnl.data());
        
        matmul::primitive_desc matmul_pd(eng, a_md, b_md, c_md);
        matmul matmul_prim(matmul_pd);
        
        matmul_prim.execute(s, {
            {DNNL_ARG_SRC, a_mem},
            {DNNL_ARG_WEIGHTS, b_mem},
            {DNNL_ARG_DST, c_mem}
        });
        
        s.wait();
        
    } catch (const dnnl::error& e) {
        FAIL() << "DNNL error: " << e.what();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "DNNL " << M << "x" << K << "x" << N 
              << " matmul: " << duration.count() << " ms\n";
    
    // Basic sanity check - results shouldn't be all zeros
    bool has_nonzero = false;
    for (const auto& val : C_dnnl) {
        if (std::abs(val) > 1e-6f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "DNNL matmul produced all-zero output";
}

#else
TEST_F(OneAPIValidationTest, DNNL_NotEnabled) {
    GTEST_SKIP() << "DNNL not enabled in build (GEMMA_USE_DNNL not defined)";
}
#endif // GEMMA_USE_DNNL

// ============================================================================
// Integration Tests - Combined oneAPI Components
// ============================================================================

TEST_F(OneAPIValidationTest, IntegrationTest_ThreadingAndVectorOps) {
    constexpr size_t N = 100000;
    std::vector<float> input = GenerateRandomMatrix(1, N);
    std::vector<float> output(N, 0.0f);
    
    #if defined(GEMMA_USE_TBB) && defined(GEMMA_USE_IPP)
    // Use TBB for parallelism and IPP for vector operations
    tbb::parallel_for(tbb::blocked_range<size_t>(0, N, 256),
        [&](const tbb::blocked_range<size_t>& r) {
            int len = static_cast<int>(r.end() - r.begin());
            // IPP: scale and offset
            ippsAddC_32f(&input[r.begin()], 1.0f, &output[r.begin()], len);
        });
    
    // Verify
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(input[i] + 1.0f, output[i])
            << "TBB+IPP integration failed at index " << i;
    }
    #else
    GTEST_SKIP() << "Integration test requires both TBB and IPP";
    #endif
}

// ============================================================================
// Performance Regression Tests
// ============================================================================

TEST_F(OneAPIValidationTest, PerformanceRegression_SmallMatMul) {
    constexpr int M = 64, K = 64, N = 64;
    std::vector<float> A = GenerateRandomMatrix(M, K);
    std::vector<float> B = GenerateRandomMatrix(K, N);
    std::vector<float> C(M * N, 0.0f);
    
    auto start = std::chrono::high_resolution_clock::now();
    MatMulReference(A.data(), B.data(), C.data(), M, K, N);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();
    
    std::cout << "Small MatMul (" << M << "x" << K << "x" << N << "): "
              << duration_us << " μs\n";
    
    // Sanity check - shouldn't take more than 10ms on any modern CPU
    EXPECT_LT(duration_us, 10000) 
        << "Small matmul performance regression detected";
}

} // namespace test
} // namespace gcpp

// Entry point
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "\n=== oneAPI Validation Test Suite ===\n";
    std::cout << "Build Configuration:\n";
    #ifdef GEMMA_USE_TBB
    std::cout << "  - TBB: ENABLED\n";
    #else
    std::cout << "  - TBB: disabled\n";
    #endif
    
    #ifdef GEMMA_USE_IPP
    std::cout << "  - IPP: ENABLED\n";
    #else
    std::cout << "  - IPP: disabled\n";
    #endif
    
    #ifdef GEMMA_USE_DPL
    std::cout << "  - DPL: ENABLED\n";
    #else
    std::cout << "  - DPL: disabled\n";
    #endif
    
    #ifdef GEMMA_USE_DNNL
    std::cout << "  - DNNL: ENABLED\n";
    #else
    std::cout << "  - DNNL: disabled\n";
    #endif
    std::cout << "====================================\n\n";
    
    return RUN_ALL_TESTS();
}
