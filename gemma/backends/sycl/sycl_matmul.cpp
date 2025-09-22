/**
 * @file sycl_matmul.cpp
 * @brief Optimized matrix multiplication kernels for SYCL backend
 *
 * Implements high-performance matrix operations using:
 * - Intel oneMKL for BLAS operations
 * - Custom SYCL kernels for specialized operations
 * - Optimizations for Intel GPUs and NPUs
 */

#include "sycl_backend.h"
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>

namespace gemma {
namespace backends {
namespace sycl {

// Matrix multiplication implementation using oneMKL

bool SyclBackend::MatrixMultiply(
    const BackendBuffer& a, const BackendBuffer& b, const BackendBuffer& c,
    int m, int n, int k, float alpha, float beta) {

    if (!IsAvailable() || !a.data || !b.data || !c.data) {
        return false;
    }

    try {
        if (profiling_enabled_) {
            BeginProfiling("MatrixMultiply");
        }

        // Cast to float pointers
        const float* a_ptr = static_cast<const float*>(a.data);
        const float* b_ptr = static_cast<const float*>(b.data);
        float* c_ptr = static_cast<float*>(c.data);

        // Perform GEMM using oneMKL: C = alpha * A * B + beta * C
        // oneMKL uses column-major order, so we need to transpose
        auto gemm_event = oneapi::mkl::blas::column_major::gemm(
            *current_queue_,
            trans_n_,  // transpose A
            trans_n_,  // transpose B
            n,         // number of rows of op(B) and C
            m,         // number of columns of op(A) and C
            k,         // number of columns of op(B) and rows of op(A)
            alpha,     // alpha scalar
            b_ptr,     // matrix B
            n,         // leading dimension of B
            a_ptr,     // matrix A
            k,         // leading dimension of A
            beta,      // beta scalar
            c_ptr,     // matrix C
            n          // leading dimension of C
        );

        // Wait for completion
        gemm_event.wait();

        if (profiling_enabled_) {
            size_t flops = 2ULL * m * n * k;  // Approximate FLOP count for GEMM
            EndProfiling("MatrixMultiply", 0, flops);
        }

        return CheckDeviceError(*current_queue_, "MatrixMultiply");

    } catch (const oneapi::mkl::exception& e) {
        std::cerr << "oneMKL Error in MatrixMultiply: " << e.what() << std::endl;
        return false;
    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "MatrixMultiply");
        return false;
    }
}

bool SyclBackend::MatrixVectorMultiply(
    const BackendBuffer& a, const BackendBuffer& x, const BackendBuffer& y,
    int m, int n) {

    if (!IsAvailable() || !a.data || !x.data || !y.data) {
        return false;
    }

    try {
        if (profiling_enabled_) {
            BeginProfiling("MatrixVectorMultiply");
        }

        const float* a_ptr = static_cast<const float*>(a.data);
        const float* x_ptr = static_cast<const float*>(x.data);
        float* y_ptr = static_cast<float*>(y.data);

        // Perform GEMV using oneMKL: y = A * x
        auto gemv_event = oneapi::mkl::blas::column_major::gemv(
            *current_queue_,
            trans_t_,  // transpose A (since we're using column-major)
            n,         // number of columns of A
            m,         // number of rows of A
            1.0f,      // alpha
            a_ptr,     // matrix A
            n,         // leading dimension of A
            x_ptr,     // vector x
            1,         // increment for x
            0.0f,      // beta
            y_ptr,     // vector y
            1          // increment for y
        );

        gemv_event.wait();

        if (profiling_enabled_) {
            size_t flops = 2ULL * m * n;  // Approximate FLOP count for GEMV
            EndProfiling("MatrixVectorMultiply", 0, flops);
        }

        return CheckDeviceError(*current_queue_, "MatrixVectorMultiply");

    } catch (const oneapi::mkl::exception& e) {
        std::cerr << "oneMKL Error in MatrixVectorMultiply: " << e.what() << std::endl;
        return false;
    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "MatrixVectorMultiply");
        return false;
    }
}

// Activation function implementations

bool SyclBackend::ApplyReLU(const BackendBuffer& input, const BackendBuffer& output, size_t size) {
    if (!IsAvailable() || !input.data || !output.data || size == 0) {
        return false;
    }

    try {
        if (profiling_enabled_) {
            BeginProfiling("ApplyReLU");
        }

        const float* input_ptr = static_cast<const float*>(input.data);
        float* output_ptr = static_cast<float*>(output.data);

        // Custom SYCL kernel for ReLU activation
        auto relu_event = current_queue_->parallel_for(
            ::sycl::range<1>(size),
            [=](::sycl::id<1> idx) {
                float val = input_ptr[idx];
                output_ptr[idx] = val > 0.0f ? val : 0.0f;
            }
        );

        relu_event.wait();

        if (profiling_enabled_) {
            EndProfiling("ApplyReLU", 0, size);  // One operation per element
        }

        return CheckDeviceError(*current_queue_, "ApplyReLU");

    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "ApplyReLU");
        return false;
    }
}

bool SyclBackend::ApplyGELU(const BackendBuffer& input, const BackendBuffer& output, size_t size) {
    if (!IsAvailable() || !input.data || !output.data || size == 0) {
        return false;
    }

    try {
        if (profiling_enabled_) {
            BeginProfiling("ApplyGELU");
        }

        const float* input_ptr = static_cast<const float*>(input.data);
        float* output_ptr = static_cast<float*>(output.data);

        // GELU activation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        // Approximation for better performance on GPU
        auto gelu_event = current_queue_->parallel_for(
            ::sycl::range<1>(size),
            [=](::sycl::id<1> idx) {
                float x = input_ptr[idx];
                float x_cubed = x * x * x;
                float tanh_arg = 0.7978845608f * (x + 0.044715f * x_cubed);
                float tanh_val = ::sycl::tanh(tanh_arg);
                output_ptr[idx] = 0.5f * x * (1.0f + tanh_val);
            }
        );

        gelu_event.wait();

        if (profiling_enabled_) {
            EndProfiling("ApplyGELU", 0, size * 8);  // Approximate FLOP count
        }

        return CheckDeviceError(*current_queue_, "ApplyGELU");

    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "ApplyGELU");
        return false;
    }
}

bool SyclBackend::ApplySoftmax(const BackendBuffer& input, const BackendBuffer& output, size_t size) {
    if (!IsAvailable() || !input.data || !output.data || size == 0) {
        return false;
    }

    try {
        if (profiling_enabled_) {
            BeginProfiling("ApplySoftmax");
        }

        const float* input_ptr = static_cast<const float*>(input.data);
        float* output_ptr = static_cast<float*>(output.data);

        // Numerically stable softmax implementation
        // Two-pass algorithm: find max, then compute exp and normalize

        // Temporary buffer for intermediate computations
        auto temp_buffer = AllocateBuffer(size * sizeof(float));
        if (!temp_buffer.data) {
            return false;
        }
        float* temp_ptr = static_cast<float*>(temp_buffer.data);

        // Pass 1: Find maximum value for numerical stability
        auto max_reduction = ::sycl::reduction(temp_ptr, ::sycl::maximum<float>());
        auto find_max_event = current_queue_->parallel_for(
            ::sycl::range<1>(size), max_reduction,
            [=](::sycl::id<1> idx, auto& max_acc) {
                max_acc.combine(input_ptr[idx]);
            }
        );
        find_max_event.wait();

        float max_val = temp_ptr[0];

        // Pass 2: Compute exp(x - max) and sum
        auto sum_reduction = ::sycl::reduction(temp_ptr, ::sycl::plus<float>());
        auto compute_exp_event = current_queue_->parallel_for(
            ::sycl::range<1>(size), sum_reduction,
            [=](::sycl::id<1> idx, auto& sum_acc) {
                float exp_val = ::sycl::exp(input_ptr[idx] - max_val);
                output_ptr[idx] = exp_val;
                sum_acc += exp_val;
            }
        );
        compute_exp_event.wait();

        float sum_exp = temp_ptr[0];

        // Pass 3: Normalize by sum
        auto normalize_event = current_queue_->parallel_for(
            ::sycl::range<1>(size),
            [=](::sycl::id<1> idx) {
                output_ptr[idx] /= sum_exp;
            }
        );
        normalize_event.wait();

        FreeBuffer(temp_buffer);

        if (profiling_enabled_) {
            EndProfiling("ApplySoftmax", 0, size * 4);  // Approximate FLOP count
        }

        return CheckDeviceError(*current_queue_, "ApplySoftmax");

    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "ApplySoftmax");
        return false;
    }
}

// Additional SYCL-specific matrix operations

namespace {

/**
 * @brief Optimized matrix multiplication for small matrices
 * Uses local memory and work group cooperation
 */
class MatMulSmallKernel {
public:
    MatMulSmallKernel(const float* a, const float* b, float* c,
                      int m, int n, int k, float alpha, float beta)
        : a_(a), b_(b), c_(c), m_(m), n_(n), k_(k), alpha_(alpha), beta_(beta) {}

    void operator()(::sycl::nd_item<2> item) const {
        constexpr int TILE_SIZE = 16;

        // Local memory for tiles
        auto local_a = ::sycl::local_accessor<float, 2>(
            ::sycl::range<2>(TILE_SIZE, TILE_SIZE), item);
        auto local_b = ::sycl::local_accessor<float, 2>(
            ::sycl::range<2>(TILE_SIZE, TILE_SIZE), item);

        int global_row = item.get_global_id(0);
        int global_col = item.get_global_id(1);
        int local_row = item.get_local_id(0);
        int local_col = item.get_local_id(1);

        float sum = 0.0f;

        // Iterate over tiles
        for (int tile = 0; tile < (k_ + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
            // Load tile into local memory
            int a_col = tile * TILE_SIZE + local_col;
            int b_row = tile * TILE_SIZE + local_row;

            if (global_row < m_ && a_col < k_) {
                local_a[local_row][local_col] = a_[global_row * k_ + a_col];
            } else {
                local_a[local_row][local_col] = 0.0f;
            }

            if (b_row < k_ && global_col < n_) {
                local_b[local_row][local_col] = b_[b_row * n_ + global_col];
            } else {
                local_b[local_row][local_col] = 0.0f;
            }

            item.barrier(::sycl::access::fence_space::local_space);

            // Compute partial sum
            for (int i = 0; i < TILE_SIZE; ++i) {
                sum += local_a[local_row][i] * local_b[i][local_col];
            }

            item.barrier(::sycl::access::fence_space::local_space);
        }

        // Store result
        if (global_row < m_ && global_col < n_) {
            int idx = global_row * n_ + global_col;
            c_[idx] = alpha_ * sum + beta_ * c_[idx];
        }
    }

private:
    const float* a_;
    const float* b_;
    float* c_;
    int m_, n_, k_;
    float alpha_, beta_;
};

} // anonymous namespace

/**
 * @brief Optimized matrix multiplication for small matrices using local memory
 */
bool SyclBackend::MatrixMultiplyOptimized(
    const BackendBuffer& a, const BackendBuffer& b, const BackendBuffer& c,
    int m, int n, int k, float alpha, float beta) {

    if (!IsAvailable() || !a.data || !b.data || !c.data) {
        return false;
    }

    // Use oneMKL for large matrices, custom kernel for small ones
    constexpr int SMALL_MATRIX_THRESHOLD = 512;

    if (m > SMALL_MATRIX_THRESHOLD || n > SMALL_MATRIX_THRESHOLD || k > SMALL_MATRIX_THRESHOLD) {
        return MatrixMultiply(a, b, c, m, n, k, alpha, beta);
    }

    try {
        if (profiling_enabled_) {
            BeginProfiling("MatrixMultiplyOptimized");
        }

        const float* a_ptr = static_cast<const float*>(a.data);
        const float* b_ptr = static_cast<const float*>(b.data);
        float* c_ptr = static_cast<float*>(c.data);

        constexpr int TILE_SIZE = 16;
        ::sycl::range<2> global_size(
            ((m + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE,
            ((n + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE
        );
        ::sycl::range<2> local_size(TILE_SIZE, TILE_SIZE);

        auto kernel_event = current_queue_->parallel_for(
            ::sycl::nd_range<2>(global_size, local_size),
            MatMulSmallKernel(a_ptr, b_ptr, c_ptr, m, n, k, alpha, beta)
        );

        kernel_event.wait();

        if (profiling_enabled_) {
            size_t flops = 2ULL * m * n * k;
            EndProfiling("MatrixMultiplyOptimized", 0, flops);
        }

        return CheckDeviceError(*current_queue_, "MatrixMultiplyOptimized");

    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "MatrixMultiplyOptimized");
        return false;
    }
}

/**
 * @brief Set memory allocation strategy for better performance
 */
void SyclBackend::SetMemoryStrategy(bool use_usm_device) {
    use_usm_device_ = use_usm_device;
    std::cout << "SYCL Backend: Memory strategy set to "
              << (use_usm_device ? "USM Device" : "USM Shared") << std::endl;
}

/**
 * @brief Get memory usage statistics
 */
std::map<std::string, size_t> SyclBackend::GetMemoryStats() const {
    std::lock_guard<std::mutex> lock(memory_mutex_);

    std::map<std::string, size_t> stats;
    stats["total_allocated"] = total_allocated_memory_;
    stats["peak_usage"] = peak_memory_usage_;
    stats["allocation_count"] = memory_allocations_.size();

    size_t device_memory = 0;
    size_t shared_memory = 0;

    for (const auto& [ptr, info] : memory_allocations_) {
        if (info.alloc_type == ::sycl::usm::alloc::device) {
            device_memory += info.size;
        } else {
            shared_memory += info.size;
        }
    }

    stats["device_memory"] = device_memory;
    stats["shared_memory"] = shared_memory;

    return stats;
}

/**
 * @brief Get profiling data for performance analysis
 */
std::vector<SyclProfileData> SyclBackend::GetProfilingData() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return profiling_data_;
}

/**
 * @brief Clear accumulated profiling data
 */
void SyclBackend::ClearProfilingData() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    profiling_data_.clear();
}

} // namespace sycl
} // namespace backends
} // namespace gemma