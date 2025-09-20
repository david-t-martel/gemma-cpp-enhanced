/**
 * @file cuda_attention.cu
 * @brief Flash Attention v2 implementation for CUDA
 *
 * This file implements optimized attention mechanisms including:
 * - Flash Attention v2 for memory-efficient attention computation
 * - Multi-head attention with KV caching
 * - Grouped query attention for efficient inference
 * - Causal (autoregressive) attention masks
 * - Rotary positional embeddings (RoPE)
 */

#include "cuda_attention.h"
#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <math_constants.h>
#include <algorithm>

namespace gemma {
namespace backends {
namespace cuda {
namespace attention {

namespace cg = cooperative_groups;

// Constants for Flash Attention
constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 1024;
constexpr int TILE_SIZE_M = 64;  // Query tile size
constexpr int TILE_SIZE_N = 64;  // Key/Value tile size
constexpr int TILE_SIZE_K = 64;  // Head dimension tile size

// Utility functions
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T block_reduce_sum(T val) {
    static __shared__ T shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(0);

    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}

template<typename T>
__device__ __forceinline__ T block_reduce_max(T val) {
    static __shared__ T shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(-INFINITY);

    if (wid == 0) val = warp_reduce_max(val);

    return val;
}

// RoPE (Rotary Positional Embedding) implementation
template<typename T>
__device__ void apply_rope(T* q, T* k, int head_dim, int pos, float rope_base = 10000.0f) {
    const int half_dim = head_dim / 2;
    const int tid = threadIdx.x;

    for (int i = tid; i < half_dim; i += blockDim.x) {
        float freq = 1.0f / powf(rope_base, (float)(2 * i) / head_dim);
        float angle = pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        T q_real = q[i];
        T q_imag = q[i + half_dim];
        T k_real = k[i];
        T k_imag = k[i + half_dim];

        q[i] = q_real * cos_val - q_imag * sin_val;
        q[i + half_dim] = q_real * sin_val + q_imag * cos_val;

        k[i] = k_real * cos_val - k_imag * sin_val;
        k[i + half_dim] = k_real * sin_val + k_imag * cos_val;
    }
}

// Flash Attention Forward Pass Kernel
template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K, bool CAUSAL>
__global__ void flash_attention_forward_kernel(
    const T* Q,     // [batch_size, num_heads, seq_len, head_dim]
    const T* K,     // [batch_size, num_heads, seq_len, head_dim]
    const T* V,     // [batch_size, num_heads, seq_len, head_dim]
    T* O,           // [batch_size, num_heads, seq_len, head_dim]
    T* L,           // [batch_size, num_heads, seq_len] - row sums for numerical stability
    T* M,           // [batch_size, num_heads, seq_len] - row maxes for numerical stability
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale) {

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int block_m = blockIdx.x;

    if (batch_idx >= batch_size || head_idx >= num_heads) return;

    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / WARP_SIZE;
    const int lane_idx = thread_idx % WARP_SIZE;

    // Calculate base pointers for this batch and head
    const T* q_base = Q + (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const T* k_base = K + (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const T* v_base = V + (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    T* o_base = O + (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    T* l_base = L + (batch_idx * num_heads + head_idx) * seq_len;
    T* m_base = M + (batch_idx * num_heads + head_idx) * seq_len;

    // Shared memory for tiles
    __shared__ T q_tile[BLOCK_M][BLOCK_K];
    __shared__ T k_tile[BLOCK_N][BLOCK_K];
    __shared__ T v_tile[BLOCK_N][BLOCK_K];
    __shared__ T s_tile[BLOCK_M][BLOCK_N];  // Attention scores
    __shared__ T o_tile[BLOCK_M][BLOCK_K];  // Output accumulator

    // Initialize output tile
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i += blockDim.x / BLOCK_K) {
        #pragma unroll
        for (int j = 0; j < BLOCK_K; j += 1) {
            if (i + thread_idx / BLOCK_K < BLOCK_M && j + thread_idx % BLOCK_K < BLOCK_K) {
                o_tile[i + thread_idx / BLOCK_K][j + thread_idx % BLOCK_K] = T(0);
            }
        }
    }

    // Load Q tile
    const int q_row_start = block_m * BLOCK_M;
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i += blockDim.x / BLOCK_K) {
        #pragma unroll
        for (int j = 0; j < BLOCK_K; j += 1) {
            int row = q_row_start + i + thread_idx / BLOCK_K;
            int col = j + thread_idx % BLOCK_K;
            if (row < seq_len && col < head_dim) {
                q_tile[i + thread_idx / BLOCK_K][j + thread_idx % BLOCK_K] =
                    q_base[row * head_dim + col];
            } else {
                q_tile[i + thread_idx / BLOCK_K][j + thread_idx % BLOCK_K] = T(0);
            }
        }
    }

    // Initialize running statistics for numerical stability
    T row_max[BLOCK_M];
    T row_sum[BLOCK_M];
    #pragma unroll
    for (int i = 0; i < BLOCK_M; ++i) {
        row_max[i] = T(-INFINITY);
        row_sum[i] = T(0);
    }

    // Process K/V tiles
    const int num_blocks_n = (seq_len + BLOCK_N - 1) / BLOCK_N;

    for (int block_n = 0; block_n < num_blocks_n; ++block_n) {
        const int k_col_start = block_n * BLOCK_N;

        // Apply causal mask if needed
        if constexpr (CAUSAL) {
            if (k_col_start > q_row_start + BLOCK_M - 1) {
                break;  // No more valid blocks due to causal mask
            }
        }

        __syncthreads();

        // Load K tile
        #pragma unroll
        for (int i = 0; i < BLOCK_N; i += blockDim.x / BLOCK_K) {
            #pragma unroll
            for (int j = 0; j < BLOCK_K; j += 1) {
                int row = k_col_start + i + thread_idx / BLOCK_K;
                int col = j + thread_idx % BLOCK_K;
                if (row < seq_len && col < head_dim) {
                    k_tile[i + thread_idx / BLOCK_K][j + thread_idx % BLOCK_K] =
                        k_base[row * head_dim + col];
                } else {
                    k_tile[i + thread_idx / BLOCK_K][j + thread_idx % BLOCK_K] = T(0);
                }
            }
        }

        // Load V tile
        #pragma unroll
        for (int i = 0; i < BLOCK_N; i += blockDim.x / BLOCK_K) {
            #pragma unroll
            for (int j = 0; j < BLOCK_K; j += 1) {
                int row = k_col_start + i + thread_idx / BLOCK_K;
                int col = j + thread_idx % BLOCK_K;
                if (row < seq_len && col < head_dim) {
                    v_tile[i + thread_idx / BLOCK_K][j + thread_idx % BLOCK_K] =
                        v_base[row * head_dim + col];
                } else {
                    v_tile[i + thread_idx / BLOCK_K][j + thread_idx % BLOCK_K] = T(0);
                }
            }
        }

        __syncthreads();

        // Compute attention scores: S = Q @ K^T
        #pragma unroll
        for (int i = 0; i < BLOCK_M; i += blockDim.x / BLOCK_N) {
            #pragma unroll
            for (int j = 0; j < BLOCK_N; j += 1) {
                if (i + thread_idx / BLOCK_N < BLOCK_M && j + thread_idx % BLOCK_N < BLOCK_N) {
                    T score = T(0);
                    #pragma unroll
                    for (int k = 0; k < BLOCK_K; ++k) {
                        score += q_tile[i + thread_idx / BLOCK_N][k] *
                                k_tile[j + thread_idx % BLOCK_N][k];
                    }
                    score *= scale;

                    // Apply causal mask
                    if constexpr (CAUSAL) {
                        int q_pos = q_row_start + i + thread_idx / BLOCK_N;
                        int k_pos = k_col_start + j + thread_idx % BLOCK_N;
                        if (k_pos > q_pos) {
                            score = T(-INFINITY);
                        }
                    }

                    s_tile[i + thread_idx / BLOCK_N][j + thread_idx % BLOCK_N] = score;
                }
            }
        }

        __syncthreads();

        // Online softmax: Update running max and sum
        #pragma unroll
        for (int i = 0; i < BLOCK_M; ++i) {
            if (q_row_start + i < seq_len) {
                T block_max = T(-INFINITY);

                // Find max in this block
                for (int j = thread_idx; j < BLOCK_N; j += blockDim.x) {
                    if (k_col_start + j < seq_len) {
                        block_max = fmaxf(block_max, s_tile[i][j]);
                    }
                }
                block_max = block_reduce_max(block_max);

                T new_max = fmaxf(row_max[i], block_max);
                T exp_sum = T(0);

                // Compute exponentials and sum
                for (int j = thread_idx; j < BLOCK_N; j += blockDim.x) {
                    if (k_col_start + j < seq_len) {
                        T exp_val = expf(s_tile[i][j] - new_max);
                        s_tile[i][j] = exp_val;
                        exp_sum += exp_val;
                    } else {
                        s_tile[i][j] = T(0);
                    }
                }
                exp_sum = block_reduce_sum(exp_sum);

                // Update running statistics
                T old_sum_scaled = row_sum[i] * expf(row_max[i] - new_max);
                row_sum[i] = old_sum_scaled + exp_sum;
                row_max[i] = new_max;

                // Scale previous output
                if (block_n > 0) {
                    T scale_factor = old_sum_scaled / row_sum[i];
                    for (int k = thread_idx; k < BLOCK_K; k += blockDim.x) {
                        o_tile[i][k] *= scale_factor;
                    }
                }
            }
        }

        __syncthreads();

        // Compute output: O += P @ V
        #pragma unroll
        for (int i = 0; i < BLOCK_M; i += blockDim.x / BLOCK_K) {
            #pragma unroll
            for (int k = 0; k < BLOCK_K; k += 1) {
                if (i + thread_idx / BLOCK_K < BLOCK_M && k + thread_idx % BLOCK_K < BLOCK_K) {
                    T acc = T(0);
                    #pragma unroll
                    for (int j = 0; j < BLOCK_N; ++j) {
                        acc += s_tile[i + thread_idx / BLOCK_K][j] *
                              v_tile[j][k + thread_idx % BLOCK_K];
                    }

                    if (q_row_start + i + thread_idx / BLOCK_K < seq_len) {
                        T scale_factor = T(1) / row_sum[i + thread_idx / BLOCK_K];
                        o_tile[i + thread_idx / BLOCK_K][k + thread_idx % BLOCK_K] +=
                            acc * scale_factor;
                    }
                }
            }
        }
    }

    __syncthreads();

    // Write output to global memory
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i += blockDim.x / BLOCK_K) {
        #pragma unroll
        for (int j = 0; j < BLOCK_K; j += 1) {
            int row = q_row_start + i + thread_idx / BLOCK_K;
            int col = j + thread_idx % BLOCK_K;
            if (row < seq_len && col < head_dim) {
                o_base[row * head_dim + col] =
                    o_tile[i + thread_idx / BLOCK_K][j + thread_idx % BLOCK_K];
            }
        }
    }

    // Write statistics to global memory
    if (thread_idx < BLOCK_M) {
        int row = q_row_start + thread_idx;
        if (row < seq_len) {
            l_base[row] = row_sum[thread_idx];
            m_base[row] = row_max[thread_idx];
        }
    }
}

// Multi-head attention with KV cache
template<typename T>
__global__ void mha_with_kv_cache_kernel(
    const T* Q,           // [batch_size, num_heads, seq_len, head_dim]
    const T* K_cache,     // [batch_size, num_kv_heads, max_seq_len, head_dim]
    const T* V_cache,     // [batch_size, num_kv_heads, max_seq_len, head_dim]
    T* O,                 // [batch_size, num_heads, seq_len, head_dim]
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int kv_seq_len,
    int head_dim,
    float scale) {

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) {
        return;
    }

    // Group query attention: map head_idx to kv_head_idx
    const int kv_head_idx = head_idx % num_kv_heads;

    // Calculate base pointers
    const T* q_ptr = Q + (batch_idx * num_heads + head_idx) * seq_len * head_dim +
                         seq_idx * head_dim;
    const T* k_cache_base = K_cache + (batch_idx * num_kv_heads + kv_head_idx) * kv_seq_len * head_dim;
    const T* v_cache_base = V_cache + (batch_idx * num_kv_heads + kv_head_idx) * kv_seq_len * head_dim;
    T* o_ptr = O + (batch_idx * num_heads + head_idx) * seq_len * head_dim +
                   seq_idx * head_dim;

    // Compute attention scores
    T max_score = T(-INFINITY);
    T sum_exp = T(0);

    // First pass: find max score for numerical stability
    for (int k_idx = 0; k_idx < kv_seq_len; ++k_idx) {
        T score = T(0);
        const T* k_ptr = k_cache_base + k_idx * head_dim;

        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            score += q_ptr[d] * k_ptr[d];
        }
        score *= scale;

        // Apply causal mask
        if (k_idx > seq_idx) {
            score = T(-INFINITY);
        }

        max_score = fmaxf(max_score, score);
    }

    // Second pass: compute softmax and output
    #pragma unroll
    for (int d = 0; d < head_dim; ++d) {
        o_ptr[d] = T(0);
    }

    for (int k_idx = 0; k_idx < kv_seq_len; ++k_idx) {
        if (k_idx > seq_idx) break;  // Causal mask

        T score = T(0);
        const T* k_ptr = k_cache_base + k_idx * head_dim;
        const T* v_ptr = v_cache_base + k_idx * head_dim;

        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            score += q_ptr[d] * k_ptr[d];
        }
        score *= scale;

        T attention_weight = expf(score - max_score);
        sum_exp += attention_weight;

        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            o_ptr[d] += attention_weight * v_ptr[d];
        }
    }

    // Normalize output
    T inv_sum = T(1) / sum_exp;
    #pragma unroll
    for (int d = 0; d < head_dim; ++d) {
        o_ptr[d] *= inv_sum;
    }
}

// Standard (non-Flash) attention kernel for comparison
template<typename T>
__global__ void standard_attention_kernel(
    const T* Q,     // [batch_size, num_heads, seq_len, head_dim]
    const T* K,     // [batch_size, num_heads, seq_len, head_dim]
    const T* V,     // [batch_size, num_heads, seq_len, head_dim]
    T* O,           // [batch_size, num_heads, seq_len, head_dim]
    T* scores,      // [batch_size, num_heads, seq_len, seq_len] - temporary
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale) {

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) {
        return;
    }

    const int offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const T* q_ptr = Q + offset + seq_idx * head_dim;
    const T* k_base = K + offset;
    const T* v_base = V + offset;
    T* o_ptr = O + offset + seq_idx * head_dim;

    const int score_offset = (batch_idx * num_heads + head_idx) * seq_len * seq_len;
    T* score_row = scores + score_offset + seq_idx * seq_len;

    // Compute attention scores
    T max_score = T(-INFINITY);
    for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
        T score = T(0);
        const T* k_ptr = k_base + k_idx * head_dim;

        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            score += q_ptr[d] * k_ptr[d];
        }
        score *= scale;

        // Apply causal mask
        if (k_idx > seq_idx) {
            score = T(-INFINITY);
        }

        score_row[k_idx] = score;
        max_score = fmaxf(max_score, score);
    }

    // Apply softmax
    T sum_exp = T(0);
    for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
        if (k_idx <= seq_idx) {
            T exp_score = expf(score_row[k_idx] - max_score);
            score_row[k_idx] = exp_score;
            sum_exp += exp_score;
        } else {
            score_row[k_idx] = T(0);
        }
    }

    T inv_sum = T(1) / sum_exp;
    for (int k_idx = 0; k_idx <= seq_idx; ++k_idx) {
        score_row[k_idx] *= inv_sum;
    }

    // Compute output
    #pragma unroll
    for (int d = 0; d < head_dim; ++d) {
        o_ptr[d] = T(0);
    }

    for (int k_idx = 0; k_idx <= seq_idx; ++k_idx) {
        const T* v_ptr = v_base + k_idx * head_dim;
        T weight = score_row[k_idx];

        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            o_ptr[d] += weight * v_ptr[d];
        }
    }
}

// Kernel launcher implementations
template<typename T>
cudaError_t launch_flash_attention_kernel(
    const T* Q, const T* K, const T* V, T* O, T* L, T* M,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal, cudaStream_t stream) {

    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 64;
    constexpr int THREADS = 256;

    dim3 grid((seq_len + BLOCK_M - 1) / BLOCK_M, num_heads, batch_size);
    dim3 block(THREADS);

    // Calculate shared memory size
    size_t shared_mem = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K * 2 +
                        BLOCK_M * BLOCK_N + BLOCK_M * BLOCK_K) * sizeof(T);

    if (causal) {
        flash_attention_forward_kernel<T, BLOCK_M, BLOCK_N, BLOCK_K, true>
            <<<grid, block, shared_mem, stream>>>(
                Q, K, V, O, L, M, batch_size, num_heads, seq_len, head_dim, scale);
    } else {
        flash_attention_forward_kernel<T, BLOCK_M, BLOCK_N, BLOCK_K, false>
            <<<grid, block, shared_mem, stream>>>(
                Q, K, V, O, L, M, batch_size, num_heads, seq_len, head_dim, scale);
    }

    return cudaGetLastError();
}

template<typename T>
cudaError_t launch_mha_with_kv_cache_kernel(
    const T* Q, const T* K_cache, const T* V_cache, T* O,
    int batch_size, int num_heads, int num_kv_heads,
    int seq_len, int kv_seq_len, int head_dim,
    float scale, cudaStream_t stream) {

    constexpr int THREADS = 256;
    dim3 grid((seq_len + THREADS - 1) / THREADS, num_heads, batch_size);
    dim3 block(THREADS);

    mha_with_kv_cache_kernel<<<grid, block, 0, stream>>>(
        Q, K_cache, V_cache, O, batch_size, num_heads, num_kv_heads,
        seq_len, kv_seq_len, head_dim, scale);

    return cudaGetLastError();
}

template<typename T>
cudaError_t launch_standard_attention_kernel(
    const T* Q, const T* K, const T* V, T* O, T* temp_scores,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, cudaStream_t stream) {

    constexpr int THREADS = 256;
    dim3 grid((seq_len + THREADS - 1) / THREADS, num_heads, batch_size);
    dim3 block(THREADS);

    standard_attention_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, O, temp_scores, batch_size, num_heads, seq_len, head_dim, scale);

    return cudaGetLastError();
}

// Explicit template instantiations
template cudaError_t launch_flash_attention_kernel<float>(
    const float*, const float*, const float*, float*, float*, float*,
    int, int, int, int, float, bool, cudaStream_t);

template cudaError_t launch_flash_attention_kernel<__half>(
    const __half*, const __half*, const __half*, __half*, __half*, __half*,
    int, int, int, int, float, bool, cudaStream_t);

template cudaError_t launch_mha_with_kv_cache_kernel<float>(
    const float*, const float*, const float*, float*,
    int, int, int, int, int, int, float, cudaStream_t);

template cudaError_t launch_mha_with_kv_cache_kernel<__half>(
    const __half*, const __half*, const __half*, __half*,
    int, int, int, int, int, int, float, cudaStream_t);

template cudaError_t launch_standard_attention_kernel<float>(
    const float*, const float*, const float*, float*, float*,
    int, int, int, int, float, cudaStream_t);

template cudaError_t launch_standard_attention_kernel<__half>(
    const __half*, const __half*, const __half*, __half*, __half*,
    int, int, int, int, float, cudaStream_t);

} // namespace attention
} // namespace cuda
} // namespace backends
} // namespace gemma