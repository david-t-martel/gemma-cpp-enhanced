/**
 * @file sycl_attention.cpp
 * @brief Optimized attention computation kernels for SYCL backend
 *
 * Implements high-performance attention mechanisms for transformer models:
 * - Scaled dot-product attention
 * - Multi-head attention with flash attention optimizations
 * - Memory-efficient attention for long sequences
 * - Causal masking for autoregressive models
 */

#include "sycl_backend.h"
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>

namespace gemma {
namespace backends {
namespace sycl {

bool SyclBackend::ComputeAttention(
    const BackendBuffer& queries, const BackendBuffer& keys,
    const BackendBuffer& values, const BackendBuffer& output,
    int batch_size, int seq_len, int head_dim, int num_heads) {

    if (!IsAvailable() || !queries.data || !keys.data || !values.data || !output.data) {
        return false;
    }

    try {
        if (profiling_enabled_) {
            BeginProfiling("ComputeAttention");
        }

        const float* q_ptr = static_cast<const float*>(queries.data);
        const float* k_ptr = static_cast<const float*>(keys.data);
        const float* v_ptr = static_cast<const float*>(values.data);
        float* o_ptr = static_cast<float*>(output.data);

        // Choose attention implementation based on sequence length
        bool success = false;
        if (seq_len <= 512) {
            success = ComputeAttentionSmall(q_ptr, k_ptr, v_ptr, o_ptr,
                                          batch_size, seq_len, head_dim, num_heads);
        } else {
            success = ComputeAttentionLarge(q_ptr, k_ptr, v_ptr, o_ptr,
                                          batch_size, seq_len, head_dim, num_heads);
        }

        if (profiling_enabled_) {
            // Attention FLOP count: 4 * batch_size * num_heads * seq_len^2 * head_dim
            size_t flops = 4ULL * batch_size * num_heads * seq_len * seq_len * head_dim;
            EndProfiling("ComputeAttention", 0, flops);
        }

        return success;

    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "ComputeAttention");
        return false;
    }
}

bool SyclBackend::ComputeAttentionSmall(
    const float* queries, const float* keys, const float* values, float* output,
    int batch_size, int seq_len, int head_dim, int num_heads) {

    try {
        // Allocate temporary buffers for attention scores
        size_t scores_size = batch_size * num_heads * seq_len * seq_len * sizeof(float);
        auto scores_buffer = AllocateBuffer(scores_size);
        if (!scores_buffer.data) {
            return false;
        }
        float* scores = static_cast<float*>(scores_buffer.data);

        float scale = 1.0f / ::sycl::sqrt(static_cast<float>(head_dim));

        // Step 1: Compute Q * K^T (scaled)
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < num_heads; ++h) {
                int offset = (b * num_heads + h) * seq_len * head_dim;
                int score_offset = (b * num_heads + h) * seq_len * seq_len;

                // Use oneMKL for Q * K^T
                auto gemm_event = oneapi::mkl::blas::column_major::gemm(
                    *current_queue_,
                    trans_n_,  // K^T
                    trans_t_,  // Q
                    seq_len,   // rows of K^T
                    seq_len,   // cols of Q
                    head_dim,  // inner dimension
                    scale,     // scaling factor
                    keys + offset,    // K matrix
                    head_dim,         // lda
                    queries + offset, // Q matrix
                    head_dim,         // ldb
                    0.0f,             // beta
                    scores + score_offset, // output
                    seq_len           // ldc
                );
                gemm_event.wait();
            }
        }

        // Step 2: Apply causal mask and softmax
        auto mask_softmax_event = current_queue_->parallel_for(
            ::sycl::range<3>(batch_size, num_heads, seq_len),
            [=](::sycl::id<3> idx) {
                int b = idx[0];
                int h = idx[1];
                int i = idx[2];

                int score_offset = (b * num_heads + h) * seq_len * seq_len + i * seq_len;

                // Apply causal mask and find max for numerical stability
                float max_val = -INFINITY;
                for (int j = 0; j <= i; ++j) {  // Causal mask: only attend to previous tokens
                    max_val = ::sycl::fmax(max_val, scores[score_offset + j]);
                }

                // Compute softmax with numerical stability
                float sum_exp = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    float exp_val = ::sycl::exp(scores[score_offset + j] - max_val);
                    scores[score_offset + j] = exp_val;
                    sum_exp += exp_val;
                }

                // Normalize
                for (int j = 0; j <= i; ++j) {
                    scores[score_offset + j] /= sum_exp;
                }

                // Zero out future positions (causal mask)
                for (int j = i + 1; j < seq_len; ++j) {
                    scores[score_offset + j] = 0.0f;
                }
            }
        );
        mask_softmax_event.wait();

        // Step 3: Compute attention output: Scores * V
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < num_heads; ++h) {
                int offset = (b * num_heads + h) * seq_len * head_dim;
                int score_offset = (b * num_heads + h) * seq_len * seq_len;

                auto output_gemm_event = oneapi::mkl::blas::column_major::gemm(
                    *current_queue_,
                    trans_n_,  // V
                    trans_n_,  // Scores
                    head_dim,  // rows of V
                    seq_len,   // cols of Scores
                    seq_len,   // inner dimension
                    1.0f,      // alpha
                    values + offset,      // V matrix
                    head_dim,             // lda
                    scores + score_offset, // Scores matrix
                    seq_len,              // ldb
                    0.0f,                 // beta
                    output + offset,      // output
                    head_dim              // ldc
                );
                output_gemm_event.wait();
            }
        }

        FreeBuffer(scores_buffer);
        return CheckDeviceError(*current_queue_, "ComputeAttentionSmall");

    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "ComputeAttentionSmall");
        return false;
    }
}

bool SyclBackend::ComputeAttentionLarge(
    const float* queries, const float* keys, const float* values, float* output,
    int batch_size, int seq_len, int head_dim, int num_heads) {

    try {
        // Flash attention-style implementation for memory efficiency
        constexpr int BLOCK_SIZE = 64;

        // Allocate working memory
        size_t block_scores_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
        size_t block_output_size = BLOCK_SIZE * head_dim * sizeof(float);

        auto block_scores_buffer = AllocateBuffer(block_scores_size * num_heads);
        auto block_output_buffer = AllocateBuffer(block_output_size * num_heads);
        auto max_buffer = AllocateBuffer(BLOCK_SIZE * sizeof(float) * num_heads);
        auto sum_buffer = AllocateBuffer(BLOCK_SIZE * sizeof(float) * num_heads);

        if (!block_scores_buffer.data || !block_output_buffer.data ||
            !max_buffer.data || !sum_buffer.data) {
            return false;
        }

        float* block_scores = static_cast<float*>(block_scores_buffer.data);
        float* block_output = static_cast<float*>(block_output_buffer.data);
        float* max_vals = static_cast<float*>(max_buffer.data);
        float* sum_vals = static_cast<float*>(sum_buffer.data);

        float scale = 1.0f / ::sycl::sqrt(static_cast<float>(head_dim));

        // Process in blocks to reduce memory usage
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < num_heads; ++h) {
                int head_offset = (b * num_heads + h) * seq_len * head_dim;
                int output_offset = head_offset;

                // Initialize output to zero
                auto zero_event = current_queue_->fill(
                    output + output_offset, 0.0f, seq_len * head_dim
                );
                zero_event.wait();

                // Process query blocks
                for (int q_start = 0; q_start < seq_len; q_start += BLOCK_SIZE) {
                    int q_end = ::sycl::min(q_start + BLOCK_SIZE, seq_len);
                    int q_size = q_end - q_start;

                    // Initialize running statistics
                    auto init_stats_event = current_queue_->parallel_for(
                        ::sycl::range<1>(q_size),
                        [=](::sycl::id<1> idx) {
                            max_vals[h * BLOCK_SIZE + idx] = -INFINITY;
                            sum_vals[h * BLOCK_SIZE + idx] = 0.0f;
                        }
                    );
                    init_stats_event.wait();

                    // Process key/value blocks
                    for (int kv_start = 0; kv_start <= q_start; kv_start += BLOCK_SIZE) {
                        int kv_end = ::sycl::min(kv_start + BLOCK_SIZE, q_start + q_size);
                        int kv_size = kv_end - kv_start;

                        // Compute Q_block * K_block^T
                        auto gemm_event = oneapi::mkl::blas::column_major::gemm(
                            *current_queue_,
                            trans_n_,  // K^T
                            trans_t_,  // Q
                            kv_size,   // rows
                            q_size,    // cols
                            head_dim,  // inner dim
                            scale,     // alpha
                            keys + head_offset + kv_start * head_dim,
                            head_dim,
                            queries + head_offset + q_start * head_dim,
                            head_dim,
                            0.0f,      // beta
                            block_scores + h * BLOCK_SIZE * BLOCK_SIZE,
                            kv_size
                        );
                        gemm_event.wait();

                        // Online softmax update
                        auto softmax_update_event = current_queue_->parallel_for(
                            ::sycl::range<1>(q_size),
                            [=](::sycl::id<1> q_idx) {
                                int global_q = q_start + q_idx;

                                // Find new maximum
                                float new_max = max_vals[h * BLOCK_SIZE + q_idx];
                                for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
                                    int global_kv = kv_start + kv_idx;
                                    if (global_kv <= global_q) {  // Causal mask
                                        float score = block_scores[h * BLOCK_SIZE * BLOCK_SIZE +
                                                                 q_idx * kv_size + kv_idx];
                                        new_max = ::sycl::fmax(new_max, score);
                                    }
                                }

                                // Update running statistics
                                float old_max = max_vals[h * BLOCK_SIZE + q_idx];
                                float scale_old = ::sycl::exp(old_max - new_max);
                                float new_sum = sum_vals[h * BLOCK_SIZE + q_idx] * scale_old;

                                for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
                                    int global_kv = kv_start + kv_idx;
                                    if (global_kv <= global_q) {
                                        float score = block_scores[h * BLOCK_SIZE * BLOCK_SIZE +
                                                                 q_idx * kv_size + kv_idx];
                                        float exp_score = ::sycl::exp(score - new_max);
                                        block_scores[h * BLOCK_SIZE * BLOCK_SIZE +
                                                   q_idx * kv_size + kv_idx] = exp_score;
                                        new_sum += exp_score;
                                    } else {
                                        block_scores[h * BLOCK_SIZE * BLOCK_SIZE +
                                                   q_idx * kv_size + kv_idx] = 0.0f;
                                    }
                                }

                                max_vals[h * BLOCK_SIZE + q_idx] = new_max;
                                sum_vals[h * BLOCK_SIZE + q_idx] = new_sum;
                            }
                        );
                        softmax_update_event.wait();

                        // Compute block output: Scores_block * V_block
                        auto output_gemm_event = oneapi::mkl::blas::column_major::gemm(
                            *current_queue_,
                            trans_n_,  // V
                            trans_n_,  // Scores
                            head_dim,  // rows
                            q_size,    // cols
                            kv_size,   // inner dim
                            1.0f,      // alpha
                            values + head_offset + kv_start * head_dim,
                            head_dim,
                            block_scores + h * BLOCK_SIZE * BLOCK_SIZE,
                            kv_size,
                            1.0f,      // beta (accumulate)
                            output + output_offset + q_start * head_dim,
                            head_dim
                        );
                        output_gemm_event.wait();
                    }

                    // Normalize final output
                    auto normalize_event = current_queue_->parallel_for(
                        ::sycl::range<2>(q_size, head_dim),
                        [=](::sycl::id<2> idx) {
                            int q_idx = idx[0];
                            int d_idx = idx[1];
                            int global_q = q_start + q_idx;

                            float sum = sum_vals[h * BLOCK_SIZE + q_idx];
                            if (sum > 0.0f) {
                                output[output_offset + global_q * head_dim + d_idx] /= sum;
                            }
                        }
                    );
                    normalize_event.wait();
                }
            }
        }

        // Free temporary buffers
        FreeBuffer(block_scores_buffer);
        FreeBuffer(block_output_buffer);
        FreeBuffer(max_buffer);
        FreeBuffer(sum_buffer);

        return CheckDeviceError(*current_queue_, "ComputeAttentionLarge");

    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "ComputeAttentionLarge");
        return false;
    }
}

// Additional attention utilities

/**
 * @brief Compute position-aware attention with rotary position embeddings
 */
bool SyclBackend::ComputeRotaryAttention(
    const BackendBuffer& queries, const BackendBuffer& keys, const BackendBuffer& values,
    const BackendBuffer& cos_cache, const BackendBuffer& sin_cache,
    const BackendBuffer& output, int batch_size, int seq_len, int head_dim, int num_heads) {

    if (!IsAvailable() || !queries.data || !keys.data || !values.data ||
        !cos_cache.data || !sin_cache.data || !output.data) {
        return false;
    }

    try {
        if (profiling_enabled_) {
            BeginProfiling("ComputeRotaryAttention");
        }

        const float* q_ptr = static_cast<const float*>(queries.data);
        const float* k_ptr = static_cast<const float*>(keys.data);
        const float* v_ptr = static_cast<const float*>(values.data);
        const float* cos_ptr = static_cast<const float*>(cos_cache.data);
        const float* sin_ptr = static_cast<const float*>(sin_cache.data);
        float* o_ptr = static_cast<float*>(output.data);

        // Allocate buffers for rotated Q and K
        size_t qk_size = batch_size * num_heads * seq_len * head_dim * sizeof(float);
        auto q_rot_buffer = AllocateBuffer(qk_size);
        auto k_rot_buffer = AllocateBuffer(qk_size);

        if (!q_rot_buffer.data || !k_rot_buffer.data) {
            return false;
        }

        float* q_rot = static_cast<float*>(q_rot_buffer.data);
        float* k_rot = static_cast<float*>(k_rot_buffer.data);

        // Apply rotary position embeddings
        auto rotary_event = current_queue_->parallel_for(
            ::sycl::range<4>(batch_size, num_heads, seq_len, head_dim / 2),
            [=](::sycl::id<4> idx) {
                int b = idx[0];
                int h = idx[1];
                int s = idx[2];
                int d = idx[3];

                int offset = ((b * num_heads + h) * seq_len + s) * head_dim;
                int cos_sin_offset = s * (head_dim / 2) + d;

                float cos_val = cos_ptr[cos_sin_offset];
                float sin_val = sin_ptr[cos_sin_offset];

                // Rotate Q
                float q0 = q_ptr[offset + 2 * d];
                float q1 = q_ptr[offset + 2 * d + 1];
                q_rot[offset + 2 * d] = q0 * cos_val - q1 * sin_val;
                q_rot[offset + 2 * d + 1] = q0 * sin_val + q1 * cos_val;

                // Rotate K
                float k0 = k_ptr[offset + 2 * d];
                float k1 = k_ptr[offset + 2 * d + 1];
                k_rot[offset + 2 * d] = k0 * cos_val - k1 * sin_val;
                k_rot[offset + 2 * d + 1] = k0 * sin_val + k1 * cos_val;
            }
        );
        rotary_event.wait();

        // Compute attention with rotated Q and K
        BackendBuffer q_rot_buf(q_rot, qk_size, true);
        BackendBuffer k_rot_buf(k_rot, qk_size, true);
        BackendBuffer v_buf(const_cast<float*>(v_ptr), qk_size, true);
        BackendBuffer o_buf(o_ptr, qk_size, true);

        bool success = ComputeAttention(q_rot_buf, k_rot_buf, v_buf, o_buf,
                                       batch_size, seq_len, head_dim, num_heads);

        FreeBuffer(q_rot_buffer);
        FreeBuffer(k_rot_buffer);

        if (profiling_enabled_) {
            size_t flops = 6ULL * batch_size * num_heads * seq_len * seq_len * head_dim;
            EndProfiling("ComputeRotaryAttention", 0, flops);
        }

        return success;

    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "ComputeRotaryAttention");
        return false;
    }
}

/**
 * @brief Multi-query attention for more efficient inference
 */
bool SyclBackend::ComputeMultiQueryAttention(
    const BackendBuffer& queries, const BackendBuffer& key, const BackendBuffer& value,
    const BackendBuffer& output, int batch_size, int seq_len, int head_dim, int num_heads) {

    if (!IsAvailable() || !queries.data || !key.data || !value.data || !output.data) {
        return false;
    }

    try {
        if (profiling_enabled_) {
            BeginProfiling("ComputeMultiQueryAttention");
        }

        const float* q_ptr = static_cast<const float*>(queries.data);
        const float* k_ptr = static_cast<const float*>(key.data);    // Single key
        const float* v_ptr = static_cast<const float*>(value.data);  // Single value
        float* o_ptr = static_cast<float*>(output.data);

        float scale = 1.0f / ::sycl::sqrt(static_cast<float>(head_dim));

        // Allocate temporary buffer for attention scores
        size_t scores_size = batch_size * num_heads * seq_len * seq_len * sizeof(float);
        auto scores_buffer = AllocateBuffer(scores_size);
        if (!scores_buffer.data) {
            return false;
        }
        float* scores = static_cast<float*>(scores_buffer.data);

        // Compute attention for each head with shared K,V
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < num_heads; ++h) {
                int q_offset = (b * num_heads + h) * seq_len * head_dim;
                int kv_offset = b * seq_len * head_dim;  // Shared across heads
                int score_offset = (b * num_heads + h) * seq_len * seq_len;
                int o_offset = (b * num_heads + h) * seq_len * head_dim;

                // Q * K^T
                auto qk_event = oneapi::mkl::blas::column_major::gemm(
                    *current_queue_,
                    trans_n_, trans_t_,
                    seq_len, seq_len, head_dim,
                    scale,
                    k_ptr + kv_offset, head_dim,
                    q_ptr + q_offset, head_dim,
                    0.0f,
                    scores + score_offset, seq_len
                );
                qk_event.wait();

                // Apply causal mask and softmax
                auto mask_softmax_event = current_queue_->parallel_for(
                    ::sycl::range<1>(seq_len),
                    [=](::sycl::id<1> i) {
                        int row_offset = score_offset + i * seq_len;

                        // Find max for numerical stability
                        float max_val = -INFINITY;
                        for (int j = 0; j <= i; ++j) {
                            max_val = ::sycl::fmax(max_val, scores[row_offset + j]);
                        }

                        // Compute softmax
                        float sum_exp = 0.0f;
                        for (int j = 0; j <= i; ++j) {
                            float exp_val = ::sycl::exp(scores[row_offset + j] - max_val);
                            scores[row_offset + j] = exp_val;
                            sum_exp += exp_val;
                        }

                        // Normalize and apply causal mask
                        for (int j = 0; j <= i; ++j) {
                            scores[row_offset + j] /= sum_exp;
                        }
                        for (int j = i + 1; j < seq_len; ++j) {
                            scores[row_offset + j] = 0.0f;
                        }
                    }
                );
                mask_softmax_event.wait();

                // Scores * V
                auto sv_event = oneapi::mkl::blas::column_major::gemm(
                    *current_queue_,
                    trans_n_, trans_n_,
                    head_dim, seq_len, seq_len,
                    1.0f,
                    v_ptr + kv_offset, head_dim,
                    scores + score_offset, seq_len,
                    0.0f,
                    o_ptr + o_offset, head_dim
                );
                sv_event.wait();
            }
        }

        FreeBuffer(scores_buffer);

        if (profiling_enabled_) {
            size_t flops = 4ULL * batch_size * num_heads * seq_len * seq_len * head_dim;
            EndProfiling("ComputeMultiQueryAttention", 0, flops);
        }

        return CheckDeviceError(*current_queue_, "ComputeMultiQueryAttention");

    } catch (const ::sycl::exception& e) {
        HandleSyclException(e, "ComputeMultiQueryAttention");
        return false;
    }
}

} // namespace sycl
} // namespace backends
} // namespace gemma