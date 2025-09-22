#pragma once

/**
 * @file cuda_attention.h
 * @brief Header for CUDA attention kernels including Flash Attention v2
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace gemma {
namespace backends {
namespace cuda {
namespace attention {

/**
 * @brief Launch Flash Attention v2 kernel
 *
 * This implements the memory-efficient Flash Attention algorithm that:
 * - Tiles the computation to fit in shared memory
 * - Uses online softmax for numerical stability
 * - Minimizes HBM memory accesses
 * - Supports causal (autoregressive) attention
 *
 * @param Q Query tensor [batch_size, num_heads, seq_len, head_dim]
 * @param K Key tensor [batch_size, num_heads, seq_len, head_dim]
 * @param V Value tensor [batch_size, num_heads, seq_len, head_dim]
 * @param O Output tensor [batch_size, num_heads, seq_len, head_dim]
 * @param L Row sums for numerical stability [batch_size, num_heads, seq_len]
 * @param M Row maxes for numerical stability [batch_size, num_heads, seq_len]
 * @param batch_size Batch size
 * @param num_heads Number of attention heads
 * @param seq_len Sequence length
 * @param head_dim Head dimension
 * @param scale Attention scale factor (typically 1/sqrt(head_dim))
 * @param causal Whether to apply causal mask
 * @param stream CUDA stream for asynchronous execution
 * @return CUDA error code
 */
template<typename T>
cudaError_t launch_flash_attention_kernel(
    const T* Q, const T* K, const T* V, T* O, T* L, T* M,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, bool causal = false, cudaStream_t stream = nullptr);

/**
 * @brief Launch multi-head attention kernel with KV caching
 *
 * This kernel is optimized for inference scenarios where:
 * - K and V are cached from previous tokens
 * - Only new queries need to attend to the full context
 * - Supports grouped query attention (GQA)
 *
 * @param Q Query tensor [batch_size, num_heads, seq_len, head_dim]
 * @param K_cache Cached keys [batch_size, num_kv_heads, max_seq_len, head_dim]
 * @param V_cache Cached values [batch_size, num_kv_heads, max_seq_len, head_dim]
 * @param O Output tensor [batch_size, num_heads, seq_len, head_dim]
 * @param batch_size Batch size
 * @param num_heads Number of query heads
 * @param num_kv_heads Number of key/value heads (for GQA)
 * @param seq_len Current sequence length
 * @param kv_seq_len Cached sequence length
 * @param head_dim Head dimension
 * @param scale Attention scale factor
 * @param stream CUDA stream for asynchronous execution
 * @return CUDA error code
 */
template<typename T>
cudaError_t launch_mha_with_kv_cache_kernel(
    const T* Q, const T* K_cache, const T* V_cache, T* O,
    int batch_size, int num_heads, int num_kv_heads,
    int seq_len, int kv_seq_len, int head_dim,
    float scale, cudaStream_t stream = nullptr);

/**
 * @brief Launch standard attention kernel (for comparison/fallback)
 *
 * This implements the standard O(n²) attention algorithm:
 * - Computes full attention matrix
 * - Applies softmax
 * - Computes output
 *
 * Note: This is memory-intensive and should only be used for:
 * - Small sequences
 * - Debugging/comparison purposes
 * - When Flash Attention is not available
 *
 * @param Q Query tensor [batch_size, num_heads, seq_len, head_dim]
 * @param K Key tensor [batch_size, num_heads, seq_len, head_dim]
 * @param V Value tensor [batch_size, num_heads, seq_len, head_dim]
 * @param O Output tensor [batch_size, num_heads, seq_len, head_dim]
 * @param temp_scores Temporary scores [batch_size, num_heads, seq_len, seq_len]
 * @param batch_size Batch size
 * @param num_heads Number of attention heads
 * @param seq_len Sequence length
 * @param head_dim Head dimension
 * @param scale Attention scale factor
 * @param stream CUDA stream for asynchronous execution
 * @return CUDA error code
 */
template<typename T>
cudaError_t launch_standard_attention_kernel(
    const T* Q, const T* K, const T* V, T* O, T* temp_scores,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, cudaStream_t stream = nullptr);

/**
 * @brief Launch RoPE (Rotary Positional Embedding) kernel
 *
 * Applies rotary positional embeddings to query and key tensors.
 * This is commonly used in modern transformer architectures.
 *
 * @param Q Query tensor to apply RoPE [batch_size, num_heads, seq_len, head_dim]
 * @param K Key tensor to apply RoPE [batch_size, num_heads, seq_len, head_dim]
 * @param positions Position indices [batch_size, seq_len]
 * @param batch_size Batch size
 * @param num_heads Number of attention heads
 * @param seq_len Sequence length
 * @param head_dim Head dimension
 * @param rope_base Base for RoPE computation (default: 10000.0)
 * @param stream CUDA stream for asynchronous execution
 * @return CUDA error code
 */
template<typename T>
cudaError_t launch_rope_kernel(
    T* Q, T* K, const int* positions,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float rope_base = 10000.0f, cudaStream_t stream = nullptr);

/**
 * @brief Configuration for attention kernels
 */
struct AttentionConfig {
    bool use_flash_attention = true;      // Use Flash Attention when possible
    bool causal = true;                   // Apply causal mask
    float scale = 0.0f;                   // Auto-compute if 0.0
    bool use_rope = false;                // Apply RoPE
    float rope_base = 10000.0f;           // RoPE base frequency
    int tile_size = 64;                   // Tile size for Flash Attention
    size_t temp_memory_size = 0;          // Temporary memory for standard attention
};

/**
 * @brief Get optimal attention configuration for given parameters
 */
AttentionConfig get_optimal_attention_config(
    int batch_size, int num_heads, int seq_len, int head_dim,
    size_t available_memory, bool prefer_speed = true);

/**
 * @brief Estimate memory requirements for attention computation
 */
struct AttentionMemoryEstimate {
    size_t flash_attention_bytes;     // Memory for Flash Attention
    size_t standard_attention_bytes;  // Memory for standard attention
    size_t kv_cache_bytes;           // Memory for KV cache
    size_t temp_buffer_bytes;        // Temporary buffers
};

AttentionMemoryEstimate estimate_attention_memory(
    int batch_size, int num_heads, int seq_len, int head_dim,
    int max_seq_len = -1, bool use_kv_cache = false);

/**
 * @brief Check if Flash Attention is supported for given parameters
 */
bool is_flash_attention_supported(
    int batch_size, int num_heads, int seq_len, int head_dim,
    int device_compute_capability_major, int device_compute_capability_minor);

/**
 * @brief Benchmark different attention implementations
 */
struct AttentionBenchmarkResult {
    float flash_attention_ms;
    float standard_attention_ms;
    float kv_cache_attention_ms;
    float memory_usage_mb;
    bool flash_attention_available;
};

AttentionBenchmarkResult benchmark_attention_kernels(
    int batch_size, int num_heads, int seq_len, int head_dim,
    int num_iterations = 10, cudaStream_t stream = nullptr);

} // namespace attention
} // namespace cuda
} // namespace backends
} // namespace gemma