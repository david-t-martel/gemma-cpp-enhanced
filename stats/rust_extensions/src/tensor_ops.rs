//! Optimized tensor operations for Gemma models
//!
//! This module provides high-performance implementations of common tensor operations
//! used in transformer models, with SIMD optimizations where available.

use crate::error::{GemmaError, GemmaResult};
use crate::utils::{math, memory::AlignedVec, simd};
use pyo3::prelude::*;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::sync::Arc;
use bytemuck::{Pod, Zeroable};

/// Configuration for tensor operations
#[pyclass]
#[derive(Debug, Clone)]
pub struct TensorConfig {
    /// Use SIMD optimizations when available
    #[pyo3(get, set)]
    pub use_simd: bool,

    /// Use parallel processing for large operations
    #[pyo3(get, set)]
    pub parallel: bool,

    /// Minimum size threshold for parallel processing
    #[pyo3(get, set)]
    pub parallel_threshold: usize,

    /// Memory alignment for SIMD operations
    #[pyo3(get, set)]
    pub memory_alignment: usize,

    /// Numerical epsilon for stability
    #[pyo3(get, set)]
    pub epsilon: f32,
}

impl Default for TensorConfig {
    fn default() -> Self {
        Self {
            use_simd: simd::has_simd(),
            parallel: true,
            parallel_threshold: 1000,
            memory_alignment: 32, // AVX2 alignment
            epsilon: 1e-8,
        }
    }
}

#[pymethods]
impl TensorConfig {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create config optimized for specific operations
    #[staticmethod]
    pub fn for_attention(seq_len: usize, hidden_dim: usize) -> Self {
        Self {
            parallel_threshold: (seq_len * hidden_dim).min(500),
            ..Self::default()
        }
    }
}

/// High-performance tensor operations
#[pyclass]
pub struct TensorOperations {
    config: TensorConfig,
}

#[pymethods]
impl TensorOperations {
    #[new]
    pub fn new(config: Option<TensorConfig>) -> Self {
        Self {
            config: config.unwrap_or_default(),
        }
    }

    /// Matrix multiplication with SIMD optimization
    pub fn matmul(
        &self,
        a: Vec<f32>,
        b: Vec<f32>,
        m: usize,
        k: usize,
        n: usize,
    ) -> PyResult<Vec<f32>> {
        if a.len() != m * k || b.len() != k * n {
            return Err(GemmaError::dimension_mismatch(
                m * k + k * n,
                a.len() + b.len(),
            ).into());
        }

        let mut result = vec![0.0f32; m * n];
        self.matmul_internal(&a, &b, &mut result, m, k, n)?;
        Ok(result)
    }

    /// Optimized softmax computation
    pub fn softmax(&self, input: Vec<f32>) -> PyResult<Vec<f32>> {
        let mut output = vec![0.0f32; input.len()];
        math::stable_softmax(&input, &mut output)
            .map_err(PyErr::from)?;
        Ok(output)
    }

    /// Layer normalization
    pub fn layer_norm(
        &self,
        input: Vec<f32>,
        gamma: Vec<f32>,
        beta: Vec<f32>,
    ) -> PyResult<Vec<f32>> {
        let mut output = vec![0.0f32; input.len()];
        math::layer_norm(&input, &mut output, &gamma, &beta, self.config.epsilon)
            .map_err(PyErr::from)?;
        Ok(output)
    }

    /// RMSNorm (Root Mean Square Normalization) used in some models
    pub fn rms_norm(&self, input: Vec<f32>, weight: Vec<f32>) -> PyResult<Vec<f32>> {
        if input.len() != weight.len() {
            return Err(GemmaError::dimension_mismatch(input.len(), weight.len()).into());
        }

        let mut output = vec![0.0f32; input.len()];
        self.rms_norm_internal(&input, &weight, &mut output)?;
        Ok(output)
    }

    /// Optimized attention computation
    pub fn scaled_dot_product_attention(
        &self,
        query: Vec<f32>,
        key: Vec<f32>,
        value: Vec<f32>,
        seq_len: usize,
        hidden_dim: usize,
        scale: Option<f32>,
    ) -> PyResult<Vec<f32>> {
        let scale = scale.unwrap_or(1.0 / (hidden_dim as f32).sqrt());

        // Q * K^T
        let mut attention_scores = vec![0.0f32; seq_len * seq_len];
        self.matmul_internal(&query, &key, &mut attention_scores, seq_len, hidden_dim, seq_len)?;

        // Scale
        if self.config.use_simd {
            simd_scale(&mut attention_scores, scale)?;
        } else {
            for score in attention_scores.iter_mut() {
                *score *= scale;
            }
        }

        // Softmax over last dimension
        let mut attention_weights = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            let start = i * seq_len;
            let end = start + seq_len;
            math::stable_softmax(
                &attention_scores[start..end],
                &mut attention_weights[start..end],
            )?;
        }

        // Attention * V
        let mut output = vec![0.0f32; seq_len * hidden_dim];
        self.matmul_internal(&attention_weights, &value, &mut output, seq_len, seq_len, hidden_dim)?;

        Ok(output)
    }

    /// GELU activation function
    pub fn gelu(&self, input: Vec<f32>) -> PyResult<Vec<f32>> {
        let output = if self.config.use_simd && input.len() >= self.config.parallel_threshold {
            simd_gelu(&input)?
        } else {
            input.into_iter().map(gelu_scalar).collect()
        };
        Ok(output)
    }

    /// SiLU/Swish activation function
    pub fn silu(&self, input: Vec<f32>) -> PyResult<Vec<f32>> {
        let output = if self.config.use_simd && input.len() >= self.config.parallel_threshold {
            simd_silu(&input)?
        } else {
            input.into_iter().map(silu_scalar).collect()
        };
        Ok(output)
    }
}

impl TensorOperations {
    /// Internal matrix multiplication implementation
    fn matmul_internal(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> GemmaResult<()> {
        if self.config.parallel && m * n >= self.config.parallel_threshold {
            // Parallel implementation
            #[cfg(feature = "parallel")]
            {
                c.par_chunks_mut(n)
            }
            #[cfg(not(feature = "parallel"))]
            {
                c.chunks_mut(n).map(|chunk| chunk.into_iter())
            }
                .enumerate()
                .for_each(|(i, c_row)| {
                    for j in 0..n {
                        let mut sum = 0.0f32;
                        for kk in 0..k {
                            sum += a[i * k + kk] * b[kk * n + j];
                        }
                        c_row[j] = sum;
                    }
                });
        } else if self.config.use_simd {
            // SIMD implementation
            self.matmul_simd(a, b, c, m, k, n)?;
        } else {
            // Standard implementation
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += a[i * k + kk] * b[kk * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
        }
        Ok(())
    }

    /// SIMD-optimized matrix multiplication
    fn matmul_simd(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> GemmaResult<()> {
        #[cfg(target_feature = "avx2")]
        {
            self.matmul_avx2(a, b, c, m, k, n)
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            // Fallback to standard implementation
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += a[i * k + kk] * b[kk * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
            Ok(())
        }
    }

    #[cfg(target_feature = "avx2")]
    fn matmul_avx2(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> GemmaResult<()> {
        use std::arch::x86_64::*;

        unsafe {
            for i in 0..m {
                for j in (0..n).step_by(8) {
                    let mut sum = _mm256_setzero_ps();

                    for kk in 0..k {
                        let a_val = _mm256_broadcast_ss(&a[i * k + kk]);
                        let b_vals = if j + 8 <= n {
                            _mm256_loadu_ps(&b[kk * n + j])
                        } else {
                            // Handle remainder
                            let mut temp = [0.0f32; 8];
                            for idx in 0..(n - j).min(8) {
                                temp[idx] = b[kk * n + j + idx];
                            }
                            _mm256_loadu_ps(temp.as_ptr())
                        };

                        sum = _mm256_fmadd_ps(a_val, b_vals, sum);
                    }

                    if j + 8 <= n {
                        _mm256_storeu_ps(&mut c[i * n + j], sum);
                    } else {
                        // Handle remainder
                        let mut temp = [0.0f32; 8];
                        _mm256_storeu_ps(temp.as_mut_ptr(), sum);
                        for idx in 0..(n - j) {
                            c[i * n + j + idx] = temp[idx];
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// RMSNorm implementation
    fn rms_norm_internal(&self, input: &[f32], weight: &[f32], output: &mut [f32]) -> GemmaResult<()> {
        let n = input.len() as f32;

        // Compute mean square
        let mean_square = input.iter().map(|&x| x * x).sum::<f32>() / n;
        let rms = (mean_square + self.config.epsilon).sqrt();
        let inv_rms = 1.0 / rms;

        // Apply normalization and scaling
        for (((&x, &w), out)) in input.iter().zip(weight.iter()).zip(output.iter_mut()) {
            *out = x * inv_rms * w;
        }

        Ok(())
    }
}

/// Standalone SIMD dot product function
#[pyfunction]
pub fn simd_dot_product(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    if a.len() != b.len() {
        return Err(GemmaError::dimension_mismatch(a.len(), b.len()).into());
    }

    if simd::has_simd() {
        simd_dot_product_internal(&a, &b).map_err(PyErr::from)
    } else {
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum())
    }
}

/// SIMD vector addition
#[pyfunction]
pub fn simd_vector_add(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    if a.len() != b.len() {
        return Err(GemmaError::dimension_mismatch(a.len(), b.len()).into());
    }

    let mut result = vec![0.0f32; a.len()];
    if simd::has_simd() {
        simd_vector_add_internal(&a, &b, &mut result)?;
    } else {
        for (((&x, &y), out)) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
            *out = x + y;
        }
    }
    Ok(result)
}

/// SIMD softmax
#[pyfunction]
pub fn simd_softmax(input: Vec<f32>) -> PyResult<Vec<f32>> {
    let mut output = vec![0.0f32; input.len()];
    math::stable_softmax(&input, &mut output)?;
    Ok(output)
}

/// Optimized attention weight computation
#[pyfunction]
pub fn optimize_attention_weights(
    query: Vec<f32>,
    key: Vec<f32>,
    seq_len: usize,
    hidden_dim: usize,
) -> PyResult<Vec<f32>> {
    if query.len() != seq_len * hidden_dim || key.len() != seq_len * hidden_dim {
        return Err(GemmaError::dimension_mismatch(
            seq_len * hidden_dim,
            query.len().max(key.len()),
        ).into());
    }

    let scale = 1.0 / (hidden_dim as f32).sqrt();
    let mut attention_scores = vec![0.0f32; seq_len * seq_len];

    // Compute Q * K^T with scaling
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot_product = 0.0f32;
            for k in 0..hidden_dim {
                dot_product += query[i * hidden_dim + k] * key[j * hidden_dim + k];
            }
            attention_scores[i * seq_len + j] = dot_product * scale;
        }
    }

    // Apply softmax row-wise
    let mut attention_weights = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        let start = i * seq_len;
        let end = start + seq_len;
        math::stable_softmax(
            &attention_scores[start..end],
            &mut attention_weights[start..end],
        )?;
    }

    Ok(attention_weights)
}

/// Batch matrix multiplication
#[pyfunction]
pub fn batch_matmul(
    a_batch: Vec<Vec<f32>>,
    b_batch: Vec<Vec<f32>>,
    m: usize,
    k: usize,
    n: usize,
) -> PyResult<Vec<Vec<f32>>> {
    if a_batch.len() != b_batch.len() {
        return Err(GemmaError::dimension_mismatch(a_batch.len(), b_batch.len()).into());
    }

    let tensor_ops = TensorOperations::new(None);
    let results: Result<Vec<_>, _> = a_batch
        .into_par_iter()
        .zip(b_batch.into_par_iter())
        .map(|(a, b)| tensor_ops.matmul(a, b, m, k, n))
        .collect();

    results.map_err(|e| e)
}

/// Fast layer normalization
#[pyfunction]
pub fn fast_layer_norm(
    input: Vec<f32>,
    gamma: Vec<f32>,
    beta: Vec<f32>,
    epsilon: Option<f32>,
) -> PyResult<Vec<f32>> {
    let eps = epsilon.unwrap_or(1e-8);
    let mut output = vec![0.0f32; input.len()];
    math::layer_norm(&input, &mut output, &gamma, &beta, eps)?;
    Ok(output)
}

// Internal SIMD implementations

fn simd_dot_product_internal(a: &[f32], b: &[f32]) -> GemmaResult<f32> {
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    {
        simd_dot_product_avx2(a, b)
    }
    #[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
    {
        simd_dot_product_neon(a, b)
    }
    #[cfg(not(any(
        all(target_feature = "avx2", target_arch = "x86_64"),
        all(target_feature = "neon", target_arch = "aarch64")
    )))]
    {
        Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum())
    }
}

#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
fn simd_dot_product_avx2(a: &[f32], b: &[f32]) -> GemmaResult<f32> {
    use std::arch::x86_64::*;

    let len = a.len();
    let mut sum = 0.0f32;

    unsafe {
        let mut acc = _mm256_setzero_ps();

        // Process 8 elements at a time
        for i in (0..len).step_by(8) {
            if i + 8 <= len {
                let va = _mm256_loadu_ps(&a[i]);
                let vb = _mm256_loadu_ps(&b[i]);
                acc = _mm256_fmadd_ps(va, vb, acc);
            } else {
                // Handle remainder
                for j in i..len {
                    sum += a[j] * b[j];
                }
                break;
            }
        }

        // Horizontal sum of acc
        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), acc);
        sum += temp.iter().sum::<f32>();
    }

    Ok(sum)
}

#[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
fn simd_dot_product_neon(a: &[f32], b: &[f32]) -> GemmaResult<f32> {
    use std::arch::aarch64::*;

    let len = a.len();
    let mut sum = 0.0f32;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);

        // Process 4 elements at a time
        for i in (0..len).step_by(4) {
            if i + 4 <= len {
                let va = vld1q_f32(&a[i]);
                let vb = vld1q_f32(&b[i]);
                acc = vfmaq_f32(acc, va, vb);
            } else {
                // Handle remainder
                for j in i..len {
                    sum += a[j] * b[j];
                }
                break;
            }
        }

        // Horizontal sum
        sum += vaddvq_f32(acc);
    }

    Ok(sum)
}

fn simd_vector_add_internal(a: &[f32], b: &[f32], result: &mut [f32]) -> GemmaResult<()> {
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    {
        simd_vector_add_avx2(a, b, result)
    }
    #[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
    {
        simd_vector_add_neon(a, b, result)
    }
    #[cfg(not(any(
        all(target_feature = "avx2", target_arch = "x86_64"),
        all(target_feature = "neon", target_arch = "aarch64")
    )))]
    {
        for (((&x, &y), out)) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
            *out = x + y;
        }
        Ok(())
    }
}

#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
fn simd_vector_add_avx2(a: &[f32], b: &[f32], result: &mut [f32]) -> GemmaResult<()> {
    use std::arch::x86_64::*;

    let len = a.len();

    unsafe {
        for i in (0..len).step_by(8) {
            if i + 8 <= len {
                let va = _mm256_loadu_ps(&a[i]);
                let vb = _mm256_loadu_ps(&b[i]);
                let sum = _mm256_add_ps(va, vb);
                _mm256_storeu_ps(&mut result[i], sum);
            } else {
                // Handle remainder
                for j in i..len {
                    result[j] = a[j] + b[j];
                }
                break;
            }
        }
    }

    Ok(())
}

#[cfg(all(target_feature = "neon", target_arch = "aarch64"))]
fn simd_vector_add_neon(a: &[f32], b: &[f32], result: &mut [f32]) -> GemmaResult<()> {
    use std::arch::aarch64::*;

    let len = a.len();

    unsafe {
        for i in (0..len).step_by(4) {
            if i + 4 <= len {
                let va = vld1q_f32(&a[i]);
                let vb = vld1q_f32(&b[i]);
                let sum = vaddq_f32(va, vb);
                vst1q_f32(&mut result[i], sum);
            } else {
                // Handle remainder
                for j in i..len {
                    result[j] = a[j] + b[j];
                }
                break;
            }
        }
    }

    Ok(())
}

fn simd_scale(data: &mut [f32], scale: f32) -> GemmaResult<()> {
    if simd::has_simd() {
        #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
        {
            use std::arch::x86_64::*;
            unsafe {
                let scale_vec = _mm256_set1_ps(scale);
                for chunk in data.chunks_exact_mut(8) {
                    let vals = _mm256_loadu_ps(chunk.as_ptr());
                    let scaled = _mm256_mul_ps(vals, scale_vec);
                    _mm256_storeu_ps(chunk.as_mut_ptr(), scaled);
                }

                // Handle remainder
                let remainder_start = (data.len() / 8) * 8;
                for val in &mut data[remainder_start..] {
                    *val *= scale;
                }
            }
        }
        #[cfg(not(all(target_feature = "avx2", target_arch = "x86_64")))]
        {
            for val in data.iter_mut() {
                *val *= scale;
            }
        }
    } else {
        for val in data.iter_mut() {
            *val *= scale;
        }
    }
    Ok(())
}

// Activation functions
fn gelu_scalar(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh())
}

fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn simd_gelu(input: &[f32]) -> GemmaResult<Vec<f32>> {
    Ok(input.par_iter().map(|&x| gelu_scalar(x)).collect())
}

fn simd_silu(input: &[f32]) -> GemmaResult<Vec<f32>> {
    Ok(input.par_iter().map(|&x| silu_scalar(x)).collect())
}

/// Register the tensor operations module with Python
pub fn register_module(py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<TensorOperations>()?;
    module.add_class::<TensorConfig>()?;
    module.add_function(wrap_pyfunction!(simd_dot_product, module)?)?;
    module.add_function(wrap_pyfunction!(simd_vector_add, module)?)?;
    module.add_function(wrap_pyfunction!(simd_softmax, module)?)?;
    module.add_function(wrap_pyfunction!(optimize_attention_weights, module)?)?;
    module.add_function(wrap_pyfunction!(batch_matmul, module)?)?;
    module.add_function(wrap_pyfunction!(fast_layer_norm, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tensor_config() {
        let config = TensorConfig::default();
        assert!(config.epsilon > 0.0);
        assert!(config.memory_alignment > 0);
    }

    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let result = simd_dot_product(&a, &b).unwrap();
        assert_relative_eq!(result, 40.0, epsilon = 1e-6);
    }

    #[test]
    fn test_simd_vector_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = simd_vector_add(&a, &b).unwrap();
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matrix_multiplication() {
        let tensor_ops = TensorOperations::new(None);
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2

        let result = tensor_ops.matmul(a, b, 2, 2, 2).unwrap();
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_softmax() {
        let tensor_ops = TensorOperations::new(None);
        let input = vec![1.0, 2.0, 3.0];
        let result = tensor_ops.softmax(input).unwrap();

        let sum: f32 = result.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_layer_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0, 0.0];

        let result = fast_layer_norm(&input, &gamma, &beta, None).unwrap();

        // Check that result has mean ~0 and std ~1
        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert_relative_eq!(mean, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gelu_activation() {
        let tensor_ops = TensorOperations::new(None);
        let input = vec![-1.0, 0.0, 1.0];
        let result = tensor_ops.gelu(input).unwrap();

        // GELU(0) should be 0
        assert_relative_eq!(result[1], 0.0, epsilon = 1e-6);
        // GELU should be monotonic
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }
}
