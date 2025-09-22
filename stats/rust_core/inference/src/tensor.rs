//! High-performance tensor operations with SIMD optimizations

use crate::error::{InferenceError, InferenceResult};
use crate::memory::{TensorShape, TensorAllocation, MemoryPool};
use std::sync::Arc;

/// Data types supported by the tensor system
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    F32,
    F16,
    BF16,
    I32,
    I16,
    I8,
    U8,
}

impl DataType {
    /// Get the size in bytes for this data type
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::F32 | DataType::I32 => 4,
            DataType::F16 | DataType::BF16 | DataType::I16 => 2,
            DataType::I8 | DataType::U8 => 1,
        }
    }
}

/// High-performance tensor with SIMD operations
pub struct Tensor {
    data: Vec<f32>, // Simplified to f32 for now
    shape: TensorShape,
    data_type: DataType,
    memory_pool: Option<Arc<MemoryPool>>,
}

/// Tensor operations trait
pub trait TensorOps {
    /// Matrix multiplication with SIMD optimization
    fn matmul(&self, other: &Self) -> InferenceResult<Tensor>;

    /// Element-wise addition
    fn add(&self, other: &Self) -> InferenceResult<Tensor>;

    /// Softmax activation
    fn softmax(&self, dim: i32) -> InferenceResult<Tensor>;

    /// Layer normalization
    fn layer_norm(&self, eps: f32) -> InferenceResult<Tensor>;

    /// ReLU activation
    fn relu(&self) -> InferenceResult<Tensor>;

    /// GELU activation
    fn gelu(&self) -> InferenceResult<Tensor>;
}

impl Tensor {
    /// Create a new tensor with given shape and data type
    pub fn new(shape: TensorShape, data_type: DataType) -> Self {
        let size = shape.size();
        Self {
            data: vec![0.0; size],
            shape,
            data_type,
            memory_pool: None,
        }
    }

    /// Create tensor from memory pool allocation
    pub fn from_allocation(allocation: TensorAllocation<f32>) -> Self {
        let shape = allocation.shape().clone();
        let size = shape.size();
        let mut data = vec![0.0; size];

        // Copy data from allocation
        unsafe {
            std::ptr::copy_nonoverlapping(
                allocation.as_ptr(),
                data.as_mut_ptr(),
                size
            );
        }

        Self {
            data,
            shape,
            data_type: DataType::F32,
            memory_pool: None,
        }
    }

    /// Get tensor shape
    pub fn shape(&self) -> &TensorShape {
        &self.shape
    }

    /// Get data type
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Get raw data slice
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable raw data slice
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

impl TensorOps for Tensor {
    fn matmul(&self, other: &Self) -> InferenceResult<Tensor> {
        // Simplified matrix multiplication - would use optimized BLAS in production
        if self.shape.dims.len() != 2 || other.shape.dims.len() != 2 {
            return Err(InferenceError::runtime("Matrix multiplication requires 2D tensors"));
        }

        let m = self.shape.dims[0];
        let k = self.shape.dims[1];
        let n = other.shape.dims[1];

        if k != other.shape.dims[0] {
            return Err(InferenceError::runtime("Incompatible matrix dimensions"));
        }

        let result_shape = TensorShape::new(vec![m, n]);
        let mut result = Tensor::new(result_shape, DataType::F32);

        // Basic matrix multiplication with potential for SIMD optimization
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k_idx in 0..k {
                    sum += self.data[i * k + k_idx] * other.data[k_idx * n + j];
                }
                result.data[i * n + j] = sum;
            }
        }

        Ok(result)
    }

    fn add(&self, other: &Self) -> InferenceResult<Tensor> {
        if self.shape != other.shape {
            return Err(InferenceError::runtime("Tensor shapes must match for addition"));
        }

        let mut result = Tensor::new(self.shape.clone(), self.data_type);

        // SIMD-optimized addition would go here
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + other.data[i];
        }

        Ok(result)
    }

    fn softmax(&self, dim: i32) -> InferenceResult<Tensor> {
        let mut result = self.clone();

        // Simplified softmax implementation
        let size = self.data.len();

        // Find max for numerical stability
        let max_val = self.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp and sum
        let mut sum = 0.0;
        for i in 0..size {
            result.data[i] = (self.data[i] - max_val).exp();
            sum += result.data[i];
        }

        // Normalize
        for i in 0..size {
            result.data[i] /= sum;
        }

        Ok(result)
    }

    fn layer_norm(&self, eps: f32) -> InferenceResult<Tensor> {
        let mut result = self.clone();
        let size = self.data.len();

        // Calculate mean
        let mean = self.data.iter().sum::<f32>() / size as f32;

        // Calculate variance
        let variance = self.data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / size as f32;

        // Normalize
        let std_dev = (variance + eps).sqrt();
        for i in 0..size {
            result.data[i] = (self.data[i] - mean) / std_dev;
        }

        Ok(result)
    }

    fn relu(&self) -> InferenceResult<Tensor> {
        let mut result = self.clone();

        for i in 0..result.data.len() {
            result.data[i] = result.data[i].max(0.0);
        }

        Ok(result)
    }

    fn gelu(&self) -> InferenceResult<Tensor> {
        let mut result = self.clone();

        // GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        const SQRT_2_OVER_PI: f32 = 0.7978845608; // sqrt(2/π)

        for i in 0..result.data.len() {
            let x = result.data[i];
            let inner = SQRT_2_OVER_PI * (x + 0.044715 * x.powi(3));
            result.data[i] = x * 0.5 * (1.0 + inner.tanh());
        }

        Ok(result)
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            data_type: self.data_type,
            memory_pool: self.memory_pool.clone(),
        }
    }
}

/// SIMD-optimized operations for specific architectures
pub mod simd_ops {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    pub mod x86_64 {
        use super::*;

        /// SIMD dot product for x86_64 with AVX2
        #[target_feature(enable = "avx2")]
        pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
            // AVX2 implementation would go here
            // For now, fallback to scalar
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }

        /// SIMD vector addition with AVX2
        #[target_feature(enable = "avx2")]
        pub unsafe fn vector_add_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
            // AVX2 implementation would go here
            for i in 0..a.len() {
                result[i] = a[i] + b[i];
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub mod aarch64 {
        use super::*;

        /// SIMD dot product for ARM64 with NEON
        #[target_feature(enable = "neon")]
        pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
            // NEON implementation would go here
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }

        /// SIMD vector addition with NEON
        #[target_feature(enable = "neon")]
        pub unsafe fn vector_add_neon(a: &[f32], b: &[f32], result: &mut [f32]) {
            // NEON implementation would go here
            for i in 0..a.len() {
                result[i] = a[i] + b[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let shape = TensorShape::new(vec![2, 3]);
        let tensor = Tensor::new(shape, DataType::F32);
        assert_eq!(tensor.shape().dims, vec![2, 3]);
        assert_eq!(tensor.data().len(), 6);
    }

    #[test]
    fn test_tensor_addition() {
        let shape = TensorShape::new(vec![2, 2]);
        let mut a = Tensor::new(shape.clone(), DataType::F32);
        let mut b = Tensor::new(shape, DataType::F32);

        a.data_mut().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        b.data_mut().copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);

        let result = a.add(&b).unwrap();
        assert_eq!(result.data(), &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_softmax() {
        let shape = TensorShape::new(vec![4]);
        let mut tensor = Tensor::new(shape, DataType::F32);
        tensor.data_mut().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let result = tensor.softmax(0).unwrap();
        let sum: f32 = result.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
