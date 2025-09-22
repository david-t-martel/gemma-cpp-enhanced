//! Utility functions and helpers for Gemma extensions

use crate::error::{GemmaError, GemmaResult};
#[allow(unused_imports)]
use std::collections::HashMap;

/// Memory alignment utilities
pub mod memory {
    use std::alloc::{alloc_zeroed, dealloc, Layout};
    use std::ptr::NonNull;

    /// Aligned memory allocation for SIMD operations
    pub struct AlignedVec<T> {
        ptr: NonNull<T>,
        len: usize,
        capacity: usize,
        alignment: usize,
    }

    impl<T> AlignedVec<T> {
        /// Create a new aligned vector with specified alignment
        pub fn with_capacity_aligned(capacity: usize, alignment: usize) -> Option<Self> {
            if capacity == 0 {
                return Some(Self {
                    ptr: NonNull::dangling(),
                    len: 0,
                    capacity: 0,
                    alignment,
                });
            }

            let layout = Layout::from_size_align(
                capacity * std::mem::size_of::<T>(),
                alignment.max(std::mem::align_of::<T>()),
            )
            .ok()?;

            let ptr = unsafe { alloc_zeroed(layout) };
            if ptr.is_null() {
                return None;
            }

            Some(Self {
                ptr: unsafe { NonNull::new_unchecked(ptr as *mut T) },
                len: 0,
                capacity,
                alignment: layout.align(),
            })
        }

        /// Get a slice of the vector
        pub fn as_slice(&self) -> &[T] {
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }

        /// Get a mutable slice of the vector
        pub fn as_mut_slice(&mut self) -> &mut [T] {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }

        /// Push an element to the vector
        pub fn push(&mut self, value: T) -> Result<(), T> {
            if self.len >= self.capacity {
                return Err(value);
            }
            unsafe {
                self.ptr.as_ptr().add(self.len).write(value);
            }
            self.len += 1;
            Ok(())
        }

        /// Get the length of the vector
        pub fn len(&self) -> usize {
            self.len
        }

        /// Check if the vector is empty
        pub fn is_empty(&self) -> bool {
            self.len == 0
        }

        /// Get the capacity of the vector
        pub fn capacity(&self) -> usize {
            self.capacity
        }

        /// Get the alignment of the vector
        pub fn alignment(&self) -> usize {
            self.alignment
        }
    }

    impl<T> Drop for AlignedVec<T> {
        fn drop(&mut self) {
            if self.capacity > 0 {
                unsafe {
                    // Drop all elements
                    for i in 0..self.len {
                        self.ptr.as_ptr().add(i).drop_in_place();
                    }

                    // Deallocate memory
                    let layout = Layout::from_size_align_unchecked(
                        self.capacity * std::mem::size_of::<T>(),
                        self.alignment,
                    );
                    dealloc(self.ptr.as_ptr() as *mut u8, layout);
                }
            }
        }
    }

    unsafe impl<T: Send> Send for AlignedVec<T> {}
    unsafe impl<T: Sync> Sync for AlignedVec<T> {}
}

/// SIMD detection and utilities
pub mod simd {
    /// Check if AVX2 is available on x86_64
    #[cfg(target_arch = "x86_64")]
    pub fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }

    /// Check if NEON is available on aarch64
    #[cfg(target_arch = "aarch64")]
    pub fn has_neon() -> bool {
        std::arch::is_aarch64_feature_detected!("neon")
    }

    /// Check if any SIMD is available
    pub fn has_simd() -> bool {
        #[cfg(target_arch = "x86_64")]
        return has_avx2();

        #[cfg(target_arch = "aarch64")]
        return has_neon();

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        return false;
    }

    /// Get the optimal chunk size for SIMD operations
    pub fn optimal_chunk_size<T>() -> usize {
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            return 256 / (8 * std::mem::size_of::<T>()).max(1);
        }

        #[cfg(target_arch = "aarch64")]
        if has_neon() {
            return 128 / (8 * std::mem::size_of::<T>()).max(1);
        }

        // Fallback for scalar operations
        16 / std::mem::size_of::<T>().max(1)
    }
}

/// String processing utilities
pub mod string {
    use super::*;

    /// Fast string normalization for tokenization
    pub fn normalize_text(text: &str) -> String {
        // Basic normalization: lowercase, trim whitespace, normalize unicode
        text.to_lowercase()
            .chars()
            .map(|c| match c {
                '\t' | '\n' | '\r' => ' ',
                c if c.is_whitespace() => ' ',
                c => c,
            })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Fast whitespace tokenization
    pub fn fast_split_whitespace(text: &str) -> Vec<&str> {
        let mut tokens = Vec::new();
        let mut start = None;

        for (i, byte) in text.bytes().enumerate() {
            match byte {
                b' ' | b'\t' | b'\n' | b'\r' => {
                    if let Some(s) = start {
                        tokens.push(&text[s..i]);
                        start = None;
                    }
                }
                _ => {
                    if start.is_none() {
                        start = Some(i);
                    }
                }
            }
        }

        if let Some(s) = start {
            tokens.push(&text[s..]);
        }

        tokens
    }

    /// Count UTF-8 characters efficiently
    pub fn count_chars(text: &str) -> usize {
        text.chars().count()
    }

    /// Truncate string to maximum bytes while preserving UTF-8 validity
    pub fn truncate_utf8(text: &str, max_bytes: usize) -> &str {
        if text.len() <= max_bytes {
            return text;
        }

        let mut boundary = max_bytes;
        while boundary > 0 && !text.is_char_boundary(boundary) {
            boundary -= 1;
        }

        &text[..boundary]
    }
}

/// Math utilities for tensor operations
pub mod math {
    use crate::error::{GemmaError, GemmaResult};
    use std::f32;

    /// Fast approximation of exp(x) using polynomial approximation
    #[inline]
    pub fn fast_exp(x: f32) -> f32 {
        if x < -87.0 {
            return 0.0;
        }
        if x > 88.0 {
            return f32::INFINITY;
        }

        // Use hardware exp if available, otherwise approximation
        #[cfg(target_feature = "fma")]
        {
            x.exp()
        }
        #[cfg(not(target_feature = "fma"))]
        {
            // Polynomial approximation for better performance
            let x = x.max(-87.0).min(88.0);
            let x2 = x * x;
            let x3 = x2 * x;
            let x4 = x2 * x2;

            1.0 + x + x2 * 0.5 + x3 * 0.16666667 + x4 * 0.041666667
        }
    }

    /// Fast log base 2
    #[inline]
    pub fn fast_log2(x: f32) -> f32 {
        if x <= 0.0 {
            return f32::NEG_INFINITY;
        }
        x.log2()
    }

    /// Compute softmax with numerical stability
    pub fn stable_softmax(input: &[f32], output: &mut [f32]) -> GemmaResult<()> {
        if input.len() != output.len() {
            return Err(GemmaError::dimension_mismatch(input.len(), output.len()));
        }

        if input.is_empty() {
            return Ok(());
        }

        // Find maximum for numerical stability
        let max_val = input.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));

        // Compute exp(x - max) and sum
        let mut sum = 0.0f32;
        for (i, &x) in input.iter().enumerate() {
            let exp_val = fast_exp(x - max_val);
            output[i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for val in output.iter_mut() {
                *val *= inv_sum;
            }
        }

        Ok(())
    }

    /// Compute layer normalization
    pub fn layer_norm(
        input: &[f32],
        output: &mut [f32],
        gamma: &[f32],
        beta: &[f32],
        epsilon: f32,
    ) -> GemmaResult<()> {
        let len = input.len();
        if output.len() != len || gamma.len() != len || beta.len() != len {
            return Err(GemmaError::dimension_mismatch(len, output.len()));
        }

        // Compute mean
        let mean = input.iter().sum::<f32>() / len as f32;

        // Compute variance
        let variance = input.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / len as f32;

        let inv_std = 1.0 / (variance + epsilon).sqrt();

        // Apply normalization
        for (((&x, out), &g), &b) in input
            .iter()
            .zip(output.iter_mut())
            .zip(gamma.iter())
            .zip(beta.iter())
        {
            *out = (x - mean) * inv_std * g + b;
        }

        Ok(())
    }
}

/// Performance monitoring utilities
pub mod perf {
    use std::collections::VecDeque;
    use std::time::{Duration, Instant};

    /// Simple performance counter
    pub struct PerfCounter {
        name: String,
        start_time: Option<Instant>,
        measurements: VecDeque<Duration>,
        max_samples: usize,
    }

    impl PerfCounter {
        /// Create a new performance counter
        pub fn new(name: impl Into<String>, max_samples: usize) -> Self {
            Self {
                name: name.into(),
                start_time: None,
                measurements: VecDeque::with_capacity(max_samples),
                max_samples,
            }
        }

        /// Start timing
        pub fn start(&mut self) {
            self.start_time = Some(Instant::now());
        }

        /// Stop timing and record measurement
        pub fn stop(&mut self) -> Option<Duration> {
            if let Some(start) = self.start_time.take() {
                let duration = start.elapsed();

                if self.measurements.len() >= self.max_samples {
                    self.measurements.pop_front();
                }
                self.measurements.push_back(duration);

                Some(duration)
            } else {
                None
            }
        }

        /// Get average duration
        pub fn average(&self) -> Option<Duration> {
            if self.measurements.is_empty() {
                return None;
            }

            let total: Duration = self.measurements.iter().sum();
            Some(total / self.measurements.len() as u32)
        }

        /// Get minimum duration
        pub fn min(&self) -> Option<Duration> {
            self.measurements.iter().min().copied()
        }

        /// Get maximum duration
        pub fn max(&self) -> Option<Duration> {
            self.measurements.iter().max().copied()
        }

        /// Get sample count
        pub fn count(&self) -> usize {
            self.measurements.len()
        }

        /// Get the name of the counter
        pub fn name(&self) -> &str {
            &self.name
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_vec() {
        let mut vec = memory::AlignedVec::<f32>::with_capacity_aligned(16, 32).unwrap();
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 16);
        assert!(vec.alignment() >= 32);

        vec.push(1.0).unwrap();
        vec.push(2.0).unwrap();
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.as_slice(), &[1.0, 2.0]);
    }

    #[test]
    fn test_string_utils() {
        assert_eq!(string::normalize_text("  Hello\tWorld\n  "), "hello world");

        let tokens = string::fast_split_whitespace("hello world test");
        assert_eq!(tokens, vec!["hello", "world", "test"]);

        assert_eq!(string::count_chars("hello 世界"), 8);
        assert_eq!(string::truncate_utf8("hello world", 5), "hello");
    }

    #[test]
    fn test_math_utils() {
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        math::stable_softmax(&input, &mut output).unwrap();

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_perf_counter() {
        let mut counter = perf::PerfCounter::new("test", 10);
        counter.start();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let duration = counter.stop().unwrap();

        assert!(duration.as_millis() >= 1);
        assert_eq!(counter.count(), 1);
    }
}
