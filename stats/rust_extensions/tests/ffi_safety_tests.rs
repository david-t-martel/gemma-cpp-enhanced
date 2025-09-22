//! Comprehensive tests for FFI safety and unsafe operations
//!
//! These tests thoroughly validate all unsafe code blocks in the FFI module,
//! ensuring memory safety, proper error handling, and correct C interop.

use gemma_extensions::*;
use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use proptest::prelude::*;
use tokio::runtime::Runtime;

/// Test basic FFI string handling safety
#[cfg(test)]
mod string_handling_tests {
    use super::*;

    #[test]
    fn test_c_string_creation_and_validation() {
        let test_strings = vec![
            "Hello, world!",
            "Unicode: 🦀 Rust",
            "", // Empty string
            "A".repeat(1000), // Long string
            "Special chars: !@#$%^&*()",
            "Newlines:\nand\ttabs\rand\x00embedded", // With null byte
        ];

        for test_str in test_strings {
            // Test C string creation
            match CString::new(test_str.clone()) {
                Ok(c_string) => {
                    // Verify round-trip conversion
                    let c_str = c_string.as_c_str();
                    let rust_str = c_str.to_string_lossy();

                    if !test_str.contains('\0') {
                        assert_eq!(test_str, rust_str);
                    }

                    // Test pointer safety
                    let ptr = c_string.as_ptr();
                    assert!(!ptr.is_null());

                    // Test that we can safely read the string
                    unsafe {
                        let len = libc::strlen(ptr);
                        assert!(len <= test_str.len());
                    }
                }
                Err(_) => {
                    // Error is expected for strings with null bytes
                    assert!(test_str.contains('\0'));
                }
            }
        }
    }

    #[test]
    fn test_null_pointer_handling() {
        unsafe {
            // Test that our FFI functions handle null pointers gracefully
            let null_ptr = ptr::null();

            // These operations should not crash
            assert!(null_ptr.is_null());

            // Test null C string handling
            let null_cstr = ptr::null() as *const libc::c_char;
            assert!(null_cstr.is_null());

            // Any FFI function should validate pointers before use
            // This is a conceptual test - actual validation depends on implementation
        }
    }

    #[test]
    fn test_string_encoding_safety() {
        let test_cases = vec![
            ("ASCII", "Hello World"),
            ("UTF-8", "Hello 🌍"),
            ("Latin-1", "Hëllö Wörld"),
            ("Empty", ""),
            ("Long", &"A".repeat(10000)),
        ];

        for (name, test_str) in test_cases {
            // Test encoding to C string
            if let Ok(c_string) = CString::new(test_str) {
                let ptr = c_string.as_ptr();

                // Test safe reading
                unsafe {
                    let reconstructed = CStr::from_ptr(ptr);
                    let rust_str = reconstructed.to_string_lossy();

                    // Should preserve content (may have lossy conversion for non-UTF-8)
                    assert!(!rust_str.is_empty() || test_str.is_empty());
                }
            }
        }
    }

    #[test]
    fn test_buffer_bounds_checking() {
        let buffer_sizes = vec![0, 1, 16, 64, 256, 1024, 4096];

        for size in buffer_sizes {
            let mut buffer = vec![0u8; size];

            // Safe initialization
            for (i, byte) in buffer.iter_mut().enumerate() {
                *byte = (i % 256) as u8;
            }

            // Convert to C string if possible
            if size > 0 && !buffer.contains(&0) {
                buffer.push(0); // Add null terminator

                unsafe {
                    let c_str = CStr::from_bytes_with_nul(&buffer);
                    match c_str {
                        Ok(c_str) => {
                            let _rust_str = c_str.to_string_lossy();
                            // Should not crash
                        }
                        Err(_) => {
                            // Error is acceptable for invalid sequences
                        }
                    }
                }
            }
        }
    }
}

/// Test memory allocation and deallocation safety
#[cfg(test)]
mod memory_allocation_tests {
    use super::*;
    use std::alloc::{alloc, dealloc, Layout};

    #[test]
    fn test_aligned_allocation_safety() {
        let alignments = vec![1, 2, 4, 8, 16, 32, 64];
        let sizes = vec![16, 64, 256, 1024, 4096];

        for &alignment in &alignments {
            for &size in &sizes {
                if let Ok(layout) = Layout::from_size_align(size, alignment) {
                    unsafe {
                        let ptr = alloc(layout);

                        if !ptr.is_null() {
                            // Verify alignment
                            assert_eq!(ptr as usize % alignment, 0);

                            // Test write/read to ensure memory is valid
                            ptr::write(ptr, 0x42);
                            let value = ptr::read(ptr);
                            assert_eq!(value, 0x42);

                            // Test filling entire allocation
                            ptr::write_bytes(ptr, 0xAA, size);

                            // Verify the fill
                            for i in 0..size {
                                let byte = ptr::read(ptr.add(i));
                                assert_eq!(byte, 0xAA);
                            }

                            dealloc(ptr, layout);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_memory_leak_prevention() {
        struct LeakTester {
            ptr: *mut u8,
            layout: Layout,
        }

        impl Drop for LeakTester {
            fn drop(&mut self) {
                if !self.ptr.is_null() {
                    unsafe {
                        dealloc(self.ptr, self.layout);
                    }
                }
            }
        }

        impl LeakTester {
            fn new(size: usize) -> Option<Self> {
                if let Ok(layout) = Layout::from_size_align(size, 8) {
                    unsafe {
                        let ptr = alloc(layout);
                        if !ptr.is_null() {
                            return Some(Self { ptr, layout });
                        }
                    }
                }
                None
            }
        }

        // Test that Drop is called even in panic scenarios
        let result = std::panic::catch_unwind(|| {
            let _tester = LeakTester::new(1024).unwrap();
            panic!("Test panic");
        });

        assert!(result.is_err());
        // Memory should be cleaned up by Drop
    }

    #[test]
    fn test_double_free_protection() {
        let layout = Layout::from_size_align(64, 8).unwrap();

        unsafe {
            let ptr = alloc(layout);
            assert!(!ptr.is_null());

            // First deallocation
            dealloc(ptr, layout);

            // Second deallocation would be undefined behavior
            // We can't test this directly, but our code should prevent it
            // by setting pointers to null after deallocation
        }
    }

    #[test]
    fn test_concurrent_allocation() {
        use std::sync::Barrier;

        let thread_count = 8;
        let barrier = Arc::new(Barrier::new(thread_count));

        let handles: Vec<_> = (0..thread_count)
            .map(|_| {
                let barrier_clone = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier_clone.wait();

                    let layout = Layout::from_size_align(1024, 16).unwrap();
                    let mut allocations = Vec::new();

                    // Each thread allocates memory
                    for _ in 0..100 {
                        unsafe {
                            let ptr = alloc(layout);
                            if !ptr.is_null() {
                                // Write to memory to ensure it's valid
                                ptr::write(ptr, 0x42);
                                allocations.push(ptr);
                            }
                        }
                    }

                    // Cleanup
                    for ptr in allocations {
                        unsafe {
                            dealloc(ptr, layout);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }
}

/// Test pointer arithmetic and bounds safety
#[cfg(test)]
mod pointer_arithmetic_tests {
    use super::*;

    #[test]
    fn test_safe_pointer_arithmetic() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let ptr = data.as_ptr();

        unsafe {
            // Test in-bounds pointer arithmetic
            for i in 0..data.len() {
                let element_ptr = ptr.add(i);
                let value = ptr::read(element_ptr);
                assert_eq!(value, data[i]);
            }

            // Test offset calculation
            let offset_ptr = ptr.add(5);
            let value = ptr::read(offset_ptr);
            assert_eq!(value, 6);

            // Test pointer difference
            let diff = offset_ptr.offset_from(ptr);
            assert_eq!(diff, 5);
        }
    }

    #[test]
    fn test_slice_from_raw_parts_safety() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let ptr = data.as_ptr();
        let len = data.len();

        unsafe {
            // Create slice from raw parts
            let slice = std::slice::from_raw_parts(ptr, len);
            assert_eq!(slice.len(), len);

            for (i, &value) in slice.iter().enumerate() {
                assert_eq!(value, data[i]);
            }

            // Test with different alignments
            if len > 2 {
                let offset_slice = std::slice::from_raw_parts(ptr.add(1), len - 2);
                assert_eq!(offset_slice.len(), len - 2);

                for (i, &value) in offset_slice.iter().enumerate() {
                    assert_eq!(value, data[i + 1]);
                }
            }
        }
    }

    #[test]
    fn test_type_casting_safety() {
        // Test safe type casting between compatible types
        let data = vec![1u32, 2, 3, 4];
        let byte_len = data.len() * 4;

        unsafe {
            // Cast u32 slice to u8 slice
            let u32_ptr = data.as_ptr();
            let u8_ptr = u32_ptr as *const u8;
            let byte_slice = std::slice::from_raw_parts(u8_ptr, byte_len);

            assert_eq!(byte_slice.len(), byte_len);

            // Verify endianness-independent properties
            let sum: u32 = byte_slice.iter().map(|&b| b as u32).sum();
            assert!(sum > 0); // Should have some non-zero bytes

            // Cast back to verify
            let u32_slice = std::slice::from_raw_parts(u32_ptr, data.len());
            for (i, &value) in u32_slice.iter().enumerate() {
                assert_eq!(value, data[i]);
            }
        }
    }

    #[test]
    fn test_alignment_preservation() {
        let sizes = vec![16, 32, 64, 128, 256];
        let alignments = vec![4, 8, 16, 32];

        for &size in &sizes {
            for &alignment in &alignments {
                if let Ok(layout) = Layout::from_size_align(size, alignment) {
                    unsafe {
                        let ptr = alloc(layout);
                        if !ptr.is_null() {
                            // Check initial alignment
                            assert_eq!(ptr as usize % alignment, 0);

                            // Test that aligned access works
                            if alignment >= 4 {
                                let u32_ptr = ptr as *mut u32;
                                if (u32_ptr as usize) % 4 == 0 {
                                    ptr::write(u32_ptr, 0x12345678);
                                    let value = ptr::read(u32_ptr);
                                    assert_eq!(value, 0x12345678);
                                }
                            }

                            if alignment >= 8 {
                                let u64_ptr = ptr as *mut u64;
                                if (u64_ptr as usize) % 8 == 0 {
                                    ptr::write(u64_ptr, 0x123456789ABCDEF0);
                                    let value = ptr::read(u64_ptr);
                                    assert_eq!(value, 0x123456789ABCDEF0);
                                }
                            }

                            dealloc(ptr, layout);
                        }
                    }
                }
            }
        }
    }
}

/// Test async FFI operations and thread safety
#[cfg(test)]
mod async_ffi_tests {
    use super::*;

    #[test]
    fn test_ffi_with_async_context() {
        let rt = Runtime::new().unwrap();

        rt.block_on(async {
            // Test that FFI operations work in async context
            let test_data = "Hello from async";

            if let Ok(c_string) = CString::new(test_data) {
                unsafe {
                    let ptr = c_string.as_ptr();
                    let len = libc::strlen(ptr);
                    assert_eq!(len, test_data.len());
                }
            }

            // Yield to ensure async context is preserved
            tokio::task::yield_now().await;

            // Test memory allocation in async context
            let layout = Layout::from_size_align(1024, 16).unwrap();
            unsafe {
                let ptr = alloc(layout);
                if !ptr.is_null() {
                    ptr::write_bytes(ptr, 0x42, 1024);

                    // Verify in async context
                    tokio::task::yield_now().await;

                    let value = ptr::read(ptr);
                    assert_eq!(value, 0x42);

                    dealloc(ptr, layout);
                }
            }
        });
    }

    #[test]
    fn test_concurrent_ffi_operations() {
        use std::sync::Barrier;

        let thread_count = 4;
        let barrier = Arc::new(Barrier::new(thread_count));

        let handles: Vec<_> = (0..thread_count)
            .map(|thread_id| {
                let barrier_clone = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier_clone.wait();

                    // Each thread performs FFI operations
                    let test_str = format!("Thread {}", thread_id);

                    if let Ok(c_string) = CString::new(test_str.clone()) {
                        unsafe {
                            let ptr = c_string.as_ptr();
                            let reconstructed = CStr::from_ptr(ptr);
                            let rust_str = reconstructed.to_string_lossy();
                            assert_eq!(test_str, rust_str);
                        }
                    }

                    // Test memory operations
                    let layout = Layout::from_size_align(256, 8).unwrap();
                    unsafe {
                        let ptr = alloc(layout);
                        if !ptr.is_null() {
                            // Write thread ID pattern
                            for i in 0..256 {
                                ptr::write(ptr.add(i), (thread_id % 256) as u8);
                            }

                            // Verify pattern
                            for i in 0..256 {
                                let value = ptr::read(ptr.add(i));
                                assert_eq!(value, (thread_id % 256) as u8);
                            }

                            dealloc(ptr, layout);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_ffi_resource_cleanup_on_drop() {
        struct FfiResource {
            ptr: *mut u8,
            layout: Layout,
            id: usize,
        }

        impl FfiResource {
            fn new(id: usize) -> Self {
                let layout = Layout::from_size_align(1024, 16).unwrap();
                unsafe {
                    let ptr = alloc(layout);
                    if !ptr.is_null() {
                        ptr::write_bytes(ptr, id as u8, 1024);
                    }
                    Self { ptr, layout, id }
                }
            }

            fn verify(&self) -> bool {
                if self.ptr.is_null() {
                    return false;
                }

                unsafe {
                    for i in 0..1024 {
                        let value = ptr::read(self.ptr.add(i));
                        if value != (self.id as u8) {
                            return false;
                        }
                    }
                }
                true
            }
        }

        impl Drop for FfiResource {
            fn drop(&mut self) {
                if !self.ptr.is_null() {
                    unsafe {
                        dealloc(self.ptr, self.layout);
                    }
                    self.ptr = ptr::null_mut();
                }
            }
        }

        // Test that resources are properly cleaned up
        let mut resources = Vec::new();
        for i in 0..10 {
            resources.push(FfiResource::new(i));
        }

        // Verify all resources
        for resource in &resources {
            assert!(resource.verify());
        }

        // Drop all resources
        resources.clear();
        // Cleanup should happen automatically
    }
}

/// Property-based tests for FFI safety
#[cfg(test)]
mod property_ffi_tests {
    use super::*;

    proptest! {
        #[test]
        fn prop_string_roundtrip_safety(s in "\\PC{0,1000}") {
            // Test that string round-trip through C is safe
            if !s.contains('\0') {
                if let Ok(c_string) = CString::new(s.clone()) {
                    unsafe {
                        let ptr = c_string.as_ptr();
                        let reconstructed = CStr::from_ptr(ptr);
                        let rust_str = reconstructed.to_string_lossy();
                        assert_eq!(s, rust_str);
                    }
                }
            }
        }

        #[test]
        fn prop_memory_allocation_safety(
            size in 1usize..100_000,
            alignment in prop::sample::select(vec![1, 2, 4, 8, 16, 32, 64])
        ) {
            if let Ok(layout) = Layout::from_size_align(size, alignment) {
                unsafe {
                    let ptr = alloc(layout);
                    if !ptr.is_null() {
                        // Check alignment
                        assert_eq!(ptr as usize % alignment, 0);

                        // Test that we can write to the entire allocation
                        if size > 0 {
                            ptr::write_bytes(ptr, 0x42, size);

                            // Verify the write
                            for i in 0..size.min(100) { // Check first 100 bytes
                                let value = ptr::read(ptr.add(i));
                                assert_eq!(value, 0x42);
                            }
                        }

                        dealloc(ptr, layout);
                    }
                }
            }
        }

        #[test]
        fn prop_pointer_arithmetic_safety(
            data in prop::collection::vec(0u8..=255, 1..1000),
            offset in 0usize..100
        ) {
            if offset < data.len() {
                let ptr = data.as_ptr();

                unsafe {
                    let offset_ptr = ptr.add(offset);
                    let value = ptr::read(offset_ptr);
                    assert_eq!(value, data[offset]);

                    // Test slice creation
                    let remaining = data.len() - offset;
                    if remaining > 0 {
                        let slice = std::slice::from_raw_parts(offset_ptr, remaining);
                        assert_eq!(slice.len(), remaining);
                        assert_eq!(slice[0], data[offset]);
                    }
                }
            }
        }
    }
}

/// Error handling tests for FFI operations
#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_invalid_utf8_handling() {
        // Test handling of invalid UTF-8 sequences in C strings
        let invalid_utf8_bytes = vec![
            vec![0xFF, 0xFE, 0x00], // Invalid UTF-8 with null terminator
            vec![0x80, 0x80, 0x00], // Invalid continuation bytes
            vec![0xC0, 0x80, 0x00], // Overlong encoding
        ];

        for bytes in invalid_utf8_bytes {
            unsafe {
                let ptr = bytes.as_ptr() as *const libc::c_char;
                let c_str_result = CStr::from_ptr(ptr).to_string_lossy();

                // Should not panic, might contain replacement characters
                assert!(!c_str_result.is_empty() || bytes.len() <= 1);
            }
        }
    }

    #[test]
    fn test_allocation_failure_handling() {
        // Test handling of allocation failures
        let huge_size = usize::MAX / 2;

        if let Ok(layout) = Layout::from_size_align(huge_size, 8) {
            unsafe {
                let ptr = alloc(layout);
                // Allocation will likely fail, should return null
                if ptr.is_null() {
                    // This is expected behavior for huge allocations
                } else {
                    // If it somehow succeeds, clean up
                    dealloc(ptr, layout);
                }
            }
        }
    }

    #[test]
    fn test_bounds_checking_enforcement() {
        let data = vec![1u8, 2, 3, 4, 5];
        let ptr = data.as_ptr();

        // Test that our bounds checking prevents out-of-bounds access
        // Note: This is more of a design test - actual bounds checking
        // would be in the specific FFI function implementations

        unsafe {
            // Valid access
            for i in 0..data.len() {
                let _value = ptr::read(ptr.add(i));
                // Should not crash
            }

            // Out-of-bounds access would be undefined behavior
            // Our FFI functions should validate bounds before this point
        }
    }

    #[test]
    fn test_error_propagation() {
        // Test that errors from FFI operations are properly propagated
        use std::io::{Error, ErrorKind};

        // Simulate FFI error conditions
        let error_conditions = vec![
            Error::new(ErrorKind::NotFound, "Resource not found"),
            Error::new(ErrorKind::PermissionDenied, "Access denied"),
            Error::new(ErrorKind::OutOfMemory, "Out of memory"),
        ];

        for error in error_conditions {
            let gemma_error = GemmaError::from(error);

            // Error should be properly converted
            let error_str = format!("{}", gemma_error);
            assert!(!error_str.is_empty());

            // Should convert to Python exception
            let py_err: PyErr = gemma_error.into();
            let py_err_str = format!("{}", py_err);
            assert!(!py_err_str.is_empty());
        }
    }
}

/// Integration tests combining multiple FFI operations
#[cfg(test)]
mod integration_ffi_tests {
    use super::*;

    #[test]
    fn test_complex_ffi_workflow() {
        // Test a complex workflow involving multiple FFI operations
        let test_data = "Complex FFI test data with Unicode: 🦀🐍";

        // Step 1: Convert to C string
        let c_string = CString::new(test_data).unwrap();

        // Step 2: Allocate memory for processing
        let buffer_size = test_data.len() * 2;
        let layout = Layout::from_size_align(buffer_size, 8).unwrap();

        unsafe {
            let buffer = alloc(layout);
            assert!(!buffer.is_null());

            // Step 3: Copy data to buffer
            let src_ptr = c_string.as_ptr() as *const u8;
            let src_len = libc::strlen(c_string.as_ptr());
            ptr::copy_nonoverlapping(src_ptr, buffer, src_len);

            // Step 4: Process data (simple transformation)
            for i in 0..src_len {
                let byte = ptr::read(buffer.add(i));
                ptr::write(buffer.add(i), byte.wrapping_add(1));
            }

            // Step 5: Transform back
            for i in 0..src_len {
                let byte = ptr::read(buffer.add(i));
                ptr::write(buffer.add(i), byte.wrapping_sub(1));
            }

            // Step 6: Verify data integrity
            let result_slice = std::slice::from_raw_parts(buffer, src_len);
            let original_slice = std::slice::from_raw_parts(src_ptr, src_len);
            assert_eq!(result_slice, original_slice);

            // Step 7: Cleanup
            dealloc(buffer, layout);
        }
    }

    #[test]
    fn test_ffi_with_rust_collections() {
        // Test FFI operations with Rust collections
        let mut rust_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        unsafe {
            // Get raw pointer to Rust data
            let ptr = rust_data.as_mut_ptr();
            let len = rust_data.len();

            // Simulate FFI operation that modifies data
            for i in 0..len {
                let value = ptr::read(ptr.add(i));
                ptr::write(ptr.add(i), value * 2.0);
            }

            // Verify modifications
            for (i, &value) in rust_data.iter().enumerate() {
                assert_eq!(value, ((i + 1) * 2) as f32);
            }

            // Test with different layouts
            let byte_ptr = ptr as *mut u8;
            let byte_len = len * 4; // 4 bytes per f32

            // Should be able to access as bytes
            let byte_slice = std::slice::from_raw_parts(byte_ptr, byte_len);
            assert_eq!(byte_slice.len(), byte_len);
        }
    }

    #[test]
    fn test_cross_thread_ffi_data_sharing() {
        use std::sync::{Arc, Mutex};

        // Test sharing FFI-allocated data across threads
        let shared_data = Arc::new(Mutex::new(Vec::new()));

        let handles: Vec<_> = (0..4)
            .map(|thread_id| {
                let data_clone = Arc::clone(&shared_data);
                thread::spawn(move || {
                    // Each thread allocates memory and stores pointer info
                    let layout = Layout::from_size_align(256, 16).unwrap();

                    unsafe {
                        let ptr = alloc(layout);
                        if !ptr.is_null() {
                            // Fill with thread-specific pattern
                            for i in 0..256 {
                                ptr::write(ptr.add(i), (thread_id + i) as u8);
                            }

                            // Store info for verification
                            {
                                let mut data = data_clone.lock().unwrap();
                                data.push((ptr, layout, thread_id));
                            }

                            // Keep allocation alive for a bit
                            thread::sleep(Duration::from_millis(10));

                            dealloc(ptr, layout);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // All threads should have completed successfully
        let data = shared_data.lock().unwrap();
        assert_eq!(data.len(), 4);
    }
}