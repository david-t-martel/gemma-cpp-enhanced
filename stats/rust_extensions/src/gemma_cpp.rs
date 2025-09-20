use crate::error::{GemmaError, GemmaResult};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, error, warn};

// FFI declarations for gemma.cpp
extern "C" {
    fn gemma_create_model(model_path: *const c_char) -> *mut c_void;
    fn gemma_destroy_model(model: *mut c_void);
    fn gemma_generate(
        model: *mut c_void,
        prompt: *const c_char,
        max_tokens: c_int,
        temperature: c_float,
        output: *mut c_char,
        output_size: c_int,
    ) -> c_int;
    fn gemma_embed(
        model: *mut c_void,
        text: *const c_char,
        embeddings: *mut c_float,
        dim: c_int,
    ) -> c_int;
}

// Constants for safety limits
const MAX_PROMPT_LENGTH: usize = 32_768; // 32KB max prompt
const MAX_OUTPUT_LENGTH: usize = 131_072; // 128KB max output
const MAX_MODEL_PATH_LENGTH: usize = 4_096; // 4KB max path
const MAX_EMBEDDING_DIM: usize = 8_192; // Max embedding dimension
const FFI_TIMEOUT_MS: u64 = 30_000; // 30 second timeout
const MIN_BUFFER_SIZE: usize = 1_024; // Minimum buffer size

// Valid model file extensions
const VALID_MODEL_EXTENSIONS: &[&str] = &[".bin", ".gguf", ".safetensors", ".pt", ".pth", ".ckpt"];

// Memory pool for frequently used buffers
type BufferPool = Arc<Mutex<HashMap<usize, Vec<Vec<u8>>>>>;

lazy_static::lazy_static! {
    static ref BUFFER_POOL: BufferPool = Arc::new(Mutex::new(HashMap::new()));
}

/// Safe wrapper around raw model pointer with RAII cleanup
struct SafeModelHandle {
    ptr: *mut c_void,
    path: PathBuf,
    created_at: Instant,
}

impl SafeModelHandle {
    fn new(path: PathBuf) -> GemmaResult<Self> {
        let c_path = validate_and_convert_path(&path)?;

        let ptr = unsafe { gemma_create_model(c_path.as_ptr()) };

        if ptr.is_null() {
            return Err(model_load_error(format!(
                "Failed to load model from {}",
                path.display()
            )));
        }

        debug!("Successfully loaded model from {}", path.display());

        Ok(Self {
            ptr,
            path,
            created_at: Instant::now(),
        })
    }

    fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    fn is_valid(&self) -> bool {
        !self.ptr.is_null()
    }
}

impl Drop for SafeModelHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            debug!("Destroying model loaded from {}", self.path.display());
            unsafe { gemma_destroy_model(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

// Send is safe because we control access through the safe wrapper
unsafe impl Send for SafeModelHandle {}

// Custom error helper functions for gemma.cpp specific errors
fn invalid_path_error(msg: String) -> GemmaError {
    GemmaError::InvalidArgument(format!("Invalid model path: {}", msg))
}

fn path_traversal_error(msg: String) -> GemmaError {
    GemmaError::InvalidArgument(format!("Path traversal attempt blocked: {}", msg))
}

fn model_load_error(msg: String) -> GemmaError {
    GemmaError::General {
        message: format!("Model loading failed: {}", msg),
    }
}

fn validation_error(msg: String) -> GemmaError {
    GemmaError::InvalidArgument(format!("Input validation failed: {}", msg))
}

fn timeout_error(timeout_ms: u64) -> GemmaError {
    GemmaError::General {
        message: format!("FFI operation timed out after {}ms", timeout_ms),
    }
}

fn generation_error(msg: String) -> GemmaError {
    GemmaError::General {
        message: format!("Generation failed: {}", msg),
    }
}

fn embedding_error(msg: String) -> GemmaError {
    GemmaError::General {
        message: format!("Embedding failed: {}", msg),
    }
}

/// Validate model path for security
fn validate_model_path(path: &str) -> GemmaResult<PathBuf> {
    // Check length
    if path.len() > MAX_MODEL_PATH_LENGTH {
        return Err(validation_error(format!(
            "Path too long: {} chars (max: {})",
            path.len(),
            MAX_MODEL_PATH_LENGTH
        )));
    }

    // Check for null bytes
    if path.contains('\0') {
        return Err(validation_error("Path contains null byte".to_string()));
    }

    let path_buf = PathBuf::from(path);

    // Canonicalize path to resolve .. and . components
    let canonical_path = path_buf
        .canonicalize()
        .map_err(|e| invalid_path_error(format!("Cannot resolve path {}: {}", path, e)))?;

    // Check if file exists
    if !canonical_path.exists() {
        return Err(invalid_path_error(format!(
            "Model file does not exist: {}",
            canonical_path.display()
        )));
    }

    // Check if it's a file (not directory)
    if !canonical_path.is_file() {
        return Err(invalid_path_error(format!(
            "Path is not a file: {}",
            canonical_path.display()
        )));
    }

    // Check file extension
    let extension = canonical_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|s| format!(".{}", s))
        .unwrap_or_default();

    if !VALID_MODEL_EXTENSIONS.contains(&extension.as_str()) {
        return Err(validation_error(format!(
            "Invalid model file extension: {} (valid: {:?})",
            extension, VALID_MODEL_EXTENSIONS
        )));
    }

    // Prevent path traversal by checking if canonical path contains parent directory references
    let path_str = canonical_path.to_string_lossy();
    if path_str.contains("..") {
        return Err(path_traversal_error(format!(
            "Path traversal detected in: {}",
            path_str
        )));
    }

    Ok(canonical_path)
}

/// Convert validated path to CString
fn validate_and_convert_path(path: &Path) -> GemmaResult<CString> {
    let path_str = path
        .to_str()
        .ok_or_else(|| validation_error("Path contains invalid UTF-8".to_string()))?;

    CString::new(path_str).map_err(|e| validation_error(format!("Invalid path string: {}", e)))
}

/// Validate and convert input text
fn validate_text_input(text: &str, max_len: usize, field_name: &str) -> GemmaResult<CString> {
    if text.len() > max_len {
        return Err(validation_error(format!(
            "{} too long: {} chars (max: {})",
            field_name,
            text.len(),
            max_len
        )));
    }

    if text.is_empty() {
        return Err(validation_error(format!("{} cannot be empty", field_name)));
    }

    CString::new(text)
        .map_err(|e| validation_error(format!("Invalid {} string: {}", field_name, e)))
}

/// Get buffer from pool or create new one
fn get_pooled_buffer(size: usize) -> Vec<u8> {
    let mut pool = BUFFER_POOL.lock().unwrap();

    if let Some(buffers) = pool.get_mut(&size) {
        if let Some(mut buffer) = buffers.pop() {
            buffer.clear();
            buffer.resize(size, 0);
            return buffer;
        }
    }

    vec![0u8; size]
}

/// Return buffer to pool for reuse
fn return_pooled_buffer(buffer: Vec<u8>) {
    let size = buffer.len();
    let mut pool = BUFFER_POOL.lock().unwrap();

    let buffers = pool.entry(size).or_insert_with(Vec::new);

    // Limit pool size to prevent unbounded growth
    if buffers.len() < 10 {
        buffers.push(buffer);
    }
}

/// Calculate optimal buffer size based on expected output
fn calculate_buffer_size(max_tokens: i32, base_size: usize) -> usize {
    let estimated_size = (max_tokens as usize * 4).max(MIN_BUFFER_SIZE); // ~4 chars per token
    let capped_size = estimated_size.min(MAX_OUTPUT_LENGTH);
    capped_size.max(base_size)
}

/// Safe string extraction from C buffer with length validation
fn extract_safe_string(buffer: &[u8]) -> GemmaResult<String> {
    // Find the first null byte
    let end_pos = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());

    if end_pos == 0 {
        return Ok(String::new());
    }

    // Ensure we don't read beyond the buffer
    let safe_slice = &buffer[..end_pos];

    // Convert to string with proper error handling
    String::from_utf8(safe_slice.to_vec())
        .map_err(|e| validation_error(format!("Invalid UTF-8 in output: {}", e)))
}

#[pyclass]
pub struct GemmaCpp {
    model: SafeModelHandle,
    config: GemmaConfig,
}

/// Configuration for Gemma model
#[pyclass]
#[derive(Debug, Clone)]
pub struct GemmaConfig {
    #[pyo3(get, set)]
    pub num_threads: u32,
    #[pyo3(get, set)]
    pub context_length: u32,
    #[pyo3(get, set)]
    pub batch_size: u32,
    #[pyo3(get, set)]
    pub use_mmap: bool,
    #[pyo3(get, set)]
    pub timeout_ms: u64,
}

#[pymethods]
impl GemmaConfig {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    fn __repr__(&self) -> String {
        format!(
            "GemmaConfig(num_threads={}, context_length={}, batch_size={}, use_mmap={}, timeout_ms={})",
            self.num_threads, self.context_length, self.batch_size, self.use_mmap, self.timeout_ms
        )
    }
}

impl Default for GemmaConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get() as u32,
            context_length: 4096,
            batch_size: 1,
            use_mmap: true,
            timeout_ms: FFI_TIMEOUT_MS,
        }
    }
}

#[pymethods]
impl GemmaCpp {
    #[new]
    #[pyo3(signature = (model_path, config = None))]
    fn new(model_path: &str, config: Option<GemmaConfig>) -> PyResult<Self> {
        let validated_path = validate_model_path(model_path)?;
        let model = SafeModelHandle::new(validated_path)?;
        let config = config.unwrap_or_default();

        Ok(Self { model, config })
    }

    /// Get model information
    fn get_model_info(&self) -> PyResult<std::collections::HashMap<String, String>> {
        let mut info = std::collections::HashMap::new();
        info.insert("path".to_string(), self.model.path.display().to_string());
        info.insert(
            "loaded_at".to_string(),
            self.model.created_at.elapsed().as_secs().to_string(),
        );
        info.insert("valid".to_string(), self.model.is_valid().to_string());
        Ok(info)
    }

    /// Check if model is still valid
    fn is_valid(&self) -> bool {
        self.model.is_valid()
    }

    #[pyo3(signature = (prompt, max_tokens = 100, temperature = 0.7))]
    fn generate(&self, prompt: &str, max_tokens: i32, temperature: f32) -> PyResult<String> {
        // Validate inputs
        if max_tokens <= 0 || max_tokens > 4096 {
            return Err(validation_error(format!(
                "max_tokens must be between 1 and 4096, got {}",
                max_tokens
            ))
            .into());
        }

        if temperature < 0.0 || temperature > 2.0 {
            return Err(validation_error(format!(
                "temperature must be between 0.0 and 2.0, got {}",
                temperature
            ))
            .into());
        }

        if !self.model.is_valid() {
            return Err(model_load_error("Model handle is invalid".to_string()).into());
        }

        let c_prompt = validate_text_input(prompt, MAX_PROMPT_LENGTH, "prompt")?;

        // Calculate optimal buffer size
        let buffer_size = calculate_buffer_size(max_tokens, MIN_BUFFER_SIZE);
        let mut output = get_pooled_buffer(buffer_size);

        let start_time = Instant::now();
        let timeout = Duration::from_millis(self.config.timeout_ms);

        // Perform FFI call with timeout check
        let result = unsafe {
            gemma_generate(
                self.model.as_ptr(),
                c_prompt.as_ptr(),
                max_tokens,
                temperature,
                output.as_mut_ptr() as *mut c_char,
                output.len() as c_int,
            )
        };

        // Check for timeout
        if start_time.elapsed() > timeout {
            return_pooled_buffer(output);
            return Err(timeout_error(self.config.timeout_ms).into());
        }

        if result < 0 {
            return_pooled_buffer(output);
            return Err(
                generation_error(format!("FFI call returned error code: {}", result)).into(),
            );
        }

        // Safely extract string
        let generated_text = extract_safe_string(&output)?;
        return_pooled_buffer(output);

        debug!(
            "Generated {} chars in {:?}",
            generated_text.len(),
            start_time.elapsed()
        );
        Ok(generated_text)
    }

    #[pyo3(signature = (text, embedding_dim = 768))]
    fn embed(&self, text: &str, embedding_dim: usize) -> PyResult<Vec<f32>> {
        // Validate inputs
        if embedding_dim == 0 || embedding_dim > MAX_EMBEDDING_DIM {
            return Err(validation_error(format!(
                "embedding_dim must be between 1 and {}, got {}",
                MAX_EMBEDDING_DIM, embedding_dim
            ))
            .into());
        }

        if !self.model.is_valid() {
            return Err(model_load_error("Model handle is invalid".to_string()).into());
        }

        let c_text = validate_text_input(text, MAX_PROMPT_LENGTH, "text")?;

        let mut embeddings = vec![0.0f32; embedding_dim];

        let start_time = Instant::now();
        let timeout = Duration::from_millis(self.config.timeout_ms);

        let result = unsafe {
            gemma_embed(
                self.model.as_ptr(),
                c_text.as_ptr(),
                embeddings.as_mut_ptr(),
                embeddings.len() as c_int,
            )
        };

        // Check for timeout
        if start_time.elapsed() > timeout {
            return Err(timeout_error(self.config.timeout_ms).into());
        }

        if result < 0 {
            return Err(
                embedding_error(format!("FFI call returned error code: {}", result)).into(),
            );
        }

        // Validate embedding output (check for NaN/Inf)
        let mut invalid_indices = Vec::new();
        for (i, &value) in embeddings.iter().enumerate() {
            if !value.is_finite() {
                warn!("Invalid embedding value at index {}: {}", i, value);
                invalid_indices.push(i);
            }
        }

        // Replace invalid values with safe ones
        for i in invalid_indices {
            embeddings[i] = 0.0;
        }

        debug!(
            "Generated embedding of dimension {} in {:?}",
            embedding_dim,
            start_time.elapsed()
        );
        Ok(embeddings)
    }

    /// Batch generate embeddings for multiple texts
    fn embed_batch(&self, texts: Vec<&str>) -> PyResult<Vec<Vec<f32>>> {
        if texts.len() > 100 {
            return Err(validation_error(format!(
                "Too many texts for batch processing: {} (max: 100)",
                texts.len()
            ))
            .into());
        }

        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            results.push(self.embed(text, 768)?);
        }

        Ok(results)
    }

    /// Update configuration
    fn update_config(&mut self, config: GemmaConfig) {
        self.config = config;
    }

    /// Get current configuration
    fn get_config(&self) -> GemmaConfig {
        self.config.clone()
    }
}

// Drop is automatically implemented by SafeModelHandle

/// Register the gemma_cpp module
pub fn register_gemma_cpp(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let gemma_module = PyModule::new(py, "gemma_cpp")?;

    // Add classes
    gemma_module.add_class::<GemmaCpp>()?;
    gemma_module.add_class::<GemmaConfig>()?;

    // Add constants
    gemma_module.add("MAX_PROMPT_LENGTH", MAX_PROMPT_LENGTH)?;
    gemma_module.add("MAX_OUTPUT_LENGTH", MAX_OUTPUT_LENGTH)?;
    gemma_module.add("MAX_EMBEDDING_DIM", MAX_EMBEDDING_DIM)?;
    gemma_module.add("FFI_TIMEOUT_MS", FFI_TIMEOUT_MS)?;

    // Add utility functions
    gemma_module.add_function(wrap_pyfunction!(validate_model_file, gemma_module)?)?;
    gemma_module.add_function(wrap_pyfunction!(clear_buffer_pool, gemma_module)?)?;

    parent_module.add_submodule(gemma_module)?;
    Ok(())
}

/// Python function to validate model file without loading it
#[pyfunction]
fn validate_model_file(path: &str) -> PyResult<bool> {
    match validate_model_path(path) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Clear the buffer pool to free memory
#[pyfunction]
fn clear_buffer_pool() -> PyResult<usize> {
    let mut pool = BUFFER_POOL.lock().unwrap();
    let cleared_count = pool.values().map(|v| v.len()).sum::<usize>();
    pool.clear();
    Ok(cleared_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_path_validation() {
        // Test valid path (assuming this test file exists)
        let temp_dir = TempDir::new().unwrap();
        let model_path = temp_dir.path().join("test_model.bin");
        File::create(&model_path)
            .unwrap()
            .write_all(b"test")
            .unwrap();

        assert!(validate_model_path(model_path.to_str().unwrap()).is_ok());

        // Test invalid extension
        let invalid_path = temp_dir.path().join("test_model.txt");
        File::create(&invalid_path).unwrap();
        assert!(validate_model_path(invalid_path.to_str().unwrap()).is_err());

        // Test non-existent file
        assert!(validate_model_path("/nonexistent/path.bin").is_err());

        // Test path traversal
        assert!(validate_model_path("../../../etc/passwd").is_err());
    }

    #[test]
    fn test_text_validation() {
        assert!(validate_text_input("hello", 100, "test").is_ok());
        assert!(validate_text_input("", 100, "test").is_err());
        assert!(validate_text_input(&"x".repeat(1000), 100, "test").is_err());
        assert!(validate_text_input("hello\0world", 100, "test").is_err());
    }

    #[test]
    fn test_buffer_pool() {
        let buffer1 = get_pooled_buffer(1024);
        assert_eq!(buffer1.len(), 1024);

        return_pooled_buffer(buffer1);

        let buffer2 = get_pooled_buffer(1024);
        assert_eq!(buffer2.len(), 1024);

        clear_buffer_pool().unwrap();
    }

    #[test]
    fn test_buffer_size_calculation() {
        assert_eq!(calculate_buffer_size(100, 1024), 1024);
        assert_eq!(calculate_buffer_size(1000, 1024), 4000);
        assert!(calculate_buffer_size(100000, 1024) <= MAX_OUTPUT_LENGTH);
    }

    #[test]
    fn test_safe_string_extraction() {
        let buffer = b"hello\0world";
        let result = extract_safe_string(buffer).unwrap();
        assert_eq!(result, "hello");

        let empty_buffer = b"\0";
        let result = extract_safe_string(empty_buffer).unwrap();
        assert_eq!(result, "");

        let no_null_buffer = b"hello";
        let result = extract_safe_string(no_null_buffer).unwrap();
        assert_eq!(result, "hello");
    }
}
