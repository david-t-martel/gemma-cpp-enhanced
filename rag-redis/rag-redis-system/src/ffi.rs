//! FFI Interface for RAG Redis System
//!
//! This module provides a C-compatible Foreign Function Interface (FFI) for the RAG Redis System.
//! It enables integration with C and C++ applications while maintaining memory safety and
//! thread safety through careful API design.

use crate::error::{Error, ErrorCode, Result};
use crate::{Config, RagSystem, SearchResult};
use dashmap::DashMap;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_uint, c_ulong};
use std::ptr;
use std::slice;
use std::sync::Arc;
use tokio::runtime::Runtime;

// Global state management
static RUNTIME: std::sync::OnceLock<Arc<Runtime>> = std::sync::OnceLock::new();
static SYSTEMS: std::sync::OnceLock<Arc<DashMap<u64, Arc<Mutex<RagSystem>>>>> =
    std::sync::OnceLock::new();
static NEXT_HANDLE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn get_runtime() -> &'static Arc<Runtime> {
    RUNTIME.get_or_init(|| Arc::new(Runtime::new().expect("Failed to create Tokio runtime")))
}

fn get_systems() -> &'static Arc<DashMap<u64, Arc<Mutex<RagSystem>>>> {
    SYSTEMS.get_or_init(|| Arc::new(DashMap::new()))
}

/// Opaque handle to a RAG system instance
pub type RagHandle = u64;

/// FFI-compatible configuration structure
#[repr(C)]
pub struct RagConfig {
    // Redis configuration
    pub redis_url: *const c_char,
    pub redis_pool_size: c_uint,
    pub redis_connection_timeout_secs: c_uint,
    pub redis_command_timeout_secs: c_uint,
    pub redis_max_retries: c_uint,
    pub redis_retry_delay_ms: c_uint,
    pub redis_enable_cluster: c_int, // 0 = false, 1 = true

    // Vector store configuration
    pub vector_dimension: c_uint,
    pub vector_max_elements: c_uint,
    pub vector_m: c_uint,
    pub vector_ef_construction: c_uint,
    pub vector_ef_search: c_uint,
    pub vector_similarity_threshold: c_float,

    // Document configuration
    pub doc_chunk_size: c_uint,
    pub doc_chunk_overlap: c_uint,
    pub doc_max_chunk_size: c_uint,
    pub doc_min_chunk_size: c_uint,
    pub doc_enable_metadata_extraction: c_int,

    // Memory configuration
    pub memory_max_mb: c_uint,
    pub memory_ttl_seconds: c_uint,
    pub memory_cleanup_interval_secs: c_uint,
    pub memory_max_items: c_uint,

    // Research configuration
    pub research_enable_web_search: c_int,
    pub research_max_results: c_uint,
    pub research_timeout_secs: c_uint,
    pub research_rate_limit_per_sec: c_uint,
}

/// FFI-compatible search result structure
#[repr(C)]
pub struct RagSearchResult {
    pub id: *mut c_char,
    pub text: *mut c_char,
    pub score: c_float,
    pub metadata_json: *mut c_char,
    pub source: *mut c_char,
    pub url: *mut c_char, // nullable
}

/// FFI-compatible search results array
#[repr(C)]
pub struct RagSearchResults {
    pub results: *mut RagSearchResult,
    pub count: c_uint,
}

/// Error information structure
#[repr(C)]
pub struct RagErrorInfo {
    pub code: c_int, // ErrorCode as integer
    pub message: *mut c_char,
}

// Helper functions for string conversion
fn str_to_cstring(s: &str) -> *mut c_char {
    match CString::new(s) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

fn cstr_to_string(cstr: *const c_char) -> Result<String> {
    if cstr.is_null() {
        return Err(Error::InvalidInput("Null string pointer".to_string()));
    }

    unsafe {
        CStr::from_ptr(cstr)
            .to_str()
            .map_err(|e| Error::InvalidInput(format!("Invalid UTF-8 string: {}", e)))
            .map(|s| s.to_string())
    }
}

fn convert_config(ffi_config: &RagConfig) -> Result<Config> {
    let redis_url = cstr_to_string(ffi_config.redis_url)?;

    let redis_config = crate::redis_backend::RedisConfig {
        url: redis_url,
        pool_size: ffi_config.redis_pool_size,
        connection_timeout: std::time::Duration::from_secs(
            ffi_config.redis_connection_timeout_secs as u64,
        ),
        command_timeout: std::time::Duration::from_secs(
            ffi_config.redis_command_timeout_secs as u64,
        ),
        max_retries: ffi_config.redis_max_retries,
        retry_delay: std::time::Duration::from_millis(ffi_config.redis_retry_delay_ms as u64),
        enable_cluster: ffi_config.redis_enable_cluster != 0,
    };

    let vector_config = crate::config::VectorStoreConfig {
        dimension: ffi_config.vector_dimension as usize,
        max_elements: ffi_config.vector_max_elements as usize,
        m: ffi_config.vector_m as usize,
        ef_construction: ffi_config.vector_ef_construction as usize,
        ef_search: ffi_config.vector_ef_search as usize,
        similarity_threshold: ffi_config.vector_similarity_threshold,
    };

    let document_config = crate::config::DocumentConfig {
        chunk_size: ffi_config.doc_chunk_size as usize,
        chunk_overlap: ffi_config.doc_chunk_overlap as usize,
        max_chunk_size: ffi_config.doc_max_chunk_size as usize,
        min_chunk_size: ffi_config.doc_min_chunk_size as usize,
        enable_metadata_extraction: ffi_config.doc_enable_metadata_extraction != 0,
        supported_formats: vec!["txt".to_string(), "md".to_string(), "pdf".to_string()],
    };

    let memory_config = crate::config::MemoryConfig {
        max_memory_mb: ffi_config.memory_max_mb as usize,
        ttl_seconds: ffi_config.memory_ttl_seconds as u64,
        cleanup_interval: std::time::Duration::from_secs(
            ffi_config.memory_cleanup_interval_secs as u64,
        ),
        max_items: ffi_config.memory_max_items as usize,
    };

    let research_config = crate::config::ResearchConfig {
        enable_web_search: ffi_config.research_enable_web_search != 0,
        max_results: ffi_config.research_max_results as usize,
        timeout: std::time::Duration::from_secs(ffi_config.research_timeout_secs as u64),
        user_agent: "RAG-Redis-FFI/1.0".to_string(),
        rate_limit_per_second: ffi_config.research_rate_limit_per_sec,
    };

    Ok(Config {
        redis: redis_config,
        vector_store: vector_config,
        document: document_config,
        memory: memory_config,
        research: research_config,
        #[cfg(feature = "metrics")]
        metrics: crate::config::MetricsConfig::default(),
    })
}

/// Initialize the RAG system with the given configuration
/// Returns a handle to the RAG system instance or 0 on error
#[no_mangle]
pub extern "C" fn rag_create(config: *const RagConfig, error_info: *mut RagErrorInfo) -> RagHandle {
    if config.is_null() {
        if !error_info.is_null() {
            unsafe {
                (*error_info).code = ErrorCode::InvalidInput as c_int;
                (*error_info).message = str_to_cstring("Configuration pointer is null");
            }
        }
        return 0;
    }

    let ffi_config = unsafe { &*config };

    let rust_config = match convert_config(ffi_config) {
        Ok(config) => config,
        Err(e) => {
            if !error_info.is_null() {
                unsafe {
                    (*error_info).code = ErrorCode::from(&e) as c_int;
                    (*error_info).message = str_to_cstring(&e.to_string());
                }
            }
            return 0;
        }
    };

    let runtime = get_runtime();
    let system = match runtime.block_on(RagSystem::new(rust_config)) {
        Ok(system) => system,
        Err(e) => {
            if !error_info.is_null() {
                unsafe {
                    (*error_info).code = ErrorCode::from(&e) as c_int;
                    (*error_info).message = str_to_cstring(&e.to_string());
                }
            }
            return 0;
        }
    };

    let handle = NEXT_HANDLE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let systems = get_systems();
    systems.insert(handle, Arc::new(Mutex::new(system)));

    handle
}

/// Destroy a RAG system instance and free associated resources
#[no_mangle]
pub extern "C" fn rag_destroy(handle: RagHandle) -> c_int {
    let systems = get_systems();
    if systems.remove(&handle).is_some() {
        1 // success
    } else {
        0 // handle not found
    }
}

/// Ingest a document into the RAG system
/// Returns 1 on success, 0 on error
#[no_mangle]
pub extern "C" fn rag_ingest_document(
    handle: RagHandle,
    content: *const c_char,
    metadata_json: *const c_char,
    document_id: *mut *mut c_char,
    error_info: *mut RagErrorInfo,
) -> c_int {
    let systems = get_systems();
    let system = match systems.get(&handle) {
        Some(system) => system,
        None => {
            if !error_info.is_null() {
                unsafe {
                    (*error_info).code = ErrorCode::InvalidInput as c_int;
                    (*error_info).message = str_to_cstring("Invalid handle");
                }
            }
            return 0;
        }
    };

    let content_str = match cstr_to_string(content) {
        Ok(s) => s,
        Err(e) => {
            if !error_info.is_null() {
                unsafe {
                    (*error_info).code = ErrorCode::from(&e) as c_int;
                    (*error_info).message = str_to_cstring(&e.to_string());
                }
            }
            return 0;
        }
    };

    let metadata = if metadata_json.is_null() {
        serde_json::Value::Null
    } else {
        match cstr_to_string(metadata_json) {
            Ok(json_str) => match serde_json::from_str(&json_str) {
                Ok(value) => value,
                Err(e) => {
                    if !error_info.is_null() {
                        unsafe {
                            (*error_info).code = ErrorCode::Serialization as c_int;
                            (*error_info).message =
                                str_to_cstring(&format!("Invalid JSON metadata: {}", e));
                        }
                    }
                    return 0;
                }
            },
            Err(e) => {
                if !error_info.is_null() {
                    unsafe {
                        (*error_info).code = ErrorCode::from(&e) as c_int;
                        (*error_info).message = str_to_cstring(&e.to_string());
                    }
                }
                return 0;
            }
        }
    };

    let runtime = get_runtime();
    let system_guard = system.lock();

    match runtime.block_on(system_guard.ingest_document(&content_str, metadata)) {
        Ok(doc_id) => {
            if !document_id.is_null() {
                unsafe {
                    *document_id = str_to_cstring(&doc_id);
                }
            }
            1
        }
        Err(e) => {
            if !error_info.is_null() {
                unsafe {
                    (*error_info).code = ErrorCode::from(&e) as c_int;
                    (*error_info).message = str_to_cstring(&e.to_string());
                }
            }
            0
        }
    }
}

/// Search for documents in the RAG system
/// Returns a RagSearchResults structure containing the results
/// The caller is responsible for calling rag_free_search_results to clean up memory
#[no_mangle]
pub extern "C" fn rag_search(
    handle: RagHandle,
    query: *const c_char,
    limit: c_uint,
    error_info: *mut RagErrorInfo,
) -> *mut RagSearchResults {
    let systems = get_systems();
    let system = match systems.get(&handle) {
        Some(system) => system,
        None => {
            if !error_info.is_null() {
                unsafe {
                    (*error_info).code = ErrorCode::InvalidInput as c_int;
                    (*error_info).message = str_to_cstring("Invalid handle");
                }
            }
            return ptr::null_mut();
        }
    };

    let query_str = match cstr_to_string(query) {
        Ok(s) => s,
        Err(e) => {
            if !error_info.is_null() {
                unsafe {
                    (*error_info).code = ErrorCode::from(&e) as c_int;
                    (*error_info).message = str_to_cstring(&e.to_string());
                }
            }
            return ptr::null_mut();
        }
    };

    let runtime = get_runtime();
    let system_guard = system.lock();

    let results = match runtime.block_on(system_guard.search(&query_str, limit as usize)) {
        Ok(results) => results,
        Err(e) => {
            if !error_info.is_null() {
                unsafe {
                    (*error_info).code = ErrorCode::from(&e) as c_int;
                    (*error_info).message = str_to_cstring(&e.to_string());
                }
            }
            return ptr::null_mut();
        }
    };

    // Convert results to FFI format
    let mut ffi_results: Vec<RagSearchResult> = Vec::new();

    for result in results {
        let metadata_json = serde_json::to_string(&result.metadata).unwrap_or_default();

        ffi_results.push(RagSearchResult {
            id: str_to_cstring(&result.id),
            text: str_to_cstring(&result.text),
            score: result.score,
            metadata_json: str_to_cstring(&metadata_json),
            source: str_to_cstring(&result.source),
            url: result
                .url
                .as_ref()
                .map_or(ptr::null_mut(), |u| str_to_cstring(u)),
        });
    }

    let results_ptr = ffi_results.as_mut_ptr();
    let count = ffi_results.len() as c_uint;
    std::mem::forget(ffi_results); // Transfer ownership to caller

    let search_results = Box::new(RagSearchResults {
        results: results_ptr,
        count,
    });

    Box::into_raw(search_results)
}

/// Research function with web search capabilities
#[no_mangle]
pub extern "C" fn rag_research(
    handle: RagHandle,
    query: *const c_char,
    sources: *const *const c_char,
    sources_count: c_uint,
    error_info: *mut RagErrorInfo,
) -> *mut RagSearchResults {
    let systems = get_systems();
    let system = match systems.get(&handle) {
        Some(system) => system,
        None => {
            if !error_info.is_null() {
                unsafe {
                    (*error_info).code = ErrorCode::InvalidInput as c_int;
                    (*error_info).message = str_to_cstring("Invalid handle");
                }
            }
            return ptr::null_mut();
        }
    };

    let query_str = match cstr_to_string(query) {
        Ok(s) => s,
        Err(e) => {
            if !error_info.is_null() {
                unsafe {
                    (*error_info).code = ErrorCode::from(&e) as c_int;
                    (*error_info).message = str_to_cstring(&e.to_string());
                }
            }
            return ptr::null_mut();
        }
    };

    let mut sources_vec = Vec::new();
    if !sources.is_null() && sources_count > 0 {
        unsafe {
            let sources_slice = slice::from_raw_parts(sources, sources_count as usize);
            for &source_ptr in sources_slice {
                if let Ok(source) = cstr_to_string(source_ptr) {
                    sources_vec.push(source);
                }
            }
        }
    }

    let runtime = get_runtime();
    let system_guard = system.lock();

    let results = match runtime.block_on(system_guard.research(&query_str, sources_vec)) {
        Ok(results) => results,
        Err(e) => {
            if !error_info.is_null() {
                unsafe {
                    (*error_info).code = ErrorCode::from(&e) as c_int;
                    (*error_info).message = str_to_cstring(&e.to_string());
                }
            }
            return ptr::null_mut();
        }
    };

    // Convert results to FFI format (same as search)
    let mut ffi_results: Vec<RagSearchResult> = Vec::new();

    for result in results {
        let metadata_json = serde_json::to_string(&result.metadata).unwrap_or_default();

        ffi_results.push(RagSearchResult {
            id: str_to_cstring(&result.id),
            text: str_to_cstring(&result.text),
            score: result.score,
            metadata_json: str_to_cstring(&metadata_json),
            source: str_to_cstring(&result.source),
            url: result
                .url
                .as_ref()
                .map_or(ptr::null_mut(), |u| str_to_cstring(u)),
        });
    }

    let results_ptr = ffi_results.as_mut_ptr();
    let count = ffi_results.len() as c_uint;
    std::mem::forget(ffi_results);

    let search_results = Box::new(RagSearchResults {
        results: results_ptr,
        count,
    });

    Box::into_raw(search_results)
}

/// Free memory allocated for search results
#[no_mangle]
pub extern "C" fn rag_free_search_results(results: *mut RagSearchResults) {
    if results.is_null() {
        return;
    }

    unsafe {
        let results_box = Box::from_raw(results);

        if !results_box.results.is_null() {
            let results_slice =
                slice::from_raw_parts_mut(results_box.results, results_box.count as usize);

            for result in results_slice {
                if !result.id.is_null() {
                    let _ = CString::from_raw(result.id);
                }
                if !result.text.is_null() {
                    let _ = CString::from_raw(result.text);
                }
                if !result.metadata_json.is_null() {
                    let _ = CString::from_raw(result.metadata_json);
                }
                if !result.source.is_null() {
                    let _ = CString::from_raw(result.source);
                }
                if !result.url.is_null() {
                    let _ = CString::from_raw(result.url);
                }
            }

            let _ = Vec::from_raw_parts(
                results_box.results,
                results_box.count as usize,
                results_box.count as usize,
            );
        }
    }
}

/// Free memory allocated for error message
#[no_mangle]
pub extern "C" fn rag_free_error_message(message: *mut c_char) {
    if !message.is_null() {
        unsafe {
            let _ = CString::from_raw(message);
        }
    }
}

/// Free memory allocated for string
#[no_mangle]
pub extern "C" fn rag_free_string(string: *mut c_char) {
    if !string.is_null() {
        unsafe {
            let _ = CString::from_raw(string);
        }
    }
}

/// Get system statistics as JSON string
/// The caller is responsible for calling rag_free_string to clean up memory
#[no_mangle]
pub extern "C" fn rag_get_stats(handle: RagHandle, error_info: *mut RagErrorInfo) -> *mut c_char {
    let systems = get_systems();
    let _system = match systems.get(&handle) {
        Some(system) => system,
        None => {
            if !error_info.is_null() {
                unsafe {
                    (*error_info).code = ErrorCode::InvalidInput as c_int;
                    (*error_info).message = str_to_cstring("Invalid handle");
                }
            }
            return ptr::null_mut();
        }
    };

    // Simplified stats - in production you'd gather real metrics
    let stats = serde_json::json!({
        "status": "active",
        "uptime_seconds": 0,
        "total_documents": 0,
        "total_vectors": 0,
        "memory_usage_mb": 0.0,
        "cache_hit_rate": 0.0
    });

    match serde_json::to_string(&stats) {
        Ok(json_str) => str_to_cstring(&json_str),
        Err(e) => {
            if !error_info.is_null() {
                unsafe {
                    (*error_info).code = ErrorCode::Serialization as c_int;
                    (*error_info).message = str_to_cstring(&e.to_string());
                }
            }
            ptr::null_mut()
        }
    }
}

/// Health check function
#[no_mangle]
pub extern "C" fn rag_health_check(handle: RagHandle) -> c_int {
    let systems = get_systems();
    if systems.contains_key(&handle) {
        1 // healthy
    } else {
        0 // unhealthy
    }
}

/// Get library version
#[no_mangle]
pub extern "C" fn rag_version() -> *const c_char {
    "1.0.0\0".as_ptr() as *const c_char
}

/// Initialize global state (should be called once at startup)
#[no_mangle]
pub extern "C" fn rag_init() -> c_int {
    // Initialize the runtime and systems containers
    let _ = get_runtime();
    let _ = get_systems();
    1 // success
}

/// Cleanup global state (should be called once at shutdown)
#[no_mangle]
pub extern "C" fn rag_cleanup() -> c_int {
    let systems = get_systems();
    systems.clear();
    1 // success
}

// Thread safety tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn create_test_config() -> RagConfig {
        let redis_url = CString::new("redis://127.0.0.1:6379").unwrap();

        RagConfig {
            redis_url: redis_url.as_ptr(),
            redis_pool_size: 10,
            redis_connection_timeout_secs: 5,
            redis_command_timeout_secs: 10,
            redis_max_retries: 3,
            redis_retry_delay_ms: 100,
            redis_enable_cluster: 0,

            vector_dimension: 768,
            vector_max_elements: 100000,
            vector_m: 16,
            vector_ef_construction: 200,
            vector_ef_search: 50,
            vector_similarity_threshold: 0.7,

            doc_chunk_size: 512,
            doc_chunk_overlap: 50,
            doc_max_chunk_size: 1024,
            doc_min_chunk_size: 100,
            doc_enable_metadata_extraction: 1,

            memory_max_mb: 1024,
            memory_ttl_seconds: 3600,
            memory_cleanup_interval_secs: 300,
            memory_max_items: 10000,

            research_enable_web_search: 1,
            research_max_results: 10,
            research_timeout_secs: 30,
            research_rate_limit_per_sec: 5,
        }
    }

    #[test]
    fn test_create_destroy() {
        rag_init();

        let config = create_test_config();
        let mut error_info = RagErrorInfo {
            code: 0,
            message: ptr::null_mut(),
        };

        let handle = rag_create(&config, &mut error_info);
        assert_ne!(handle, 0);
        assert_eq!(error_info.code, 0);

        let result = rag_destroy(handle);
        assert_eq!(result, 1);

        rag_cleanup();
    }

    #[test]
    fn test_health_check() {
        rag_init();

        let config = create_test_config();
        let handle = rag_create(&config, ptr::null_mut());
        assert_ne!(handle, 0);

        assert_eq!(rag_health_check(handle), 1);
        assert_eq!(rag_health_check(999), 0); // invalid handle

        rag_destroy(handle);
        rag_cleanup();
    }

    #[test]
    fn test_version() {
        let version_ptr = rag_version();
        assert!(!version_ptr.is_null());

        unsafe {
            let version_cstr = CStr::from_ptr(version_ptr);
            let version_str = version_cstr.to_str().unwrap();
            assert_eq!(version_str, "1.0.0");
        }
    }
}
