//! FFI (Foreign Function Interface) definitions for C++ integration
//!
//! This module provides C-compatible interfaces for all major RAG system components:
//! - Redis memory management operations
//! - Vector storage and similarity search
//! - Document processing pipeline
//! - Research client functionality
//! - Error handling and result types
//! - Memory management with proper allocation/deallocation

use crate::{
    document_pipeline::{ChunkingConfig, DocumentFormat, DocumentPipeline, EmbeddingConfig},
    error::{GemmaError, GemmaResult},
    redis_manager::{DocumentMetadata, RedisConfig, RedisManager},
    research_client::{ResearchClient, ResearchConfig, ResearchQuery},
    vector_store::{DistanceMetric, VectorStore, VectorStoreConfig, SearchResult},
};
use libc::{c_char, c_float, c_int, c_uint, c_void, size_t};
use std::{
    collections::HashMap,
    ffi::{CStr, CString as StdCString},
    ptr,
    sync::Arc,
};
use tokio::runtime::Runtime;
use tracing::error;

/// Result codes for C++ integration
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CResultCode {
    Success = 0,
    InvalidArgument = -1,
    NullPointer = -2,
    OutOfMemory = -3,
    IoError = -4,
    NetworkError = -5,
    SerializationError = -6,
    RedisError = -7,
    VectorStoreError = -8,
    DocumentParsingError = -9,
    NotImplemented = -10,
    UnknownError = -99,
}

impl From<GemmaError> for CResultCode {
    fn from(error: GemmaError) -> Self {
        match error {
            GemmaError::InvalidArgument(_) => CResultCode::InvalidArgument,
            GemmaError::Io(_) => CResultCode::IoError,
            GemmaError::Network(_) => CResultCode::NetworkError,
            GemmaError::Serialization(_) => CResultCode::SerializationError,
            GemmaError::Redis(_) | GemmaError::RedisConnection(_) => CResultCode::RedisError,
            GemmaError::VectorStore(_) | GemmaError::InvalidVectorDimension { .. } => CResultCode::VectorStoreError,
            GemmaError::DocumentParsing(_) => CResultCode::DocumentParsingError,
            GemmaError::NotImplemented(_) => CResultCode::NotImplemented,
            _ => CResultCode::UnknownError,
        }
    }
}

/// C-compatible string structure
#[repr(C)]
pub struct CString {
    pub data: *mut c_char,
    pub length: size_t,
}

impl CString {
    fn from_rust_string(s: String) -> Self {
        let c_str = StdCString::new(s).unwrap_or_default();
        let data = c_str.into_raw();
        let length = unsafe { libc::strlen(data) };
        Self { data, length }
    }

    fn to_rust_string(&self) -> Option<String> {
        if self.data.is_null() {
            return None;
        }
        unsafe {
            let c_str = CStr::from_ptr(self.data);
            c_str.to_string_lossy().into_owned().into()
        }
    }
}

/// C-compatible vector structure
#[repr(C)]
pub struct CFloatVector {
    pub data: *mut c_float,
    pub length: size_t,
    pub capacity: size_t,
}

impl CFloatVector {
    fn from_rust_vec(vec: Vec<f32>) -> Self {
        let mut vec = vec.into_boxed_slice();
        let data = vec.as_mut_ptr();
        let length = vec.len();
        let capacity = vec.len();
        std::mem::forget(vec);
        Self {
            data,
            length,
            capacity,
        }
    }

    fn to_rust_vec(&self) -> Vec<f32> {
        if self.data.is_null() || self.length == 0 {
            return Vec::new();
        }
        unsafe {
            Vec::from_raw_parts(self.data, self.length, self.capacity)
        }
    }
}

/// C-compatible search result
#[repr(C)]
pub struct CSearchResult {
    pub document_id: CString,
    pub chunk_id: c_uint,
    pub text: CString,
    pub similarity: c_float,
    pub distance: c_float,
    pub vector: CFloatVector,
}

impl From<SearchResult> for CSearchResult {
    fn from(result: SearchResult) -> Self {
        Self {
            document_id: CString::from_rust_string(result.metadata.document_id),
            chunk_id: result.metadata.chunk_id as c_uint,
            text: CString::from_rust_string(result.metadata.text),
            similarity: result.similarity,
            distance: result.distance,
            vector: CFloatVector::from_rust_vec(result.vector.unwrap_or_default()),
        }
    }
}

/// Opaque handles for Rust objects
pub type RedisManagerHandle = *mut c_void;
pub type VectorStoreHandle = *mut c_void;
pub type DocumentPipelineHandle = *mut c_void;
pub type ResearchClientHandle = *mut c_void;
pub type RuntimeHandle = *mut c_void;

/// Helper macro to convert Rust results to C results with error handling
macro_rules! handle_result {
    ($expr:expr) => {
        match $expr {
            Ok(value) => (CResultCode::Success, value),
            Err(e) => {
                error!("FFI error: {:?}", e);
                (CResultCode::from(e), Default::default())
            }
        }
    };
}

/// Helper function to create a Tokio runtime
fn create_runtime() -> Result<Runtime, std::io::Error> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
}

/// Initialize the Rust RAG system
#[no_mangle]
pub extern "C" fn rag_init() -> RuntimeHandle {
    match create_runtime() {
        Ok(runtime) => Box::into_raw(Box::new(runtime)) as RuntimeHandle,
        Err(_) => ptr::null_mut(),
    }
}

/// Cleanup the Rust RAG system
#[no_mangle]
pub extern "C" fn rag_cleanup(runtime: RuntimeHandle) {
    if !runtime.is_null() {
        unsafe {
            let _ = Box::from_raw(runtime as *mut Runtime);
        }
    }
}

// Redis Manager FFI functions

/// Create a Redis manager
#[no_mangle]
pub extern "C" fn redis_manager_create(
    runtime: RuntimeHandle,
    url: *const c_char,
    max_connections: c_uint,
    timeout_ms: c_uint,
) -> (CResultCode, RedisManagerHandle) {
    if runtime.is_null() || url.is_null() {
        return (CResultCode::NullPointer, ptr::null_mut());
    }

    let runtime = unsafe { &*(runtime as *mut Runtime) };
    let url_str = unsafe { CStr::from_ptr(url) }.to_string_lossy().into_owned();

    let config = RedisConfig {
        url: url_str,
        max_connections,
        connection_timeout_ms: timeout_ms as u64,
        ..Default::default()
    };

    let result = runtime.block_on(async {
        RedisManager::new(config).await
    });

    match result {
        Ok(manager) => {
            let handle = Box::into_raw(Box::new(Arc::new(manager))) as RedisManagerHandle;
            (CResultCode::Success, handle)
        }
        Err(e) => (CResultCode::from(e), ptr::null_mut()),
    }
}

/// Destroy a Redis manager
#[no_mangle]
pub extern "C" fn redis_manager_destroy(handle: RedisManagerHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle as *mut Arc<RedisManager>);
        }
    }
}

/// Store a document in Redis
#[no_mangle]
pub extern "C" fn redis_store_document(
    runtime: RuntimeHandle,
    handle: RedisManagerHandle,
    doc_id: *const c_char,
    content: *const c_char,
    content_type: *const c_char,
) -> CResultCode {
    if runtime.is_null() || handle.is_null() || doc_id.is_null() || content.is_null() {
        return CResultCode::NullPointer;
    }

    let runtime = unsafe { &*(runtime as *mut Runtime) };
    let manager = unsafe { &*(handle as *mut Arc<RedisManager>) };

    let doc_id_str = unsafe { CStr::from_ptr(doc_id) }.to_string_lossy().into_owned();
    let content_str = unsafe { CStr::from_ptr(content) }.to_string_lossy().into_owned();
    let content_type_str = if content_type.is_null() {
        "text/plain".to_string()
    } else {
        unsafe { CStr::from_ptr(content_type) }.to_string_lossy().into_owned()
    };

    let metadata = DocumentMetadata::new(doc_id_str, content_type_str);

    let result = runtime.block_on(async {
        manager.store_document(&content_str, &metadata).await
    });

    match result {
        Ok(_) => CResultCode::Success,
        Err(e) => CResultCode::from(e),
    }
}

/// Retrieve a document from Redis
#[no_mangle]
pub extern "C" fn redis_get_document(
    runtime: RuntimeHandle,
    handle: RedisManagerHandle,
    doc_id: *const c_char,
    content_out: *mut CString,
) -> CResultCode {
    if runtime.is_null() || handle.is_null() || doc_id.is_null() || content_out.is_null() {
        return CResultCode::NullPointer;
    }

    let runtime = unsafe { &*(runtime as *mut Runtime) };
    let manager = unsafe { &*(handle as *mut Arc<RedisManager>) };
    let doc_id_str = unsafe { CStr::from_ptr(doc_id) }.to_string_lossy().into_owned();

    let result = runtime.block_on(async {
        manager.get_document(&doc_id_str).await
    });

    match result {
        Ok(Some((content, _metadata))) => {
            unsafe {
                *content_out = CString::from_rust_string(content);
            }
            CResultCode::Success
        }
        Ok(None) => CResultCode::InvalidArgument, // Document not found
        Err(e) => CResultCode::from(e),
    }
}

// Vector Store FFI functions

/// Create a vector store
#[no_mangle]
pub extern "C" fn vector_store_create(
    dimension: c_uint,
    metric: c_int, // 0=Cosine, 1=Euclidean, 2=DotProduct, 3=Manhattan
) -> (CResultCode, VectorStoreHandle) {
    let distance_metric = match metric {
        0 => DistanceMetric::Cosine,
        1 => DistanceMetric::Euclidean,
        2 => DistanceMetric::DotProduct,
        3 => DistanceMetric::Manhattan,
        _ => return (CResultCode::InvalidArgument, ptr::null_mut()),
    };

    let config = VectorStoreConfig {
        dimension: dimension as usize,
        metric: distance_metric,
        ..Default::default()
    };

    match VectorStore::new(config) {
        Ok(store) => {
            let handle = Box::into_raw(Box::new(Arc::new(store))) as VectorStoreHandle;
            (CResultCode::Success, handle)
        }
        Err(e) => (CResultCode::from(e), ptr::null_mut()),
    }
}

/// Destroy a vector store
#[no_mangle]
pub extern "C" fn vector_store_destroy(handle: VectorStoreHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle as *mut Arc<VectorStore>);
        }
    }
}

/// Add a vector to the store
#[no_mangle]
pub extern "C" fn vector_store_add_vector(
    runtime: RuntimeHandle,
    handle: VectorStoreHandle,
    vector: *const c_float,
    vector_size: size_t,
    doc_id: *const c_char,
    chunk_id: c_uint,
    text: *const c_char,
) -> (CResultCode, c_uint) {
    if runtime.is_null() || handle.is_null() || vector.is_null() || doc_id.is_null() || text.is_null() {
        return (CResultCode::NullPointer, 0);
    }

    let runtime = unsafe { &*(runtime as *mut Runtime) };
    let store = unsafe { &*(handle as *mut Arc<VectorStore>) };

    let vector_data = unsafe {
        std::slice::from_raw_parts(vector, vector_size).to_vec()
    };

    let doc_id_str = unsafe { CStr::from_ptr(doc_id) }.to_string_lossy().into_owned();
    let text_str = unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned();

    let metadata = crate::vector_store::VectorMetadata::new(
        doc_id_str,
        chunk_id as usize,
        text_str,
    );

    let result = runtime.block_on(async {
        store.add_vector(vector_data, metadata).await
    });

    match result {
        Ok(id) => (CResultCode::Success, id as c_uint),
        Err(e) => (CResultCode::from(e), 0),
    }
}

/// Search for similar vectors
#[no_mangle]
pub extern "C" fn vector_store_search(
    runtime: RuntimeHandle,
    handle: VectorStoreHandle,
    query_vector: *const c_float,
    vector_size: size_t,
    k: c_uint,
    include_vectors: c_int, // 0=false, 1=true
    results_out: *mut *mut CSearchResult,
    results_count: *mut size_t,
) -> CResultCode {
    if runtime.is_null() || handle.is_null() || query_vector.is_null() || results_out.is_null() || results_count.is_null() {
        return CResultCode::NullPointer;
    }

    let runtime = unsafe { &*(runtime as *mut Runtime) };
    let store = unsafe { &*(handle as *mut Arc<VectorStore>) };

    let query_data = unsafe {
        std::slice::from_raw_parts(query_vector, vector_size).to_vec()
    };

    let result = runtime.block_on(async {
        store.search(&query_data, k as usize, include_vectors != 0).await
    });

    match result {
        Ok(search_results) => {
            let c_results: Vec<CSearchResult> = search_results
                .into_iter()
                .map(CSearchResult::from)
                .collect();

            let results_ptr = c_results.into_boxed_slice();
            let count = results_ptr.len();
            let ptr = Box::into_raw(results_ptr) as *mut CSearchResult;

            unsafe {
                *results_out = ptr;
                *results_count = count;
            }

            CResultCode::Success
        }
        Err(e) => CResultCode::from(e),
    }
}

/// Free search results memory
#[no_mangle]
pub extern "C" fn vector_store_free_search_results(
    results: *mut CSearchResult,
    count: size_t,
) {
    if !results.is_null() && count > 0 {
        unsafe {
            let results_slice = std::slice::from_raw_parts_mut(results, count);

            // Free individual strings in each result
            for result in results_slice.iter_mut() {
                if !result.document_id.data.is_null() {
                    let _ = std::ffi::CString::from_raw(result.document_id.data);
                }
                if !result.text.data.is_null() {
                    let _ = std::ffi::CString::from_raw(result.text.data);
                }
                if !result.vector.data.is_null() {
                    let _ = Vec::from_raw_parts(
                        result.vector.data,
                        result.vector.length,
                        result.vector.capacity,
                    );
                }
            }

            // Free the results array
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(results, count));
        }
    }
}

// Document Pipeline FFI functions

/// Create a document pipeline
#[no_mangle]
pub extern "C" fn document_pipeline_create(
    max_chunk_size: c_uint,
    chunk_overlap: c_uint,
    embedding_dimension: c_uint,
) -> (CResultCode, DocumentPipelineHandle) {
    let chunking_config = ChunkingConfig {
        max_chunk_size: max_chunk_size as usize,
        chunk_overlap: chunk_overlap as usize,
        ..Default::default()
    };

    let embedding_config = EmbeddingConfig {
        dimension: embedding_dimension as usize,
        ..Default::default()
    };

    match DocumentPipeline::new(chunking_config, embedding_config) {
        Ok(pipeline) => {
            let handle = Box::into_raw(Box::new(Arc::new(pipeline))) as DocumentPipelineHandle;
            (CResultCode::Success, handle)
        }
        Err(e) => (CResultCode::from(e), ptr::null_mut()),
    }
}

/// Destroy a document pipeline
#[no_mangle]
pub extern "C" fn document_pipeline_destroy(handle: DocumentPipelineHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle as *mut Arc<DocumentPipeline>);
        }
    }
}

/// Process a document through the pipeline
#[no_mangle]
pub extern "C" fn document_pipeline_process(
    runtime: RuntimeHandle,
    handle: DocumentPipelineHandle,
    content: *const c_char,
    doc_id: *const c_char,
    format: c_int, // DocumentFormat enum
    vector_ids_out: *mut *mut *mut c_char,
    vector_ids_count: *mut size_t,
) -> CResultCode {
    if runtime.is_null() || handle.is_null() || content.is_null() || doc_id.is_null() {
        return CResultCode::NullPointer;
    }

    let runtime = unsafe { &*(runtime as *mut Runtime) };
    let pipeline = unsafe { &*(handle as *mut Arc<DocumentPipeline>) };

    let content_str = unsafe { CStr::from_ptr(content) }.to_string_lossy().into_owned();
    let doc_id_str = unsafe { CStr::from_ptr(doc_id) }.to_string_lossy().into_owned();

    let doc_format = match format {
        0 => DocumentFormat::PlainText,
        1 => DocumentFormat::Markdown,
        2 => DocumentFormat::Html,
        3 => DocumentFormat::Pdf,
        4 => DocumentFormat::Json,
        _ => return CResultCode::InvalidArgument,
    };

    let metadata = DocumentMetadata::new(doc_id_str, "application/octet-stream".to_string());

    let result = runtime.block_on(async {
        pipeline.process_document(&content_str, metadata, doc_format).await
    });

    match result {
        Ok(vector_ids) => {
            if !vector_ids_out.is_null() && !vector_ids_count.is_null() {
                let c_strings: Vec<*mut c_char> = vector_ids
                    .into_iter()
                    .map(|id| std::ffi::CString::new(id).unwrap().into_raw())
                    .collect();

                let count = c_strings.len();
                let ptr = Box::into_raw(c_strings.into_boxed_slice()) as *mut *mut c_char;

                unsafe {
                    *vector_ids_out = ptr;
                    *vector_ids_count = count;
                }
            }
            CResultCode::Success
        }
        Err(e) => CResultCode::from(e),
    }
}

/// Free vector IDs array
#[no_mangle]
pub extern "C" fn document_pipeline_free_vector_ids(
    vector_ids: *mut *mut c_char,
    count: size_t,
) {
    if !vector_ids.is_null() && count > 0 {
        unsafe {
            let ids_slice = std::slice::from_raw_parts_mut(vector_ids, count);

            // Free individual strings
            for id_ptr in ids_slice.iter_mut() {
                if !id_ptr.is_null() {
                    let _ = std::ffi::CString::from_raw(*id_ptr);
                }
            }

            // Free the array
            let _ = Box::from_raw(ids_slice);
        }
    }
}

// Research Client FFI functions

/// Create a research client
#[no_mangle]
pub extern "C" fn research_client_create(
    max_concurrent: c_uint,
    timeout_ms: c_uint,
    rate_limit_rps: c_float,
) -> (CResultCode, ResearchClientHandle) {
    let config = ResearchConfig {
        max_concurrent_requests: max_concurrent as usize,
        timeout_ms: timeout_ms as u64,
        rate_limit_rps: rate_limit_rps as f64,
        ..Default::default()
    };

    match ResearchClient::new(config) {
        Ok(client) => {
            let handle = Box::into_raw(Box::new(Arc::new(client))) as ResearchClientHandle;
            (CResultCode::Success, handle)
        }
        Err(e) => (CResultCode::from(e), ptr::null_mut()),
    }
}

/// Destroy a research client
#[no_mangle]
pub extern "C" fn research_client_destroy(handle: ResearchClientHandle) {
    if !handle.is_null() {
        unsafe {
            let _ = Box::from_raw(handle as *mut Arc<ResearchClient>);
        }
    }
}

/// Perform a research query
#[no_mangle]
pub extern "C" fn research_client_query(
    runtime: RuntimeHandle,
    handle: ResearchClientHandle,
    query: *const c_char,
    sources: *const *const c_char,
    sources_count: size_t,
    max_results: c_uint,
    results_json: *mut CString,
) -> CResultCode {
    if runtime.is_null() || handle.is_null() || query.is_null() || results_json.is_null() {
        return CResultCode::NullPointer;
    }

    let runtime = unsafe { &*(runtime as *mut Runtime) };
    let client = unsafe { &*(handle as *mut Arc<ResearchClient>) };

    let query_str = unsafe { CStr::from_ptr(query) }.to_string_lossy().into_owned();

    let sources_vec = if !sources.is_null() && sources_count > 0 {
        unsafe {
            std::slice::from_raw_parts(sources, sources_count)
                .iter()
                .filter_map(|&ptr| {
                    if ptr.is_null() { None } else {
                        Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
                    }
                })
                .collect()
        }
    } else {
        Vec::new()
    };

    let research_query = ResearchQuery::new(query_str)
        .with_sources(sources_vec)
        .with_max_results(max_results as usize);

    let result = runtime.block_on(async {
        client.research(research_query).await
    });

    match result {
        Ok(response) => {
            match serde_json::to_string(&response) {
                Ok(json) => {
                    unsafe {
                        *results_json = CString::from_rust_string(json);
                    }
                    CResultCode::Success
                }
                Err(_) => CResultCode::SerializationError,
            }
        }
        Err(e) => CResultCode::from(e),
    }
}

/// Free a C string
#[no_mangle]
pub extern "C" fn free_c_string(s: CString) {
    if !s.data.is_null() {
        unsafe {
            let _ = std::ffi::CString::from_raw(s.data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_code_conversion() {
        assert_eq!(CResultCode::from(GemmaError::InvalidArgument("test".to_string())), CResultCode::InvalidArgument);
        assert_eq!(CResultCode::from(GemmaError::Redis("test".to_string())), CResultCode::RedisError);
    }

    #[test]
    fn test_c_string_conversion() {
        let rust_string = "Hello, World!".to_string();
        let c_string = CString::from_rust_string(rust_string.clone());

        assert!(!c_string.data.is_null());
        assert_eq!(c_string.length, rust_string.len());

        if let Some(converted_back) = c_string.to_rust_string() {
            assert_eq!(converted_back, rust_string);
        }
    }

    #[test]
    fn test_float_vector_conversion() {
        let rust_vec = vec![1.0, 2.0, 3.0, 4.0];
        let c_vector = CFloatVector::from_rust_vec(rust_vec.clone());

        assert!(!c_vector.data.is_null());
        assert_eq!(c_vector.length, rust_vec.len());

        // Note: to_rust_vec() would consume the data, so we can't test the conversion back
    }
}