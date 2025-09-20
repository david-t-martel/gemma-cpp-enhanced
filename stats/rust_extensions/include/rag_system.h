/**
 * @file rag_system.h
 * @brief C++ header for Redis-backed RAG system FFI interface
 *
 * This header provides C++ bindings for the Rust-based RAG system components:
 * - Redis memory management
 * - Vector storage and similarity search
 * - Document processing pipeline
 * - Internet research client
 *
 * All functions are thread-safe and designed for high-performance concurrent access.
 */

#ifndef RAG_SYSTEM_H
#define RAG_SYSTEM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Types and Constants
// =============================================================================

/// Result codes for all RAG system operations
typedef enum {
    RAG_SUCCESS = 0,
    RAG_INVALID_ARGUMENT = -1,
    RAG_NULL_POINTER = -2,
    RAG_OUT_OF_MEMORY = -3,
    RAG_IO_ERROR = -4,
    RAG_NETWORK_ERROR = -5,
    RAG_SERIALIZATION_ERROR = -6,
    RAG_REDIS_ERROR = -7,
    RAG_VECTOR_STORE_ERROR = -8,
    RAG_DOCUMENT_PARSING_ERROR = -9,
    RAG_NOT_IMPLEMENTED = -10,
    RAG_UNKNOWN_ERROR = -99
} RagResultCode;

/// Distance metrics for vector similarity
typedef enum {
    DISTANCE_COSINE = 0,
    DISTANCE_EUCLIDEAN = 1,
    DISTANCE_DOT_PRODUCT = 2,
    DISTANCE_MANHATTAN = 3
} RagDistanceMetric;

/// Document formats supported by the pipeline
typedef enum {
    FORMAT_PLAIN_TEXT = 0,
    FORMAT_MARKDOWN = 1,
    FORMAT_HTML = 2,
    FORMAT_PDF = 3,
    FORMAT_JSON = 4
} RagDocumentFormat;

/// C-compatible string structure
typedef struct {
    char* data;
    size_t length;
} RagString;

/// C-compatible float vector structure
typedef struct {
    float* data;
    size_t length;
    size_t capacity;
} RagFloatVector;

/// Search result from vector store
typedef struct {
    RagString document_id;
    uint32_t chunk_id;
    RagString text;
    float similarity;
    float distance;
    RagFloatVector vector;
} RagSearchResult;

// Opaque handles for Rust objects
typedef void* RagRuntimeHandle;
typedef void* RagRedisManagerHandle;
typedef void* RagVectorStoreHandle;
typedef void* RagDocumentPipelineHandle;
typedef void* RagResearchClientHandle;

// =============================================================================
// System Management Functions
// =============================================================================

/**
 * Initialize the RAG system with Tokio runtime.
 * Must be called before using any other functions.
 *
 * @return Handle to the runtime, or NULL on failure
 */
RagRuntimeHandle rag_init(void);

/**
 * Cleanup the RAG system and free all resources.
 * Should be called when shutting down.
 *
 * @param runtime Runtime handle from rag_init()
 */
void rag_cleanup(RagRuntimeHandle runtime);

// =============================================================================
// Redis Manager Functions
// =============================================================================

/**
 * Create a new Redis manager with connection pooling.
 *
 * @param runtime Runtime handle
 * @param url Redis connection URL (e.g., "redis://localhost:6379")
 * @param max_connections Maximum number of connections in pool
 * @param timeout_ms Connection timeout in milliseconds
 * @return Tuple of (result_code, handle)
 */
typedef struct {
    RagResultCode result;
    RagRedisManagerHandle handle;
} RagRedisManagerCreateResult;

RagRedisManagerCreateResult redis_manager_create(
    RagRuntimeHandle runtime,
    const char* url,
    uint32_t max_connections,
    uint32_t timeout_ms
);

/**
 * Destroy a Redis manager and free its resources.
 *
 * @param handle Redis manager handle
 */
void redis_manager_destroy(RagRedisManagerHandle handle);

/**
 * Store a document in Redis with metadata.
 *
 * @param runtime Runtime handle
 * @param handle Redis manager handle
 * @param doc_id Document ID (must be unique)
 * @param content Document content
 * @param content_type MIME type of content (optional, defaults to "text/plain")
 * @return Result code
 */
RagResultCode redis_store_document(
    RagRuntimeHandle runtime,
    RagRedisManagerHandle handle,
    const char* doc_id,
    const char* content,
    const char* content_type
);

/**
 * Retrieve a document from Redis by ID.
 *
 * @param runtime Runtime handle
 * @param handle Redis manager handle
 * @param doc_id Document ID to retrieve
 * @param content_out Output parameter for document content (caller must free)
 * @return Result code
 */
RagResultCode redis_get_document(
    RagRuntimeHandle runtime,
    RagRedisManagerHandle handle,
    const char* doc_id,
    RagString* content_out
);

// =============================================================================
// Vector Store Functions
// =============================================================================

/**
 * Create a new vector store with HNSW index.
 *
 * @param dimension Vector dimension (must match embedding model)
 * @param metric Distance metric for similarity calculation
 * @return Tuple of (result_code, handle)
 */
typedef struct {
    RagResultCode result;
    RagVectorStoreHandle handle;
} RagVectorStoreCreateResult;

RagVectorStoreCreateResult vector_store_create(
    uint32_t dimension,
    int32_t metric
);

/**
 * Destroy a vector store and free its resources.
 *
 * @param handle Vector store handle
 */
void vector_store_destroy(RagVectorStoreHandle handle);

/**
 * Add a vector to the store with associated metadata.
 *
 * @param runtime Runtime handle
 * @param handle Vector store handle
 * @param vector Pointer to vector data
 * @param vector_size Number of elements in vector
 * @param doc_id Document ID this vector belongs to
 * @param chunk_id Chunk ID within the document
 * @param text Text content associated with this vector
 * @return Tuple of (result_code, vector_id)
 */
typedef struct {
    RagResultCode result;
    uint32_t vector_id;
} RagVectorAddResult;

RagVectorAddResult vector_store_add_vector(
    RagRuntimeHandle runtime,
    RagVectorStoreHandle handle,
    const float* vector,
    size_t vector_size,
    const char* doc_id,
    uint32_t chunk_id,
    const char* text
);

/**
 * Search for similar vectors in the store.
 *
 * @param runtime Runtime handle
 * @param handle Vector store handle
 * @param query_vector Query vector to search for
 * @param vector_size Number of elements in query vector
 * @param k Number of results to return
 * @param include_vectors Whether to include vector data in results
 * @param results_out Output array of search results (caller must free)
 * @param results_count Output number of results
 * @return Result code
 */
RagResultCode vector_store_search(
    RagRuntimeHandle runtime,
    RagVectorStoreHandle handle,
    const float* query_vector,
    size_t vector_size,
    uint32_t k,
    int32_t include_vectors,
    RagSearchResult** results_out,
    size_t* results_count
);

/**
 * Free search results returned by vector_store_search.
 *
 * @param results Results array to free
 * @param count Number of results in array
 */
void vector_store_free_search_results(
    RagSearchResult* results,
    size_t count
);

// =============================================================================
// Document Pipeline Functions
// =============================================================================

/**
 * Create a document processing pipeline.
 *
 * @param max_chunk_size Maximum size of each chunk in tokens
 * @param chunk_overlap Overlap between adjacent chunks in tokens
 * @param embedding_dimension Dimension of generated embeddings
 * @return Tuple of (result_code, handle)
 */
typedef struct {
    RagResultCode result;
    RagDocumentPipelineHandle handle;
} RagDocumentPipelineCreateResult;

RagDocumentPipelineCreateResult document_pipeline_create(
    uint32_t max_chunk_size,
    uint32_t chunk_overlap,
    uint32_t embedding_dimension
);

/**
 * Destroy a document pipeline and free its resources.
 *
 * @param handle Document pipeline handle
 */
void document_pipeline_destroy(RagDocumentPipelineHandle handle);

/**
 * Process a document through the complete pipeline.
 * Parses, chunks, generates embeddings, and stores in vector store.
 *
 * @param runtime Runtime handle
 * @param handle Document pipeline handle
 * @param content Document content to process
 * @param doc_id Unique document identifier
 * @param format Document format (see RagDocumentFormat enum)
 * @param vector_ids_out Output array of vector IDs created (caller must free)
 * @param vector_ids_count Output number of vector IDs
 * @return Result code
 */
RagResultCode document_pipeline_process(
    RagRuntimeHandle runtime,
    RagDocumentPipelineHandle handle,
    const char* content,
    const char* doc_id,
    int32_t format,
    char*** vector_ids_out,
    size_t* vector_ids_count
);

/**
 * Free vector IDs array returned by document_pipeline_process.
 *
 * @param vector_ids Vector IDs array to free
 * @param count Number of vector IDs in array
 */
void document_pipeline_free_vector_ids(
    char** vector_ids,
    size_t count
);

// =============================================================================
// Research Client Functions
// =============================================================================

/**
 * Create a research client for web scraping and API integration.
 *
 * @param max_concurrent Maximum number of concurrent requests
 * @param timeout_ms Request timeout in milliseconds
 * @param rate_limit_rps Rate limit in requests per second
 * @return Tuple of (result_code, handle)
 */
typedef struct {
    RagResultCode result;
    RagResearchClientHandle handle;
} RagResearchClientCreateResult;

RagResearchClientCreateResult research_client_create(
    uint32_t max_concurrent,
    uint32_t timeout_ms,
    float rate_limit_rps
);

/**
 * Destroy a research client and free its resources.
 *
 * @param handle Research client handle
 */
void research_client_destroy(RagResearchClientHandle handle);

/**
 * Perform a research query across multiple sources.
 *
 * @param runtime Runtime handle
 * @param handle Research client handle
 * @param query Query string to search for
 * @param sources Array of source URLs or API endpoints
 * @param sources_count Number of sources in array
 * @param max_results Maximum number of results to return
 * @param results_json Output JSON string with results (caller must free)
 * @return Result code
 */
RagResultCode research_client_query(
    RagRuntimeHandle runtime,
    RagResearchClientHandle handle,
    const char* query,
    const char* const* sources,
    size_t sources_count,
    uint32_t max_results,
    RagString* results_json
);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Free a RagString structure.
 *
 * @param s String to free
 */
void free_c_string(RagString s);

// =============================================================================
// C++ Convenience Wrappers
// =============================================================================

#ifdef __cplusplus
} // extern "C"

#include <memory>
#include <string>
#include <vector>
#include <stdexcept>

namespace rag {

/**
 * Exception class for RAG system errors
 */
class RagException : public std::runtime_error {
public:
    explicit RagException(RagResultCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    RagResultCode code() const noexcept { return code_; }

private:
    RagResultCode code_;
};

/**
 * RAII wrapper for RAG system runtime
 */
class Runtime {
public:
    Runtime() : handle_(rag_init()) {
        if (!handle_) {
            throw RagException(RAG_UNKNOWN_ERROR, "Failed to initialize RAG runtime");
        }
    }

    ~Runtime() {
        if (handle_) {
            rag_cleanup(handle_);
        }
    }

    // Non-copyable, but movable
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    Runtime(Runtime&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    Runtime& operator=(Runtime&& other) noexcept {
        if (this != &other) {
            if (handle_) rag_cleanup(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    RagRuntimeHandle handle() const noexcept { return handle_; }

private:
    RagRuntimeHandle handle_;
};

/**
 * RAII wrapper for Redis manager
 */
class RedisManager {
public:
    RedisManager(const Runtime& runtime, const std::string& url,
                uint32_t max_connections = 20, uint32_t timeout_ms = 5000) {
        auto result = redis_manager_create(runtime.handle(), url.c_str(),
                                         max_connections, timeout_ms);
        if (result.result != RAG_SUCCESS) {
            throw RagException(result.result, "Failed to create Redis manager");
        }
        handle_ = result.handle;
    }

    ~RedisManager() {
        if (handle_) {
            redis_manager_destroy(handle_);
        }
    }

    // Non-copyable, but movable
    RedisManager(const RedisManager&) = delete;
    RedisManager& operator=(const RedisManager&) = delete;
    RedisManager(RedisManager&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    RedisManager& operator=(RedisManager&& other) noexcept {
        if (this != &other) {
            if (handle_) redis_manager_destroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    void store_document(const Runtime& runtime, const std::string& doc_id,
                       const std::string& content,
                       const std::string& content_type = "text/plain") {
        auto result = redis_store_document(runtime.handle(), handle_,
                                         doc_id.c_str(), content.c_str(),
                                         content_type.c_str());
        if (result != RAG_SUCCESS) {
            throw RagException(result, "Failed to store document");
        }
    }

    std::string get_document(const Runtime& runtime, const std::string& doc_id) {
        RagString content;
        auto result = redis_get_document(runtime.handle(), handle_,
                                       doc_id.c_str(), &content);
        if (result != RAG_SUCCESS) {
            throw RagException(result, "Failed to get document");
        }

        std::string content_str(content.data, content.length);
        free_c_string(content);
        return content_str;
    }

    RagRedisManagerHandle handle() const noexcept { return handle_; }

private:
    RagRedisManagerHandle handle_;
};

/**
 * RAII wrapper for vector store
 */
class VectorStore {
public:
    VectorStore(uint32_t dimension, RagDistanceMetric metric = DISTANCE_COSINE) {
        auto result = vector_store_create(dimension, static_cast<int32_t>(metric));
        if (result.result != RAG_SUCCESS) {
            throw RagException(result.result, "Failed to create vector store");
        }
        handle_ = result.handle;
    }

    ~VectorStore() {
        if (handle_) {
            vector_store_destroy(handle_);
        }
    }

    // Non-copyable, but movable
    VectorStore(const VectorStore&) = delete;
    VectorStore& operator=(const VectorStore&) = delete;
    VectorStore(VectorStore&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    VectorStore& operator=(VectorStore&& other) noexcept {
        if (this != &other) {
            if (handle_) vector_store_destroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    uint32_t add_vector(const Runtime& runtime, const std::vector<float>& vector,
                       const std::string& doc_id, uint32_t chunk_id,
                       const std::string& text) {
        auto result = vector_store_add_vector(runtime.handle(), handle_,
                                            vector.data(), vector.size(),
                                            doc_id.c_str(), chunk_id, text.c_str());
        if (result.result != RAG_SUCCESS) {
            throw RagException(result.result, "Failed to add vector");
        }
        return result.vector_id;
    }

    std::vector<RagSearchResult> search(const Runtime& runtime,
                                       const std::vector<float>& query_vector,
                                       uint32_t k, bool include_vectors = false) {
        RagSearchResult* results;
        size_t count;
        auto result = vector_store_search(runtime.handle(), handle_,
                                        query_vector.data(), query_vector.size(),
                                        k, include_vectors ? 1 : 0,
                                        &results, &count);
        if (result != RAG_SUCCESS) {
            throw RagException(result, "Failed to search vectors");
        }

        std::vector<RagSearchResult> search_results(results, results + count);
        vector_store_free_search_results(results, count);
        return search_results;
    }

    RagVectorStoreHandle handle() const noexcept { return handle_; }

private:
    RagVectorStoreHandle handle_;
};

} // namespace rag

#endif // __cplusplus

#endif // RAG_SYSTEM_H
