/**
 * @file rag_redis.h
 * @brief C API for RAG Redis System
 *
 * This header provides a complete C interface for the RAG (Retrieval-Augmented Generation)
 * Redis System. The API is thread-safe and provides comprehensive error handling.
 *
 * @author RAG-Redis Development Team
 * @version 1.0.0
 * @date 2024
 *
 * @example Basic Usage:
 * @code
 * #include "rag_redis.h"
 *
 * int main() {
 *     // Initialize the library
 *     if (!rag_init()) {
 *         return -1;
 *     }
 *
 *     // Create configuration
 *     RagConfig config = {
 *         .redis_url = "redis://127.0.0.1:6379",
 *         .redis_pool_size = 10,
 *         .vector_dimension = 768,
 *         // ... other fields
 *     };
 *
 *     // Create RAG system
 *     RagErrorInfo error = {0};
 *     RagHandle handle = rag_create(&config, &error);
 *     if (handle == 0) {
 *         printf("Error creating RAG system: %s\n", error.message);
 *         rag_free_error_message(error.message);
 *         return -1;
 *     }
 *
 *     // Ingest document
 *     char* doc_id = NULL;
 *     int result = rag_ingest_document(handle, "Sample document content",
 *                                     "{\"title\": \"Sample\"}", &doc_id, &error);
 *     if (result) {
 *         printf("Document ingested with ID: %s\n", doc_id);
 *         rag_free_string(doc_id);
 *     }
 *
 *     // Search
 *     RagSearchResults* search_results = rag_search(handle, "sample query", 10, &error);
 *     if (search_results) {
 *         printf("Found %u results\n", search_results->count);
 *         for (unsigned int i = 0; i < search_results->count; i++) {
 *             printf("Result %u: %s (score: %.3f)\n",
 *                    i, search_results->results[i].text, search_results->results[i].score);
 *         }
 *         rag_free_search_results(search_results);
 *     }
 *
 *     // Cleanup
 *     rag_destroy(handle);
 *     rag_cleanup();
 *     return 0;
 * }
 * @endcode
 */

#ifndef RAG_REDIS_H
#define RAG_REDIS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* ================================================================================================
 * TYPE DEFINITIONS
 * ================================================================================================ */

/**
 * @brief Opaque handle to a RAG system instance
 *
 * This handle represents an active RAG system instance. All operations
 * require a valid handle obtained from rag_create().
 */
typedef uint64_t RagHandle;

/**
 * @brief Configuration structure for RAG system initialization
 *
 * This structure contains all configuration parameters needed to initialize
 * a RAG system instance. All string parameters must be null-terminated.
 *
 * @note String pointers must remain valid during the lifetime of rag_create() call
 */
typedef struct {
    /* Redis Configuration */
    const char* redis_url;                    ///< Redis connection URL (e.g., "redis://127.0.0.1:6379")
    unsigned int redis_pool_size;             ///< Number of connections in Redis pool (default: 10)
    unsigned int redis_connection_timeout_secs; ///< Connection timeout in seconds (default: 5)
    unsigned int redis_command_timeout_secs;  ///< Command timeout in seconds (default: 10)
    unsigned int redis_max_retries;           ///< Maximum retry attempts (default: 3)
    unsigned int redis_retry_delay_ms;        ///< Delay between retries in milliseconds (default: 100)
    int redis_enable_cluster;                 ///< Enable cluster mode (0=false, 1=true)

    /* Vector Store Configuration */
    unsigned int vector_dimension;            ///< Embedding vector dimension (default: 768)
    unsigned int vector_max_elements;         ///< Maximum number of vectors (default: 100000)
    unsigned int vector_m;                    ///< HNSW M parameter (default: 16)
    unsigned int vector_ef_construction;      ///< HNSW ef_construction parameter (default: 200)
    unsigned int vector_ef_search;            ///< HNSW ef_search parameter (default: 50)
    float vector_similarity_threshold;        ///< Minimum similarity threshold (default: 0.7)

    /* Document Processing Configuration */
    unsigned int doc_chunk_size;              ///< Document chunk size in tokens (default: 512)
    unsigned int doc_chunk_overlap;           ///< Overlap between chunks (default: 50)
    unsigned int doc_max_chunk_size;          ///< Maximum chunk size (default: 1024)
    unsigned int doc_min_chunk_size;          ///< Minimum chunk size (default: 100)
    int doc_enable_metadata_extraction;       ///< Enable metadata extraction (0=false, 1=true)

    /* Memory Management Configuration */
    unsigned int memory_max_mb;               ///< Maximum memory usage in MB (default: 1024)
    unsigned int memory_ttl_seconds;          ///< Memory item TTL in seconds (default: 3600)
    unsigned int memory_cleanup_interval_secs; ///< Cleanup interval in seconds (default: 300)
    unsigned int memory_max_items;            ///< Maximum number of memory items (default: 10000)

    /* Research Configuration */
    int research_enable_web_search;           ///< Enable web search (0=false, 1=true)
    unsigned int research_max_results;        ///< Maximum research results (default: 10)
    unsigned int research_timeout_secs;       ///< Research timeout in seconds (default: 30)
    unsigned int research_rate_limit_per_sec; ///< Rate limit per second (default: 5)
} RagConfig;

/**
 * @brief Single search result structure
 *
 * Contains information about a single search result. All string fields
 * are null-terminated and owned by the result structure.
 */
typedef struct {
    char* id;            ///< Unique result identifier (never NULL)
    char* text;          ///< Result text content (never NULL)
    float score;         ///< Similarity score (0.0-1.0, higher is better)
    char* metadata_json; ///< Metadata as JSON string (never NULL, may be "{}")
    char* source;        ///< Source identifier (never NULL)
    char* url;           ///< Optional URL (may be NULL)
} RagSearchResult;

/**
 * @brief Search results array structure
 *
 * Contains an array of search results with count information.
 * Memory is managed by the library - use rag_free_search_results() to clean up.
 */
typedef struct {
    RagSearchResult* results; ///< Array of search results
    unsigned int count;       ///< Number of results in array
} RagSearchResults;

/**
 * @brief Error information structure
 *
 * Contains error code and message for detailed error reporting.
 * The message string is owned by the library and should be freed with rag_free_error_message().
 */
typedef struct {
    int code;        ///< Error code (see ErrorCode enum values)
    char* message;   ///< Human-readable error message (may be NULL)
} RagErrorInfo;

/* ================================================================================================
 * ERROR CODES
 * ================================================================================================ */

/**
 * @brief Error code constants
 *
 * These constants match the Rust ErrorCode enum and can be used
 * to identify specific error conditions.
 */
#define RAG_ERROR_SUCCESS              0   ///< Operation completed successfully
#define RAG_ERROR_REDIS               1   ///< Redis operation error
#define RAG_ERROR_SERIALIZATION       2   ///< Serialization/deserialization error
#define RAG_ERROR_IO                  3   ///< I/O operation error
#define RAG_ERROR_CONFIG              4   ///< Configuration error
#define RAG_ERROR_VECTOR              5   ///< Vector operation error
#define RAG_ERROR_DOCUMENT            6   ///< Document processing error
#define RAG_ERROR_EMBEDDING           7   ///< Embedding generation error
#define RAG_ERROR_NETWORK             8   ///< Network operation error
#define RAG_ERROR_NOT_FOUND           9   ///< Resource not found error
#define RAG_ERROR_INVALID_INPUT      10   ///< Invalid input parameter error
#define RAG_ERROR_MEMORY             11   ///< Memory allocation error
#define RAG_ERROR_TIMEOUT            12   ///< Operation timeout error
#define RAG_ERROR_UNKNOWN           999   ///< Unknown error

/* ================================================================================================
 * LIBRARY MANAGEMENT FUNCTIONS
 * ================================================================================================ */

/**
 * @brief Initialize the RAG Redis System library
 *
 * This function must be called once before using any other library functions.
 * It initializes global state including the async runtime.
 *
 * @return 1 on success, 0 on failure
 *
 * @thread_safety This function is thread-safe
 * @example
 * @code
 * if (!rag_init()) {
 *     fprintf(stderr, "Failed to initialize RAG library\n");
 *     exit(1);
 * }
 * @endcode
 */
int rag_init(void);

/**
 * @brief Clean up global library state
 *
 * This function should be called once at program shutdown to clean up
 * global resources. After calling this function, rag_init() must be called
 * again before using other library functions.
 *
 * @return 1 on success, 0 on failure
 *
 * @thread_safety This function is thread-safe but should not be called concurrently with other library functions
 */
int rag_cleanup(void);

/**
 * @brief Get library version string
 *
 * Returns a pointer to a static string containing the library version.
 * The returned pointer is valid for the lifetime of the program.
 *
 * @return Pointer to null-terminated version string
 *
 * @thread_safety This function is thread-safe
 */
const char* rag_version(void);

/* ================================================================================================
 * RAG SYSTEM MANAGEMENT FUNCTIONS
 * ================================================================================================ */

/**
 * @brief Create a new RAG system instance
 *
 * Creates and initializes a new RAG system with the provided configuration.
 * The returned handle must be destroyed with rag_destroy() to prevent resource leaks.
 *
 * @param config Pointer to configuration structure (must not be NULL)
 * @param error_info Optional pointer to error information structure (may be NULL)
 * @return Valid handle (> 0) on success, 0 on failure
 *
 * @thread_safety This function is thread-safe
 *
 * @example
 * @code
 * RagConfig config = {
 *     .redis_url = "redis://localhost:6379",
 *     .redis_pool_size = 10,
 *     .vector_dimension = 768,
 *     // ... initialize other fields
 * };
 *
 * RagErrorInfo error = {0};
 * RagHandle handle = rag_create(&config, &error);
 * if (handle == 0) {
 *     printf("Error: %s\n", error.message);
 *     rag_free_error_message(error.message);
 * }
 * @endcode
 */
RagHandle rag_create(const RagConfig* config, RagErrorInfo* error_info);

/**
 * @brief Destroy a RAG system instance
 *
 * Destroys the RAG system instance and frees all associated resources.
 * After calling this function, the handle becomes invalid and must not be used.
 *
 * @param handle Valid RAG system handle
 * @return 1 if handle was valid and destroyed, 0 if handle was invalid
 *
 * @thread_safety This function is thread-safe
 */
int rag_destroy(RagHandle handle);

/**
 * @brief Check health status of a RAG system instance
 *
 * Performs a quick health check on the RAG system instance.
 *
 * @param handle RAG system handle to check
 * @return 1 if healthy, 0 if unhealthy or invalid handle
 *
 * @thread_safety This function is thread-safe
 */
int rag_health_check(RagHandle handle);

/* ================================================================================================
 * DOCUMENT MANAGEMENT FUNCTIONS
 * ================================================================================================ */

/**
 * @brief Ingest a document into the RAG system
 *
 * Processes and ingests a document into the RAG system. The document will be
 * chunked, embedded, and stored for future retrieval.
 *
 * @param handle Valid RAG system handle
 * @param content Document content as null-terminated string (must not be NULL)
 * @param metadata_json Optional metadata as JSON string (may be NULL)
 * @param document_id Optional pointer to receive document ID (may be NULL)
 * @param error_info Optional pointer to error information (may be NULL)
 * @return 1 on success, 0 on failure
 *
 * @note If document_id is provided, the caller must free it with rag_free_string()
 * @thread_safety This function is thread-safe
 *
 * @example
 * @code
 * char* doc_id = NULL;
 * RagErrorInfo error = {0};
 * int result = rag_ingest_document(handle,
 *                                  "This is a sample document content.",
 *                                  "{\"title\": \"Sample Document\", \"author\": \"John Doe\"}",
 *                                  &doc_id, &error);
 * if (result) {
 *     printf("Document ingested with ID: %s\n", doc_id);
 *     rag_free_string(doc_id);
 * } else {
 *     printf("Error ingesting document: %s\n", error.message);
 *     rag_free_error_message(error.message);
 * }
 * @endcode
 */
int rag_ingest_document(RagHandle handle,
                       const char* content,
                       const char* metadata_json,
                       char** document_id,
                       RagErrorInfo* error_info);

/* ================================================================================================
 * SEARCH FUNCTIONS
 * ================================================================================================ */

/**
 * @brief Search for documents in the RAG system
 *
 * Performs a similarity search against the ingested documents using the provided query.
 * Results are ranked by similarity score in descending order.
 *
 * @param handle Valid RAG system handle
 * @param query Search query as null-terminated string (must not be NULL)
 * @param limit Maximum number of results to return
 * @param error_info Optional pointer to error information (may be NULL)
 * @return Pointer to search results structure on success, NULL on failure
 *
 * @note The caller must free the returned results with rag_free_search_results()
 * @thread_safety This function is thread-safe
 *
 * @example
 * @code
 * RagErrorInfo error = {0};
 * RagSearchResults* results = rag_search(handle, "machine learning", 5, &error);
 * if (results) {
 *     printf("Found %u results:\n", results->count);
 *     for (unsigned int i = 0; i < results->count; i++) {
 *         RagSearchResult* result = &results->results[i];
 *         printf("%u. %s (score: %.3f)\n", i+1, result->text, result->score);
 *     }
 *     rag_free_search_results(results);
 * } else {
 *     printf("Search failed: %s\n", error.message);
 *     rag_free_error_message(error.message);
 * }
 * @endcode
 */
RagSearchResults* rag_search(RagHandle handle,
                             const char* query,
                             unsigned int limit,
                             RagErrorInfo* error_info);

/**
 * @brief Perform research with web search capabilities
 *
 * Performs both local document search and optional web search using the provided sources.
 * Results from both sources are combined and ranked by relevance.
 *
 * @param handle Valid RAG system handle
 * @param query Research query as null-terminated string (must not be NULL)
 * @param sources Array of source identifiers (may be NULL)
 * @param sources_count Number of sources in the array
 * @param error_info Optional pointer to error information (may be NULL)
 * @return Pointer to search results structure on success, NULL on failure
 *
 * @note The caller must free the returned results with rag_free_search_results()
 * @thread_safety This function is thread-safe
 *
 * @example
 * @code
 * const char* sources[] = {"wikipedia", "arxiv", "github"};
 * RagSearchResults* results = rag_research(handle, "neural networks",
 *                                          sources, 3, NULL);
 * if (results) {
 *     for (unsigned int i = 0; i < results->count; i++) {
 *         printf("Source: %s, Score: %.3f\n",
 *                results->results[i].source, results->results[i].score);
 *     }
 *     rag_free_search_results(results);
 * }
 * @endcode
 */
RagSearchResults* rag_research(RagHandle handle,
                              const char* query,
                              const char* const* sources,
                              unsigned int sources_count,
                              RagErrorInfo* error_info);

/* ================================================================================================
 * SYSTEM INFORMATION FUNCTIONS
 * ================================================================================================ */

/**
 * @brief Get system statistics as JSON string
 *
 * Returns detailed system statistics including memory usage, performance metrics,
 * and operational counters as a JSON-formatted string.
 *
 * @param handle Valid RAG system handle
 * @param error_info Optional pointer to error information (may be NULL)
 * @return Pointer to JSON string on success, NULL on failure
 *
 * @note The caller must free the returned string with rag_free_string()
 * @thread_safety This function is thread-safe
 *
 * @example
 * @code
 * char* stats = rag_get_stats(handle, NULL);
 * if (stats) {
 *     printf("System stats: %s\n", stats);
 *     rag_free_string(stats);
 * }
 * @endcode
 */
char* rag_get_stats(RagHandle handle, RagErrorInfo* error_info);

/* ================================================================================================
 * MEMORY MANAGEMENT FUNCTIONS
 * ================================================================================================ */

/**
 * @brief Free memory allocated for search results
 *
 * Frees all memory associated with a RagSearchResults structure,
 * including the results array and all string fields.
 *
 * @param results Pointer to search results structure (may be NULL)
 *
 * @thread_safety This function is thread-safe
 */
void rag_free_search_results(RagSearchResults* results);

/**
 * @brief Free memory allocated for error message
 *
 * Frees memory allocated for an error message string.
 *
 * @param message Pointer to error message string (may be NULL)
 *
 * @thread_safety This function is thread-safe
 */
void rag_free_error_message(char* message);

/**
 * @brief Free memory allocated for string
 *
 * Frees memory allocated for a string returned by library functions.
 *
 * @param string Pointer to string (may be NULL)
 *
 * @thread_safety This function is thread-safe
 */
void rag_free_string(char* string);

/* ================================================================================================
 * UTILITY MACROS
 * ================================================================================================ */

/**
 * @brief Check if error occurred and get error message
 *
 * @param error_info Pointer to RagErrorInfo structure
 * @return Non-zero if error occurred
 */
#define RAG_HAS_ERROR(error_info) ((error_info) && (error_info)->code != RAG_ERROR_SUCCESS)

/**
 * @brief Get error message from error info
 *
 * @param error_info Pointer to RagErrorInfo structure
 * @return Error message string or "Unknown error" if NULL
 */
#define RAG_ERROR_MESSAGE(error_info) \
    (((error_info) && (error_info)->message) ? (error_info)->message : "Unknown error")

/* ================================================================================================
 * CONVENIENCE FUNCTIONS (IMPLEMENTATION OPTIONAL)
 * ================================================================================================ */

/**
 * @brief Create default configuration
 *
 * Fills a RagConfig structure with sensible default values.
 * The redis_url field must still be set by the caller.
 *
 * @param config Pointer to configuration structure to initialize
 */
static inline void rag_config_default(RagConfig* config) {
    if (!config) return;

    config->redis_url = "redis://127.0.0.1:6379";
    config->redis_pool_size = 10;
    config->redis_connection_timeout_secs = 5;
    config->redis_command_timeout_secs = 10;
    config->redis_max_retries = 3;
    config->redis_retry_delay_ms = 100;
    config->redis_enable_cluster = 0;

    config->vector_dimension = 768;
    config->vector_max_elements = 100000;
    config->vector_m = 16;
    config->vector_ef_construction = 200;
    config->vector_ef_search = 50;
    config->vector_similarity_threshold = 0.7f;

    config->doc_chunk_size = 512;
    config->doc_chunk_overlap = 50;
    config->doc_max_chunk_size = 1024;
    config->doc_min_chunk_size = 100;
    config->doc_enable_metadata_extraction = 1;

    config->memory_max_mb = 1024;
    config->memory_ttl_seconds = 3600;
    config->memory_cleanup_interval_secs = 300;
    config->memory_max_items = 10000;

    config->research_enable_web_search = 1;
    config->research_max_results = 10;
    config->research_timeout_secs = 30;
    config->research_rate_limit_per_sec = 5;
}

#ifdef __cplusplus
}
#endif

#endif /* RAG_REDIS_H */
