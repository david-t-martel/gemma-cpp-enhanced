/**
 * @file rag_redis.hpp
 * @brief C++ wrapper for RAG Redis System
 *
 * This header provides a modern C++ interface for the RAG (Retrieval-Augmented Generation)
 * Redis System. It uses RAII patterns for automatic resource management, provides
 * exception safety, and offers a more ergonomic API than the C interface.
 *
 * @author RAG-Redis Development Team
 * @version 1.0.0
 * @date 2024
 *
 * @example Basic Usage:
 * @code
 * #include "rag_redis.hpp"
 * #include <iostream>
 *
 * int main() {
 *     try {
 *         // Initialize library (RAII)
 *         rag::Library library;
 *
 *         // Create configuration
 *         rag::Config config;
 *         config.redis_url("redis://127.0.0.1:6379")
 *               .vector_dimension(768)
 *               .document_chunk_size(512);
 *
 *         // Create RAG system
 *         rag::System system(config);
 *
 *         // Ingest document
 *         std::string doc_id = system.ingest_document(
 *             "This is a sample document content.",
 *             {{"title", "Sample Document"}, {"author", "John Doe"}}
 *         );
 *         std::cout << "Document ingested with ID: " << doc_id << std::endl;
 *
 *         // Search
 *         auto results = system.search("sample query", 10);
 *         std::cout << "Found " << results.size() << " results" << std::endl;
 *         for (const auto& result : results) {
 *             std::cout << "Score: " << result.score << " - " << result.text << std::endl;
 *         }
 *
 *         return 0;
 *     } catch (const rag::Exception& e) {
 *         std::cerr << "RAG Error: " << e.what() << std::endl;
 *         return 1;
 *     }
 * }
 * @endcode
 */

#ifndef RAG_REDIS_HPP
#define RAG_REDIS_HPP

#include "rag_redis.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <optional>
#include <chrono>
#include <atomic>
#include <mutex>
#include <future>

namespace rag {

/* ================================================================================================
 * FORWARD DECLARATIONS
 * ================================================================================================ */

class Exception;
class Config;
class SearchResult;
class SearchResults;
class System;
class Library;

/* ================================================================================================
 * TYPE ALIASES
 * ================================================================================================ */

/// @brief Type alias for metadata as key-value pairs
using Metadata = std::unordered_map<std::string, std::string>;

/// @brief Type alias for source list
using Sources = std::vector<std::string>;

/// @brief Type alias for duration
using Duration = std::chrono::seconds;

/* ================================================================================================
 * EXCEPTION CLASSES
 * ================================================================================================ */

/**
 * @brief Base exception class for all RAG Redis System errors
 *
 * This exception class provides structured error information including
 * error codes that correspond to the C API error codes.
 */
class Exception : public std::runtime_error {
public:
    /**
     * @brief Error code enumeration matching the C API
     */
    enum Code {
        Success = RAG_ERROR_SUCCESS,
        Redis = RAG_ERROR_REDIS,
        Serialization = RAG_ERROR_SERIALIZATION,
        IO = RAG_ERROR_IO,
        Config = RAG_ERROR_CONFIG,
        Vector = RAG_ERROR_VECTOR,
        Document = RAG_ERROR_DOCUMENT,
        Embedding = RAG_ERROR_EMBEDDING,
        Network = RAG_ERROR_NETWORK,
        NotFound = RAG_ERROR_NOT_FOUND,
        InvalidInput = RAG_ERROR_INVALID_INPUT,
        Memory = RAG_ERROR_MEMORY,
        Timeout = RAG_ERROR_TIMEOUT,
        Unknown = RAG_ERROR_UNKNOWN
    };

private:
    Code code_;

public:
    /**
     * @brief Construct exception with error code and message
     * @param code Error code
     * @param message Error message
     */
    Exception(Code code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    /**
     * @brief Construct exception from C API error info
     * @param error_info C API error information
     */
    explicit Exception(const RagErrorInfo& error_info)
        : std::runtime_error(error_info.message ? error_info.message : "Unknown error")
        , code_(static_cast<Code>(error_info.code)) {}

    /**
     * @brief Get error code
     * @return Error code
     */
    Code code() const noexcept { return code_; }

    /**
     * @brief Get error code as string
     * @return String representation of error code
     */
    std::string code_string() const {
        switch (code_) {
            case Success: return "Success";
            case Redis: return "Redis";
            case Serialization: return "Serialization";
            case IO: return "IO";
            case Config: return "Config";
            case Vector: return "Vector";
            case Document: return "Document";
            case Embedding: return "Embedding";
            case Network: return "Network";
            case NotFound: return "NotFound";
            case InvalidInput: return "InvalidInput";
            case Memory: return "Memory";
            case Timeout: return "Timeout";
            case Unknown: default: return "Unknown";
        }
    }
};

/* ================================================================================================
 * UTILITY CLASSES
 * ================================================================================================ */

/**
 * @brief Thread-safe string pool for managing C string lifetimes
 *
 * This class helps manage the lifetime of C strings when interfacing
 * with the C API, ensuring they remain valid during async operations.
 */
class StringPool {
private:
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<char[]>> strings_;

public:
    /**
     * @brief Add a string to the pool and get a C-style pointer
     * @param str String to add
     * @return Pointer to null-terminated C string
     */
    const char* add(const std::string& str) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto cstr = std::make_unique<char[]>(str.size() + 1);
        std::strcpy(cstr.get(), str.c_str());
        const char* ptr = cstr.get();
        strings_.push_back(std::move(cstr));
        return ptr;
    }

    /**
     * @brief Clear all strings from the pool
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        strings_.clear();
    }

    /**
     * @brief Get current pool size
     * @return Number of strings in pool
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return strings_.size();
    }
};

/* ================================================================================================
 * CONFIGURATION CLASS
 * ================================================================================================ */

/**
 * @brief Configuration builder class with fluent interface
 *
 * This class provides a type-safe, fluent interface for building
 * RAG system configuration with sensible defaults.
 */
class Config {
private:
    RagConfig config_;
    mutable StringPool string_pool_;

public:
    /**
     * @brief Construct configuration with default values
     */
    Config() {
        rag_config_default(&config_);
    }

    // Redis configuration methods
    Config& redis_url(const std::string& url) {
        config_.redis_url = string_pool_.add(url);
        return *this;
    }

    Config& redis_pool_size(unsigned int size) {
        config_.redis_pool_size = size;
        return *this;
    }

    Config& redis_connection_timeout(Duration timeout) {
        config_.redis_connection_timeout_secs = static_cast<unsigned int>(timeout.count());
        return *this;
    }

    Config& redis_command_timeout(Duration timeout) {
        config_.redis_command_timeout_secs = static_cast<unsigned int>(timeout.count());
        return *this;
    }

    Config& redis_max_retries(unsigned int retries) {
        config_.redis_max_retries = retries;
        return *this;
    }

    Config& redis_retry_delay(std::chrono::milliseconds delay) {
        config_.redis_retry_delay_ms = static_cast<unsigned int>(delay.count());
        return *this;
    }

    Config& redis_enable_cluster(bool enable = true) {
        config_.redis_enable_cluster = enable ? 1 : 0;
        return *this;
    }

    // Vector store configuration methods
    Config& vector_dimension(unsigned int dimension) {
        config_.vector_dimension = dimension;
        return *this;
    }

    Config& vector_max_elements(unsigned int max_elements) {
        config_.vector_max_elements = max_elements;
        return *this;
    }

    Config& vector_m(unsigned int m) {
        config_.vector_m = m;
        return *this;
    }

    Config& vector_ef_construction(unsigned int ef_construction) {
        config_.vector_ef_construction = ef_construction;
        return *this;
    }

    Config& vector_ef_search(unsigned int ef_search) {
        config_.vector_ef_search = ef_search;
        return *this;
    }

    Config& vector_similarity_threshold(float threshold) {
        config_.vector_similarity_threshold = threshold;
        return *this;
    }

    // Document processing configuration methods
    Config& document_chunk_size(unsigned int size) {
        config_.doc_chunk_size = size;
        return *this;
    }

    Config& document_chunk_overlap(unsigned int overlap) {
        config_.doc_chunk_overlap = overlap;
        return *this;
    }

    Config& document_max_chunk_size(unsigned int max_size) {
        config_.doc_max_chunk_size = max_size;
        return *this;
    }

    Config& document_min_chunk_size(unsigned int min_size) {
        config_.doc_min_chunk_size = min_size;
        return *this;
    }

    Config& document_enable_metadata_extraction(bool enable = true) {
        config_.doc_enable_metadata_extraction = enable ? 1 : 0;
        return *this;
    }

    // Memory management configuration methods
    Config& memory_max_mb(unsigned int max_mb) {
        config_.memory_max_mb = max_mb;
        return *this;
    }

    Config& memory_ttl(Duration ttl) {
        config_.memory_ttl_seconds = static_cast<unsigned int>(ttl.count());
        return *this;
    }

    Config& memory_cleanup_interval(Duration interval) {
        config_.memory_cleanup_interval_secs = static_cast<unsigned int>(interval.count());
        return *this;
    }

    Config& memory_max_items(unsigned int max_items) {
        config_.memory_max_items = max_items;
        return *this;
    }

    // Research configuration methods
    Config& research_enable_web_search(bool enable = true) {
        config_.research_enable_web_search = enable ? 1 : 0;
        return *this;
    }

    Config& research_max_results(unsigned int max_results) {
        config_.research_max_results = max_results;
        return *this;
    }

    Config& research_timeout(Duration timeout) {
        config_.research_timeout_secs = static_cast<unsigned int>(timeout.count());
        return *this;
    }

    Config& research_rate_limit_per_second(unsigned int rate_limit) {
        config_.research_rate_limit_per_sec = rate_limit;
        return *this;
    }

    /**
     * @brief Get the underlying C configuration structure
     * @return Reference to C configuration
     */
    const RagConfig& c_config() const { return config_; }

    /**
     * @brief Validate configuration parameters
     * @throws Exception if configuration is invalid
     */
    void validate() const {
        if (!config_.redis_url || std::string(config_.redis_url).empty()) {
            throw Exception(Exception::Config, "Redis URL cannot be empty");
        }
        if (config_.vector_dimension == 0) {
            throw Exception(Exception::Config, "Vector dimension must be greater than 0");
        }
        if (config_.doc_chunk_size < config_.doc_min_chunk_size) {
            throw Exception(Exception::Config, "Document chunk size must be >= min_chunk_size");
        }
        if (config_.doc_chunk_size > config_.doc_max_chunk_size) {
            throw Exception(Exception::Config, "Document chunk size must be <= max_chunk_size");
        }
    }
};

/* ================================================================================================
 * SEARCH RESULT CLASSES
 * ================================================================================================ */

/**
 * @brief Search result wrapper with automatic memory management
 *
 * This class wraps a single search result and provides convenient
 * access to result fields with automatic string conversion.
 */
class SearchResult {
private:
    std::string id_;
    std::string text_;
    float score_;
    Metadata metadata_;
    std::string source_;
    std::optional<std::string> url_;

public:
    /**
     * @brief Construct search result from C API result
     * @param c_result C API search result
     */
    explicit SearchResult(const RagSearchResult& c_result)
        : id_(c_result.id ? c_result.id : "")
        , text_(c_result.text ? c_result.text : "")
        , score_(c_result.score)
        , source_(c_result.source ? c_result.source : "")
    {
        // Parse metadata JSON
        if (c_result.metadata_json) {
            try {
                // Simple JSON parsing for key-value pairs
                // In production, you'd use a proper JSON library
                std::string json(c_result.metadata_json);
                // This is a simplified parser - implement proper JSON parsing
                metadata_["raw_json"] = json;
            } catch (...) {
                // Ignore JSON parsing errors
                metadata_["raw_json"] = c_result.metadata_json;
            }
        }

        // Set URL if present
        if (c_result.url) {
            url_ = std::string(c_result.url);
        }
    }

    // Getters
    const std::string& id() const { return id_; }
    const std::string& text() const { return text_; }
    float score() const { return score_; }
    const Metadata& metadata() const { return metadata_; }
    const std::string& source() const { return source_; }
    const std::optional<std::string>& url() const { return url_; }

    /**
     * @brief Get metadata value by key
     * @param key Metadata key
     * @return Optional metadata value
     */
    std::optional<std::string> metadata(const std::string& key) const {
        auto it = metadata_.find(key);
        return (it != metadata_.end()) ? std::optional<std::string>(it->second) : std::nullopt;
    }

    /**
     * @brief Check if result has URL
     * @return True if URL is present
     */
    bool has_url() const { return url_.has_value(); }

    /**
     * @brief Convert result to string representation
     * @return String representation of result
     */
    std::string to_string() const {
        std::ostringstream oss;
        oss << "SearchResult{id='" << id_ << "', score=" << score_
            << ", source='" << source_ << "', text='" << text_.substr(0, 50);
        if (text_.length() > 50) oss << "...";
        oss << "'}";
        return oss.str();
    }
};

/**
 * @brief Search results container with RAII memory management
 *
 * This class manages a collection of search results with automatic
 * cleanup and provides STL-compatible iteration.
 */
class SearchResults {
private:
    std::vector<SearchResult> results_;

public:
    /**
     * @brief Construct from C API search results
     * @param c_results C API search results (takes ownership)
     */
    explicit SearchResults(RagSearchResults* c_results) {
        if (c_results) {
            results_.reserve(c_results->count);
            for (unsigned int i = 0; i < c_results->count; ++i) {
                results_.emplace_back(c_results->results[i]);
            }
            rag_free_search_results(c_results);
        }
    }

    /**
     * @brief Move constructor
     */
    SearchResults(SearchResults&& other) noexcept = default;

    /**
     * @brief Move assignment operator
     */
    SearchResults& operator=(SearchResults&& other) noexcept = default;

    // Disable copying to prevent double-free
    SearchResults(const SearchResults&) = delete;
    SearchResults& operator=(const SearchResults&) = delete;

    // STL-compatible interface
    using iterator = std::vector<SearchResult>::iterator;
    using const_iterator = std::vector<SearchResult>::const_iterator;

    iterator begin() { return results_.begin(); }
    iterator end() { return results_.end(); }
    const_iterator begin() const { return results_.begin(); }
    const_iterator end() const { return results_.end(); }
    const_iterator cbegin() const { return results_.cbegin(); }
    const_iterator cend() const { return results_.cend(); }

    bool empty() const { return results_.empty(); }
    size_t size() const { return results_.size(); }
    const SearchResult& operator[](size_t index) const { return results_[index]; }
    const SearchResult& at(size_t index) const { return results_.at(index); }

    /**
     * @brief Get results as vector
     * @return Vector of search results
     */
    const std::vector<SearchResult>& as_vector() const { return results_; }

    /**
     * @brief Filter results by minimum score
     * @param min_score Minimum score threshold
     * @return New SearchResults containing filtered results
     */
    SearchResults filter_by_score(float min_score) const {
        SearchResults filtered_results(nullptr);
        for (const auto& result : results_) {
            if (result.score() >= min_score) {
                filtered_results.results_.push_back(result);
            }
        }
        return filtered_results;
    }

    /**
     * @brief Filter results by source
     * @param source Source identifier to filter by
     * @return New SearchResults containing filtered results
     */
    SearchResults filter_by_source(const std::string& source) const {
        SearchResults filtered_results(nullptr);
        for (const auto& result : results_) {
            if (result.source() == source) {
                filtered_results.results_.push_back(result);
            }
        }
        return filtered_results;
    }

    /**
     * @brief Get top N results
     * @param n Number of top results to return
     * @return New SearchResults containing top N results
     */
    SearchResults top(size_t n) const {
        SearchResults top_results(nullptr);
        size_t limit = std::min(n, results_.size());
        top_results.results_.reserve(limit);
        for (size_t i = 0; i < limit; ++i) {
            top_results.results_.push_back(results_[i]);
        }
        return top_results;
    }
};

/* ================================================================================================
 * SYSTEM CLASS
 * ================================================================================================ */

/**
 * @brief Main RAG system class with RAII resource management
 *
 * This class provides the main interface to the RAG system with
 * automatic resource management, exception safety, and modern C++ features.
 */
class System {
private:
    RagHandle handle_;
    mutable StringPool string_pool_;

    /**
     * @brief Convert metadata map to JSON string
     * @param metadata Metadata map
     * @return JSON string representation
     */
    std::string metadata_to_json(const Metadata& metadata) const {
        if (metadata.empty()) {
            return "{}";
        }

        std::ostringstream oss;
        oss << "{";
        bool first = true;
        for (const auto& [key, value] : metadata) {
            if (!first) oss << ",";
            oss << "\"" << key << "\":\"" << value << "\"";
            first = false;
        }
        oss << "}";
        return oss.str();
    }

    /**
     * @brief Check for errors and throw exception if present
     * @param error_info Error information from C API
     */
    void check_error(const RagErrorInfo& error_info) const {
        if (error_info.code != RAG_ERROR_SUCCESS) {
            Exception ex(error_info);
            if (error_info.message) {
                rag_free_error_message(error_info.message);
            }
            throw ex;
        }
        if (error_info.message) {
            rag_free_error_message(error_info.message);
        }
    }

public:
    /**
     * @brief Construct system with configuration
     * @param config System configuration
     * @throws Exception on initialization failure
     */
    explicit System(const Config& config) : handle_(0) {
        config.validate();

        RagErrorInfo error_info = {0};
        handle_ = rag_create(&config.c_config(), &error_info);

        if (handle_ == 0) {
            check_error(error_info);
            throw Exception(Exception::Unknown, "Failed to create RAG system");
        }
    }

    /**
     * @brief Move constructor
     */
    System(System&& other) noexcept : handle_(other.handle_) {
        other.handle_ = 0;
    }

    /**
     * @brief Move assignment operator
     */
    System& operator=(System&& other) noexcept {
        if (this != &other) {
            if (handle_ != 0) {
                rag_destroy(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = 0;
        }
        return *this;
    }

    // Disable copying
    System(const System&) = delete;
    System& operator=(const System&) = delete;

    /**
     * @brief Destructor with automatic cleanup
     */
    ~System() {
        if (handle_ != 0) {
            rag_destroy(handle_);
        }
    }

    /**
     * @brief Check if system is valid
     * @return True if system is initialized and valid
     */
    bool is_valid() const {
        return handle_ != 0 && rag_health_check(handle_) != 0;
    }

    /**
     * @brief Get system handle (for advanced usage)
     * @return Raw system handle
     */
    RagHandle handle() const { return handle_; }

    /**
     * @brief Ingest document into the system
     * @param content Document content
     * @param metadata Optional metadata
     * @return Document ID
     * @throws Exception on ingestion failure
     */
    std::string ingest_document(const std::string& content, const Metadata& metadata = {}) {
        if (handle_ == 0) {
            throw Exception(Exception::InvalidInput, "System not initialized");
        }

        std::string metadata_json = metadata_to_json(metadata);
        const char* metadata_ptr = metadata_json.empty() ? nullptr : string_pool_.add(metadata_json);

        char* doc_id_ptr = nullptr;
        RagErrorInfo error_info = {0};

        int result = rag_ingest_document(
            handle_,
            string_pool_.add(content),
            metadata_ptr,
            &doc_id_ptr,
            &error_info
        );

        if (result == 0) {
            check_error(error_info);
            throw Exception(Exception::Document, "Failed to ingest document");
        }

        std::string doc_id;
        if (doc_id_ptr) {
            doc_id = std::string(doc_id_ptr);
            rag_free_string(doc_id_ptr);
        }

        return doc_id;
    }

    /**
     * @brief Search for documents
     * @param query Search query
     * @param limit Maximum number of results
     * @return Search results
     * @throws Exception on search failure
     */
    SearchResults search(const std::string& query, unsigned int limit = 10) {
        if (handle_ == 0) {
            throw Exception(Exception::InvalidInput, "System not initialized");
        }

        RagErrorInfo error_info = {0};
        RagSearchResults* c_results = rag_search(
            handle_,
            string_pool_.add(query),
            limit,
            &error_info
        );

        if (!c_results) {
            check_error(error_info);
            throw Exception(Exception::Vector, "Search failed");
        }

        return SearchResults(c_results);
    }

    /**
     * @brief Perform research with web search
     * @param query Research query
     * @param sources Source list for web search
     * @return Research results
     * @throws Exception on research failure
     */
    SearchResults research(const std::string& query, const Sources& sources = {}) {
        if (handle_ == 0) {
            throw Exception(Exception::InvalidInput, "System not initialized");
        }

        // Prepare sources array
        std::vector<const char*> source_ptrs;
        for (const auto& source : sources) {
            source_ptrs.push_back(string_pool_.add(source));
        }

        RagErrorInfo error_info = {0};
        RagSearchResults* c_results = rag_research(
            handle_,
            string_pool_.add(query),
            source_ptrs.empty() ? nullptr : source_ptrs.data(),
            static_cast<unsigned int>(source_ptrs.size()),
            &error_info
        );

        if (!c_results) {
            check_error(error_info);
            throw Exception(Exception::Network, "Research failed");
        }

        return SearchResults(c_results);
    }

    /**
     * @brief Get system statistics
     * @return Statistics as JSON string
     * @throws Exception on failure
     */
    std::string get_stats() {
        if (handle_ == 0) {
            throw Exception(Exception::InvalidInput, "System not initialized");
        }

        RagErrorInfo error_info = {0};
        char* stats_ptr = rag_get_stats(handle_, &error_info);

        if (!stats_ptr) {
            check_error(error_info);
            throw Exception(Exception::Unknown, "Failed to get statistics");
        }

        std::string stats(stats_ptr);
        rag_free_string(stats_ptr);
        return stats;
    }

    /**
     * @brief Perform health check
     * @return True if system is healthy
     */
    bool health_check() const {
        return handle_ != 0 && rag_health_check(handle_) != 0;
    }

    /**
     * @brief Asynchronous search operation
     * @param query Search query
     * @param limit Maximum number of results
     * @return Future containing search results
     */
    std::future<SearchResults> search_async(const std::string& query, unsigned int limit = 10) {
        return std::async(std::launch::async, [this, query, limit]() {
            return search(query, limit);
        });
    }

    /**
     * @brief Asynchronous document ingestion
     * @param content Document content
     * @param metadata Optional metadata
     * @return Future containing document ID
     */
    std::future<std::string> ingest_document_async(const std::string& content, const Metadata& metadata = {}) {
        return std::async(std::launch::async, [this, content, metadata]() {
            return ingest_document(content, metadata);
        });
    }
};

/* ================================================================================================
 * LIBRARY MANAGEMENT CLASS
 * ================================================================================================ */

/**
 * @brief RAII library initialization and cleanup
 *
 * This class ensures proper initialization and cleanup of the
 * RAG Redis System library using RAII principles.
 */
class Library {
private:
    static std::atomic<int> instance_count_;
    static std::mutex init_mutex_;

public:
    /**
     * @brief Initialize library
     * @throws Exception on initialization failure
     */
    Library() {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (instance_count_.fetch_add(1) == 0) {
            if (rag_init() == 0) {
                instance_count_.fetch_sub(1);
                throw Exception(Exception::Unknown, "Failed to initialize RAG library");
            }
        }
    }

    /**
     * @brief Cleanup library resources
     */
    ~Library() {
        std::lock_guard<std::mutex> lock(init_mutex_);
        if (instance_count_.fetch_sub(1) == 1) {
            rag_cleanup();
        }
    }

    // Disable copying and moving
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
    Library(Library&&) = delete;
    Library& operator=(Library&&) = delete;

    /**
     * @brief Get library version
     * @return Version string
     */
    static std::string version() {
        return std::string(rag_version());
    }
};

// Static member definitions
std::atomic<int> Library::instance_count_{0};
std::mutex Library::init_mutex_;

/* ================================================================================================
 * UTILITY FUNCTIONS
 * ================================================================================================ */

/**
 * @brief Create default configuration
 * @return Default configuration instance
 */
inline Config create_default_config() {
    return Config();
}

/**
 * @brief Create configuration for local development
 * @param redis_url Optional Redis URL (defaults to localhost)
 * @return Development configuration
 */
inline Config create_dev_config(const std::string& redis_url = "redis://127.0.0.1:6379") {
    return Config()
        .redis_url(redis_url)
        .redis_pool_size(5)
        .vector_dimension(384)  // Smaller dimension for development
        .vector_max_elements(10000)
        .document_chunk_size(256)
        .memory_max_mb(512)
        .research_enable_web_search(false);  // Disable web search in dev
}

/**
 * @brief Create configuration for production use
 * @param redis_url Redis URL
 * @return Production configuration
 */
inline Config create_production_config(const std::string& redis_url) {
    return Config()
        .redis_url(redis_url)
        .redis_pool_size(20)
        .redis_connection_timeout(Duration(10))
        .redis_command_timeout(Duration(30))
        .vector_dimension(768)
        .vector_max_elements(1000000)
        .document_chunk_size(512)
        .memory_max_mb(4096)
        .research_enable_web_search(true);
}

} // namespace rag

/* ================================================================================================
 * IOSTREAM SUPPORT
 * ================================================================================================ */

/**
 * @brief Stream output operator for SearchResult
 */
inline std::ostream& operator<<(std::ostream& os, const rag::SearchResult& result) {
    return os << result.to_string();
}

/**
 * @brief Stream output operator for SearchResults
 */
inline std::ostream& operator<<(std::ostream& os, const rag::SearchResults& results) {
    os << "SearchResults{count=" << results.size() << ", results=[";
    bool first = true;
    for (const auto& result : results) {
        if (!first) os << ", ";
        os << result;
        first = false;
    }
    os << "]}";
    return os;
}

/**
 * @brief Stream output operator for Exception
 */
inline std::ostream& operator<<(std::ostream& os, const rag::Exception& ex) {
    return os << "RAG Exception [" << ex.code_string() << "]: " << ex.what();
}

#endif /* RAG_REDIS_HPP */
