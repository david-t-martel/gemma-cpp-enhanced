#pragma once

#include "Session.h"
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <optional>
#include <chrono>
#include <list>
#include <nlohmann/json.hpp>

// Forward declare SQLite types to avoid including sqlite3.h in header
struct sqlite3;
struct sqlite3_stmt;

namespace gemma {
namespace session {

/**
 * @brief LRU (Least Recently Used) cache for in-memory session storage
 * 
 * This class provides efficient O(1) access and eviction for session caching.
 */
class LRUCache {
public:
    /**
     * @brief Construct a new LRUCache object
     * 
     * @param capacity Maximum number of items to cache
     */
    explicit LRUCache(size_t capacity);
    
    /**
     * @brief Get a session from the cache
     * 
     * @param session_id Session identifier
     * @return std::shared_ptr<Session> Session pointer or nullptr if not found
     */
    std::shared_ptr<Session> get(const std::string& session_id);
    
    /**
     * @brief Put a session into the cache
     * 
     * @param session_id Session identifier
     * @param session Session to cache
     */
    void put(const std::string& session_id, std::shared_ptr<Session> session);
    
    /**
     * @brief Remove a session from the cache
     * 
     * @param session_id Session identifier
     */
    void remove(const std::string& session_id);
    
    /**
     * @brief Clear all sessions from the cache
     */
    void clear();
    
    /**
     * @brief Get the current number of cached sessions
     * 
     * @return size_t Number of cached sessions
     */
    size_t size() const;
    
    /**
     * @brief Get the cache capacity
     * 
     * @return size_t Maximum cache capacity
     */
    size_t capacity() const;

private:
    struct CacheNode {
        std::string session_id;
        std::shared_ptr<Session> session;
    };
    
    size_t capacity_;
    std::list<CacheNode> cache_list_;
    std::unordered_map<std::string, std::list<CacheNode>::iterator> cache_map_;
    mutable std::mutex mutex_;
    
    /**
     * @brief Evict the least recently used item
     */
    void evict();
};

/**
 * @brief SessionStorage class providing persistent storage for sessions
 * 
 * This class manages the persistence layer for sessions using SQLite as the backend.
 * It provides in-memory caching with LRU eviction, JSON export/import capabilities,
 * and automatic cleanup of expired sessions.
 */
class SessionStorage {
public:
    /**
     * @brief Storage configuration options
     */
    struct Config {
        std::string db_path = "sessions.db";           // SQLite database file path
        size_t cache_capacity = 100;                   // LRU cache capacity
        std::chrono::hours session_ttl{24};            // Session time-to-live
        bool enable_auto_cleanup = true;               // Enable automatic cleanup
        std::chrono::minutes cleanup_interval{60};     // Cleanup interval
    };
    
    /**
     * @brief Construct a new SessionStorage object
     * 
     * @param config Storage configuration
     */
    explicit SessionStorage(const Config& config = Config{});
    
    /**
     * @brief Destroy the SessionStorage object
     */
    ~SessionStorage();
    
    // Disable copy constructor and assignment
    SessionStorage(const SessionStorage&) = delete;
    SessionStorage& operator=(const SessionStorage&) = delete;
    
    // Enable move constructor and assignment
    SessionStorage(SessionStorage&&) = default;
    SessionStorage& operator=(SessionStorage&&) = default;
    
    /**
     * @brief Initialize the storage backend
     * 
     * @return bool True if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Save a session to persistent storage
     * 
     * @param session Session to save
     * @return bool True if save successful
     */
    bool save_session(std::shared_ptr<Session> session);
    
    /**
     * @brief Load a session from storage
     * 
     * @param session_id Session identifier
     * @return std::shared_ptr<Session> Session pointer or nullptr if not found
     */
    std::shared_ptr<Session> load_session(const std::string& session_id);
    
    /**
     * @brief Delete a session from storage
     * 
     * @param session_id Session identifier
     * @return bool True if deletion successful
     */
    bool delete_session(const std::string& session_id);
    
    /**
     * @brief Check if a session exists in storage
     * 
     * @param session_id Session identifier
     * @return bool True if session exists
     */
    bool session_exists(const std::string& session_id);
    
    /**
     * @brief Get metadata for all sessions
     * 
     * @return std::vector<nlohmann::json> List of session metadata
     */
    std::vector<nlohmann::json> list_sessions();
    
    /**
     * @brief Get metadata for sessions with optional filtering
     * 
     * @param limit Maximum number of sessions to return (0 = no limit)
     * @param offset Number of sessions to skip
     * @param sort_by Field to sort by ("created_at", "last_activity", "session_id")
     * @param ascending Sort direction
     * @return std::vector<nlohmann::json> List of session metadata
     */
    std::vector<nlohmann::json> list_sessions(size_t limit, size_t offset, 
                                             const std::string& sort_by = "last_activity", 
                                             bool ascending = false);
    
    /**
     * @brief Export all sessions to JSON
     * 
     * @param file_path Output file path
     * @return bool True if export successful
     */
    bool export_to_json(const std::string& file_path);
    
    /**
     * @brief Import sessions from JSON
     * 
     * @param file_path Input file path
     * @param overwrite_existing Whether to overwrite existing sessions
     * @return bool True if import successful
     */
    bool import_from_json(const std::string& file_path, bool overwrite_existing = false);
    
    /**
     * @brief Clean up expired sessions
     * 
     * @return size_t Number of sessions cleaned up
     */
    size_t cleanup_expired_sessions();
    
    /**
     * @brief Get storage statistics
     * 
     * @return nlohmann::json Storage statistics
     */
    nlohmann::json get_statistics();
    
    /**
     * @brief Close the storage backend
     */
    void close();
    
    /**
     * @brief Get the current configuration
     * 
     * @return const Config& Current configuration
     */
    const Config& get_config() const;

private:
    Config config_;
    sqlite3* db_;
    std::unique_ptr<LRUCache> cache_;
    mutable std::mutex db_mutex_;
    bool initialized_;
    std::chrono::system_clock::time_point last_cleanup_;
    
    /**
     * @brief Initialize the SQLite database schema
     * 
     * @return bool True if initialization successful
     */
    bool init_database();
    
    /**
     * @brief Prepare SQL statements
     * 
     * @return bool True if preparation successful
     */
    bool prepare_statements();
    
    /**
     * @brief Execute automatic cleanup if needed
     */
    void maybe_cleanup();
    
    /**
     * @brief Convert SQLite row to session metadata JSON
     * 
     * @param stmt SQLite statement
     * @return nlohmann::json Session metadata
     */
    nlohmann::json sqlite_row_to_metadata(sqlite3_stmt* stmt);
    
    /**
     * @brief Get the current timestamp as SQLite-compatible integer
     * 
     * @return int64_t Timestamp in milliseconds since epoch
     */
    int64_t current_timestamp();
    
    /**
     * @brief Check if a timestamp is expired
     * 
     * @param timestamp Timestamp to check
     * @return bool True if expired
     */
    bool is_expired(int64_t timestamp);
};

} // namespace session
} // namespace gemma