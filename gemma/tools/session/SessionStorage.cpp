#include "SessionStorage.h"
#include <sqlite3.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace gemma {
namespace session {

// LRUCache implementation
LRUCache::LRUCache(size_t capacity) : capacity_(capacity) {
    if (capacity_ == 0) {
        throw std::invalid_argument("Cache capacity must be greater than 0");
    }
}

std::shared_ptr<Session> LRUCache::get(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_map_.find(session_id);
    if (it == cache_map_.end()) {
        return nullptr;
    }
    
    // Move accessed item to front
    auto list_it = it->second;
    cache_list_.splice(cache_list_.begin(), cache_list_, list_it);
    
    return list_it->session;
}

void LRUCache::put(const std::string& session_id, std::shared_ptr<Session> session) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_map_.find(session_id);
    if (it != cache_map_.end()) {
        // Update existing entry
        auto list_it = it->second;
        list_it->session = session;
        cache_list_.splice(cache_list_.begin(), cache_list_, list_it);
        return;
    }
    
    // Add new entry
    if (cache_list_.size() >= capacity_) {
        evict();
    }
    
    cache_list_.emplace_front(CacheNode{session_id, session});
    cache_map_[session_id] = cache_list_.begin();
}

void LRUCache::remove(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cache_map_.find(session_id);
    if (it != cache_map_.end()) {
        cache_list_.erase(it->second);
        cache_map_.erase(it);
    }
}

void LRUCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_list_.clear();
    cache_map_.clear();
}

size_t LRUCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_list_.size();
}

size_t LRUCache::capacity() const {
    return capacity_;
}

void LRUCache::evict() {
    if (!cache_list_.empty()) {
        auto& last = cache_list_.back();
        cache_map_.erase(last.session_id);
        cache_list_.pop_back();
    }
}

// SessionStorage implementation
SessionStorage::SessionStorage(const Config& config)
    : config_(config)
    , db_(nullptr)
    , cache_(std::make_unique<LRUCache>(config_.cache_capacity))
    , initialized_(false)
    , last_cleanup_(std::chrono::system_clock::now()) {
}

SessionStorage::~SessionStorage() {
    close();
}

bool SessionStorage::initialize() {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    if (initialized_) {
        return true;
    }
    
    if (!init_database()) {
        return false;
    }
    
    if (!prepare_statements()) {
        return false;
    }
    
    initialized_ = true;
    return true;
}

bool SessionStorage::save_session(std::shared_ptr<Session> session) {
    if (!session) {
        return false;
    }
    
    maybe_cleanup();
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    if (!initialized_) {
        return false;
    }
    
    const char* sql = R"(
        INSERT OR REPLACE INTO sessions (
            session_id, session_data, created_at, last_activity, total_tokens
        ) VALUES (?, ?, ?, ?, ?)
    )";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }
    
    auto json_data = session->to_json();
    std::string json_str = json_data.dump();
    
    auto created_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        session->get_created_at().time_since_epoch()).count();
    auto activity_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        session->get_last_activity().time_since_epoch()).count();
    
    sqlite3_bind_text(stmt, 1, session->get_session_id().c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, json_str.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 3, created_ms);
    sqlite3_bind_int64(stmt, 4, activity_ms);
    sqlite3_bind_int64(stmt, 5, static_cast<int64_t>(session->get_total_tokens()));
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc == SQLITE_DONE) {
        // Update cache
        cache_->put(session->get_session_id(), session);
        return true;
    }
    
    return false;
}

std::shared_ptr<Session> SessionStorage::load_session(const std::string& session_id) {
    // Check cache first
    auto cached_session = cache_->get(session_id);
    if (cached_session) {
        cached_session->touch();
        return cached_session;
    }
    
    maybe_cleanup();
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    if (!initialized_) {
        return nullptr;
    }
    
    const char* sql = "SELECT session_data FROM sessions WHERE session_id = ? AND NOT ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return nullptr;
    }
    
    sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_STATIC);
    
    auto current_time = current_timestamp();
    auto expiry_time = current_time - std::chrono::duration_cast<std::chrono::milliseconds>(config_.session_ttl).count();
    sqlite3_bind_int(stmt, 2, static_cast<int>(current_time < expiry_time));
    
    std::shared_ptr<Session> session = nullptr;
    
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        const char* json_data = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (json_data) {
            try {
                auto json = nlohmann::json::parse(json_data);
                session = std::make_shared<Session>(json);
                session->touch();
                
                // Add to cache
                cache_->put(session_id, session);
            } catch (const std::exception& e) {
                // Invalid JSON data, return nullptr
                session = nullptr;
            }
        }
    }
    
    sqlite3_finalize(stmt);
    return session;
}

bool SessionStorage::delete_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    if (!initialized_) {
        return false;
    }
    
    const char* sql = "DELETE FROM sessions WHERE session_id = ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }
    
    sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc == SQLITE_DONE) {
        // Remove from cache
        cache_->remove(session_id);
        return true;
    }
    
    return false;
}

bool SessionStorage::session_exists(const std::string& session_id) {
    // Check cache first
    if (cache_->get(session_id)) {
        return true;
    }
    
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    if (!initialized_) {
        return false;
    }
    
    const char* sql = "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }
    
    sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_STATIC);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    return rc == SQLITE_ROW;
}

std::vector<nlohmann::json> SessionStorage::list_sessions() {
    return list_sessions(0, 0, "last_activity", false);
}

std::vector<nlohmann::json> SessionStorage::list_sessions(size_t limit, size_t offset, 
                                                         const std::string& sort_by, 
                                                         bool ascending) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    std::vector<nlohmann::json> sessions;
    
    if (!initialized_) {
        return sessions;
    }
    
    std::string sql = "SELECT session_id, created_at, last_activity, total_tokens FROM sessions";
    
    // Add ORDER BY clause
    if (sort_by == "created_at" || sort_by == "last_activity" || sort_by == "session_id" || sort_by == "total_tokens") {
        sql += " ORDER BY " + sort_by + (ascending ? " ASC" : " DESC");
    } else {
        sql += " ORDER BY last_activity DESC";
    }
    
    // Add LIMIT and OFFSET
    if (limit > 0) {
        sql += " LIMIT " + std::to_string(limit);
        if (offset > 0) {
            sql += " OFFSET " + std::to_string(offset);
        }
    }
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return sessions;
    }
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        sessions.push_back(sqlite_row_to_metadata(stmt));
    }
    
    sqlite3_finalize(stmt);
    return sessions;
}

bool SessionStorage::export_to_json(const std::string& file_path) {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    if (!initialized_) {
        return false;
    }
    
    const char* sql = "SELECT session_data FROM sessions";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return false;
    }
    
    nlohmann::json export_data = nlohmann::json::array();
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const char* json_data = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (json_data) {
            try {
                auto session_json = nlohmann::json::parse(json_data);
                export_data.push_back(session_json);
            } catch (const std::exception& e) {
                // Skip invalid JSON entries
                continue;
            }
        }
    }
    
    sqlite3_finalize(stmt);
    
    // Write to file
    std::ofstream file(file_path);
    if (!file.is_open()) {
        return false;
    }
    
    file << export_data.dump(2);
    return file.good();
}

bool SessionStorage::import_from_json(const std::string& file_path, bool overwrite_existing) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return false;
    }
    
    std::string json_content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    
    try {
        auto import_data = nlohmann::json::parse(json_content);
        
        if (!import_data.is_array()) {
            return false;
        }
        
        size_t imported_count = 0;
        for (const auto& session_json : import_data) {
            try {
                auto session = std::make_shared<Session>(session_json);
                
                if (!overwrite_existing && session_exists(session->get_session_id())) {
                    continue;
                }
                
                if (save_session(session)) {
                    imported_count++;
                }
            } catch (const std::exception& e) {
                // Skip invalid session entries
                continue;
            }
        }
        
        return imported_count > 0;
    } catch (const std::exception& e) {
        return false;
    }
}

size_t SessionStorage::cleanup_expired_sessions() {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    if (!initialized_) {
        return 0;
    }
    
    auto current_time = current_timestamp();
    auto expiry_time = current_time - std::chrono::duration_cast<std::chrono::milliseconds>(config_.session_ttl).count();
    
    const char* sql = "DELETE FROM sessions WHERE last_activity < ?";
    
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        return 0;
    }
    
    sqlite3_bind_int64(stmt, 1, expiry_time);
    
    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    
    if (rc == SQLITE_DONE) {
        last_cleanup_ = std::chrono::system_clock::now();
        return sqlite3_changes(db_);
    }
    
    return 0;
}

nlohmann::json SessionStorage::get_statistics() {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    nlohmann::json stats;
    
    if (!initialized_) {
        return stats;
    }
    
    // Count total sessions
    const char* count_sql = "SELECT COUNT(*) FROM sessions";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db_, count_sql, -1, &stmt, nullptr);
    if (rc == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            stats["total_sessions"] = sqlite3_column_int64(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    
    // Sum total tokens
    const char* tokens_sql = "SELECT SUM(total_tokens) FROM sessions";
    rc = sqlite3_prepare_v2(db_, tokens_sql, -1, &stmt, nullptr);
    if (rc == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            stats["total_tokens"] = sqlite3_column_int64(stmt, 0);
        }
        sqlite3_finalize(stmt);
    }
    
    // Cache statistics
    stats["cache_size"] = cache_->size();
    stats["cache_capacity"] = cache_->capacity();
    
    // Configuration
    stats["session_ttl_hours"] = config_.session_ttl.count();
    stats["auto_cleanup_enabled"] = config_.enable_auto_cleanup;
    
    return stats;
}

void SessionStorage::close() {
    std::lock_guard<std::mutex> lock(db_mutex_);
    
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
    
    if (cache_) {
        cache_->clear();
    }
    
    initialized_ = false;
}

const SessionStorage::Config& SessionStorage::get_config() const {
    return config_;
}

bool SessionStorage::init_database() {
    int rc = sqlite3_open(config_.db_path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        return false;
    }
    
    const char* create_table_sql = R"(
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            session_data TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            last_activity INTEGER NOT NULL,
            total_tokens INTEGER DEFAULT 0
        );
        
        CREATE INDEX IF NOT EXISTS idx_last_activity ON sessions(last_activity);
        CREATE INDEX IF NOT EXISTS idx_created_at ON sessions(created_at);
    )";
    
    char* error_msg = nullptr;
    rc = sqlite3_exec(db_, create_table_sql, nullptr, nullptr, &error_msg);
    
    if (rc != SQLITE_OK) {
        if (error_msg) {
            sqlite3_free(error_msg);
        }
        return false;
    }
    
    return true;
}

bool SessionStorage::prepare_statements() {
    // For now, we prepare statements dynamically in each method
    // This could be optimized by preparing common statements once
    return true;
}

void SessionStorage::maybe_cleanup() {
    if (!config_.enable_auto_cleanup) {
        return;
    }
    
    auto now = std::chrono::system_clock::now();
    if (now - last_cleanup_ >= config_.cleanup_interval) {
        cleanup_expired_sessions();
    }
}

nlohmann::json SessionStorage::sqlite_row_to_metadata(sqlite3_stmt* stmt) {
    return nlohmann::json{
        {"session_id", reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0))},
        {"created_at", sqlite3_column_int64(stmt, 1)},
        {"last_activity", sqlite3_column_int64(stmt, 2)},
        {"total_tokens", sqlite3_column_int64(stmt, 3)}
    };
}

int64_t SessionStorage::current_timestamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

bool SessionStorage::is_expired(int64_t timestamp) {
    auto current_time = current_timestamp();
    auto expiry_time = current_time - std::chrono::duration_cast<std::chrono::milliseconds>(config_.session_ttl).count();
    return timestamp < expiry_time;
}

} // namespace session
} // namespace gemma