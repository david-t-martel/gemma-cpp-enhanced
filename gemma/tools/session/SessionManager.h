#pragma once

#include "Session.h"
#include "i_session_storage.h"
#include <string>
#include <memory>
#include <mutex>
#include <random>
#include <chrono>
#include <vector>
#include <functional>
// Avoid pulling heavy JSON dependency into public header; use opaque strings.

namespace gemma {
namespace session {

/**
 * @brief SessionManager class providing high-level session management
 * 
 * This class provides a thread-safe interface for managing conversation sessions.
 * It handles session creation with UUID generation, automatic cleanup, and
 * provides various convenience methods for session operations.
 */
class SessionManager {
public:
    /**
     * @brief Session manager configuration
     */
    struct Config {
        // Keep original concrete config temporarily; will be decoupled in later phase
        // Forward use SessionStorage::Config if concrete storage present
        // (We declare a minimal placeholder here to avoid hard dependency loop.)
        struct StorageConfigShim {
            std::string db_path = "sessions.db";
            size_t cache_capacity = 100;
            std::chrono::hours session_ttl{24};
            bool enable_auto_cleanup = true;
            std::chrono::minutes cleanup_interval{60};
        } storage_config;                         // Temporary shim
        size_t default_max_context_tokens = 8192;  // Default context window size
        bool enable_metrics = true;                // Enable performance metrics
        std::chrono::minutes metrics_interval{5};  // Metrics collection interval
    };
    
    /**
     * @brief Session creation options
     */
    struct CreateOptions {
        std::string session_id;              // Custom session ID (empty = auto-generate)
        size_t max_context_tokens = 0;       // Context window size (0 = use default)
        std::string metadata_json;           // Raw JSON string (deferred parse) to avoid header dep
    };
    
    /**
     * @brief Session metrics
     */
    struct Metrics {
        size_t total_sessions_created = 0;
        size_t total_sessions_deleted = 0;
        size_t total_messages_processed = 0;
        size_t total_tokens_processed = 0;
        std::chrono::system_clock::time_point last_reset;
        double avg_session_duration_minutes = 0.0;
        double avg_tokens_per_session = 0.0;
        double avg_messages_per_session = 0.0;
    };
    
    /**
     * @brief Construct a new SessionManager object
     * 
     * @param config Manager configuration
     */
    explicit SessionManager(const Config& config = Config{});
    
    /**
     * @brief Destroy the SessionManager object
     */
    ~SessionManager();
    
    // Disable copy constructor and assignment
    SessionManager(const SessionManager&) = delete;
    SessionManager& operator=(const SessionManager&) = delete;
    
    // Enable move constructor and assignment
    // Non-movable for now (mutex & pointers); can be revisited after full interface DI.
    SessionManager(SessionManager&&) = delete;
    SessionManager& operator=(SessionManager&&) = delete;
    
    /**
     * @brief Initialize the session manager
     * 
     * @return bool True if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Create a new session
     * 
     * @param options Session creation options
     * @return std::string Session ID of the created session
     */
    std::string create_session(const CreateOptions& options);
    std::string create_session(); // convenience overload using defaults
    
    /**
     * @brief Get an existing session
     * 
     * @param session_id Session identifier
     * @return std::shared_ptr<Session> Session pointer or nullptr if not found
     */
    std::shared_ptr<Session> get_session(const std::string& session_id);
    
    /**
     * @brief Delete a session
     * 
     * @param session_id Session identifier
     * @return bool True if deletion successful
     */
    bool delete_session(const std::string& session_id);
    
    /**
     * @brief Check if a session exists
     * 
     * @param session_id Session identifier
     * @return bool True if session exists
     */
    bool session_exists(const std::string& session_id);
    
    /**
     * @brief Add a message to a session
     * 
     * @param session_id Session identifier
     * @param role Message role
     * @param content Message content
     * @param token_count Number of tokens in the message
     * @return bool True if message was added successfully
     */
    bool add_message(const std::string& session_id, 
                    ConversationMessage::Role role, 
                    const std::string& content, 
                    size_t token_count);
    
    /**
     * @brief Get conversation history for a session
     * 
     * @param session_id Session identifier
     * @return std::vector<ConversationMessage> Conversation history
     */
    std::vector<ConversationMessage> get_conversation_history(const std::string& session_id);
    
    /**
     * @brief Get context messages for a session
     * 
     * @param session_id Session identifier
     * @return std::vector<ConversationMessage> Messages within context window
     */
    std::vector<ConversationMessage> get_context_messages(const std::string& session_id);
    
    /**
     * @brief Clear conversation history for a session
     * 
     * @param session_id Session identifier
     * @return bool True if history was cleared successfully
     */
    bool clear_session_history(const std::string& session_id);
    
    /**
     * @brief Update session context window size
     * 
     * @param session_id Session identifier
     * @param max_context_tokens New context window size
     * @return bool True if update successful
     */
    bool update_session_context_size(const std::string& session_id, size_t max_context_tokens);
    
    /**
     * @brief List all sessions with optional filtering
     * 
     * @param limit Maximum number of sessions to return (0 = no limit)
     * @param offset Number of sessions to skip
     * @param sort_by Field to sort by
     * @param ascending Sort direction
     * @return std::vector<nlohmann::json> List of session metadata
     */
    std::vector<std::string> list_sessions(size_t limit = 0, size_t offset = 0,
                                           const std::string& sort_by = "last_activity",
                                           bool ascending = false);
    
    /**
     * @brief Export sessions to JSON file
     * 
     * @param file_path Output file path
     * @param session_ids Specific session IDs to export (empty = all sessions)
     * @return bool True if export successful
     */
    bool export_sessions(const std::string& file_path, 
                        const std::vector<std::string>& session_ids = {});
    
    /**
     * @brief Import sessions from JSON file
     * 
     * @param file_path Input file path
     * @param overwrite_existing Whether to overwrite existing sessions
     * @return size_t Number of sessions imported
     */
    size_t import_sessions(const std::string& file_path, bool overwrite_existing = false);
    
    /**
     * @brief Clean up expired sessions
     * 
     * @return size_t Number of sessions cleaned up
     */
    size_t cleanup_expired_sessions();
    
    /**
     * @brief Get session manager statistics
     * 
     * @return nlohmann::json Statistics including storage and performance metrics
     */
    std::string get_statistics(); // JSON string
    
    /**
     * @brief Get current performance metrics
     * 
     * @return Metrics Current metrics
     */
    Metrics get_metrics() const;
    
    /**
     * @brief Reset performance metrics
     */
    void reset_metrics();
    
    /**
     * @brief Set a callback for session events
     * 
     * @param callback Function to call on session events
     */
    void set_event_callback(std::function<void(const std::string& event, const std::string& data_json)> callback);
    
    /**
     * @brief Get the current configuration
     * 
     * @return const Config& Current configuration
     */
    const Config& get_config() const;
    
    /**
     * @brief Shutdown the session manager
     */
    void shutdown();

private:
    Config config_;
    std::unique_ptr<ISessionStorage> storage_;
    mutable std::mutex manager_mutex_;
    std::mt19937 uuid_generator_;
    bool initialized_;
    Metrics metrics_;
    std::chrono::system_clock::time_point last_metrics_update_;
    std::function<void(const std::string&, const std::string&)> event_callback_;
    
    /**
     * @brief Generate a unique session ID
     * 
     * @return std::string Generated UUID
     */
    std::string generate_session_id();
    
    /**
     * @brief Update performance metrics
     */
    void update_metrics();
    
    /**
     * @brief Fire an event callback if registered
     * 
     * @param event Event name
     * @param data Event data
     */
    void fire_event(const std::string& event, const std::string& data_json);
    
    /**
     * @brief Generate UUID v4 string
     * 
     * @return std::string UUID string
     */
    std::string generate_uuid();
    
    /**
     * @brief Validate session ID format
     * 
     * @param session_id Session ID to validate
     * @return bool True if valid
     */
    bool is_valid_session_id(const std::string& session_id);
};

} // namespace session
} // namespace gemma