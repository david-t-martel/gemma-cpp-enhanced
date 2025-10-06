#include "SessionManager.h"
#include "SessionStorage.h" // concrete implementation (temporary until DI introduced)
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <fstream>
#include <nlohmann/json.hpp>

namespace gemma {
namespace session {

SessionManager::SessionManager(const Config& config)
    : config_(config)
    , storage_([&config]() {
        SessionStorage::Config cfg; // aggregate with defaults
        cfg.db_path = config.storage_config.db_path;
        cfg.cache_capacity = config.storage_config.cache_capacity;
        cfg.session_ttl = config.storage_config.session_ttl;
        cfg.enable_auto_cleanup = config.storage_config.enable_auto_cleanup;
        cfg.cleanup_interval = config.storage_config.cleanup_interval;
        return std::make_unique<SessionStorage>(cfg);
      }())
    , uuid_generator_(std::random_device{}())
    , initialized_(false)
    , last_metrics_update_(std::chrono::system_clock::now()) {
    
    // Initialize metrics
    metrics_.last_reset = std::chrono::system_clock::now();
}

SessionManager::~SessionManager() {
    shutdown();
}

bool SessionManager::initialize() {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    if (initialized_) {
        return true;
    }
    
    if (!storage_->initialize()) {
        return false;
    }
    
    initialized_ = true;
    
    nlohmann::json init_evt{{"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count()}};
    fire_event("manager_initialized", init_evt.dump());
    
    return true;
}

std::string SessionManager::create_session(const CreateOptions& options) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    if (!initialized_) {
        throw std::runtime_error("SessionManager not initialized");
    }
    
    std::string session_id = options.session_id;
    if (session_id.empty()) {
        session_id = generate_session_id();
    } else if (!is_valid_session_id(session_id)) {
        throw std::invalid_argument("Invalid session ID format");
    }
    
    // Check if session already exists
    if (storage_->session_exists(session_id)) {
        throw std::runtime_error("Session with ID '" + session_id + "' already exists");
    }
    
    size_t max_context_tokens = options.max_context_tokens;
    if (max_context_tokens == 0) {
        max_context_tokens = config_.default_max_context_tokens;
    }
    
    auto session = std::make_shared<Session>(session_id, max_context_tokens);
    
    if (!storage_->save_session(session)) {
        throw std::runtime_error("Failed to save session to storage");
    }
    
    // Update metrics
    metrics_.total_sessions_created++;
    update_metrics();
    
    nlohmann::json created_evt{{"session_id", session_id},
                               {"max_context_tokens", max_context_tokens},
                               {"metadata", options.metadata_json},
                               {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::system_clock::now().time_since_epoch()).count()}};
    fire_event("session_created", created_evt.dump());
    
    return session_id;
}

    // Convenience overload using defaults defined in CreateSessionOptions' default constructor
    std::string SessionManager::create_session() {
        return create_session(CreateOptions{});
    }

std::shared_ptr<Session> SessionManager::get_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    if (!initialized_) {
        return nullptr;
    }
    
    if (!is_valid_session_id(session_id)) {
        return nullptr;
    }
    
    auto session = storage_->load_session(session_id);
    if (session) {
        nlohmann::json accessed_evt{{"session_id", session_id},
                                    {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::system_clock::now().time_since_epoch()).count()}};
        fire_event("session_accessed", accessed_evt.dump());
    }
    
    return session;
}

bool SessionManager::delete_session(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    if (!initialized_) {
        return false;
    }
    
    if (!is_valid_session_id(session_id)) {
        return false;
    }
    
    bool success = storage_->delete_session(session_id);
    
    if (success) {
        metrics_.total_sessions_deleted++;
        update_metrics();
        
        nlohmann::json deleted_evt{{"session_id", session_id},
                                   {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                                       std::chrono::system_clock::now().time_since_epoch()).count()}};
        fire_event("session_deleted", deleted_evt.dump());
    }
    
    return success;
}

bool SessionManager::session_exists(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    if (!initialized_) {
        return false;
    }
    
    if (!is_valid_session_id(session_id)) {
        return false;
    }
    
    return storage_->session_exists(session_id);
}

bool SessionManager::add_message(const std::string& session_id, 
                                ConversationMessage::Role role, 
                                const std::string& content, 
                                size_t token_count) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    if (!initialized_) {
        return false;
    }
    
    auto session = storage_->load_session(session_id);
    if (!session) {
        return false;
    }
    
    session->add_message(role, content, token_count);
    
    bool success = storage_->save_session(session);
    
    if (success) {
        metrics_.total_messages_processed++;
        metrics_.total_tokens_processed += token_count;
        update_metrics();
        
        nlohmann::json msg_evt{{"session_id", session_id},
                               {"role", static_cast<int>(role)},
                               {"content_length", content.length()},
                               {"token_count", token_count},
                               {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::system_clock::now().time_since_epoch()).count()}};
        fire_event("message_added", msg_evt.dump());
    }
    
    return success;
}

std::vector<ConversationMessage> SessionManager::get_conversation_history(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(manager_mutex_);

    if (!initialized_) {
        return {};
    }

    auto session = storage_->load_session(session_id);
    if (!session) {
        return {};
    }

    // Convert deque to vector for the public API
    const auto& history = session->get_conversation_history();
    return std::vector<ConversationMessage>(history.begin(), history.end());
}

std::vector<ConversationMessage> SessionManager::get_context_messages(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    if (!initialized_) {
        return {};
    }
    
    auto session = storage_->load_session(session_id);
    if (!session) {
        return {};
    }
    
    return session->get_context_messages();
}

bool SessionManager::clear_session_history(const std::string& session_id) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    if (!initialized_) {
        return false;
    }
    
    auto session = storage_->load_session(session_id);
    if (!session) {
        return false;
    }
    
    session->clear_history();
    bool success = storage_->save_session(session);
    
    if (success) {
        nlohmann::json clr_evt{{"session_id", session_id},
                               {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::system_clock::now().time_since_epoch()).count()}};
        fire_event("session_history_cleared", clr_evt.dump());
    }
    
    return success;
}

bool SessionManager::update_session_context_size(const std::string& session_id, size_t max_context_tokens) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    if (!initialized_) {
        return false;
    }
    
    auto session = storage_->load_session(session_id);
    if (!session) {
        return false;
    }
    
    session->set_max_context_tokens(max_context_tokens);
    bool success = storage_->save_session(session);
    
    if (success) {
        nlohmann::json ctx_evt{{"session_id", session_id},
                               {"max_context_tokens", max_context_tokens},
                               {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::system_clock::now().time_since_epoch()).count()}};
        fire_event("session_context_updated", ctx_evt.dump());
    }
    
    return success;
}

std::vector<std::string> SessionManager::list_sessions(size_t limit, size_t offset,
                                                       const std::string& sort_by,
                                                       bool ascending) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    if (!initialized_) {
        return {};
    }
    
    return storage_->list_sessions(limit, offset, sort_by, ascending);
}

bool SessionManager::export_sessions(const std::string& file_path, 
                                    const std::vector<std::string>& session_ids) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    if (!initialized_) {
        return false;
    }
    
    if (session_ids.empty()) {
        // Export all sessions
        return storage_->export_to_json(file_path);
    }
    
    // Export specific sessions
    nlohmann::json export_data = nlohmann::json::array();
    
    for (const auto& session_id : session_ids) {
        auto session = storage_->load_session(session_id);
        if (session) {
            export_data.push_back(session->to_json());
        }
    }
    
    std::ofstream file(file_path);
    if (!file.is_open()) {
        return false;
    }
    
    file << export_data.dump(2);
    return file.good();
}

size_t SessionManager::import_sessions(const std::string& file_path, bool overwrite_existing) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    if (!initialized_) {
        return 0;
    }
    
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return 0;
    }
    
    std::string json_content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    
    try {
        auto import_data = nlohmann::json::parse(json_content);
        
        if (!import_data.is_array()) {
            return 0;
        }
        
        size_t imported_count = 0;
        for (const auto& session_json : import_data) {
            try {
                    auto session = std::make_shared<Session>(session_json);
                
                if (!overwrite_existing && storage_->session_exists(session->get_session_id())) {
                    continue;
                }
                
                if (storage_->save_session(session)) {
                    imported_count++;
                    
                    nlohmann::json imp_evt{{"session_id", session->get_session_id()},
                                           {"overwrite", overwrite_existing},
                                           {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                                               std::chrono::system_clock::now().time_since_epoch()).count()}};
                    fire_event("session_imported", imp_evt.dump());
                }
            } catch (const std::exception&) {
                // Skip invalid session entries
                continue;
            }
        }
        
        return imported_count;
    } catch (const std::exception&) {
        return 0;
    }
}

size_t SessionManager::cleanup_expired_sessions() {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    if (!initialized_) {
        return 0;
    }
    
    size_t cleaned_count = storage_->cleanup_expired_sessions();
    
    if (cleaned_count > 0) {
        nlohmann::json clean_evt{{"count", cleaned_count},
                                 {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::system_clock::now().time_since_epoch()).count()}};
        fire_event("sessions_cleaned_up", clean_evt.dump());
    }
    
    return cleaned_count;
}

std::string SessionManager::get_statistics() {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    if (!initialized_) {
        return std::string("{}");
    }
    
    auto storage_stats = storage_->get_statistics();
    auto metrics = get_metrics();
    
    nlohmann::json stats = nlohmann::json{
        {"storage", storage_stats},
        {"metrics", {
            {"total_sessions_created", metrics.total_sessions_created},
            {"total_sessions_deleted", metrics.total_sessions_deleted},
            {"total_messages_processed", metrics.total_messages_processed},
            {"total_tokens_processed", metrics.total_tokens_processed},
            {"avg_session_duration_minutes", metrics.avg_session_duration_minutes},
            {"avg_tokens_per_session", metrics.avg_tokens_per_session},
            {"avg_messages_per_session", metrics.avg_messages_per_session},
            {"last_reset", std::chrono::duration_cast<std::chrono::milliseconds>(
                metrics.last_reset.time_since_epoch()).count()}
        }},
        {"config", {
            {"default_max_context_tokens", config_.default_max_context_tokens},
            {"enable_metrics", config_.enable_metrics},
            {"metrics_interval_minutes", config_.metrics_interval.count()}
        }}
    };
    return stats.dump();
}

SessionManager::Metrics SessionManager::get_metrics() const {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    return metrics_;
}

void SessionManager::reset_metrics() {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    metrics_ = Metrics{};
    metrics_.last_reset = std::chrono::system_clock::now();
    
    nlohmann::json reset_evt{{"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count()}};
    fire_event("metrics_reset", reset_evt.dump());
}

void SessionManager::set_event_callback(std::function<void(const std::string&, const std::string&)> callback) {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    event_callback_ = std::move(callback);
}

const SessionManager::Config& SessionManager::get_config() const {
    return config_;
}

void SessionManager::shutdown() {
    std::lock_guard<std::mutex> lock(manager_mutex_);
    
    if (storage_) {
        storage_->close();
    }
    
    initialized_ = false;
    
    nlohmann::json shut_evt{{"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count()}};
    fire_event("manager_shutdown", shut_evt.dump());
}

std::string SessionManager::generate_session_id() {
    return generate_uuid();
}

void SessionManager::update_metrics() {
    if (!config_.enable_metrics) {
        return;
    }
    
    auto now = std::chrono::system_clock::now();
    if (now - last_metrics_update_ >= config_.metrics_interval) {
        // Calculate derived metrics
        if (metrics_.total_sessions_created > 0) {
            metrics_.avg_tokens_per_session = 
                static_cast<double>(metrics_.total_tokens_processed) / metrics_.total_sessions_created;
            metrics_.avg_messages_per_session = 
                static_cast<double>(metrics_.total_messages_processed) / metrics_.total_sessions_created;
        }
        
        last_metrics_update_ = now;
    }
}

void SessionManager::fire_event(const std::string& event, const std::string& data_json) {
    if (event_callback_) {
        try {
            event_callback_(event, data_json);
        } catch (const std::exception&) {
            // Ignore callback exceptions to prevent affecting core functionality
        }
    }
}

std::string SessionManager::generate_uuid() {
    // Generate UUID v4
    std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFF);
    
    uint32_t data[4];
    for (int i = 0; i < 4; ++i) {
        data[i] = dis(uuid_generator_);
    }
    
    // Set version (4) and variant bits
    data[1] = (data[1] & 0x0FFFFFFF) | 0x40000000;  // Version 4
    data[2] = (data[2] & 0x3FFFFFFF) | 0x80000000;  // Variant 10
    
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    oss << std::setw(8) << data[0] << "-";
    oss << std::setw(4) << (data[1] >> 16) << "-";
    oss << std::setw(4) << (data[1] & 0xFFFF) << "-";
    oss << std::setw(4) << (data[2] >> 16) << "-";
    oss << std::setw(4) << (data[2] & 0xFFFF);
    oss << std::setw(8) << data[3];
    
    return oss.str();
}

bool SessionManager::is_valid_session_id(const std::string& session_id) {
    // Basic validation - non-empty and reasonable length
    if (session_id.empty() || session_id.length() > 255) {
        return false;
    }
    
    // Check for basic UUID format (optional - could be made more strict)
    if (session_id.length() == 36) {
        // UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        if (session_id[8] == '-' && session_id[13] == '-' && 
            session_id[18] == '-' && session_id[23] == '-') {
            return true;
        }
    }
    
    // Allow other reasonable session ID formats
    return std::all_of(session_id.begin(), session_id.end(), [](char c) {
        return std::isalnum(c) || c == '-' || c == '_';
    });
}

} // namespace session
} // namespace gemma