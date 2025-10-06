#pragma once

#include <string>
#include <vector>
#include <deque>
#include <chrono>
#include <memory>
#include <atomic>
#include <cstddef>
// Include JSON forward declarations (light) or full header to expose nlohmann::json in API.
#if __has_include(<nlohmann/json.hpp>)
    #include <nlohmann/json.hpp>
#elif __has_include(<nlohmann/json_fwd.hpp>)
    #include <nlohmann/json_fwd.hpp>
#else
    #include "../../third_party/nlohmann_json/single_include/nlohmann/json.hpp"
#endif
#include "conversation_types.h"

namespace gemma {
namespace session {

// ConversationMessage has been moved to conversation_types.h (Phase 1 refactor)

/**
 * @brief Session class representing an individual conversation session
 *
 * This class manages the conversation history, context window, and token counting
 * for a single session. It provides serialization capabilities and efficient
 * memory management for long conversations.
 *
 * Performance optimizations:
 * - O(1) context token tracking via caching
 * - Efficient deque-based message management
 * - Smart memory management with reserve() hints
 */
class Session {
public:
    /**
     * @brief Construct a new Session object
     *
     * @param session_id Unique identifier for the session
     * @param max_context_tokens Maximum number of tokens to keep in context
     */
    explicit Session(const std::string& session_id, size_t max_context_tokens = 8192);

    // JSON-based constructor for deserialization
    explicit Session(const nlohmann::json& json_data);

    // Disable copy constructor and assignment
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    // Enable move constructor and assignment
    Session(Session&&) = default;
    Session& operator=(Session&&) = default;

    ~Session() = default;

    /**
     * @brief Add a message to the conversation history
     *
     * @param role Message role (USER, ASSISTANT, SYSTEM)
     * @param content Message content
     * @param token_count Number of tokens in the message
     */
    void add_message(ConversationMessage::Role role, const std::string& content, size_t token_count);

    /**
     * @brief Get the conversation history
     *
     * @return const std::deque<ConversationMessage>& Reference to the conversation history
     */
    const std::deque<ConversationMessage>& get_conversation_history() const;

    /**
     * @brief Get messages within the current context window
     *
     * @return std::vector<ConversationMessage> Messages that fit in the context window
     */
    std::vector<ConversationMessage> get_context_messages() const;

    /**
     * @brief Get the session ID
     *
     * @return const std::string& Session identifier
     */
    const std::string& get_session_id() const;

    /**
     * @brief Get the total number of tokens in the session
     *
     * @return size_t Total token count
     */
    size_t get_total_tokens() const;

    /**
     * @brief Get the number of tokens in the current context window (cached)
     *
     * @return size_t Context token count
     */
    size_t get_context_tokens() const;

    /**
     * @brief Get the creation timestamp
     *
     * @return std::chrono::system_clock::time_point When the session was created
     */
    std::chrono::system_clock::time_point get_created_at() const;

    /**
     * @brief Get the last activity timestamp
     *
     * @return std::chrono::system_clock::time_point When the session was last accessed
     */
    std::chrono::system_clock::time_point get_last_activity() const;

    /**
     * @brief Update the last activity timestamp to now
     */
    void touch();

    /**
     * @brief Clear the conversation history
     */
    void clear_history();

    /**
     * @brief Set the maximum context tokens
     *
     * @param max_tokens New maximum context size
     */
    void set_max_context_tokens(size_t max_tokens);

    /**
     * @brief Get the maximum context tokens
     *
     * @return size_t Maximum context size
     */
    size_t get_max_context_tokens() const;

    // Serialization API
    nlohmann::json to_json() const;
    void from_json(const nlohmann::json& json_data);
    nlohmann::json get_metadata() const;

private:
    std::string session_id_;
    std::deque<ConversationMessage> conversation_history_;
    size_t max_context_tokens_;
    size_t total_tokens_;
    mutable size_t cached_context_tokens_;
    mutable bool context_cache_valid_;
    mutable size_t context_start_index_;
    std::chrono::system_clock::time_point created_at_;
    std::chrono::system_clock::time_point last_activity_;

    void trim_context();
    size_t calculate_context_tokens() const;
    void invalidate_context_cache() const;
    void update_context_cache() const;
};

} // namespace session
} // namespace gemma