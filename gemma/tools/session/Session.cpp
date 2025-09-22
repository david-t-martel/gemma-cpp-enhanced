#include "Session.h"
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace gemma {
namespace session {

// ConversationMessage serialization
void ConversationMessage::to_json(nlohmann::json& j) const {
    j = nlohmann::json{
        {"role", static_cast<int>(role)},
        {"content", content},
        {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
            timestamp.time_since_epoch()).count()},
        {"token_count", token_count}
    };
}

void ConversationMessage::from_json(const nlohmann::json& j) {
    role = static_cast<Role>(j.at("role").get<int>());
    content = j.at("content").get<std::string>();

    auto timestamp_ms = j.at("timestamp").get<int64_t>();
    timestamp = std::chrono::system_clock::time_point(
        std::chrono::milliseconds(timestamp_ms));

    token_count = j.at("token_count").get<size_t>();
}

// Session implementation
Session::Session(const std::string& session_id, size_t max_context_tokens)
    : session_id_(session_id)
    , max_context_tokens_(max_context_tokens)
    , total_tokens_(0)
    , cached_context_tokens_(0)
    , context_cache_valid_(false)
    , context_start_index_(0)
    , created_at_(std::chrono::system_clock::now())
    , last_activity_(std::chrono::system_clock::now()) {

    if (session_id_.empty()) {
        throw std::invalid_argument("Session ID cannot be empty");
    }

    if (max_context_tokens_ == 0) {
        throw std::invalid_argument("Max context tokens must be greater than 0");
    }
}

Session::Session(const nlohmann::json& json_data)
    : cached_context_tokens_(0)
    , context_cache_valid_(false)
    , context_start_index_(0) {
    from_json(json_data);
}

void Session::add_message(ConversationMessage::Role role, const std::string& content, size_t token_count) {
    if (content.empty()) {
        throw std::invalid_argument("Message content cannot be empty");
    }

    ConversationMessage message;
    message.role = role;
    message.content = content;
    message.token_count = token_count;
    message.timestamp = std::chrono::system_clock::now();

    conversation_history_.push_back(std::move(message));
    total_tokens_ += token_count;
    last_activity_ = std::chrono::system_clock::now();

    // Invalidate cache since we added a new message
    invalidate_context_cache();

    // Trim context if necessary
    trim_context();
}

const std::deque<ConversationMessage>& Session::get_conversation_history() const {
    return conversation_history_;
}

std::vector<ConversationMessage> Session::get_context_messages() const {
    // Update cache if needed
    update_context_cache();

    // Build result from cached indices - O(n) where n is messages in context
    std::vector<ConversationMessage> context_messages;

    // Reserve space to avoid reallocations
    size_t estimated_messages = conversation_history_.size() - context_start_index_;
    context_messages.reserve(estimated_messages);

    // Copy messages starting from cached index
    for (size_t i = context_start_index_; i < conversation_history_.size(); ++i) {
        context_messages.push_back(conversation_history_[i]);
    }

    return context_messages;
}

const std::string& Session::get_session_id() const {
    return session_id_;
}

size_t Session::get_total_tokens() const {
    return total_tokens_;
}

size_t Session::get_context_tokens() const {
    // Use cached value, update if needed
    update_context_cache();
    return cached_context_tokens_;
}

std::chrono::system_clock::time_point Session::get_created_at() const {
    return created_at_;
}

std::chrono::system_clock::time_point Session::get_last_activity() const {
    return last_activity_;
}

void Session::touch() {
    last_activity_ = std::chrono::system_clock::now();
}

void Session::clear_history() {
    conversation_history_.clear();
    total_tokens_ = 0;
    last_activity_ = std::chrono::system_clock::now();

    // Reset cache
    invalidate_context_cache();
}

void Session::set_max_context_tokens(size_t max_tokens) {
    if (max_tokens == 0) {
        throw std::invalid_argument("Max context tokens must be greater than 0");
    }

    max_context_tokens_ = max_tokens;
    last_activity_ = std::chrono::system_clock::now();

    // Context window changed, invalidate cache
    invalidate_context_cache();

    // Trim if needed with new limit
    trim_context();
}

size_t Session::get_max_context_tokens() const {
    return max_context_tokens_;
}

nlohmann::json Session::to_json() const {
    nlohmann::json j;
    j["session_id"] = session_id_;
    j["max_context_tokens"] = max_context_tokens_;
    j["total_tokens"] = total_tokens_;
    j["created_at"] = std::chrono::duration_cast<std::chrono::milliseconds>(
        created_at_.time_since_epoch()).count();
    j["last_activity"] = std::chrono::duration_cast<std::chrono::milliseconds>(
        last_activity_.time_since_epoch()).count();

    j["conversation_history"] = nlohmann::json::array();
    for (const auto& message : conversation_history_) {
        nlohmann::json msg_json;
        message.to_json(msg_json);
        j["conversation_history"].push_back(msg_json);
    }

    return j;
}

void Session::from_json(const nlohmann::json& json_data) {
    session_id_ = json_data.at("session_id").get<std::string>();
    max_context_tokens_ = json_data.at("max_context_tokens").get<size_t>();
    total_tokens_ = json_data.at("total_tokens").get<size_t>();

    auto created_ms = json_data.at("created_at").get<int64_t>();
    created_at_ = std::chrono::system_clock::time_point(std::chrono::milliseconds(created_ms));

    auto activity_ms = json_data.at("last_activity").get<int64_t>();
    last_activity_ = std::chrono::system_clock::time_point(std::chrono::milliseconds(activity_ms));

    conversation_history_.clear();
    if (json_data.contains("conversation_history")) {
        for (const auto& msg_json : json_data.at("conversation_history")) {
            ConversationMessage message;
            message.from_json(msg_json);
            conversation_history_.push_back(std::move(message));
        }
    }

    // Validate session after deserialization
    if (session_id_.empty()) {
        throw std::invalid_argument("Deserialized session ID cannot be empty");
    }

    if (max_context_tokens_ == 0) {
        throw std::invalid_argument("Deserialized max context tokens must be greater than 0");
    }

    // Invalidate cache after loading
    invalidate_context_cache();
}

nlohmann::json Session::get_metadata() const {
    // Use cached context tokens
    update_context_cache();

    return nlohmann::json{
        {"session_id", session_id_},
        {"created_at", std::chrono::duration_cast<std::chrono::milliseconds>(
            created_at_.time_since_epoch()).count()},
        {"last_activity", std::chrono::duration_cast<std::chrono::milliseconds>(
            last_activity_.time_since_epoch()).count()},
        {"total_tokens", total_tokens_},
        {"context_tokens", cached_context_tokens_},
        {"max_context_tokens", max_context_tokens_},
        {"message_count", conversation_history_.size()},
        {"context_message_count", conversation_history_.size() - context_start_index_}
    };
}

void Session::trim_context() {
    // This function now actually trims old messages that are completely outside the context window
    // We maintain a reasonable buffer to avoid frequent trimming

    const size_t TRIM_THRESHOLD_MULTIPLIER = 2;  // Keep up to 2x max context in history
    const size_t MIN_MESSAGES_TO_KEEP = 10;       // Always keep at least 10 messages

    // Don't trim if we have few messages
    if (conversation_history_.size() <= MIN_MESSAGES_TO_KEEP) {
        return;
    }

    // Calculate total tokens
    size_t total_history_tokens = std::accumulate(
        conversation_history_.begin(),
        conversation_history_.end(),
        size_t(0),
        [](size_t sum, const ConversationMessage& msg) {
            return sum + msg.token_count;
        }
    );

    // Only trim if we exceed threshold
    if (total_history_tokens <= max_context_tokens_ * TRIM_THRESHOLD_MULTIPLIER) {
        return;
    }

    // Find how many messages to remove from the front
    size_t tokens_to_remove = total_history_tokens - (max_context_tokens_ * TRIM_THRESHOLD_MULTIPLIER);
    size_t removed_tokens = 0;
    size_t messages_to_remove = 0;

    for (const auto& msg : conversation_history_) {
        if (removed_tokens >= tokens_to_remove) {
            break;
        }
        removed_tokens += msg.token_count;
        messages_to_remove++;
    }

    // Keep at least MIN_MESSAGES_TO_KEEP
    messages_to_remove = std::min(messages_to_remove,
                                  conversation_history_.size() - MIN_MESSAGES_TO_KEEP);

    if (messages_to_remove > 0) {
        // Efficiently remove from front using deque
        for (size_t i = 0; i < messages_to_remove; ++i) {
            conversation_history_.pop_front();
        }

        // Invalidate cache since we removed messages
        invalidate_context_cache();
    }
}

size_t Session::calculate_context_tokens() const {
    if (conversation_history_.empty()) {
        return 0;
    }

    size_t context_tokens = 0;
    context_start_index_ = conversation_history_.size();  // Start from the end

    // Iterate from the end to find messages that fit in context
    for (auto it = conversation_history_.rbegin(); it != conversation_history_.rend(); ++it) {
        if (context_tokens + it->token_count <= max_context_tokens_) {
            context_tokens += it->token_count;
            context_start_index_--;
        } else {
            break;
        }
    }

    return context_tokens;
}

void Session::invalidate_context_cache() const {
    context_cache_valid_ = false;
}

void Session::update_context_cache() const {
    if (context_cache_valid_) {
        return;  // Cache is already valid
    }

    // Recalculate and cache the context tokens
    cached_context_tokens_ = calculate_context_tokens();
    context_cache_valid_ = true;
}

} // namespace session
} // namespace gemma