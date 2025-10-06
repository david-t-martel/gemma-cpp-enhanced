#include "Session.h"
#include "session_serialization.h"
#if __has_include(<nlohmann/json.hpp>)
  #include <nlohmann/json.hpp>
#else
  #include "../../third_party/nlohmann_json/single_include/nlohmann/json.hpp"
#endif
#include <stdexcept>

namespace gemma { namespace session {

Session::Session(const nlohmann::json& json_data)
    : cached_context_tokens_(0)
    , context_cache_valid_(false)
    , context_start_index_(0) {
    from_json(json_data);
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
        nlohmann::json msg_json; message_to_json(msg_json, message);
        j["conversation_history"].push_back(msg_json);
    }
    return j;
}

void Session::from_json(const nlohmann::json& json_data) {
    session_id_ = json_data.at("session_id").get<std::string>();
    max_context_tokens_ = json_data.at("max_context_tokens").get<size_t>();
    total_tokens_ = json_data.value("total_tokens", size_t{0});
    auto created_ms = json_data.at("created_at").get<int64_t>();
    created_at_ = std::chrono::system_clock::time_point(std::chrono::milliseconds(created_ms));
    auto activity_ms = json_data.at("last_activity").get<int64_t>();
    last_activity_ = std::chrono::system_clock::time_point(std::chrono::milliseconds(activity_ms));
    conversation_history_.clear();
    if (json_data.contains("conversation_history")) {
        for (const auto& msg_json : json_data.at("conversation_history")) {
            ConversationMessage m; message_from_json(msg_json, m); conversation_history_.push_back(std::move(m));
        }
    }
    if (session_id_.empty()) throw std::invalid_argument("Deserialized session ID empty");
    if (max_context_tokens_ == 0) throw std::invalid_argument("Deserialized max_context_tokens zero");
    invalidate_context_cache();
}

nlohmann::json Session::get_metadata() const {
    update_context_cache();
    return nlohmann::json{{"session_id", session_id_},
                          {"created_at", std::chrono::duration_cast<std::chrono::milliseconds>(created_at_.time_since_epoch()).count()},
                          {"last_activity", std::chrono::duration_cast<std::chrono::milliseconds>(last_activity_.time_since_epoch()).count()},
                          {"total_tokens", total_tokens_},
                          {"context_tokens", cached_context_tokens_},
                          {"max_context_tokens", max_context_tokens_},
                          {"message_count", conversation_history_.size()},
                          {"context_message_count", conversation_history_.size() - context_start_index_}};
}

} } // namespace gemma::session
