#include "session_serialization.h"
#include <nlohmann/json.hpp>

namespace gemma {
namespace session {

void message_to_json(nlohmann::json& j, const ConversationMessage& msg) {
    j = nlohmann::json{
        {"role", static_cast<int>(msg.role)},
        {"content", msg.content},
        {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
            msg.timestamp.time_since_epoch()).count()},
        {"token_count", msg.token_count}
    };
}

void message_from_json(const nlohmann::json& j, ConversationMessage& msg) {
    msg.role = static_cast<ConversationMessage::Role>(j.at("role").get<int>());
    msg.content = j.at("content").get<std::string>();

    auto timestamp_ms = j.at("timestamp").get<int64_t>();
    msg.timestamp = std::chrono::system_clock::time_point(
        std::chrono::milliseconds(timestamp_ms));

    msg.token_count = j.at("token_count").get<size_t>();
}

} // namespace session
} // namespace gemma
