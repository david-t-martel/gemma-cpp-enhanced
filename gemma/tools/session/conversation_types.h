// Conversation-related lightweight types extracted from Session.h
// Phase 1 refactor: this header intentionally contains ONLY simple POD-like
// structures and enums without any JSON or storage concerns.

#pragma once

#include <string>
#include <chrono>

namespace gemma {
namespace session {

struct ConversationMessage {
    enum class Role {
        USER,
        ASSISTANT,
        SYSTEM
    };

    Role role{};                       // Speaker role
    std::string content;               // Raw message content
    std::chrono::system_clock::time_point timestamp; // Creation time
    size_t token_count{};              // Count of tokens (model-dependent)
};

} // namespace session
} // namespace gemma
