#pragma once

#include "conversation_types.h"
#if __has_include(<nlohmann/json.hpp>)
	#include <nlohmann/json.hpp>
#else
	#include "../../third_party/nlohmann_json/single_include/nlohmann/json.hpp"
#endif

namespace gemma {
namespace session {

// Serialization helpers separated from core types to enforce SRP.
void message_to_json(nlohmann::json& j, const ConversationMessage& msg);
void message_from_json(const nlohmann::json& j, ConversationMessage& msg);

} // namespace session
} // namespace gemma
