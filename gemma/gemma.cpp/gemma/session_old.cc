// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gemma/session.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gemma/gemma.h"
#include "gemma/configs.h"
#include "gemma/gemma_args.h"
#include "gemma/kv_cache.h"
#include "ops/matmul.h"
#include "util/basics.h"
#include "util/threading_context.h"
#include "hwy/base.h"

#include <nlohmann/json.hpp>

namespace gcpp {

// Anonymous namespace for internal utilities
namespace {

// Helper function to generate unique IDs
std::string GenerateRandomId() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> dis(0, 15);

  const char* hex_chars = "0123456789abcdef";
  std::string id(16, '0');

  for (size_t i = 0; i < 16; ++i) {
    id[i] = hex_chars[dis(gen)];
  }

  return id;
}

// Convert message role to string
std::string MessageRoleToString(MessageRole role) {
  switch (role) {
    case MessageRole::USER: return "user";
    case MessageRole::ASSISTANT: return "assistant";
    case MessageRole::SYSTEM: return "system";
    default: return "unknown";
  }
}

// Convert string to message role
MessageRole StringToMessageRole(const std::string& role_str) {
  if (role_str == "user") return MessageRole::USER;
  if (role_str == "assistant") return MessageRole::ASSISTANT;
  if (role_str == "system") return MessageRole::SYSTEM;
  return MessageRole::USER; // Default fallback
}

// Convert time point to ISO string
std::string TimeToISOString(const std::chrono::system_clock::time_point& tp) {
  auto time_t = std::chrono::system_clock::to_time_t(tp);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      tp.time_since_epoch()) % 1000;

  std::stringstream ss;
  ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
  ss << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
  return ss.str();
}

// Convert ISO string to time point
std::chrono::system_clock::time_point TimeFromISOString(const std::string& iso_str) {
  // Simple implementation - in production, use a proper ISO parser
  return std::chrono::system_clock::now(); // Placeholder
}

} // anonymous namespace

// =============================================================================
// GemmaSession Implementation
// =============================================================================

gcpp::GemmaSession::GemmaSession(const std::string& session_id,
                                 const ModelConfig& config,
                                 const InferenceArgs& inference_args,
                                 ThreadingContext& ctx,
                                 const SessionConfig& session_config)
    : session_id_(session_id.empty() ? GenerateRandomId() : session_id),
      config_(session_config),
      owner_thread_(std::this_thread::get_id()),
      model_config_(config),
      inference_args_(inference_args),
      threading_context_(ctx),
      last_auto_save_(std::chrono::steady_clock::now()) {

  conversation_history_.reserve(config_.max_history_turns * 2); // User + Assistant pairs
}

gcpp::GemmaSession::~GemmaSession() {
  Terminate();
}

bool gcpp::GemmaSession::Initialize(const Gemma& gemma) {
  std::lock_guard<std::recursive_mutex> lock(session_mutex_);

  if (GetState() != SessionState::IDLE) {
    return false;
  }

  try {
    InitializeKVCache();
    SetState(SessionState::ACTIVE);
    return true;
  } catch (const std::exception& e) {
    fprintf(stderr, "Failed to initialize session %s: %s\n",
            session_id_.c_str(), e.what());
    SetState(SessionState::ERROR);
    return false;
  }
}

void GemmaSession::Terminate() {
  std::lock_guard<std::recursive_mutex> lock(session_mutex_);

  if (GetState() == SessionState::TERMINATED) {
    return;
  }

  // Perform auto-save if enabled
  if (config_.auto_save && !conversation_history_.empty()) {
    AutoSave();
  }

  SetState(SessionState::TERMINATED);
}

void GemmaSession::Reset() {
  std::lock_guard<std::recursive_mutex> lock(session_mutex_);

  conversation_history_.clear();
  current_token_count_.store(0);
  current_position_ = 0;

  if (kv_cache_) {
    kv_cache_->ZeroGriffinCache();
  }

  // Reset statistics but keep session metadata
  stats_.total_turns.store(0);
  stats_.total_input_tokens.store(0);
  stats_.total_output_tokens.store(0);
  stats_.cache_hits.store(0);
  stats_.cache_misses.store(0);
  stats_.avg_response_time_ms.store(0.0);

  SetState(SessionState::ACTIVE);
}

void GemmaSession::AddUserMessage(const std::string& message) {
  std::lock_guard<std::recursive_mutex> lock(session_mutex_);

  conversation_history_.emplace_back(MessageRole::USER, message);
  UpdateTokenCount();

  stats_.last_active = std::chrono::system_clock::now();

  // Check if we need memory cleanup
  if (current_token_count_.load() > config_.memory_cleanup_threshold) {
    PerformMemoryCleanup();
  }
}

void GemmaSession::AddSystemMessage(const std::string& message) {
  std::lock_guard<std::recursive_mutex> lock(session_mutex_);

  conversation_history_.emplace_back(MessageRole::SYSTEM, message);
  UpdateTokenCount();

  stats_.last_active = std::chrono::system_clock::now();
}

std::string GemmaSession::GenerateResponse(const Gemma& gemma,
                                          MatMulEnv& env,
                                          const RuntimeConfig& runtime_config) {
  auto start_time = std::chrono::steady_clock::now();
  std::string response;

  // Use streaming interface internally but collect all tokens
  GenerateResponseStream(gemma, env, runtime_config,
    [&response](const std::string& token, bool is_final) -> bool {
      response += token;
      return true; // Continue generation
    });

  // Update timing statistics
  auto end_time = std::chrono::steady_clock::now();
  auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time).count();
  stats_.UpdateResponseTime(static_cast<double>(duration_ms));

  return response;
}

void GemmaSession::GenerateResponseStream(const Gemma& gemma,
                                         MatMulEnv& env,
                                         const RuntimeConfig& runtime_config,
                                         StreamCallback callback) {
  std::lock_guard<std::recursive_mutex> lock(session_mutex_);

  if (GetState() != SessionState::ACTIVE) {
    callback("Error: Session not in active state", true);
    return;
  }

  try {
    SetState(SessionState::ACTIVE);

    // Build context tokens from conversation history
    PromptTokens context_tokens = BuildContextTokens(gemma);

    if (context_tokens.empty()) {
      callback("Error: No context available", true);
      return;
    }

    // Prepare for generation
    TimingInfo timing_info;
    timing_info.verbosity = 0; // Suppress internal timing output
    timing_info.prefill_start = hwy::platform::Now();

    std::string response_text;
    size_t tokens_generated = 0;

    // Create runtime config with streaming callback
    RuntimeConfig stream_config = runtime_config;
    stream_config.stream_token = [&](int token, float prob) -> bool {
      std::string token_text;
      if (gemma.Tokenizer().Decode({token}, &token_text)) {
        response_text += token_text;
        tokens_generated++;

        // Check generation limits
        if (tokens_generated >= config_.max_generation_tokens) {
          return false;
        }

        // Call user callback
        return callback(token_text, false);
      }
      return true;
    };

    // Generate response using Gemma
    gemma.Generate(stream_config, context_tokens, current_position_,
                   *kv_cache_, env, timing_info);

    // Add assistant response to conversation history
    conversation_history_.emplace_back(MessageRole::ASSISTANT, response_text);

    // Update statistics
    stats_.total_turns.fetch_add(1);
    stats_.total_output_tokens.fetch_add(tokens_generated);
    stats_.last_active = std::chrono::system_clock::now();

    // Update position and token count
    current_position_ += tokens_generated;
    UpdateTokenCount();

    // Final callback
    callback("", true);

    // Check if auto-save is needed
    if (ShouldAutoSave()) {
      AutoSave();
    }

  } catch (const std::exception& e) {
    SetState(SessionState::ERROR);
    callback(std::string("Error during generation: ") + e.what(), true);
  }
}

size_t GemmaSession::GetConversationLength() const {
  std::lock_guard<std::recursive_mutex> lock(session_mutex_);
  return conversation_history_.size();
}

std::vector<ConversationMessage> GemmaSession::GetHistory(size_t max_messages) const {
  std::lock_guard<std::recursive_mutex> lock(session_mutex_);

  if (max_messages == 0 || max_messages >= conversation_history_.size()) {
    return conversation_history_;
  }

  // Return the most recent messages
  auto start_it = conversation_history_.end() - max_messages;
  return std::vector<ConversationMessage>(start_it, conversation_history_.end());
}

void GemmaSession::ClearHistory() {
  std::lock_guard<std::recursive_mutex> lock(session_mutex_);

  conversation_history_.clear();
  current_token_count_.store(0);
  current_position_ = 0;

  if (kv_cache_) {
    kv_cache_->ZeroGriffinCache();
  }
}

void GemmaSession::TrimHistory(size_t max_tokens) {
  std::lock_guard<std::recursive_mutex> lock(session_mutex_);

  if (current_token_count_.load() <= max_tokens) {
    return;
  }

  // Remove oldest messages until we're under the limit
  size_t current_tokens = 0;
  auto it = conversation_history_.rbegin();

  while (it != conversation_history_.rend() &&
         current_tokens < max_tokens) {
    current_tokens += it->tokens.size();
    ++it;
  }

  // Keep messages from 'it' position onward
  if (it != conversation_history_.rend()) {
    conversation_history_.erase(conversation_history_.begin(), it.base());
    UpdateTokenCount();

    // Reset KV cache since we removed history
    current_position_ = 0;
    if (kv_cache_) {
      kv_cache_->ZeroGriffinCache();
    }
  }
}

bool GemmaSession::SaveToFile(const Path& file_path) const {
  std::lock_guard<std::recursive_mutex> lock(session_mutex_);

  try {
    auto serialized = SerializeSession();

    nlohmann::json j;
    j["session_id"] = serialized.session_id;
    j["saved_at"] = TimeToISOString(serialized.saved_at);
    j["current_position"] = serialized.current_position;

    // Serialize config
    j["config"]["max_history_turns"] = serialized.config.max_history_turns;
    j["config"]["context_window_tokens"] = serialized.config.context_window_tokens;
    j["config"]["memory_cleanup_threshold"] = serialized.config.memory_cleanup_threshold;
    j["config"]["auto_save"] = serialized.config.auto_save;
    j["config"]["save_interval"] = serialized.config.save_interval.count();
    j["config"]["compress_history"] = serialized.config.compress_history;
    j["config"]["prefill_batch_size"] = serialized.config.prefill_batch_size;
    j["config"]["enable_kv_cache_reuse"] = serialized.config.enable_kv_cache_reuse;
    j["config"]["enable_streaming"] = serialized.config.enable_streaming;
    j["config"]["max_generation_tokens"] = serialized.config.max_generation_tokens;
    j["config"]["generation_timeout"] = serialized.config.generation_timeout.count();
    j["config"]["enable_content_filtering"] = serialized.config.enable_content_filtering;

    // Serialize conversation history
    j["history"] = nlohmann::json::array();
    for (const auto& msg : serialized.history) {
      nlohmann::json msg_json;
      msg_json["role"] = MessageRoleToString(msg.role);
      msg_json["content"] = msg.content;
      msg_json["timestamp"] = TimeToISOString(msg.timestamp);
      msg_json["tokens"] = msg.tokens;
      j["history"].push_back(msg_json);
    }

    // Serialize statistics
    j["stats"]["total_turns"] = serialized.stats.total_turns.load();
    j["stats"]["total_input_tokens"] = serialized.stats.total_input_tokens.load();
    j["stats"]["total_output_tokens"] = serialized.stats.total_output_tokens.load();
    j["stats"]["cache_hits"] = serialized.stats.cache_hits.load();
    j["stats"]["cache_misses"] = serialized.stats.cache_misses.load();
    j["stats"]["created_at"] = TimeToISOString(serialized.stats.created_at);
    j["stats"]["last_active"] = TimeToISOString(serialized.stats.last_active);
    j["stats"]["avg_response_time_ms"] = serialized.stats.avg_response_time_ms.load();

    // Write to file
    std::ofstream ofs(file_path.path);
    if (!ofs.is_open()) {
      return false;
    }

    ofs << j.dump(2);
    return ofs.good();

  } catch (const std::exception& e) {
    fprintf(stderr, "Failed to save session to %s: %s\n",
            file_path.path.c_str(), e.what());
    return false;
  }
}

bool GemmaSession::LoadFromFile(const Path& file_path, const Gemma& gemma) {
  std::lock_guard<std::recursive_mutex> lock(session_mutex_);

  try {
    std::ifstream ifs(file_path.path);
    if (!ifs.is_open()) {
      return false;
    }

    nlohmann::json j;
    ifs >> j;

    // Create serialized data structure
    SerializedSession data;
    data.session_id = j["session_id"];
    data.current_position = j["current_position"];
    data.saved_at = TimeFromISOString(j["saved_at"]);

    // Deserialize config
    if (j.contains("config")) {
      auto& cfg = j["config"];
      data.config.max_history_turns = cfg["max_history_turns"];
      data.config.context_window_tokens = cfg["context_window_tokens"];
      data.config.memory_cleanup_threshold = cfg["memory_cleanup_threshold"];
      data.config.auto_save = cfg["auto_save"];
      data.config.save_interval = std::chrono::seconds(cfg["save_interval"]);
      data.config.compress_history = cfg["compress_history"];
      data.config.prefill_batch_size = cfg["prefill_batch_size"];
      data.config.enable_kv_cache_reuse = cfg["enable_kv_cache_reuse"];
      data.config.enable_streaming = cfg["enable_streaming"];
      data.config.max_generation_tokens = cfg["max_generation_tokens"];
      data.config.generation_timeout = std::chrono::seconds(cfg["generation_timeout"]);
      data.config.enable_content_filtering = cfg["enable_content_filtering"];
    }

    // Deserialize conversation history
    if (j.contains("history")) {
      for (const auto& msg_json : j["history"]) {
        ConversationMessage msg(
          StringToMessageRole(msg_json["role"]),
          msg_json["content"]
        );
        msg.timestamp = TimeFromISOString(msg_json["timestamp"]);
        msg.tokens = msg_json["tokens"];
        data.history.push_back(msg);
      }
    }

    // Deserialize statistics
    if (j.contains("stats")) {
      auto& stats_json = j["stats"];
      data.stats.total_turns.store(stats_json["total_turns"]);
      data.stats.total_input_tokens.store(stats_json["total_input_tokens"]);
      data.stats.total_output_tokens.store(stats_json["total_output_tokens"]);
      data.stats.cache_hits.store(stats_json["cache_hits"]);
      data.stats.cache_misses.store(stats_json["cache_misses"]);
      data.stats.created_at = TimeFromISOString(stats_json["created_at"]);
      data.stats.last_active = TimeFromISOString(stats_json["last_active"]);
      data.stats.avg_response_time_ms.store(stats_json["avg_response_time_ms"]);
    }

    return DeserializeSession(data, gemma);

  } catch (const std::exception& e) {
    fprintf(stderr, "Failed to load session from %s: %s\n",
            file_path.path.c_str(), e.what());
    return false;
  }
}

bool GemmaSession::AutoSave() const {
  if (!config_.auto_save) {
    return true;
  }

  // Generate auto-save filename
  std::string filename = session_id_ + "_autosave.json";
  Path auto_save_path("sessions/" + filename);

  return SaveToFile(auto_save_path);
}

void GemmaSession::ClearKVCache() {
  std::lock_guard<std::recursive_mutex> lock(session_mutex_);

  if (kv_cache_) {
    kv_cache_->ZeroGriffinCache();
  }

  current_position_ = 0;
}

void GemmaSession::OptimizeKVCache() {
  std::lock_guard<std::recursive_mutex> lock(session_mutex_);

  // Implementation could include:
  // - Compacting unused cache space
  // - Optimizing memory layout
  // - Prefetching commonly used patterns

  // For now, just ensure cache is properly initialized
  if (!kv_cache_) {
    InitializeKVCache();
  }
}

size_t GemmaSession::GetKVCacheSize() const {
  std::lock_guard<std::recursive_mutex> lock(session_mutex_);

  if (!kv_cache_) {
    return 0;
  }

  return kv_cache_->SeqLen();
}

void GemmaSession::UpdateConfig(const SessionConfig& new_config) {
  std::lock_guard<std::recursive_mutex> lock(session_mutex_);

  config_ = new_config;

  // Apply new configuration immediately where possible
  if (current_token_count_.load() > config_.memory_cleanup_threshold) {
    PerformMemoryCleanup();
  }
}

// Private methods implementation

void GemmaSession::InitializeKVCache() {
  kv_cache_ = std::make_unique<KVCache>(
    model_config_, inference_args_, threading_context_.allocator
  );
}

PromptTokens GemmaSession::TokenizeMessage(const Gemma& gemma,
                                          const ConversationMessage& msg) const {
  std::vector<int> tokens;
  if (gemma.Tokenizer().Encode(msg.content, &tokens)) {
    return tokens;
  }
  return {};
}

PromptTokens GemmaSession::BuildContextTokens(const Gemma& gemma) const {
  PromptTokens context_tokens;
  context_tokens.reserve(config_.context_window_tokens);

  // Process conversation history in order
  for (const auto& msg : conversation_history_) {
    PromptTokens msg_tokens = TokenizeMessage(gemma, msg);

    // Check if adding this message would exceed context window
    if (context_tokens.size() + msg_tokens.size() > config_.context_window_tokens) {
      break;
    }

    context_tokens.insert(context_tokens.end(), msg_tokens.begin(), msg_tokens.end());
  }

  return context_tokens;
}

void GemmaSession::UpdateTokenCount() {
  size_t total_tokens = 0;
  for (const auto& msg : conversation_history_) {
    total_tokens += msg.tokens.size();
  }
  current_token_count_.store(total_tokens);
}

void GemmaSession::PerformMemoryCleanup() {
  if (conversation_history_.empty()) {
    return;
  }

  // Remove oldest messages to stay within limits
  size_t target_tokens = config_.context_window_tokens * 0.8; // Keep 80% capacity
  TrimHistory(target_tokens);

  // Compact KV cache
  OptimizeKVCache();
}

bool GemmaSession::ShouldAutoSave() const {
  if (!config_.auto_save) {
    return false;
  }

  auto now = std::chrono::steady_clock::now();
  auto elapsed = now - last_auto_save_;

  return elapsed >= config_.save_interval;
}

void GemmaSession::SetState(SessionState new_state) {
  state_.store(new_state);
  stats_.last_active = std::chrono::system_clock::now();
}

GemmaSession::SerializedSession GemmaSession::SerializeSession() const {
  SerializedSession data;
  data.session_id = session_id_;
  data.config = config_;
  data.history = conversation_history_;
  data.stats = stats_;
  data.current_position = current_position_;
  data.saved_at = std::chrono::system_clock::now();

  return data;
}

bool GemmaSession::DeserializeSession(const SerializedSession& data,
                                     const Gemma& gemma) {
  try {
    session_id_ = data.session_id;
    config_ = data.config;
    conversation_history_ = data.history;
    stats_ = data.stats;
    current_position_ = data.current_position;

    // Re-tokenize messages to ensure consistency
    for (auto& msg : conversation_history_) {
      if (msg.tokens.empty()) {
        msg.tokens = TokenizeMessage(gemma, msg);
      }
    }

    UpdateTokenCount();

    // Reinitialize KV cache
    InitializeKVCache();

    SetState(SessionState::ACTIVE);
    return true;

  } catch (const std::exception& e) {
    fprintf(stderr, "Failed to deserialize session: %s\n", e.what());
    SetState(SessionState::ERROR);
    return false;
  }
}

// =============================================================================
// SessionManager Implementation
// =============================================================================

SessionManager::SessionManager(const ModelConfig& config,
                               const InferenceArgs& inference_args,
                               ThreadingContext& ctx)
    : model_config_(config),
      inference_args_(inference_args),
      threading_context_(ctx) {
  stats_.created_at = std::chrono::system_clock::now();
}

SessionManager::~SessionManager() {
  RemoveAllSessions();
}

std::shared_ptr<GemmaSession> SessionManager::CreateSession(
    const std::string& session_id,
    const SessionConfig& config) {
  std::unique_lock<std::shared_mutex> lock(sessions_mutex_);

  std::string actual_id = session_id.empty() ? GenerateSessionId() : session_id;

  // Check if session ID already exists
  if (sessions_.find(actual_id) != sessions_.end()) {
    return nullptr;
  }

  auto session = std::make_shared<GemmaSession>(
    actual_id, model_config_, inference_args_, threading_context_, config
  );

  sessions_[actual_id] = session;
  stats_.total_sessions_created++;
  stats_.current_active_sessions++;

  return session;
}

std::shared_ptr<GemmaSession> SessionManager::GetSession(const std::string& session_id) {
  std::shared_lock<std::shared_mutex> lock(sessions_mutex_);

  auto it = sessions_.find(session_id);
  return (it != sessions_.end()) ? it->second : nullptr;
}

bool SessionManager::RemoveSession(const std::string& session_id) {
  std::unique_lock<std::shared_mutex> lock(sessions_mutex_);

  auto it = sessions_.find(session_id);
  if (it == sessions_.end()) {
    return false;
  }

  // Terminate the session gracefully
  it->second->Terminate();
  sessions_.erase(it);
  stats_.current_active_sessions--;

  return true;
}

void SessionManager::RemoveAllSessions() {
  std::unique_lock<std::shared_mutex> lock(sessions_mutex_);

  for (auto& [id, session] : sessions_) {
    session->Terminate();
  }

  sessions_.clear();
  stats_.current_active_sessions = 0;
}

std::vector<std::string> SessionManager::ListSessions() const {
  std::shared_lock<std::shared_mutex> lock(sessions_mutex_);

  std::vector<std::string> session_ids;
  session_ids.reserve(sessions_.size());

  for (const auto& [id, session] : sessions_) {
    session_ids.push_back(id);
  }

  return session_ids;
}

size_t SessionManager::GetActiveSessionCount() const {
  std::shared_lock<std::shared_mutex> lock(sessions_mutex_);
  return sessions_.size();
}

bool SessionManager::SaveAllSessions(const Path& directory_path) const {
  std::shared_lock<std::shared_mutex> lock(sessions_mutex_);

  bool all_success = true;

  for (const auto& [id, session] : sessions_) {
    Path session_file(directory_path.path + "/" + id + ".json");
    if (!session->SaveToFile(session_file)) {
      all_success = false;
    }
  }

  return all_success;
}

bool SessionManager::LoadAllSessions(const Path& directory_path,
                                    const Gemma& gemma) {
  // This is a simplified implementation
  // In a full implementation, you would scan the directory for .json files
  // and load each one as a session

  std::unique_lock<std::shared_mutex> lock(sessions_mutex_);

  // Placeholder implementation
  return true;
}

void SessionManager::CleanupInactiveSessions(std::chrono::seconds max_idle_time) {
  std::unique_lock<std::shared_mutex> lock(sessions_mutex_);

  auto now = std::chrono::system_clock::now();
  auto it = sessions_.begin();

  while (it != sessions_.end()) {
    auto& [id, session] = *it;
    auto last_active = session->GetStats().last_active;
    auto idle_time = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_active);

    if (idle_time > max_idle_time) {
      session->Terminate();
      it = sessions_.erase(it);
      stats_.current_active_sessions--;
    } else {
      ++it;
    }
  }
}

void SessionManager::OptimizeAllSessions() {
  std::shared_lock<std::shared_mutex> lock(sessions_mutex_);

  for (auto& [id, session] : sessions_) {
    session->OptimizeKVCache();
  }
}

SessionManager::ManagerStats SessionManager::GetStats() const {
  std::shared_lock<std::shared_mutex> lock(sessions_mutex_);

  ManagerStats current_stats = stats_;
  current_stats.current_active_sessions = sessions_.size();

  // Aggregate statistics from all sessions
  current_stats.total_conversations = 0;
  current_stats.total_tokens_processed = 0;

  for (const auto& [id, session] : sessions_) {
    auto session_stats = session->GetStats();
    current_stats.total_conversations += session_stats.total_turns.load();
    current_stats.total_tokens_processed +=
        session_stats.total_input_tokens.load() +
        session_stats.total_output_tokens.load();
  }

  return current_stats;
}

std::string SessionManager::GenerateSessionId() {
  std::string prefix = "session_";
  std::string id = std::to_string(next_session_id_.fetch_add(1));
  return prefix + id + "_" + GenerateRandomId().substr(0, 8);
}

// =============================================================================
// Utility Functions Implementation
// =============================================================================

namespace session_utils {

std::string GenerateUniqueId() {
  return GenerateRandomId();
}

bool ValidateSessionConfig(const SessionConfig& config) {
  // Validate reasonable limits
  if (config.max_history_turns == 0) return false;
  if (config.context_window_tokens == 0) return false;
  if (config.memory_cleanup_threshold < config.context_window_tokens) return false;
  if (config.prefill_batch_size == 0) return false;
  if (config.max_generation_tokens == 0) return false;

  return true;
}

size_t CalculateHistoryMemoryUsage(const std::vector<ConversationMessage>& history) {
  size_t total_memory = 0;

  for (const auto& msg : history) {
    total_memory += msg.content.size();
    total_memory += msg.tokens.size() * sizeof(int);
    total_memory += sizeof(ConversationMessage);
  }

  return total_memory;
}

std::vector<ConversationMessage> CompressHistory(
    const std::vector<ConversationMessage>& history,
    size_t max_messages,
    size_t max_tokens) {

  if (history.empty()) {
    return {};
  }

  std::vector<ConversationMessage> compressed;
  compressed.reserve(std::min(max_messages, history.size()));

  size_t token_count = 0;
  size_t message_count = 0;

  // Process history in reverse order (most recent first)
  auto it = history.rbegin();
  while (it != history.rend() &&
         message_count < max_messages &&
         token_count < max_tokens) {

    if (token_count + it->tokens.size() <= max_tokens) {
      compressed.insert(compressed.begin(), *it);
      token_count += it->tokens.size();
      message_count++;
    } else {
      break;
    }
    ++it;
  }

  return compressed;
}

SessionConfig CreateInteractiveConfig() {
  SessionConfig config;
  config.max_history_turns = 50;
  config.context_window_tokens = 4096;
  config.memory_cleanup_threshold = 8192;
  config.auto_save = true;
  config.save_interval = std::chrono::seconds(300);
  config.enable_streaming = true;
  config.max_generation_tokens = 2048;
  config.generation_timeout = std::chrono::seconds(120);

  return config;
}

SessionConfig CreateBatchConfig() {
  SessionConfig config;
  config.max_history_turns = 10;
  config.context_window_tokens = 2048;
  config.memory_cleanup_threshold = 4096;
  config.auto_save = false;
  config.enable_streaming = false;
  config.max_generation_tokens = 1024;
  config.generation_timeout = std::chrono::seconds(30);
  config.prefill_batch_size = 1024;

  return config;
}

SessionConfig CreateHighThroughputConfig() {
  SessionConfig config;
  config.max_history_turns = 20;
  config.context_window_tokens = 1024;
  config.memory_cleanup_threshold = 2048;
  config.auto_save = false;
  config.enable_streaming = false;
  config.enable_kv_cache_reuse = true;
  config.max_generation_tokens = 512;
  config.generation_timeout = std::chrono::seconds(15);
  config.prefill_batch_size = 512;

  return config;
}

} // namespace session_utils

} // namespace gcpp