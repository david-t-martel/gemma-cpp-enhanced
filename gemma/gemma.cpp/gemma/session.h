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

#ifndef THIRD_PARTY_GEMMA_CPP_GEMMA_SESSION_H_
#define THIRD_PARTY_GEMMA_CPP_GEMMA_SESSION_H_

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gemma/configs.h"
#include "gemma/gemma_args.h"
#include "gemma/kv_cache.h"
#include "io/io.h"  // Path
#include "util/basics.h"  // PromptTokens
#include "util/threading_context.h"

namespace gcpp {

// Forward declarations
class Gemma;
class MatMulEnv;
struct TimingInfo;

// Session state for multi-turn conversations
enum class SessionState {
  IDLE,           // Session created but not started
  ACTIVE,         // Session actively generating responses
  PAUSED,         // Session temporarily paused
  TERMINATED,     // Session ended normally
  ERROR           // Session in error state
};

// Message role in conversation
enum class MessageRole {
  USER,
  ASSISTANT,
  SYSTEM
};

// Individual message in conversation history
struct ConversationMessage {
  MessageRole role;
  std::string content;
  PromptTokens tokens;
  std::chrono::system_clock::time_point timestamp;

  ConversationMessage(MessageRole r, const std::string& text)
      : role(r), content(text), timestamp(std::chrono::system_clock::now()) {}

  ConversationMessage(MessageRole r, const std::string& text,
                     const PromptTokens& token_ids)
      : role(r), content(text), tokens(token_ids),
        timestamp(std::chrono::system_clock::now()) {}
};

// Configuration for session behavior
struct SessionConfig {
  // Memory management
  size_t max_history_turns = 50;        // Maximum conversation turns to keep
  size_t context_window_tokens = 4096;  // Maximum tokens in context
  size_t memory_cleanup_threshold = 8192; // Trigger cleanup at this token count

  // Persistence settings
  bool auto_save = true;                 // Automatically save session state
  std::chrono::seconds save_interval{300}; // Auto-save every 5 minutes
  bool compress_history = true;          // Compress old conversation history

  // Performance settings
  size_t prefill_batch_size = 512;      // Batch size for prefill operations
  bool enable_kv_cache_reuse = true;    // Reuse KV cache between turns
  bool enable_streaming = true;          // Enable token streaming

  // Safety settings
  size_t max_generation_tokens = 2048;  // Maximum tokens per generation
  std::chrono::seconds generation_timeout{120}; // Generation timeout
  bool enable_content_filtering = false; // Content safety filtering
};

// Statistics for session performance monitoring
struct SessionStats {
  std::atomic<size_t> total_turns{0};
  std::atomic<size_t> total_input_tokens{0};
  std::atomic<size_t> total_output_tokens{0};
  std::atomic<size_t> cache_hits{0};
  std::atomic<size_t> cache_misses{0};
  std::chrono::system_clock::time_point created_at;
  std::chrono::system_clock::time_point last_active;
  std::atomic<double> avg_response_time_ms{0.0};

  SessionStats() : created_at(std::chrono::system_clock::now()),
                   last_active(std::chrono::system_clock::now()) {}

  void UpdateResponseTime(double time_ms) {
    double current = avg_response_time_ms.load();
    double new_avg = (current == 0.0) ? time_ms : (current + time_ms) / 2.0;
    avg_response_time_ms.store(new_avg);
  }
};

// Thread-safe session manager for multi-turn conversations
class GemmaSession {
public:
  explicit GemmaSession(const std::string& session_id,
                       const ModelConfig& config,
                       const InferenceArgs& inference_args,
                       ThreadingContext& ctx,
                       const SessionConfig& session_config = SessionConfig{});

  ~GemmaSession();

  // Non-copyable but movable
  GemmaSession(const GemmaSession&) = delete;
  GemmaSession& operator=(const GemmaSession&) = delete;
  GemmaSession(GemmaSession&&) = default;
  GemmaSession& operator=(GemmaSession&&) = default;

  // Session lifecycle management
  bool Initialize(const Gemma& gemma);
  void Terminate();
  void Reset();  // Clear conversation history but keep session alive

  // Conversation management
  void AddUserMessage(const std::string& message);
  void AddSystemMessage(const std::string& message);
  std::string GenerateResponse(const Gemma& gemma, MatMulEnv& env,
                              const RuntimeConfig& runtime_config);

  // Streaming interface
  using StreamCallback = std::function<bool(const std::string&, bool)>;
  void GenerateResponseStream(const Gemma& gemma, MatMulEnv& env,
                             const RuntimeConfig& runtime_config,
                             StreamCallback callback);

  // Session state and information
  SessionState GetState() const { return state_.load(); }
  const std::string& GetSessionId() const { return session_id_; }
  size_t GetConversationLength() const;
  size_t GetCurrentTokenCount() const { return current_token_count_.load(); }
  const SessionStats& GetStats() const { return stats_; }

  // History management
  std::vector<ConversationMessage> GetHistory(size_t max_messages = 0) const;
  void ClearHistory();
  void TrimHistory(size_t max_tokens);

  // Persistence operations
  bool SaveToFile(const Path& file_path) const;
  bool LoadFromFile(const Path& file_path, const Gemma& gemma);
  bool AutoSave() const;

  // KV Cache management
  void ClearKVCache();
  void OptimizeKVCache();
  size_t GetKVCacheSize() const;

  // Thread safety
  bool IsThreadSafe() const { return true; }
  std::thread::id GetOwnerThread() const { return owner_thread_; }

  // Configuration access
  const SessionConfig& GetConfig() const { return config_; }
  void UpdateConfig(const SessionConfig& new_config);

private:
  // Core session data
  std::string session_id_;
  SessionConfig config_;
  mutable std::recursive_mutex session_mutex_;
  std::atomic<SessionState> state_{SessionState::IDLE};
  std::thread::id owner_thread_;

  // Conversation state
  std::vector<ConversationMessage> conversation_history_;
  std::atomic<size_t> current_token_count_{0};
  std::unique_ptr<KVCache> kv_cache_;
  size_t current_position_{0};  // Current position in KV cache

  // Model configuration
  const ModelConfig& model_config_;
  const InferenceArgs& inference_args_;
  ThreadingContext& threading_context_;

  // Performance monitoring
  SessionStats stats_;
  mutable std::chrono::steady_clock::time_point last_auto_save_;

  // Internal methods
  void InitializeKVCache();
  PromptTokens TokenizeMessage(const Gemma& gemma, const ConversationMessage& msg) const;
  PromptTokens BuildContextTokens(const Gemma& gemma) const;
  void UpdateTokenCount();
  void PerformMemoryCleanup();
  bool ShouldAutoSave() const;
  void SetState(SessionState new_state);

  // Serialization helpers
  struct SerializedSession {
    std::string session_id;
    SessionConfig config;
    std::vector<ConversationMessage> history;
    SessionStats stats;
    size_t current_position;
    std::chrono::system_clock::time_point saved_at;
  };

  SerializedSession SerializeSession() const;
  bool DeserializeSession(const SerializedSession& data, const Gemma& gemma);
};

// Session manager for handling multiple concurrent sessions
class SessionManager {
public:
  explicit SessionManager(const ModelConfig& config,
                         const InferenceArgs& inference_args,
                         ThreadingContext& ctx);

  ~SessionManager();

  // Session lifecycle
  std::shared_ptr<GemmaSession> CreateSession(
      const std::string& session_id = "",
      const SessionConfig& config = SessionConfig{});

  std::shared_ptr<GemmaSession> GetSession(const std::string& session_id);
  bool RemoveSession(const std::string& session_id);
  void RemoveAllSessions();

  // Session discovery
  std::vector<std::string> ListSessions() const;
  size_t GetActiveSessionCount() const;

  // Persistence operations
  bool SaveAllSessions(const Path& directory_path) const;
  bool LoadAllSessions(const Path& directory_path, const Gemma& gemma);

  // Resource management
  void CleanupInactiveSessions(std::chrono::seconds max_idle_time);
  void OptimizeAllSessions();

  // Statistics
  struct ManagerStats {
    size_t total_sessions_created{0};
    size_t current_active_sessions{0};
    size_t total_conversations{0};
    size_t total_tokens_processed{0};
    std::chrono::system_clock::time_point created_at;
  };

  ManagerStats GetStats() const;

private:
  const ModelConfig& model_config_;
  const InferenceArgs& inference_args_;
  ThreadingContext& threading_context_;

  mutable std::shared_mutex sessions_mutex_;
  std::unordered_map<std::string, std::shared_ptr<GemmaSession>> sessions_;

  mutable ManagerStats stats_;
  std::atomic<size_t> next_session_id_{1};

  std::string GenerateSessionId();
};

// Utility functions for session management
namespace session_utils {

// Generate a unique session ID
std::string GenerateUniqueId();

// Validate session configuration
bool ValidateSessionConfig(const SessionConfig& config);

// Calculate memory usage for conversation history
size_t CalculateHistoryMemoryUsage(const std::vector<ConversationMessage>& history);

// Compress conversation history for long-term storage
std::vector<ConversationMessage> CompressHistory(
    const std::vector<ConversationMessage>& history,
    size_t max_messages, size_t max_tokens);

// Create default session configuration for different use cases
SessionConfig CreateInteractiveConfig();    // For interactive chat
SessionConfig CreateBatchConfig();         // For batch processing
SessionConfig CreateHighThroughputConfig(); // For high-throughput scenarios

}  // namespace session_utils

}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_GEMMA_SESSION_H_