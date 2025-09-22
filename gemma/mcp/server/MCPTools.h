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

#ifndef THIRD_PARTY_GEMMA_CPP_INTERFACES_MCP_MCPTOOLS_H_
#define THIRD_PARTY_GEMMA_CPP_INTERFACES_MCP_MCPTOOLS_H_

#include "MCPProtocol.h"
#include "gemma/gemma.h"
#include "gemma/configs.h"
#include "util/threading_context.h"
#include "ops/matmul.h"

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <future>
#include <atomic>
#include <mutex>

namespace gcpp {
namespace mcp {

// Forward declarations
class MCPServer;
class ToolRegistry;

// Tool execution context
struct ToolContext {
  // Reference to the Gemma instance
  std::shared_ptr<Gemma> gemma;
  
  // Threading and computation contexts
  std::shared_ptr<ThreadingContext> threading_ctx;
  std::shared_ptr<MatMulEnv> matmul_env;
  
  // Current runtime configuration
  RuntimeConfig runtime_config;
  
  // Session management
  std::string session_id;
  std::shared_ptr<KVCache> kv_cache;
  size_t current_pos = 0;
  
  // Tool registry reference
  ToolRegistry* registry = nullptr;
  
  // Timing and performance info
  mutable TimingInfo timing_info;
};

// Tool execution result
struct ToolResult {
  bool success = false;
  std::vector<TextContent> content;
  std::optional<JsonRpcError> error;
  
  // Performance metrics
  double execution_time_ms = 0.0;
  size_t tokens_processed = 0;
  size_t tokens_generated = 0;
  
  // Create success result
  static ToolResult Success(const std::vector<TextContent>& content, 
                           double execution_time_ms = 0.0);
  static ToolResult Success(const std::string& text_content, 
                           double execution_time_ms = 0.0);
  
  // Create error result
  static ToolResult Error(const JsonRpcError& error);
  static ToolResult Error(ErrorCode code, const std::string& message, 
                         const std::string& details = "");
};

// Stream callback for real-time token generation
using StreamCallback = std::function<void(const std::string& token, bool is_final)>;

// Base tool interface
class MCPTool {
public:
  virtual ~MCPTool() = default;
  
  // Tool metadata
  virtual std::string GetName() const = 0;
  virtual std::string GetDescription() const = 0;
  virtual ToolSchema GetInputSchema() const = 0;
  
  // Tool execution
  virtual std::future<ToolResult> ExecuteAsync(const nlohmann::json& args, 
                                               ToolContext& context) = 0;
  
  // Synchronous execution (default implementation uses async)
  virtual ToolResult Execute(const nlohmann::json& args, ToolContext& context);
  
  // Streaming execution (optional)
  virtual std::future<ToolResult> ExecuteStreamAsync(const nlohmann::json& args, 
                                                     ToolContext& context,
                                                     StreamCallback stream_callback);
  
  // Validation
  virtual bool ValidateArguments(const nlohmann::json& args) const;
  
  // Tool capabilities
  virtual bool SupportsStreaming() const { return false; }
  virtual bool RequiresSession() const { return false; }
  virtual bool IsStateful() const { return false; }
  
  // Convert to MCP Tool format
  Tool ToMCPTool() const;
};

// Tool registration and management
class ToolRegistry {
public:
  ToolRegistry() = default;
  ~ToolRegistry() = default;

  // Tool registration
  bool RegisterTool(std::unique_ptr<MCPTool> tool);
  bool UnregisterTool(const std::string& name);
  
  // Tool discovery
  std::vector<Tool> GetAllTools() const;
  MCPTool* GetTool(const std::string& name) const;
  bool HasTool(const std::string& name) const;
  std::vector<std::string> GetToolNames() const;
  
  // Tool execution
  std::future<ToolResult> ExecuteToolAsync(const std::string& name, 
                                          const nlohmann::json& args,
                                          ToolContext& context);
  
  ToolResult ExecuteTool(const std::string& name, 
                        const nlohmann::json& args,
                        ToolContext& context);
  
  // Streaming execution
  std::future<ToolResult> ExecuteToolStreamAsync(const std::string& name,
                                                 const nlohmann::json& args,
                                                 ToolContext& context,
                                                 StreamCallback stream_callback);
  
  // Tool filtering and categorization
  std::vector<Tool> GetToolsByCategory(const std::string& category) const;
  std::vector<Tool> GetStreamingTools() const;
  std::vector<Tool> GetStatefulTools() const;

private:
  std::unordered_map<std::string, std::unique_ptr<MCPTool>> tools_;
  mutable std::shared_mutex tools_mutex_;
};

// Concrete tool implementations

// Text generation tool
class GenerateTextTool : public MCPTool {
public:
  std::string GetName() const override { return "generate_text"; }
  std::string GetDescription() const override;
  ToolSchema GetInputSchema() const override;
  
  std::future<ToolResult> ExecuteAsync(const nlohmann::json& args, 
                                      ToolContext& context) override;
  
  std::future<ToolResult> ExecuteStreamAsync(const nlohmann::json& args, 
                                            ToolContext& context,
                                            StreamCallback stream_callback) override;
  
  bool SupportsStreaming() const override { return true; }
  bool RequiresSession() const override { return true; }
  bool IsStateful() const override { return true; }

private:
  ToolResult GenerateText(const std::string& prompt, 
                         const nlohmann::json& config,
                         ToolContext& context,
                         StreamCallback stream_callback = nullptr);
};

// Token counting tool
class CountTokensTool : public MCPTool {
public:
  std::string GetName() const override { return "count_tokens"; }
  std::string GetDescription() const override;
  ToolSchema GetInputSchema() const override;
  
  std::future<ToolResult> ExecuteAsync(const nlohmann::json& args, 
                                      ToolContext& context) override;

private:
  ToolResult CountTokens(const std::string& text, ToolContext& context);
};

// Model information tool
class GetModelInfoTool : public MCPTool {
public:
  std::string GetName() const override { return "get_model_info"; }
  std::string GetDescription() const override;
  ToolSchema GetInputSchema() const override;
  
  std::future<ToolResult> ExecuteAsync(const nlohmann::json& args, 
                                      ToolContext& context) override;

private:
  ToolResult GetModelInfo(ToolContext& context);
};

// Session management tool
class ListSessionsTool : public MCPTool {
public:
  std::string GetName() const override { return "list_sessions"; }
  std::string GetDescription() const override;
  ToolSchema GetInputSchema() const override;
  
  std::future<ToolResult> ExecuteAsync(const nlohmann::json& args, 
                                      ToolContext& context) override;

private:
  ToolResult ListSessions(ToolContext& context);
};

// Backend switching tool
class SetBackendTool : public MCPTool {
public:
  std::string GetName() const override { return "set_backend"; }
  std::string GetDescription() const override;
  ToolSchema GetInputSchema() const override;
  
  std::future<ToolResult> ExecuteAsync(const nlohmann::json& args, 
                                      ToolContext& context) override;

private:
  ToolResult SetBackend(const std::string& backend, ToolContext& context);
};

// Chat completion tool (multi-turn conversation)
class ChatCompletionTool : public MCPTool {
public:
  std::string GetName() const override { return "chat_completion"; }
  std::string GetDescription() const override;
  ToolSchema GetInputSchema() const override;
  
  std::future<ToolResult> ExecuteAsync(const nlohmann::json& args, 
                                      ToolContext& context) override;
  
  std::future<ToolResult> ExecuteStreamAsync(const nlohmann::json& args, 
                                            ToolContext& context,
                                            StreamCallback stream_callback) override;
  
  bool SupportsStreaming() const override { return true; }
  bool RequiresSession() const override { return true; }
  bool IsStateful() const override { return true; }

private:
  ToolResult ProcessChatCompletion(const nlohmann::json& messages,
                                  const nlohmann::json& config,
                                  ToolContext& context,
                                  StreamCallback stream_callback = nullptr);
  
  std::string FormatChatPrompt(const nlohmann::json& messages, ToolContext& context);
};

// Embedding generation tool (if supported by model)
class GenerateEmbeddingsTool : public MCPTool {
public:
  std::string GetName() const override { return "generate_embeddings"; }
  std::string GetDescription() const override;
  ToolSchema GetInputSchema() const override;
  
  std::future<ToolResult> ExecuteAsync(const nlohmann::json& args, 
                                      ToolContext& context) override;

private:
  ToolResult GenerateEmbeddings(const std::string& text, ToolContext& context);
};

// Model benchmark tool
class BenchmarkTool : public MCPTool {
public:
  std::string GetName() const override { return "benchmark"; }
  std::string GetDescription() const override;
  ToolSchema GetInputSchema() const override;
  
  std::future<ToolResult> ExecuteAsync(const nlohmann::json& args, 
                                      ToolContext& context) override;

private:
  ToolResult RunBenchmark(const nlohmann::json& config, ToolContext& context);
};

// Session management utilities
class SessionManager {
public:
  SessionManager() = default;
  ~SessionManager() = default;

  // Session lifecycle
  std::string CreateSession(std::shared_ptr<Gemma> gemma, 
                           std::shared_ptr<ThreadingContext> threading_ctx);
  bool DestroySession(const std::string& session_id);
  bool HasSession(const std::string& session_id) const;
  
  // Session access
  std::shared_ptr<KVCache> GetKVCache(const std::string& session_id) const;
  size_t GetPosition(const std::string& session_id) const;
  void SetPosition(const std::string& session_id, size_t pos);
  
  // Session info
  std::vector<std::string> GetSessionIds() const;
  size_t GetSessionCount() const;
  nlohmann::json GetSessionInfo(const std::string& session_id) const;
  
  // Cleanup
  void CleanupExpiredSessions(std::chrono::seconds max_idle_time);
  void CleanupAllSessions();

private:
  struct SessionData {
    std::shared_ptr<KVCache> kv_cache;
    std::chrono::steady_clock::time_point last_access;
    size_t current_pos = 0;
    size_t max_seq_len = 0;
  };
  
  std::unordered_map<std::string, SessionData> sessions_;
  mutable std::shared_mutex sessions_mutex_;
  std::atomic<size_t> session_counter_{0};
};

// Default tool factory for creating standard tools
class DefaultToolFactory {
public:
  static std::unique_ptr<ToolRegistry> CreateStandardRegistry();
  static std::vector<std::unique_ptr<MCPTool>> CreateStandardTools();
  
  // Individual tool creators
  static std::unique_ptr<GenerateTextTool> CreateGenerateTextTool();
  static std::unique_ptr<CountTokensTool> CreateCountTokensTool();
  static std::unique_ptr<GetModelInfoTool> CreateGetModelInfoTool();
  static std::unique_ptr<ListSessionsTool> CreateListSessionsTool();
  static std::unique_ptr<SetBackendTool> CreateSetBackendTool();
  static std::unique_ptr<ChatCompletionTool> CreateChatCompletionTool();
  static std::unique_ptr<GenerateEmbeddingsTool> CreateGenerateEmbeddingsTool();
  static std::unique_ptr<BenchmarkTool> CreateBenchmarkTool();
};

// Utility functions for tool development
namespace tool_utils {

// Argument validation helpers
bool ValidateStringArg(const nlohmann::json& args, const std::string& key, 
                      bool required = true);
bool ValidateIntArg(const nlohmann::json& args, const std::string& key, 
                   int min_val = INT_MIN, int max_val = INT_MAX, bool required = true);
bool ValidateFloatArg(const nlohmann::json& args, const std::string& key, 
                     double min_val = -std::numeric_limits<double>::infinity(), 
                     double max_val = std::numeric_limits<double>::infinity(), 
                     bool required = true);
bool ValidateArrayArg(const nlohmann::json& args, const std::string& key, 
                     bool required = true);

// Argument extraction helpers
std::string GetStringArg(const nlohmann::json& args, const std::string& key, 
                        const std::string& default_value = "");
int GetIntArg(const nlohmann::json& args, const std::string& key, int default_value = 0);
double GetFloatArg(const nlohmann::json& args, const std::string& key, double default_value = 0.0);
nlohmann::json GetArrayArg(const nlohmann::json& args, const std::string& key, 
                          const nlohmann::json& default_value = nlohmann::json::array());

// Runtime config helpers
RuntimeConfig CreateRuntimeConfig(const nlohmann::json& config);
void UpdateRuntimeConfigFromArgs(RuntimeConfig& config, const nlohmann::json& args);

// Error creation helpers
JsonRpcError CreateValidationError(const std::string& field, const std::string& issue);
JsonRpcError CreateExecutionError(const std::string& operation, const std::string& details);

// Performance measurement
class PerformanceTimer {
public:
  PerformanceTimer();
  void Start();
  void Stop();
  double GetElapsedMs() const;
  void Reset();

private:
  std::chrono::high_resolution_clock::time_point start_time_;
  std::chrono::high_resolution_clock::time_point end_time_;
  bool running_ = false;
};

}  // namespace tool_utils

}  // namespace mcp
}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_INTERFACES_MCP_MCPTOOLS_H_