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

#ifndef THIRD_PARTY_GEMMA_CPP_INTERFACES_MCP_MCPSERVER_H_
#define THIRD_PARTY_GEMMA_CPP_INTERFACES_MCP_MCPSERVER_H_

#include "MCPProtocol.h"
#include "MCPTransport.h"
#include "MCPTools.h"

#include "gemma/gemma.h"
#include "gemma/configs.h"
#include "gemma/kv_cache.h"
#include "util/threading_context.h"
#include "ops/matmul.h"

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <future>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <chrono>
#include <functional>

namespace gcpp {
namespace mcp {

// Server configuration
struct MCPServerConfig {
  // Server identification
  std::string server_name = "gemma-mcp-server";
  std::string server_version = "1.0.0";
  std::string protocol_version = "2024-11-05";
  
  // Gemma configuration
  std::string model_path;
  std::string tokenizer_path;
  InferenceArgs inference_args;
  
  // Runtime defaults
  RuntimeConfig default_runtime_config;
  
  // Session management
  std::chrono::seconds session_timeout = std::chrono::seconds(3600);  // 1 hour
  size_t max_sessions = 100;
  bool enable_session_cleanup = true;
  
  // Performance settings
  size_t max_concurrent_requests = 10;
  std::chrono::milliseconds request_timeout = std::chrono::milliseconds(30000);  // 30 seconds
  bool enable_streaming = true;
  
  // Logging and monitoring
  int log_level = 1;  // 0=none, 1=basic, 2=verbose
  bool enable_performance_metrics = true;
  bool enable_request_logging = false;
  
  // Transport settings
  std::vector<std::string> transport_uris = {"stdio:"};
  bool auto_detect_transports = true;
  
  // Security settings
  std::vector<std::string> allowed_tools;  // Empty = all tools allowed
  bool require_tool_validation = true;
  size_t max_request_size = 1024 * 1024;  // 1MB
};

// Server capabilities
struct MCPServerCapabilities {
  bool supports_tools = true;
  bool supports_resources = false;
  bool supports_prompts = false;
  bool supports_sampling = false;
  bool supports_logging = false;
  
  // Tool-specific capabilities
  bool supports_streaming = true;
  bool supports_sessions = true;
  bool supports_backend_switching = false;
  
  nlohmann::json to_json() const;
  static MCPServerCapabilities from_json(const nlohmann::json& json);
};

// Request statistics
struct RequestStats {
  std::atomic<size_t> total_requests{0};
  std::atomic<size_t> successful_requests{0};
  std::atomic<size_t> failed_requests{0};
  std::atomic<size_t> active_requests{0};
  
  std::atomic<double> total_execution_time{0.0};
  std::atomic<size_t> total_tokens_processed{0};
  std::atomic<size_t> total_tokens_generated{0};
  
  std::chrono::steady_clock::time_point start_time;
  
  RequestStats() : start_time(std::chrono::steady_clock::now()) {}
  
  nlohmann::json to_json() const;
  void reset();
};

// Server status
enum class ServerStatus {
  STOPPED,
  STARTING,
  RUNNING,
  STOPPING,
  ERROR
};

// Request context for tracking and management
struct RequestContext {
  std::string request_id;
  std::chrono::steady_clock::time_point start_time;
  std::string method;
  std::string transport_name;
  std::future<JsonRpcResponse> response_future;
  bool is_streaming = false;
  
  RequestContext(const std::string& id, const std::string& method_name, 
                const std::string& transport)
    : request_id(id), start_time(std::chrono::steady_clock::now()), 
      method(method_name), transport_name(transport) {}
};

// Main MCP Server class
class MCPServer {
public:
  explicit MCPServer(const MCPServerConfig& config);
  ~MCPServer();

  // Server lifecycle
  bool Initialize();
  bool Start();
  void Stop();
  void Shutdown();
  
  // Status and information
  ServerStatus GetStatus() const;
  std::string GetStatusString() const;
  MCPServerConfig GetConfig() const;
  MCPServerCapabilities GetCapabilities() const;
  RequestStats GetStats() const;
  
  // Model management
  bool LoadModel(const std::string& model_path, const std::string& tokenizer_path = "");
  bool IsModelLoaded() const;
  std::string GetModelInfo() const;
  
  // Transport management
  bool AddTransport(std::unique_ptr<MCPTransport> transport, const std::string& name = "");
  bool RemoveTransport(const std::string& name);
  std::vector<std::string> GetTransportNames() const;
  std::map<std::string, TransportStatus> GetTransportStatuses() const;
  
  // Tool management
  bool RegisterTool(std::unique_ptr<MCPTool> tool);
  bool UnregisterTool(const std::string& name);
  std::vector<Tool> GetRegisteredTools() const;
  
  // Session management
  std::string CreateSession();
  bool DestroySession(const std::string& session_id);
  std::vector<std::string> GetSessionIds() const;
  nlohmann::json GetSessionInfo(const std::string& session_id) const;
  void CleanupExpiredSessions();
  
  // Request handling
  void ProcessRequest(const std::string& request_data, const std::string& transport_name);
  std::future<JsonRpcResponse> HandleRequestAsync(const JsonRpcRequest& request, 
                                                  const std::string& transport_name);
  
  // Batch request handling
  std::future<BatchResponse> HandleBatchRequestAsync(const BatchRequest& batch_request,
                                                     const std::string& transport_name);
  
  // Streaming support
  void StartStream(const std::string& request_id, StreamCallback callback);
  void StopStream(const std::string& request_id);
  bool IsStreaming(const std::string& request_id) const;
  
  // Configuration updates
  void UpdateRuntimeConfig(const RuntimeConfig& config);
  RuntimeConfig GetRuntimeConfig() const;
  
  // Monitoring and diagnostics
  nlohmann::json GetDiagnostics() const;
  void ResetStats();
  
  // Event callbacks
  using ErrorCallback = std::function<void(const std::string& error, const std::string& details)>;
  using RequestCallback = std::function<void(const std::string& method, double duration_ms)>;
  
  void SetErrorCallback(ErrorCallback callback);
  void SetRequestCallback(RequestCallback callback);

private:
  // Core request handlers
  JsonRpcResponse HandleInitialize(const nlohmann::json& params);
  JsonRpcResponse HandleListTools(const nlohmann::json& params);
  JsonRpcResponse HandleCallTool(const nlohmann::json& params, const std::string& transport_name);
  JsonRpcResponse HandleListResources(const nlohmann::json& params);
  JsonRpcResponse HandleNotification(const std::string& method, const nlohmann::json& params);
  
  // Tool execution
  std::future<JsonRpcResponse> ExecuteToolAsync(const std::string& tool_name, 
                                               const nlohmann::json& args,
                                               const RequestId& request_id,
                                               const std::string& transport_name);
  
  // Streaming tool execution
  std::future<JsonRpcResponse> ExecuteToolStreamAsync(const std::string& tool_name,
                                                     const nlohmann::json& args,
                                                     const RequestId& request_id,
                                                     const std::string& transport_name);
  
  // Transport callbacks
  void OnTransportMessage(const std::string& message, const std::string& transport_name);
  void OnTransportStatus(TransportStatus status, const std::string& details);
  
  // Request management
  std::string GenerateRequestId();
  void AddActiveRequest(const std::string& request_id, std::unique_ptr<RequestContext> context);
  void RemoveActiveRequest(const std::string& request_id);
  std::shared_ptr<RequestContext> GetActiveRequest(const std::string& request_id) const;
  
  // Session management helpers
  ToolContext CreateToolContext(const std::string& session_id = "");
  void UpdateSessionPosition(const std::string& session_id, size_t new_pos);
  
  // Error handling
  void LogError(const std::string& error, const std::string& details = "");
  void LogRequest(const std::string& method, double duration_ms, bool success);
  JsonRpcResponse CreateErrorResponse(const RequestId& id, ErrorCode code, 
                                     const std::string& message, 
                                     const std::string& details = "");
  
  // Threading and async management
  void RequestWorker();
  void SessionCleanupWorker();
  void StatsUpdateWorker();
  
  // Validation
  bool ValidateRequest(const JsonRpcRequest& request) const;
  bool IsToolAllowed(const std::string& tool_name) const;
  bool ValidateRequestSize(const std::string& data) const;
  
  // Configuration and state
  MCPServerConfig config_;
  MCPServerCapabilities capabilities_;
  std::atomic<ServerStatus> status_{ServerStatus::STOPPED};
  
  // Core components
  std::shared_ptr<Gemma> gemma_;
  std::shared_ptr<ThreadingContext> threading_ctx_;
  std::shared_ptr<MatMulEnv> matmul_env_;
  std::unique_ptr<ToolRegistry> tool_registry_;
  std::unique_ptr<SessionManager> session_manager_;
  std::unique_ptr<ConnectionManager> connection_manager_;
  
  // Runtime state
  RuntimeConfig runtime_config_;
  std::string default_session_id_;
  
  // Request handling
  std::unordered_map<std::string, std::unique_ptr<RequestContext>> active_requests_;
  std::unordered_map<std::string, StreamCallback> active_streams_;
  mutable std::shared_mutex requests_mutex_;
  mutable std::shared_mutex streams_mutex_;
  
  // Statistics and monitoring
  RequestStats stats_;
  mutable std::mutex stats_mutex_;
  
  // Threading
  std::vector<std::thread> worker_threads_;
  std::queue<std::function<void()>> request_queue_;
  std::mutex request_queue_mutex_;
  std::condition_variable request_queue_cv_;
  std::atomic<bool> shutdown_requested_{false};
  
  // Callbacks
  ErrorCallback error_callback_;
  RequestCallback request_callback_;
  mutable std::mutex callback_mutex_;
  
  // Request ID generation
  std::atomic<size_t> request_counter_{0};
  
  // Background tasks
  std::thread session_cleanup_thread_;
  std::thread stats_update_thread_;
};

// Utility functions for server management
namespace server_utils {

// Configuration helpers
MCPServerConfig CreateDefaultConfig();
MCPServerConfig LoadConfigFromFile(const std::string& config_file);
bool SaveConfigToFile(const MCPServerConfig& config, const std::string& config_file);

// Server factory
std::unique_ptr<MCPServer> CreateStandardServer(const MCPServerConfig& config);
std::unique_ptr<MCPServer> CreateServerFromConfig(const std::string& config_file);

// Model loading helpers
bool ValidateModelPath(const std::string& model_path);
bool ValidateTokenizerPath(const std::string& tokenizer_path);
InferenceArgs CreateDefaultInferenceArgs();

// Transport helpers
std::vector<std::unique_ptr<MCPTransport>> CreateTransportsFromUris(
    const std::vector<std::string>& uris);

// Logging helpers
void SetupLogging(int log_level);
void LogServerEvent(const std::string& event, const std::string& details = "");

// Performance helpers
void WarmupModel(Gemma& gemma, ThreadingContext& ctx, MatMulEnv& env);
void OptimizeRuntimeConfig(RuntimeConfig& config, const ModelConfig& model_config);

}  // namespace server_utils

}  // namespace mcp
}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_INTERFACES_MCP_MCPSERVER_H_