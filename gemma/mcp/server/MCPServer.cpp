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

#include "MCPServer.h"
#include "gemma/model_store.h"
#include "io/blob_store.h"
#include "util/threading.h"

#include <chrono>
#include <sstream>
#include <algorithm>
#include <random>
#include <iomanip>
#include <fstream>

namespace gcpp {
namespace mcp {

// MCPServerCapabilities implementation
nlohmann::json MCPServerCapabilities::to_json() const {
  return nlohmann::json{
    {"tools", supports_tools},
    {"resources", supports_resources},
    {"prompts", supports_prompts},
    {"sampling", supports_sampling},
    {"logging", supports_logging},
    {"streaming", supports_streaming},
    {"sessions", supports_sessions},
    {"backend_switching", supports_backend_switching}
  };
}

MCPServerCapabilities MCPServerCapabilities::from_json(const nlohmann::json& json) {
  MCPServerCapabilities caps;
  if (json.contains("tools")) caps.supports_tools = json["tools"].get<bool>();
  if (json.contains("resources")) caps.supports_resources = json["resources"].get<bool>();
  if (json.contains("prompts")) caps.supports_prompts = json["prompts"].get<bool>();
  if (json.contains("sampling")) caps.supports_sampling = json["sampling"].get<bool>();
  if (json.contains("logging")) caps.supports_logging = json["logging"].get<bool>();
  if (json.contains("streaming")) caps.supports_streaming = json["streaming"].get<bool>();
  if (json.contains("sessions")) caps.supports_sessions = json["sessions"].get<bool>();
  if (json.contains("backend_switching")) caps.supports_backend_switching = json["backend_switching"].get<bool>();
  return caps;
}

// RequestStats implementation
nlohmann::json RequestStats::to_json() const {
  auto now = std::chrono::steady_clock::now();
  auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
  
  size_t total = total_requests.load();
  double avg_time = total > 0 ? total_execution_time.load() / total : 0.0;
  
  return nlohmann::json{
    {"total_requests", total},
    {"successful_requests", successful_requests.load()},
    {"failed_requests", failed_requests.load()},
    {"active_requests", active_requests.load()},
    {"average_execution_time_ms", avg_time},
    {"total_tokens_processed", total_tokens_processed.load()},
    {"total_tokens_generated", total_tokens_generated.load()},
    {"uptime_seconds", uptime.count()}
  };
}

void RequestStats::reset() {
  total_requests.store(0);
  successful_requests.store(0);
  failed_requests.store(0);
  active_requests.store(0);
  total_execution_time.store(0.0);
  total_tokens_processed.store(0);
  total_tokens_generated.store(0);
  start_time = std::chrono::steady_clock::now();
}

// MCPServer implementation
MCPServer::MCPServer(const MCPServerConfig& config) 
    : config_(config), runtime_config_(config.default_runtime_config) {
  
  // Initialize components
  tool_registry_ = DefaultToolFactory::CreateStandardRegistry();
  session_manager_ = std::make_unique<SessionManager>();
  connection_manager_ = std::make_unique<ConnectionManager>();
  
  // Setup capabilities
  capabilities_.supports_tools = true;
  capabilities_.supports_streaming = config_.enable_streaming;
  capabilities_.supports_sessions = true;
  
  // Setup default runtime config
  if (runtime_config_.max_tokens == 0) {
    runtime_config_.max_tokens = 512;
  }
  if (runtime_config_.temperature == 0.0f) {
    runtime_config_.temperature = 0.8f;
  }
  if (runtime_config_.top_k == 0) {
    runtime_config_.top_k = 40;
  }
  if (runtime_config_.top_p == 0.0f) {
    runtime_config_.top_p = 0.95f;
  }
}

MCPServer::~MCPServer() {
  Shutdown();
}

bool MCPServer::Initialize() {
  if (status_.load() != ServerStatus::STOPPED) {
    return false;
  }
  
  status_.store(ServerStatus::STARTING);
  
  try {
    // Load model if specified
    if (!config_.model_path.empty()) {
      if (!LoadModel(config_.model_path, config_.tokenizer_path)) {
        LogError("Failed to load model", config_.model_path);
        status_.store(ServerStatus::ERROR);
        return false;
      }
    }
    
    // Setup transports
    for (const auto& uri : config_.transport_uris) {
      auto transport = TransportFactory::CreateTransport(uri);
      if (transport) {
        connection_manager_->AddTransport(std::move(transport));
      } else {
        LogError("Failed to create transport", uri);
      }
    }
    
    // Setup transport callbacks
    connection_manager_->SetMessageCallback([this](const std::string& message) {
      OnTransportMessage(message, "default");
    });
    
    connection_manager_->SetStatusCallback([this](TransportStatus status, const std::string& details) {
      OnTransportStatus(status, details);
    });
    
    // Start worker threads
    size_t num_workers = std::max(1u, std::thread::hardware_concurrency() / 2);
    for (size_t i = 0; i < num_workers; ++i) {
      worker_threads_.emplace_back(&MCPServer::RequestWorker, this);
    }
    
    // Start background tasks
    if (config_.enable_session_cleanup) {
      session_cleanup_thread_ = std::thread(&MCPServer::SessionCleanupWorker, this);
    }
    
    if (config_.enable_performance_metrics) {
      stats_update_thread_ = std::thread(&MCPServer::StatsUpdateWorker, this);
    }
    
    status_.store(ServerStatus::RUNNING);
    return true;
    
  } catch (const std::exception& e) {
    LogError("Initialization failed", e.what());
    status_.store(ServerStatus::ERROR);
    return false;
  }
}

bool MCPServer::Start() {
  if (status_.load() != ServerStatus::RUNNING) {
    if (!Initialize()) {
      return false;
    }
  }
  
  // Start all transports
  if (!connection_manager_->StartAll()) {
    LogError("Failed to start all transports");
    return false;
  }
  
  return true;
}

void MCPServer::Stop() {
  if (status_.load() == ServerStatus::STOPPED) {
    return;
  }
  
  status_.store(ServerStatus::STOPPING);
  
  // Stop accepting new requests
  shutdown_requested_.store(true);
  
  // Stop transports
  connection_manager_->StopAll();
  
  // Wait for active requests to complete (with timeout)
  auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (stats_.active_requests.load() > 0 && 
         std::chrono::steady_clock::now() < timeout) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  
  status_.store(ServerStatus::STOPPED);
}

void MCPServer::Shutdown() {
  Stop();
  
  // Stop worker threads
  {
    std::lock_guard<std::mutex> lock(request_queue_mutex_);
    request_queue_cv_.notify_all();
  }
  
  for (auto& thread : worker_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  worker_threads_.clear();
  
  // Stop background threads
  if (session_cleanup_thread_.joinable()) {
    session_cleanup_thread_.join();
  }
  
  if (stats_update_thread_.joinable()) {
    stats_update_thread_.join();
  }
  
  // Cleanup sessions
  if (session_manager_) {
    session_manager_->CleanupAllSessions();
  }
  
  // Clear active requests
  {
    std::unique_lock<std::shared_mutex> lock(requests_mutex_);
    active_requests_.clear();
  }
}

ServerStatus MCPServer::GetStatus() const {
  return status_.load();
}

std::string MCPServer::GetStatusString() const {
  switch (status_.load()) {
    case ServerStatus::STOPPED: return "stopped";
    case ServerStatus::STARTING: return "starting";
    case ServerStatus::RUNNING: return "running";
    case ServerStatus::STOPPING: return "stopping";
    case ServerStatus::ERROR: return "error";
    default: return "unknown";
  }
}

MCPServerConfig MCPServer::GetConfig() const {
  return config_;
}

MCPServerCapabilities MCPServer::GetCapabilities() const {
  return capabilities_;
}

RequestStats MCPServer::GetStats() const {
  std::lock_guard<std::mutex> lock(stats_mutex_);
  return stats_;
}

bool MCPServer::LoadModel(const std::string& model_path, const std::string& tokenizer_path) {
  try {
    // Create loading arguments
    LoaderArgs loader_args;
    loader_args.model_type = ModelType::GEMMA_2B;  // Default, will be auto-detected
    
    // Setup blob store
    BlobStore blob_store(NestedPools{});
    BlobReader blob_reader(&blob_store, "model");
    
    // Create threading context
    threading_ctx_ = std::make_shared<ThreadingContext>();
    
    // Create MatMul environment
    matmul_env_ = std::make_shared<MatMulEnv>(*threading_ctx_);
    
    // Load model
    gemma_ = std::make_shared<Gemma>(loader_args, config_.inference_args, *threading_ctx_);
    
    // Create default session
    default_session_id_ = session_manager_->CreateSession(gemma_, threading_ctx_);
    
    return true;
    
  } catch (const std::exception& e) {
    LogError("Model loading failed", e.what());
    return false;
  }
}

bool MCPServer::IsModelLoaded() const {
  return gemma_ != nullptr;
}

std::string MCPServer::GetModelInfo() const {
  if (!gemma_) {
    return "No model loaded";
  }
  
  const auto& config = gemma_->Config();
  nlohmann::json info{
    {"model_type", ToString(config.model)},
    {"vocab_size", config.vocab_size},
    {"seq_len", config.seq_len},
    {"model_dim", config.model_dim},
    {"num_layers", config.num_layers}
  };
  
  return info.dump(2);
}

bool MCPServer::AddTransport(std::unique_ptr<MCPTransport> transport, const std::string& name) {
  return connection_manager_->AddTransport(std::move(transport), name);
}

bool MCPServer::RemoveTransport(const std::string& name) {
  // Note: ConnectionManager doesn't have RemoveTransport method in current design
  // This would need to be added to the ConnectionManager class
  return false;
}

std::vector<std::string> MCPServer::GetTransportNames() const {
  return connection_manager_->GetTransportNames();
}

std::map<std::string, TransportStatus> MCPServer::GetTransportStatuses() const {
  return connection_manager_->GetAllStatuses();
}

bool MCPServer::RegisterTool(std::unique_ptr<MCPTool> tool) {
  return tool_registry_->RegisterTool(std::move(tool));
}

bool MCPServer::UnregisterTool(const std::string& name) {
  return tool_registry_->UnregisterTool(name);
}

std::vector<Tool> MCPServer::GetRegisteredTools() const {
  return tool_registry_->GetAllTools();
}

std::string MCPServer::CreateSession() {
  if (!gemma_ || !threading_ctx_) {
    return "";
  }
  
  return session_manager_->CreateSession(gemma_, threading_ctx_);
}

bool MCPServer::DestroySession(const std::string& session_id) {
  return session_manager_->DestroySession(session_id);
}

std::vector<std::string> MCPServer::GetSessionIds() const {
  return session_manager_->GetSessionIds();
}

nlohmann::json MCPServer::GetSessionInfo(const std::string& session_id) const {
  return session_manager_->GetSessionInfo(session_id);
}

void MCPServer::CleanupExpiredSessions() {
  session_manager_->CleanupExpiredSessions(config_.session_timeout);
}

void MCPServer::ProcessRequest(const std::string& request_data, const std::string& transport_name) {
  if (shutdown_requested_.load()) {
    return;
  }
  
  stats_.total_requests.fetch_add(1);
  stats_.active_requests.fetch_add(1);
  
  // Queue request for processing
  {
    std::lock_guard<std::mutex> lock(request_queue_mutex_);
    request_queue_.push([this, request_data, transport_name]() {
      auto start_time = std::chrono::steady_clock::now();
      
      try {
        // Parse request
        auto parse_result = ProtocolParser::ParseRequest(request_data);
        
        if (std::holds_alternative<JsonRpcError>(parse_result)) {
          // Parse error
          auto error = std::get<JsonRpcError>(parse_result);
          std::string error_response = ProtocolParser::FormatError(error);
          connection_manager_->SendMessage(error_response, transport_name);
          
          stats_.failed_requests.fetch_add(1);
          return;
        }
        
        if (std::holds_alternative<JsonRpcRequest>(parse_result)) {
          // Single request
          auto request = std::get<JsonRpcRequest>(parse_result);
          auto response_future = HandleRequestAsync(request, transport_name);
          
          // Handle response
          auto response = response_future.get();
          std::string response_data = ProtocolParser::FormatResponse(response);
          connection_manager_->SendMessage(response_data, transport_name);
          
        } else if (std::holds_alternative<BatchRequest>(parse_result)) {
          // Batch request
          auto batch_request = std::get<BatchRequest>(parse_result);
          auto batch_response_future = HandleBatchRequestAsync(batch_request, transport_name);
          
          auto batch_response = batch_response_future.get();
          std::string response_data = ProtocolParser::FormatResponse(batch_response);
          connection_manager_->SendMessage(response_data, transport_name);
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        stats_.successful_requests.fetch_add(1);
        stats_.total_execution_time.fetch_add(duration.count());
        
      } catch (const std::exception& e) {
        LogError("Request processing failed", e.what());
        stats_.failed_requests.fetch_add(1);
      }
      
      stats_.active_requests.fetch_sub(1);
    });
    request_queue_cv_.notify_one();
  }
}

std::future<JsonRpcResponse> MCPServer::HandleRequestAsync(const JsonRpcRequest& request, 
                                                          const std::string& transport_name) {
  return std::async(std::launch::async, [this, request, transport_name]() {
    if (!ValidateRequest(request)) {
      return CreateErrorResponse(request.id.value_or(nullptr), 
                               ErrorCode::INVALID_REQUEST, 
                               "Invalid request format");
    }
    
    try {
      // Handle different methods
      if (request.method == "initialize") {
        return HandleInitialize(request.params.value_or(nlohmann::json::object()));
      } else if (request.method == "tools/list") {
        return HandleListTools(request.params.value_or(nlohmann::json::object()));
      } else if (request.method == "tools/call") {
        return HandleCallTool(request.params.value_or(nlohmann::json::object()), transport_name);
      } else if (request.method == "resources/list") {
        return HandleListResources(request.params.value_or(nlohmann::json::object()));
      } else if (request.is_notification()) {
        return HandleNotification(request.method, request.params.value_or(nlohmann::json::object()));
      } else {
        return CreateErrorResponse(request.id.value_or(nullptr), 
                                 ErrorCode::METHOD_NOT_FOUND, 
                                 "Method not found: " + request.method);
      }
      
    } catch (const std::exception& e) {
      return CreateErrorResponse(request.id.value_or(nullptr), 
                               ErrorCode::INTERNAL_ERROR, 
                               "Internal error", e.what());
    }
  });
}

std::future<BatchResponse> MCPServer::HandleBatchRequestAsync(const BatchRequest& batch_request,
                                                             const std::string& transport_name) {
  return std::async(std::launch::async, [this, batch_request, transport_name]() {
    BatchResponse batch_response;
    std::vector<std::future<JsonRpcResponse>> futures;
    
    // Process all requests asynchronously
    for (const auto& request : batch_request.requests) {
      futures.push_back(HandleRequestAsync(request, transport_name));
    }
    
    // Collect responses
    for (auto& future : futures) {
      batch_response.responses.push_back(future.get());
    }
    
    return batch_response;
  });
}

JsonRpcResponse MCPServer::HandleInitialize(const nlohmann::json& params) {
  if (!ProtocolParser::ValidateInitializeRequest(params)) {
    return CreateErrorResponse(nullptr, ErrorCode::INVALID_PARAMS, "Invalid initialize request");
  }
  
  InitializeResponse init_response;
  init_response.protocol_version = config_.protocol_version;
  init_response.capabilities = capabilities_.to_json();
  init_response.server_info = nlohmann::json{
    {"name", config_.server_name},
    {"version", config_.server_version}
  };
  
  return ProtocolParser::CreateSuccessResponse(nullptr, init_response.to_json());
}

JsonRpcResponse MCPServer::HandleListTools(const nlohmann::json& params) {
  if (!ProtocolParser::ValidateListToolsRequest(params)) {
    return CreateErrorResponse(nullptr, ErrorCode::INVALID_PARAMS, "Invalid list tools request");
  }
  
  ListToolsResponse response;
  response.tools = tool_registry_->GetAllTools();
  
  return ProtocolParser::CreateSuccessResponse(nullptr, response.to_json());
}

JsonRpcResponse MCPServer::HandleCallTool(const nlohmann::json& params, const std::string& transport_name) {
  if (!ProtocolParser::ValidateCallToolRequest(params)) {
    return CreateErrorResponse(nullptr, ErrorCode::INVALID_PARAMS, "Invalid call tool request");
  }
  
  CallToolRequest call_request = CallToolRequest::from_json(params);
  
  if (!IsToolAllowed(call_request.name)) {
    return CreateErrorResponse(nullptr, ErrorCode::INVALID_TOOL, 
                             "Tool not allowed: " + call_request.name);
  }
  
  // Execute tool
  auto tool_context = CreateToolContext(default_session_id_);
  auto result_future = tool_registry_->ExecuteToolAsync(call_request.name, 
                                                       call_request.arguments, 
                                                       tool_context);
  
  auto result = result_future.get();
  
  CallToolResponse response;
  response.content = result.content;
  response.is_error = !result.success;
  
  if (result.success) {
    // Update stats
    stats_.total_tokens_processed.fetch_add(result.tokens_processed);
    stats_.total_tokens_generated.fetch_add(result.tokens_generated);
    
    return ProtocolParser::CreateSuccessResponse(nullptr, response.to_json());
  } else {
    return CreateErrorResponse(nullptr, 
                             result.error.value_or(CreateInternalError()).code,
                             result.error.value_or(CreateInternalError()).message);
  }
}

JsonRpcResponse MCPServer::HandleListResources(const nlohmann::json& params) {
  // Resources not implemented yet
  ListResourcesResponse response;
  return ProtocolParser::CreateSuccessResponse(nullptr, response.to_json());
}

JsonRpcResponse MCPServer::HandleNotification(const std::string& method, const nlohmann::json& params) {
  // Handle notifications (no response needed)
  if (method == "notifications/cancelled") {
    // Handle request cancellation
  }
  
  // Notifications don't return responses
  JsonRpcResponse response;
  response.jsonrpc = "2.0";
  response.id = nullptr;
  return response;
}

ToolContext MCPServer::CreateToolContext(const std::string& session_id) {
  ToolContext context;
  context.gemma = gemma_;
  context.threading_ctx = threading_ctx_;
  context.matmul_env = matmul_env_;
  context.runtime_config = runtime_config_;
  context.registry = tool_registry_.get();
  
  std::string effective_session_id = session_id.empty() ? default_session_id_ : session_id;
  context.session_id = effective_session_id;
  context.kv_cache = session_manager_->GetKVCache(effective_session_id);
  context.current_pos = session_manager_->GetPosition(effective_session_id);
  
  return context;
}

void MCPServer::OnTransportMessage(const std::string& message, const std::string& transport_name) {
  ProcessRequest(message, transport_name);
}

void MCPServer::OnTransportStatus(TransportStatus status, const std::string& details) {
  if (status == TransportStatus::ERROR) {
    LogError("Transport error", details);
  }
}

std::string MCPServer::GenerateRequestId() {
  return "req_" + std::to_string(request_counter_.fetch_add(1));
}

bool MCPServer::ValidateRequest(const JsonRpcRequest& request) const {
  if (request.jsonrpc != "2.0") {
    return false;
  }
  
  if (request.method.empty()) {
    return false;
  }
  
  return true;
}

bool MCPServer::IsToolAllowed(const std::string& tool_name) const {
  if (config_.allowed_tools.empty()) {
    return true;  // All tools allowed
  }
  
  return std::find(config_.allowed_tools.begin(), config_.allowed_tools.end(), tool_name) 
         != config_.allowed_tools.end();
}

JsonRpcResponse MCPServer::CreateErrorResponse(const RequestId& id, ErrorCode code, 
                                              const std::string& message, 
                                              const std::string& details) {
  return ProtocolParser::CreateErrorResponse(code, message, id, 
                                           details.empty() ? std::nullopt : 
                                           std::optional<nlohmann::json>(details));
}

void MCPServer::LogError(const std::string& error, const std::string& details) {
  if (config_.log_level >= 1) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    
    std::ostringstream oss;
    oss << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "] "
        << "ERROR: " << error;
    if (!details.empty()) {
      oss << " - " << details;
    }
    
    std::cerr << oss.str() << std::endl;
  }
  
  std::lock_guard<std::mutex> lock(callback_mutex_);
  if (error_callback_) {
    error_callback_(error, details);
  }
}

void MCPServer::RequestWorker() {
  while (!shutdown_requested_.load()) {
    std::function<void()> task;
    
    {
      std::unique_lock<std::mutex> lock(request_queue_mutex_);
      request_queue_cv_.wait(lock, [this] { 
        return !request_queue_.empty() || shutdown_requested_.load(); 
      });
      
      if (shutdown_requested_.load()) {
        break;
      }
      
      if (!request_queue_.empty()) {
        task = std::move(request_queue_.front());
        request_queue_.pop();
      }
    }
    
    if (task) {
      task();
    }
  }
}

void MCPServer::SessionCleanupWorker() {
  while (!shutdown_requested_.load()) {
    std::this_thread::sleep_for(std::chrono::minutes(5));  // Cleanup every 5 minutes
    
    if (!shutdown_requested_.load()) {
      CleanupExpiredSessions();
    }
  }
}

void MCPServer::StatsUpdateWorker() {
  while (!shutdown_requested_.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(30));  // Update every 30 seconds
    
    // Could implement periodic stats logging or metrics export here
  }
}

// server_utils implementation
namespace server_utils {

MCPServerConfig CreateDefaultConfig() {
  MCPServerConfig config;
  
  // Set reasonable defaults
  config.server_name = "gemma-mcp-server";
  config.server_version = "1.0.0";
  config.protocol_version = "2024-11-05";
  
  config.default_runtime_config.max_tokens = 512;
  config.default_runtime_config.temperature = 0.8f;
  config.default_runtime_config.top_k = 40;
  config.default_runtime_config.top_p = 0.95f;
  
  config.session_timeout = std::chrono::seconds(3600);
  config.max_sessions = 100;
  config.max_concurrent_requests = 10;
  config.request_timeout = std::chrono::milliseconds(30000);
  
  config.log_level = 1;
  config.enable_streaming = true;
  config.enable_performance_metrics = true;
  config.enable_session_cleanup = true;
  
  config.transport_uris = {"stdio:"};
  
  return config;
}

std::unique_ptr<MCPServer> CreateStandardServer(const MCPServerConfig& config) {
  return std::make_unique<MCPServer>(config);
}

InferenceArgs CreateDefaultInferenceArgs() {
  InferenceArgs args;
  // Set reasonable defaults for inference
  return args;
}

void SetupLogging(int log_level) {
  // Could setup more sophisticated logging here
}

void LogServerEvent(const std::string& event, const std::string& details) {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  
  std::cout << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "] "
            << event;
  if (!details.empty()) {
    std::cout << " - " << details;
  }
  std::cout << std::endl;
}

}  // namespace server_utils

}  // namespace mcp
}  // namespace gcpp