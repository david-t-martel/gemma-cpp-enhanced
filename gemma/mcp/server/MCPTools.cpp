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

#include "MCPTools.h"
#include "util/basics.h"
#include <chrono>
#include <sstream>
#include <algorithm>
#include <random>
#include <iomanip>

namespace gcpp {
namespace mcp {

// ToolResult implementation
ToolResult ToolResult::Success(const std::vector<TextContent>& content, double execution_time_ms) {
  ToolResult result;
  result.success = true;
  result.content = content;
  result.execution_time_ms = execution_time_ms;
  return result;
}

ToolResult ToolResult::Success(const std::string& text_content, double execution_time_ms) {
  return Success({TextContent{"text", text_content}}, execution_time_ms);
}

ToolResult ToolResult::Error(const JsonRpcError& error) {
  ToolResult result;
  result.success = false;
  result.error = error;
  return result;
}

ToolResult ToolResult::Error(ErrorCode code, const std::string& message, const std::string& details) {
  JsonRpcError error{code, message, details.empty() ? std::nullopt : 
                     std::optional<nlohmann::json>(details)};
  return Error(error);
}

// MCPTool base implementation
ToolResult MCPTool::Execute(const nlohmann::json& args, ToolContext& context) {
  auto future = ExecuteAsync(args, context);
  return future.get();
}

std::future<ToolResult> MCPTool::ExecuteStreamAsync(const nlohmann::json& args, 
                                                   ToolContext& context,
                                                   StreamCallback stream_callback) {
  // Default implementation: execute normally and call stream callback once with final result
  return std::async(std::launch::async, [this, args, &context, stream_callback]() {
    auto result = Execute(args, context);
    if (stream_callback && result.success && !result.content.empty()) {
      stream_callback(result.content[0].text, true);
    }
    return result;
  });
}

bool MCPTool::ValidateArguments(const nlohmann::json& args) const {
  if (!args.is_object()) {
    return false;
  }
  
  // Basic validation - subclasses should override for specific validation
  return true;
}

Tool MCPTool::ToMCPTool() const {
  Tool tool;
  tool.name = GetName();
  tool.description = GetDescription();
  tool.input_schema = GetInputSchema();
  return tool;
}

// ToolRegistry implementation
bool ToolRegistry::RegisterTool(std::unique_ptr<MCPTool> tool) {
  if (!tool) {
    return false;
  }
  
  std::unique_lock<std::shared_mutex> lock(tools_mutex_);
  std::string name = tool->GetName();
  tools_[name] = std::move(tool);
  return true;
}

bool ToolRegistry::UnregisterTool(const std::string& name) {
  std::unique_lock<std::shared_mutex> lock(tools_mutex_);
  return tools_.erase(name) > 0;
}

std::vector<Tool> ToolRegistry::GetAllTools() const {
  std::shared_lock<std::shared_mutex> lock(tools_mutex_);
  std::vector<Tool> tools;
  
  for (const auto& pair : tools_) {
    tools.push_back(pair.second->ToMCPTool());
  }
  
  return tools;
}

MCPTool* ToolRegistry::GetTool(const std::string& name) const {
  std::shared_lock<std::shared_mutex> lock(tools_mutex_);
  auto it = tools_.find(name);
  return (it != tools_.end()) ? it->second.get() : nullptr;
}

bool ToolRegistry::HasTool(const std::string& name) const {
  std::shared_lock<std::shared_mutex> lock(tools_mutex_);
  return tools_.find(name) != tools_.end();
}

std::vector<std::string> ToolRegistry::GetToolNames() const {
  std::shared_lock<std::shared_mutex> lock(tools_mutex_);
  std::vector<std::string> names;
  
  for (const auto& pair : tools_) {
    names.push_back(pair.first);
  }
  
  return names;
}

std::future<ToolResult> ToolRegistry::ExecuteToolAsync(const std::string& name, 
                                                       const nlohmann::json& args,
                                                       ToolContext& context) {
  std::shared_lock<std::shared_mutex> lock(tools_mutex_);
  auto it = tools_.find(name);
  
  if (it == tools_.end()) {
    auto promise = std::make_shared<std::promise<ToolResult>>();
    promise->set_value(ToolResult::Error(ErrorCode::METHOD_NOT_FOUND, 
                                        "Tool not found: " + name));
    return promise->get_future();
  }
  
  MCPTool* tool = it->second.get();
  lock.unlock();
  
  return tool->ExecuteAsync(args, context);
}

ToolResult ToolRegistry::ExecuteTool(const std::string& name, 
                                    const nlohmann::json& args,
                                    ToolContext& context) {
  auto future = ExecuteToolAsync(name, args, context);
  return future.get();
}

std::future<ToolResult> ToolRegistry::ExecuteToolStreamAsync(const std::string& name,
                                                            const nlohmann::json& args,
                                                            ToolContext& context,
                                                            StreamCallback stream_callback) {
  std::shared_lock<std::shared_mutex> lock(tools_mutex_);
  auto it = tools_.find(name);
  
  if (it == tools_.end()) {
    auto promise = std::make_shared<std::promise<ToolResult>>();
    promise->set_value(ToolResult::Error(ErrorCode::METHOD_NOT_FOUND, 
                                        "Tool not found: " + name));
    return promise->get_future();
  }
  
  MCPTool* tool = it->second.get();
  lock.unlock();
  
  return tool->ExecuteStreamAsync(args, context, stream_callback);
}

std::vector<Tool> ToolRegistry::GetToolsByCategory(const std::string& category) const {
  // For now, return all tools. Could be enhanced with categorization
  return GetAllTools();
}

std::vector<Tool> ToolRegistry::GetStreamingTools() const {
  std::shared_lock<std::shared_mutex> lock(tools_mutex_);
  std::vector<Tool> streaming_tools;
  
  for (const auto& pair : tools_) {
    if (pair.second->SupportsStreaming()) {
      streaming_tools.push_back(pair.second->ToMCPTool());
    }
  }
  
  return streaming_tools;
}

std::vector<Tool> ToolRegistry::GetStatefulTools() const {
  std::shared_lock<std::shared_mutex> lock(tools_mutex_);
  std::vector<Tool> stateful_tools;
  
  for (const auto& pair : tools_) {
    if (pair.second->IsStateful()) {
      stateful_tools.push_back(pair.second->ToMCPTool());
    }
  }
  
  return stateful_tools;
}

// GenerateTextTool implementation
std::string GenerateTextTool::GetDescription() const {
  return "Generate text using the Gemma model with customizable parameters. "
         "Supports streaming output and maintains conversation context.";
}

ToolSchema GenerateTextTool::GetInputSchema() const {
  ToolSchema schema;
  schema.type = "object";
  schema.properties = nlohmann::json{
    {"prompt", {
      {"type", "string"},
      {"description", "The input prompt for text generation"}
    }},
    {"max_tokens", {
      {"type", "integer"},
      {"description", "Maximum number of tokens to generate"},
      {"default", 512},
      {"minimum", 1},
      {"maximum", 4096}
    }},
    {"temperature", {
      {"type", "number"},
      {"description", "Sampling temperature (0.0 to 2.0)"},
      {"default", 0.8},
      {"minimum", 0.0},
      {"maximum", 2.0}
    }},
    {"top_k", {
      {"type", "integer"},
      {"description", "Top-k sampling parameter"},
      {"default", 40},
      {"minimum", 1}
    }},
    {"top_p", {
      {"type", "number"},
      {"description", "Top-p (nucleus) sampling parameter"},
      {"default", 0.95},
      {"minimum", 0.0},
      {"maximum", 1.0}
    }},
    {"stream", {
      {"type", "boolean"},
      {"description", "Enable streaming output"},
      {"default", false}
    }}
  };
  schema.required = {"prompt"};
  return schema;
}

std::future<ToolResult> GenerateTextTool::ExecuteAsync(const nlohmann::json& args, 
                                                      ToolContext& context) {
  return ExecuteStreamAsync(args, context, nullptr);
}

std::future<ToolResult> GenerateTextTool::ExecuteStreamAsync(const nlohmann::json& args, 
                                                            ToolContext& context,
                                                            StreamCallback stream_callback) {
  return std::async(std::launch::async, [this, args, &context, stream_callback]() {
    if (!ValidateArguments(args)) {
      return ToolResult::Error(ErrorCode::INVALID_PARAMS, "Invalid arguments");
    }
    
    std::string prompt = tool_utils::GetStringArg(args, "prompt");
    if (prompt.empty()) {
      return ToolResult::Error(ErrorCode::INVALID_PARAMS, "Prompt cannot be empty");
    }
    
    return GenerateText(prompt, args, context, stream_callback);
  });
}

ToolResult GenerateTextTool::GenerateText(const std::string& prompt, 
                                         const nlohmann::json& config,
                                         ToolContext& context,
                                         StreamCallback stream_callback) {
  tool_utils::PerformanceTimer timer;
  timer.Start();
  
  try {
    if (!context.gemma) {
      return ToolResult::Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    
    // Update runtime config from arguments
    RuntimeConfig runtime_config = context.runtime_config;
    tool_utils::UpdateRuntimeConfigFromArgs(runtime_config, config);
    
    // Tokenize prompt
    PromptTokens prompt_tokens;
    if (!context.gemma->Tokenizer().Encode(prompt, &prompt_tokens)) {
      return ToolResult::Error(ErrorCode::TOKENIZATION_FAILED, "Failed to tokenize prompt");
    }
    
    // Setup generation
    std::string generated_text;
    size_t tokens_generated = 0;
    
    // Stream callback wrapper
    auto stream_func = [&](int token, float) -> bool {
      std::string token_text;
      context.gemma->Tokenizer().Decode({token}, &token_text);
      generated_text += token_text;
      tokens_generated++;
      
      if (stream_callback) {
        stream_callback(token_text, false);
      }
      
      return true;  // Continue generation
    };
    
    runtime_config.stream_token = stream_func;
    
    // Perform generation
    context.timing_info = TimingInfo{};
    context.timing_info.verbosity = 0;  // Suppress default output
    
    context.gemma->Generate(runtime_config, prompt_tokens, context.current_pos, 
                           *context.kv_cache, *context.matmul_env, context.timing_info);
    
    // Update position
    context.current_pos += prompt_tokens.size() + tokens_generated;
    
    timer.Stop();
    
    // Final stream callback
    if (stream_callback) {
      stream_callback("", true);  // Signal completion
    }
    
    // Create result
    ToolResult result = ToolResult::Success(generated_text, timer.GetElapsedMs());
    result.tokens_processed = prompt_tokens.size();
    result.tokens_generated = tokens_generated;
    
    return result;
    
  } catch (const std::exception& e) {
    timer.Stop();
    return ToolResult::Error(ErrorCode::GENERATION_FAILED, 
                            "Generation failed: " + std::string(e.what()));
  }
}

// CountTokensTool implementation
std::string CountTokensTool::GetDescription() const {
  return "Count the number of tokens in a given text using the model's tokenizer.";
}

ToolSchema CountTokensTool::GetInputSchema() const {
  ToolSchema schema;
  schema.type = "object";
  schema.properties = nlohmann::json{
    {"text", {
      {"type", "string"},
      {"description", "The text to count tokens for"}
    }}
  };
  schema.required = {"text"};
  return schema;
}

std::future<ToolResult> CountTokensTool::ExecuteAsync(const nlohmann::json& args, 
                                                     ToolContext& context) {
  return std::async(std::launch::async, [this, args, &context]() {
    if (!ValidateArguments(args)) {
      return ToolResult::Error(ErrorCode::INVALID_PARAMS, "Invalid arguments");
    }
    
    std::string text = tool_utils::GetStringArg(args, "text");
    return CountTokens(text, context);
  });
}

ToolResult CountTokensTool::CountTokens(const std::string& text, ToolContext& context) {
  tool_utils::PerformanceTimer timer;
  timer.Start();
  
  try {
    if (!context.gemma) {
      return ToolResult::Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    
    PromptTokens tokens;
    if (!context.gemma->Tokenizer().Encode(text, &tokens)) {
      return ToolResult::Error(ErrorCode::TOKENIZATION_FAILED, "Failed to tokenize text");
    }
    
    timer.Stop();
    
    nlohmann::json result_data{
      {"token_count", tokens.size()},
      {"text_length", text.length()},
      {"tokens_per_char", text.empty() ? 0.0 : static_cast<double>(tokens.size()) / text.length()}
    };
    
    return ToolResult::Success(result_data.dump(2), timer.GetElapsedMs());
    
  } catch (const std::exception& e) {
    timer.Stop();
    return ToolResult::Error(ErrorCode::TOKENIZATION_FAILED, 
                            "Token counting failed: " + std::string(e.what()));
  }
}

// GetModelInfoTool implementation
std::string GetModelInfoTool::GetDescription() const {
  return "Get information about the currently loaded model including configuration and capabilities.";
}

ToolSchema GetModelInfoTool::GetInputSchema() const {
  ToolSchema schema;
  schema.type = "object";
  schema.properties = nlohmann::json::object();
  return schema;
}

std::future<ToolResult> GetModelInfoTool::ExecuteAsync(const nlohmann::json& args, 
                                                      ToolContext& context) {
  return std::async(std::launch::async, [this, &context]() {
    return GetModelInfo(context);
  });
}

ToolResult GetModelInfoTool::GetModelInfo(ToolContext& context) {
  tool_utils::PerformanceTimer timer;
  timer.Start();
  
  try {
    if (!context.gemma) {
      return ToolResult::Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    
    const ModelConfig& config = context.gemma->Config();
    
    nlohmann::json model_info{
      {"model_type", ToString(config.model)},
      {"vocab_size", config.vocab_size},
      {"seq_len", config.seq_len},
      {"hidden_dim", config.model_dim},
      {"num_layers", config.num_layers},
      {"num_heads", config.num_heads},
      {"num_kv_heads", config.num_kv_heads},
      {"head_dim", config.head_dim},
      {"weight_read_mode", ToString(context.gemma->WeightReadMode())},
      {"supports_vision", config.vision.enabled},
      {"supports_griffin", config.griffin.enabled}
    };
    
    if (config.vision.enabled) {
      model_info["vision_config"] = nlohmann::json{
        {"image_size", config.vision.image_size},
        {"patch_size", config.vision.patch_size}
      };
    }
    
    timer.Stop();
    
    return ToolResult::Success(model_info.dump(2), timer.GetElapsedMs());
    
  } catch (const std::exception& e) {
    timer.Stop();
    return ToolResult::Error(ErrorCode::INTERNAL_ERROR, 
                            "Failed to get model info: " + std::string(e.what()));
  }
}

// ListSessionsTool implementation
std::string ListSessionsTool::GetDescription() const {
  return "List all active inference sessions with their status and statistics.";
}

ToolSchema ListSessionsTool::GetInputSchema() const {
  ToolSchema schema;
  schema.type = "object";
  schema.properties = nlohmann::json::object();
  return schema;
}

std::future<ToolResult> ListSessionsTool::ExecuteAsync(const nlohmann::json& args, 
                                                      ToolContext& context) {
  return std::async(std::launch::async, [this, &context]() {
    return ListSessions(context);
  });
}

ToolResult ListSessionsTool::ListSessions(ToolContext& context) {
  tool_utils::PerformanceTimer timer;
  timer.Start();
  
  // For now, return current session info. Full session management would require
  // integration with a session manager
  nlohmann::json sessions_info{
    {"current_session", {
      {"id", context.session_id},
      {"position", context.current_pos},
      {"max_seq_len", context.kv_cache ? context.kv_cache->SeqLen() : 0}
    }},
    {"total_sessions", 1}
  };
  
  timer.Stop();
  
  return ToolResult::Success(sessions_info.dump(2), timer.GetElapsedMs());
}

// SetBackendTool implementation
std::string SetBackendTool::GetDescription() const {
  return "Switch the inference backend (CPU, CUDA, SYCL, etc.) if multiple backends are available.";
}

ToolSchema SetBackendTool::GetInputSchema() const {
  ToolSchema schema;
  schema.type = "object";
  schema.properties = nlohmann::json{
    {"backend", {
      {"type", "string"},
      {"description", "Backend name (cpu, cuda, sycl, vulkan)"},
      {"enum", nlohmann::json::array({"cpu", "cuda", "sycl", "vulkan"})}
    }}
  };
  schema.required = {"backend"};
  return schema;
}

std::future<ToolResult> SetBackendTool::ExecuteAsync(const nlohmann::json& args, 
                                                    ToolContext& context) {
  return std::async(std::launch::async, [this, args, &context]() {
    if (!ValidateArguments(args)) {
      return ToolResult::Error(ErrorCode::INVALID_PARAMS, "Invalid arguments");
    }
    
    std::string backend = tool_utils::GetStringArg(args, "backend");
    return SetBackend(backend, context);
  });
}

ToolResult SetBackendTool::SetBackend(const std::string& backend, ToolContext& context) {
  tool_utils::PerformanceTimer timer;
  timer.Start();
  
  // For now, this is a placeholder. Actual backend switching would require
  // integration with the backend system
  timer.Stop();
  
  nlohmann::json result{
    {"previous_backend", "cpu"},
    {"new_backend", backend},
    {"status", "Backend switching not implemented in this version"}
  };
  
  return ToolResult::Success(result.dump(2), timer.GetElapsedMs());
}

// SessionManager implementation
std::string SessionManager::CreateSession(std::shared_ptr<Gemma> gemma, 
                                         std::shared_ptr<ThreadingContext> threading_ctx) {
  std::unique_lock<std::shared_mutex> lock(sessions_mutex_);
  
  std::string session_id = "session_" + std::to_string(session_counter_++);
  
  SessionData data;
  data.kv_cache = std::make_shared<KVCache>(gemma->Config(), *threading_ctx->GetAllocator());
  data.last_access = std::chrono::steady_clock::now();
  data.current_pos = 0;
  data.max_seq_len = gemma->Config().seq_len;
  
  sessions_[session_id] = std::move(data);
  
  return session_id;
}

bool SessionManager::DestroySession(const std::string& session_id) {
  std::unique_lock<std::shared_mutex> lock(sessions_mutex_);
  return sessions_.erase(session_id) > 0;
}

bool SessionManager::HasSession(const std::string& session_id) const {
  std::shared_lock<std::shared_mutex> lock(sessions_mutex_);
  return sessions_.find(session_id) != sessions_.end();
}

std::shared_ptr<KVCache> SessionManager::GetKVCache(const std::string& session_id) const {
  std::shared_lock<std::shared_mutex> lock(sessions_mutex_);
  auto it = sessions_.find(session_id);
  return (it != sessions_.end()) ? it->second.kv_cache : nullptr;
}

size_t SessionManager::GetPosition(const std::string& session_id) const {
  std::shared_lock<std::shared_mutex> lock(sessions_mutex_);
  auto it = sessions_.find(session_id);
  return (it != sessions_.end()) ? it->second.current_pos : 0;
}

void SessionManager::SetPosition(const std::string& session_id, size_t pos) {
  std::unique_lock<std::shared_mutex> lock(sessions_mutex_);
  auto it = sessions_.find(session_id);
  if (it != sessions_.end()) {
    it->second.current_pos = pos;
    it->second.last_access = std::chrono::steady_clock::now();
  }
}

std::vector<std::string> SessionManager::GetSessionIds() const {
  std::shared_lock<std::shared_mutex> lock(sessions_mutex_);
  std::vector<std::string> ids;
  for (const auto& pair : sessions_) {
    ids.push_back(pair.first);
  }
  return ids;
}

size_t SessionManager::GetSessionCount() const {
  std::shared_lock<std::shared_mutex> lock(sessions_mutex_);
  return sessions_.size();
}

nlohmann::json SessionManager::GetSessionInfo(const std::string& session_id) const {
  std::shared_lock<std::shared_mutex> lock(sessions_mutex_);
  auto it = sessions_.find(session_id);
  
  if (it == sessions_.end()) {
    return nlohmann::json::object();
  }
  
  const auto& data = it->second;
  auto now = std::chrono::steady_clock::now();
  auto idle_time = std::chrono::duration_cast<std::chrono::seconds>(now - data.last_access);
  
  return nlohmann::json{
    {"id", session_id},
    {"position", data.current_pos},
    {"max_seq_len", data.max_seq_len},
    {"idle_time_seconds", idle_time.count()}
  };
}

void SessionManager::CleanupExpiredSessions(std::chrono::seconds max_idle_time) {
  std::unique_lock<std::shared_mutex> lock(sessions_mutex_);
  
  auto now = std::chrono::steady_clock::now();
  auto it = sessions_.begin();
  
  while (it != sessions_.end()) {
    auto idle_time = std::chrono::duration_cast<std::chrono::seconds>(now - it->second.last_access);
    if (idle_time > max_idle_time) {
      it = sessions_.erase(it);
    } else {
      ++it;
    }
  }
}

void SessionManager::CleanupAllSessions() {
  std::unique_lock<std::shared_mutex> lock(sessions_mutex_);
  sessions_.clear();
}

// DefaultToolFactory implementation
std::unique_ptr<ToolRegistry> DefaultToolFactory::CreateStandardRegistry() {
  auto registry = std::make_unique<ToolRegistry>();
  
  auto tools = CreateStandardTools();
  for (auto& tool : tools) {
    registry->RegisterTool(std::move(tool));
  }
  
  return registry;
}

std::vector<std::unique_ptr<MCPTool>> DefaultToolFactory::CreateStandardTools() {
  std::vector<std::unique_ptr<MCPTool>> tools;
  
  tools.push_back(CreateGenerateTextTool());
  tools.push_back(CreateCountTokensTool());
  tools.push_back(CreateGetModelInfoTool());
  tools.push_back(CreateListSessionsTool());
  tools.push_back(CreateSetBackendTool());
  
  return tools;
}

std::unique_ptr<GenerateTextTool> DefaultToolFactory::CreateGenerateTextTool() {
  return std::make_unique<GenerateTextTool>();
}

std::unique_ptr<CountTokensTool> DefaultToolFactory::CreateCountTokensTool() {
  return std::make_unique<CountTokensTool>();
}

std::unique_ptr<GetModelInfoTool> DefaultToolFactory::CreateGetModelInfoTool() {
  return std::make_unique<GetModelInfoTool>();
}

std::unique_ptr<ListSessionsTool> DefaultToolFactory::CreateListSessionsTool() {
  return std::make_unique<ListSessionsTool>();
}

std::unique_ptr<SetBackendTool> DefaultToolFactory::CreateSetBackendTool() {
  return std::make_unique<SetBackendTool>();
}

// tool_utils implementation
namespace tool_utils {

bool ValidateStringArg(const nlohmann::json& args, const std::string& key, bool required) {
  if (!args.contains(key)) {
    return !required;
  }
  return args[key].is_string();
}

bool ValidateIntArg(const nlohmann::json& args, const std::string& key, 
                   int min_val, int max_val, bool required) {
  if (!args.contains(key)) {
    return !required;
  }
  if (!args[key].is_number_integer()) {
    return false;
  }
  int val = args[key].get<int>();
  return val >= min_val && val <= max_val;
}

bool ValidateFloatArg(const nlohmann::json& args, const std::string& key, 
                     double min_val, double max_val, bool required) {
  if (!args.contains(key)) {
    return !required;
  }
  if (!args[key].is_number()) {
    return false;
  }
  double val = args[key].get<double>();
  return val >= min_val && val <= max_val;
}

bool ValidateArrayArg(const nlohmann::json& args, const std::string& key, bool required) {
  if (!args.contains(key)) {
    return !required;
  }
  return args[key].is_array();
}

std::string GetStringArg(const nlohmann::json& args, const std::string& key, 
                        const std::string& default_value) {
  if (args.contains(key) && args[key].is_string()) {
    return args[key].get<std::string>();
  }
  return default_value;
}

int GetIntArg(const nlohmann::json& args, const std::string& key, int default_value) {
  if (args.contains(key) && args[key].is_number_integer()) {
    return args[key].get<int>();
  }
  return default_value;
}

double GetFloatArg(const nlohmann::json& args, const std::string& key, double default_value) {
  if (args.contains(key) && args[key].is_number()) {
    return args[key].get<double>();
  }
  return default_value;
}

nlohmann::json GetArrayArg(const nlohmann::json& args, const std::string& key, 
                          const nlohmann::json& default_value) {
  if (args.contains(key) && args[key].is_array()) {
    return args[key];
  }
  return default_value;
}

RuntimeConfig CreateRuntimeConfig(const nlohmann::json& config) {
  RuntimeConfig runtime_config;
  
  runtime_config.max_tokens = GetIntArg(config, "max_tokens", 512);
  runtime_config.temperature = static_cast<float>(GetFloatArg(config, "temperature", 0.8));
  runtime_config.top_k = GetIntArg(config, "top_k", 40);
  runtime_config.top_p = static_cast<float>(GetFloatArg(config, "top_p", 0.95));
  
  return runtime_config;
}

void UpdateRuntimeConfigFromArgs(RuntimeConfig& config, const nlohmann::json& args) {
  if (args.contains("max_tokens")) {
    config.max_tokens = GetIntArg(args, "max_tokens", config.max_tokens);
  }
  if (args.contains("temperature")) {
    config.temperature = static_cast<float>(GetFloatArg(args, "temperature", config.temperature));
  }
  if (args.contains("top_k")) {
    config.top_k = GetIntArg(args, "top_k", config.top_k);
  }
  if (args.contains("top_p")) {
    config.top_p = static_cast<float>(GetFloatArg(args, "top_p", config.top_p));
  }
}

JsonRpcError CreateValidationError(const std::string& field, const std::string& issue) {
  return {ErrorCode::INVALID_PARAMS, "Validation error", 
          nlohmann::json{{"field", field}, {"issue", issue}}};
}

JsonRpcError CreateExecutionError(const std::string& operation, const std::string& details) {
  return {ErrorCode::TOOL_EXECUTION_FAILED, "Execution error", 
          nlohmann::json{{"operation", operation}, {"details", details}}};
}

// PerformanceTimer implementation
PerformanceTimer::PerformanceTimer() {
  Reset();
}

void PerformanceTimer::Start() {
  start_time_ = std::chrono::high_resolution_clock::now();
  running_ = true;
}

void PerformanceTimer::Stop() {
  if (running_) {
    end_time_ = std::chrono::high_resolution_clock::now();
    running_ = false;
  }
}

double PerformanceTimer::GetElapsedMs() const {
  auto end = running_ ? std::chrono::high_resolution_clock::now() : end_time_;
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_time_);
  return duration.count() / 1000.0;
}

void PerformanceTimer::Reset() {
  start_time_ = std::chrono::high_resolution_clock::now();
  end_time_ = start_time_;
  running_ = false;
}

}  // namespace tool_utils

}  // namespace mcp
}  // namespace gcpp