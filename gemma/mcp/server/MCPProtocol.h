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

#ifndef THIRD_PARTY_GEMMA_CPP_INTERFACES_MCP_MCPPROTOCOL_H_
#define THIRD_PARTY_GEMMA_CPP_INTERFACES_MCP_MCPPROTOCOL_H_

#include <string>
#include <variant>
#include <vector>
#include <memory>
#include <optional>
#include <functional>
#include <cstdint>

#include <nlohmann/json.hpp> // Provided via nlohmann_json target (FetchContent if absent)

namespace gcpp {
namespace mcp {

// JSON-RPC 2.0 Error Codes (standard and MCP-specific)
enum class ErrorCode : int {
  // JSON-RPC 2.0 standard errors
  PARSE_ERROR = -32700,
  INVALID_REQUEST = -32600,
  METHOD_NOT_FOUND = -32601,
  INVALID_PARAMS = -32602,
  INTERNAL_ERROR = -32603,

  // MCP protocol errors
  INVALID_TOOL = -32000,
  TOOL_EXECUTION_FAILED = -32001,
  RESOURCE_NOT_FOUND = -32002,
  TRANSPORT_ERROR = -32003,
  
  // Gemma-specific errors
  MODEL_NOT_LOADED = -32100,
  MODEL_LOAD_FAILED = -32101,
  GENERATION_FAILED = -32102,
  TOKENIZATION_FAILED = -32103,
  BACKEND_ERROR = -32104
};

// Request ID can be string, number, or null
using RequestId = std::variant<std::string, int64_t, std::nullptr_t>;

// Base JSON-RPC 2.0 structures
struct JsonRpcError {
  ErrorCode code;
  std::string message;
  std::optional<nlohmann::json> data;

  nlohmann::json to_json() const;
  static JsonRpcError from_json(const nlohmann::json& json);
};

struct JsonRpcRequest {
  std::string jsonrpc = "2.0";
  std::string method;
  std::optional<nlohmann::json> params;
  std::optional<RequestId> id;

  nlohmann::json to_json() const;
  static JsonRpcRequest from_json(const nlohmann::json& json);
  bool is_notification() const { return !id.has_value(); }
};

struct JsonRpcResponse {
  std::string jsonrpc = "2.0";
  RequestId id;
  std::optional<nlohmann::json> result;
  std::optional<JsonRpcError> error;

  nlohmann::json to_json() const;
  static JsonRpcResponse from_json(const nlohmann::json& json);
  bool is_error() const { return error.has_value(); }
};

// MCP-specific structures
struct ToolSchema {
  std::string type = "object";
  nlohmann::json properties;
  std::vector<std::string> required;

  nlohmann::json to_json() const;
  static ToolSchema from_json(const nlohmann::json& json);
};

struct Tool {
  std::string name;
  std::string description;
  ToolSchema input_schema;

  nlohmann::json to_json() const;
  static Tool from_json(const nlohmann::json& json);
};

struct TextContent {
  std::string type = "text";
  std::string text;

  nlohmann::json to_json() const;
  static TextContent from_json(const nlohmann::json& json);
};

struct Resource {
  std::string uri;
  std::string name;
  std::optional<std::string> description;
  std::optional<std::string> mime_type;

  nlohmann::json to_json() const;
  static Resource from_json(const nlohmann::json& json);
};

// MCP Protocol messages
struct InitializeRequest {
  std::string protocol_version = "2024-11-05";
  nlohmann::json capabilities;
  nlohmann::json client_info;

  static InitializeRequest from_json(const nlohmann::json& json);
};

struct InitializeResponse {
  std::string protocol_version = "2024-11-05";
  nlohmann::json capabilities;
  nlohmann::json server_info;
  std::optional<std::string> instructions;

  nlohmann::json to_json() const;
};

struct ListToolsRequest {
  std::optional<std::string> cursor;

  static ListToolsRequest from_json(const nlohmann::json& json);
};

struct ListToolsResponse {
  std::vector<Tool> tools;
  std::optional<std::string> next_cursor;

  nlohmann::json to_json() const;
};

struct CallToolRequest {
  std::string name;
  nlohmann::json arguments;

  static CallToolRequest from_json(const nlohmann::json& json);
};

struct CallToolResponse {
  std::vector<TextContent> content;
  bool is_error = false;

  nlohmann::json to_json() const;
};

struct ListResourcesRequest {
  std::optional<std::string> cursor;

  static ListResourcesRequest from_json(const nlohmann::json& json);
};

struct ListResourcesResponse {
  std::vector<Resource> resources;
  std::optional<std::string> next_cursor;

  nlohmann::json to_json() const;
};

// Batch request/response support
struct BatchRequest {
  std::vector<JsonRpcRequest> requests;

  nlohmann::json to_json() const;
  static BatchRequest from_json(const nlohmann::json& json);
};

struct BatchResponse {
  std::vector<JsonRpcResponse> responses;

  nlohmann::json to_json() const;
  static BatchResponse from_json(const nlohmann::json& json);
};

// Protocol message parser and formatter
class ProtocolParser {
public:
  // Parse incoming JSON-RPC message
  static std::variant<JsonRpcRequest, BatchRequest, JsonRpcError> 
    ParseRequest(const std::string& json_str);

  // Format outgoing JSON-RPC response
  static std::string FormatResponse(const JsonRpcResponse& response);
  static std::string FormatResponse(const BatchResponse& response);
  static std::string FormatError(const JsonRpcError& error, 
                                const std::optional<RequestId>& id = std::nullopt);

  // Create standard error responses
  static JsonRpcResponse CreateErrorResponse(ErrorCode code, 
                                           const std::string& message,
                                           const RequestId& id,
                                           const std::optional<nlohmann::json>& data = std::nullopt);

  static JsonRpcResponse CreateSuccessResponse(const RequestId& id,
                                             const nlohmann::json& result);

  // Validate MCP protocol messages
  static bool ValidateInitializeRequest(const nlohmann::json& params);
  static bool ValidateCallToolRequest(const nlohmann::json& params);
  static bool ValidateListToolsRequest(const nlohmann::json& params);
  static bool ValidateListResourcesRequest(const nlohmann::json& params);
};

// Helper functions for request ID conversion
std::string RequestIdToString(const RequestId& id);
RequestId StringToRequestId(const std::string& str);

// Error helper functions
JsonRpcError CreateParseError(const std::string& details = "");
JsonRpcError CreateInvalidRequestError(const std::string& details = "");
JsonRpcError CreateMethodNotFoundError(const std::string& method);
JsonRpcError CreateInvalidParamsError(const std::string& details = "");
JsonRpcError CreateInternalError(const std::string& details = "");
JsonRpcError CreateToolExecutionError(const std::string& details);
JsonRpcError CreateModelError(ErrorCode code, const std::string& details);

}  // namespace mcp
}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_INTERFACES_MCP_MCPPROTOCOL_H_