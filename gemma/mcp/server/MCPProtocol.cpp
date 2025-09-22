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

#include "MCPProtocol.h"
#include <sstream>
#include <stdexcept>

namespace gcpp {
namespace mcp {

// JsonRpcError implementation
nlohmann::json JsonRpcError::to_json() const {
  nlohmann::json j;
  j["code"] = static_cast<int>(code);
  j["message"] = message;
  if (data.has_value()) {
    j["data"] = data.value();
  }
  return j;
}

JsonRpcError JsonRpcError::from_json(const nlohmann::json& json) {
  JsonRpcError error;
  error.code = static_cast<ErrorCode>(json.at("code").get<int>());
  error.message = json.at("message").get<std::string>();
  if (json.contains("data")) {
    error.data = json.at("data");
  }
  return error;
}

// JsonRpcRequest implementation
nlohmann::json JsonRpcRequest::to_json() const {
  nlohmann::json j;
  j["jsonrpc"] = jsonrpc;
  j["method"] = method;
  
  if (params.has_value()) {
    j["params"] = params.value();
  }
  
  if (id.has_value()) {
    std::visit([&j](auto&& arg) {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, std::string>) {
        j["id"] = arg;
      } else if constexpr (std::is_same_v<T, int64_t>) {
        j["id"] = arg;
      } else if constexpr (std::is_same_v<T, std::nullptr_t>) {
        j["id"] = nullptr;
      }
    }, id.value());
  }
  
  return j;
}

JsonRpcRequest JsonRpcRequest::from_json(const nlohmann::json& json) {
  JsonRpcRequest request;
  request.jsonrpc = json.at("jsonrpc").get<std::string>();
  request.method = json.at("method").get<std::string>();
  
  if (json.contains("params")) {
    request.params = json.at("params");
  }
  
  if (json.contains("id")) {
    const auto& id_json = json.at("id");
    if (id_json.is_string()) {
      request.id = id_json.get<std::string>();
    } else if (id_json.is_number_integer()) {
      request.id = id_json.get<int64_t>();
    } else if (id_json.is_null()) {
      request.id = nullptr;
    }
  }
  
  return request;
}

// JsonRpcResponse implementation
nlohmann::json JsonRpcResponse::to_json() const {
  nlohmann::json j;
  j["jsonrpc"] = jsonrpc;
  
  std::visit([&j](auto&& arg) {
    using T = std::decay_t<decltype(arg)>;
    if constexpr (std::is_same_v<T, std::string>) {
      j["id"] = arg;
    } else if constexpr (std::is_same_v<T, int64_t>) {
      j["id"] = arg;
    } else if constexpr (std::is_same_v<T, std::nullptr_t>) {
      j["id"] = nullptr;
    }
  }, id);
  
  if (result.has_value()) {
    j["result"] = result.value();
  }
  
  if (error.has_value()) {
    j["error"] = error.value().to_json();
  }
  
  return j;
}

JsonRpcResponse JsonRpcResponse::from_json(const nlohmann::json& json) {
  JsonRpcResponse response;
  response.jsonrpc = json.at("jsonrpc").get<std::string>();
  
  const auto& id_json = json.at("id");
  if (id_json.is_string()) {
    response.id = id_json.get<std::string>();
  } else if (id_json.is_number_integer()) {
    response.id = id_json.get<int64_t>();
  } else if (id_json.is_null()) {
    response.id = nullptr;
  }
  
  if (json.contains("result")) {
    response.result = json.at("result");
  }
  
  if (json.contains("error")) {
    response.error = JsonRpcError::from_json(json.at("error"));
  }
  
  return response;
}

// ToolSchema implementation
nlohmann::json ToolSchema::to_json() const {
  nlohmann::json j;
  j["type"] = type;
  j["properties"] = properties;
  if (!required.empty()) {
    j["required"] = required;
  }
  return j;
}

ToolSchema ToolSchema::from_json(const nlohmann::json& json) {
  ToolSchema schema;
  schema.type = json.at("type").get<std::string>();
  schema.properties = json.at("properties");
  if (json.contains("required")) {
    schema.required = json.at("required").get<std::vector<std::string>>();
  }
  return schema;
}

// Tool implementation
nlohmann::json Tool::to_json() const {
  nlohmann::json j;
  j["name"] = name;
  j["description"] = description;
  j["inputSchema"] = input_schema.to_json();
  return j;
}

Tool Tool::from_json(const nlohmann::json& json) {
  Tool tool;
  tool.name = json.at("name").get<std::string>();
  tool.description = json.at("description").get<std::string>();
  tool.input_schema = ToolSchema::from_json(json.at("inputSchema"));
  return tool;
}

// TextContent implementation
nlohmann::json TextContent::to_json() const {
  nlohmann::json j;
  j["type"] = type;
  j["text"] = text;
  return j;
}

TextContent TextContent::from_json(const nlohmann::json& json) {
  TextContent content;
  content.type = json.at("type").get<std::string>();
  content.text = json.at("text").get<std::string>();
  return content;
}

// Resource implementation
nlohmann::json Resource::to_json() const {
  nlohmann::json j;
  j["uri"] = uri;
  j["name"] = name;
  if (description.has_value()) {
    j["description"] = description.value();
  }
  if (mime_type.has_value()) {
    j["mimeType"] = mime_type.value();
  }
  return j;
}

Resource Resource::from_json(const nlohmann::json& json) {
  Resource resource;
  resource.uri = json.at("uri").get<std::string>();
  resource.name = json.at("name").get<std::string>();
  if (json.contains("description")) {
    resource.description = json.at("description").get<std::string>();
  }
  if (json.contains("mimeType")) {
    resource.mime_type = json.at("mimeType").get<std::string>();
  }
  return resource;
}

// MCP Protocol messages implementation
InitializeRequest InitializeRequest::from_json(const nlohmann::json& json) {
  InitializeRequest request;
  request.protocol_version = json.at("protocolVersion").get<std::string>();
  request.capabilities = json.at("capabilities");
  request.client_info = json.at("clientInfo");
  return request;
}

nlohmann::json InitializeResponse::to_json() const {
  nlohmann::json j;
  j["protocolVersion"] = protocol_version;
  j["capabilities"] = capabilities;
  j["serverInfo"] = server_info;
  if (instructions.has_value()) {
    j["instructions"] = instructions.value();
  }
  return j;
}

ListToolsRequest ListToolsRequest::from_json(const nlohmann::json& json) {
  ListToolsRequest request;
  if (json.contains("cursor")) {
    request.cursor = json.at("cursor").get<std::string>();
  }
  return request;
}

nlohmann::json ListToolsResponse::to_json() const {
  nlohmann::json j;
  nlohmann::json tools_json = nlohmann::json::array();
  for (const auto& tool : tools) {
    tools_json.push_back(tool.to_json());
  }
  j["tools"] = tools_json;
  if (next_cursor.has_value()) {
    j["nextCursor"] = next_cursor.value();
  }
  return j;
}

CallToolRequest CallToolRequest::from_json(const nlohmann::json& json) {
  CallToolRequest request;
  request.name = json.at("name").get<std::string>();
  request.arguments = json.at("arguments");
  return request;
}

nlohmann::json CallToolResponse::to_json() const {
  nlohmann::json j;
  nlohmann::json content_json = nlohmann::json::array();
  for (const auto& content : content) {
    content_json.push_back(content.to_json());
  }
  j["content"] = content_json;
  j["isError"] = is_error;
  return j;
}

ListResourcesRequest ListResourcesRequest::from_json(const nlohmann::json& json) {
  ListResourcesRequest request;
  if (json.contains("cursor")) {
    request.cursor = json.at("cursor").get<std::string>();
  }
  return request;
}

nlohmann::json ListResourcesResponse::to_json() const {
  nlohmann::json j;
  nlohmann::json resources_json = nlohmann::json::array();
  for (const auto& resource : resources) {
    resources_json.push_back(resource.to_json());
  }
  j["resources"] = resources_json;
  if (next_cursor.has_value()) {
    j["nextCursor"] = next_cursor.value();
  }
  return j;
}

// Batch request/response implementation
nlohmann::json BatchRequest::to_json() const {
  nlohmann::json j = nlohmann::json::array();
  for (const auto& request : requests) {
    j.push_back(request.to_json());
  }
  return j;
}

BatchRequest BatchRequest::from_json(const nlohmann::json& json) {
  BatchRequest batch;
  for (const auto& request_json : json) {
    batch.requests.push_back(JsonRpcRequest::from_json(request_json));
  }
  return batch;
}

nlohmann::json BatchResponse::to_json() const {
  nlohmann::json j = nlohmann::json::array();
  for (const auto& response : responses) {
    j.push_back(response.to_json());
  }
  return j;
}

BatchResponse BatchResponse::from_json(const nlohmann::json& json) {
  BatchResponse batch;
  for (const auto& response_json : json) {
    batch.responses.push_back(JsonRpcResponse::from_json(response_json));
  }
  return batch;
}

// ProtocolParser implementation
std::variant<JsonRpcRequest, BatchRequest, JsonRpcError> 
ProtocolParser::ParseRequest(const std::string& json_str) {
  try {
    nlohmann::json json = nlohmann::json::parse(json_str);
    
    if (json.is_array()) {
      // Batch request
      return BatchRequest::from_json(json);
    } else if (json.is_object()) {
      // Single request
      if (!json.contains("jsonrpc") || json["jsonrpc"] != "2.0") {
        return CreateInvalidRequestError("Missing or invalid 'jsonrpc' field");
      }
      if (!json.contains("method") || !json["method"].is_string()) {
        return CreateInvalidRequestError("Missing or invalid 'method' field");
      }
      return JsonRpcRequest::from_json(json);
    } else {
      return CreateInvalidRequestError("Request must be object or array");
    }
  } catch (const nlohmann::json::parse_error& e) {
    return CreateParseError(e.what());
  } catch (const nlohmann::json::exception& e) {
    return CreateInvalidRequestError(e.what());
  } catch (const std::exception& e) {
    return CreateInternalError(e.what());
  }
}

std::string ProtocolParser::FormatResponse(const JsonRpcResponse& response) {
  return response.to_json().dump();
}

std::string ProtocolParser::FormatResponse(const BatchResponse& response) {
  return response.to_json().dump();
}

std::string ProtocolParser::FormatError(const JsonRpcError& error, 
                                       const std::optional<RequestId>& id) {
  JsonRpcResponse response;
  response.jsonrpc = "2.0";
  response.id = id.value_or(nullptr);
  response.error = error;
  return FormatResponse(response);
}

JsonRpcResponse ProtocolParser::CreateErrorResponse(ErrorCode code, 
                                                   const std::string& message,
                                                   const RequestId& id,
                                                   const std::optional<nlohmann::json>& data) {
  JsonRpcResponse response;
  response.jsonrpc = "2.0";
  response.id = id;
  response.error = JsonRpcError{code, message, data};
  return response;
}

JsonRpcResponse ProtocolParser::CreateSuccessResponse(const RequestId& id,
                                                     const nlohmann::json& result) {
  JsonRpcResponse response;
  response.jsonrpc = "2.0";
  response.id = id;
  response.result = result;
  return response;
}

bool ProtocolParser::ValidateInitializeRequest(const nlohmann::json& params) {
  return params.is_object() && 
         params.contains("protocolVersion") && 
         params.contains("capabilities") &&
         params.contains("clientInfo");
}

bool ProtocolParser::ValidateCallToolRequest(const nlohmann::json& params) {
  return params.is_object() && 
         params.contains("name") && 
         params["name"].is_string() &&
         params.contains("arguments") &&
         params["arguments"].is_object();
}

bool ProtocolParser::ValidateListToolsRequest(const nlohmann::json& params) {
  if (!params.is_object()) return false;
  if (params.contains("cursor") && !params["cursor"].is_string()) return false;
  return true;
}

bool ProtocolParser::ValidateListResourcesRequest(const nlohmann::json& params) {
  if (!params.is_object()) return false;
  if (params.contains("cursor") && !params["cursor"].is_string()) return false;
  return true;
}

// Helper functions implementation
std::string RequestIdToString(const RequestId& id) {
  return std::visit([](auto&& arg) -> std::string {
    using T = std::decay_t<decltype(arg)>;
    if constexpr (std::is_same_v<T, std::string>) {
      return arg;
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return std::to_string(arg);
    } else if constexpr (std::is_same_v<T, std::nullptr_t>) {
      return "null";
    }
    return "";
  }, id);
}

RequestId StringToRequestId(const std::string& str) {
  if (str == "null") {
    return nullptr;
  }
  
  try {
    return std::stoll(str);
  } catch (const std::exception&) {
    return str;
  }
}

// Error helper functions
JsonRpcError CreateParseError(const std::string& details) {
  return {ErrorCode::PARSE_ERROR, "Parse error", 
          details.empty() ? std::nullopt : std::optional<nlohmann::json>(details)};
}

JsonRpcError CreateInvalidRequestError(const std::string& details) {
  return {ErrorCode::INVALID_REQUEST, "Invalid Request", 
          details.empty() ? std::nullopt : std::optional<nlohmann::json>(details)};
}

JsonRpcError CreateMethodNotFoundError(const std::string& method) {
  return {ErrorCode::METHOD_NOT_FOUND, "Method not found", 
          nlohmann::json{{"method", method}}};
}

JsonRpcError CreateInvalidParamsError(const std::string& details) {
  return {ErrorCode::INVALID_PARAMS, "Invalid params", 
          details.empty() ? std::nullopt : std::optional<nlohmann::json>(details)};
}

JsonRpcError CreateInternalError(const std::string& details) {
  return {ErrorCode::INTERNAL_ERROR, "Internal error", 
          details.empty() ? std::nullopt : std::optional<nlohmann::json>(details)};
}

JsonRpcError CreateToolExecutionError(const std::string& details) {
  return {ErrorCode::TOOL_EXECUTION_FAILED, "Tool execution failed", 
          nlohmann::json{{"details", details}}};
}

JsonRpcError CreateModelError(ErrorCode code, const std::string& details) {
  std::string message;
  switch (code) {
    case ErrorCode::MODEL_NOT_LOADED:
      message = "Model not loaded";
      break;
    case ErrorCode::MODEL_LOAD_FAILED:
      message = "Model load failed";
      break;
    case ErrorCode::GENERATION_FAILED:
      message = "Generation failed";
      break;
    case ErrorCode::TOKENIZATION_FAILED:
      message = "Tokenization failed";
      break;
    case ErrorCode::BACKEND_ERROR:
      message = "Backend error";
      break;
    default:
      message = "Unknown model error";
      break;
  }
  
  return {code, message, 
          details.empty() ? std::nullopt : std::optional<nlohmann::json>(details)};
}

}  // namespace mcp
}  // namespace gcpp