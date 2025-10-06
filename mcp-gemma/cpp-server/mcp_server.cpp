/**
 * @file mcp_server.cpp
 * @brief Implementation of MCP Server for Gemma.cpp
 */

#include "mcp_server.h"
#include "inference_handler.h"
#include "model_manager.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <regex>

// WebSocket++ includes
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <websocketpp/common/thread.hpp>
#include <websocketpp/common/memory.hpp>

namespace gemma {
namespace mcp {

// WebSocket server type definition
typedef websocketpp::server<websocketpp::config::asio> ws_server;
typedef websocketpp::connection_hdl connection_hdl;
typedef ws_server::message_ptr message_ptr;

/**
 * @brief Connection metadata for tracking client state
 */
struct ConnectionMetadata {
    std::string connection_id;
    connection_hdl hdl;
    std::string client_info;
    bool is_authenticated;
    std::chrono::steady_clock::time_point connect_time;
    std::chrono::steady_clock::time_point last_activity;
    size_t messages_sent;
    size_t messages_received;
    bool mcp_handshake_complete;
    std::string client_version;

    ConnectionMetadata(const std::string& id, connection_hdl h)
        : connection_id(id), hdl(h), is_authenticated(false),
          connect_time(std::chrono::steady_clock::now()),
          last_activity(std::chrono::steady_clock::now()),
          messages_sent(0), messages_received(0),
          mcp_handshake_complete(false) {}
};

/**
 * @brief Private implementation of MCPServer
 */
class MCPServer::Impl {
public:
    explicit Impl(const Config& config)
        : config_(config), running_(false), next_connection_id_(0) {
        // Initialize inference handler and model manager
        inference_handler_ = std::make_unique<InferenceHandler>(config);
        model_manager_ = std::make_unique<ModelManager>(config);

        // Initialize WebSocket server
        InitializeWebSocketServer();
    }

    bool Initialize() {
        try {
            // Initialize model manager
            if (!model_manager_->LoadModel(config_.model_path, config_.tokenizer_path)) {
                std::cerr << "Failed to load model" << std::endl;
                return false;
            }

            // Initialize inference handler
            if (!inference_handler_->Initialize(model_manager_.get())) {
                std::cerr << "Failed to initialize inference handler" << std::endl;
                return false;
            }

            // Register default tools
            RegisterDefaultTools();

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error during initialization: " << e.what() << std::endl;
            return false;
        }
    }

    bool Start() {
        if (running_) {
            return true;
        }

        try {
            // Configure and start WebSocket server
            ws_server_.set_reuse_addr(true);
            ws_server_.listen(config_.host, config_.port);
            ws_server_.start_accept();

            // Start server thread
            server_thread_ = std::thread([this]() {
                try {
                    ws_server_.run();
                } catch (const std::exception& e) {
                    LogError("WebSocket server error: " + std::string(e.what()));
                }
            });

            // Start heartbeat thread
            if (config_.heartbeat_interval_seconds > 0) {
                heartbeat_thread_ = std::thread([this]() {
                    RunHeartbeatLoop();
                });
            }

            running_ = true;

            if (config_.enable_logging) {
                LogInfo("MCP Server started on " + config_.host + ":" + std::to_string(config_.port));
                LogInfo("Max connections: " + std::to_string(config_.max_connections));
                LogInfo("Authentication required: " + std::string(config_.require_authentication ? "yes" : "no"));
            }

            return true;
        } catch (const std::exception& e) {
            LogError("Error starting server: " + std::string(e.what()));
            return false;
        }
    }

    void Stop() {
        if (!running_) {
            return;
        }

        running_ = false;

        try {
            // Stop WebSocket server
            ws_server_.stop();

            // Wait for threads to finish
            if (server_thread_.joinable()) {
                server_thread_.join();
            }

            if (heartbeat_thread_.joinable()) {
                heartbeat_thread_.join();
            }

            // Close all connections
            {
                std::lock_guard<std::mutex> lock(connections_mutex_);
                for (const auto& conn : connections_) {
                    try {
                        ws_server_.close(conn.second.hdl, websocketpp::close::status::going_away, "Server shutting down");
                    } catch (const std::exception& e) {
                        LogError("Error closing connection: " + std::string(e.what()));
                    }
                }
                connections_.clear();
            }

            if (config_.enable_logging) {
                LogInfo("MCP Server stopped");
            }
        } catch (const std::exception& e) {
            LogError("Error stopping server: " + std::string(e.what()));
        }
    }

    bool IsRunning() const {
        return running_;
    }

    MCPResponse HandleRequest(const MCPRequest& request) {
        MCPResponse response;
        response.id = request.id;

        try {
            // Route request to appropriate handler
            if (request.method == "tools/call") {
                response.result = HandleToolCall(request.params);
            } else if (request.method == "tools/list") {
                response.result = HandleToolList(request.params);
            } else {
                response.error = {
                    {"code", -32601},
                    {"message", "Method not found"},
                    {"data", request.method}
                };
            }
        } catch (const std::exception& e) {
            response.error = {
                {"code", -32603},
                {"message", "Internal error"},
                {"data", e.what()}
            };
        }

        return response;
    }

private:
    Config config_;
    std::atomic<bool> running_;
    std::unique_ptr<InferenceHandler> inference_handler_;
    std::unique_ptr<ModelManager> model_manager_;
    std::map<std::string, std::function<nlohmann::json(const nlohmann::json&)>> tool_handlers_;

    // WebSocket server components
    ws_server ws_server_;
    std::thread server_thread_;
    std::thread heartbeat_thread_;
    std::mutex connections_mutex_;
    std::unordered_map<std::string, ConnectionMetadata> connections_;
    std::atomic<uint64_t> next_connection_id_;

    // Statistics
    std::atomic<uint64_t> total_connections_{0};
    std::atomic<uint64_t> total_messages_{0};
    std::atomic<uint64_t> total_errors_{0};
    std::chrono::steady_clock::time_point server_start_time_;

    void RegisterDefaultTools() {
        tool_handlers_["generate_text"] = [this](const nlohmann::json& params) {
            return inference_handler_->GenerateText(params);
        };

        tool_handlers_["get_model_info"] = [this](const nlohmann::json& params) {
            return model_manager_->GetModelInfo(params);
        };

        tool_handlers_["tokenize_text"] = [this](const nlohmann::json& params) {
            return model_manager_->TokenizeText(params);
        };

        tool_handlers_["set_generation_params"] = [this](const nlohmann::json& params) {
            return inference_handler_->SetGenerationParams(params);
        };

        tool_handlers_["get_server_status"] = [this](const nlohmann::json& params) {
            return GetServerStatusInternal();
        };
    }

    // WebSocket server initialization and event handlers
    void InitializeWebSocketServer() {
        try {
            // Set logging
            ws_server_.set_access_channels(websocketpp::log::alevel::all);
            ws_server_.clear_access_channels(websocketpp::log::alevel::frame_payload);
            ws_server_.set_error_channels(websocketpp::log::elevel::all);

            // Initialize Asio
            ws_server_.init_asio();

            // Set message size limit
            ws_server_.set_max_message_size(config_.max_message_size);

            // Set handlers
            ws_server_.set_validate_handler([this](connection_hdl hdl) {
                return OnValidate(hdl);
            });

            ws_server_.set_open_handler([this](connection_hdl hdl) {
                OnOpen(hdl);
            });

            ws_server_.set_close_handler([this](connection_hdl hdl) {
                OnClose(hdl);
            });

            ws_server_.set_message_handler([this](connection_hdl hdl, message_ptr msg) {
                OnMessage(hdl, msg);
            });

            ws_server_.set_fail_handler([this](connection_hdl hdl) {
                OnFail(hdl);
            });

            server_start_time_ = std::chrono::steady_clock::now();

        } catch (const std::exception& e) {
            LogError("Failed to initialize WebSocket server: " + std::string(e.what()));
            throw;
        }
    }

    // WebSocket event handlers
    bool OnValidate(connection_hdl hdl) {
        try {
            std::lock_guard<std::mutex> lock(connections_mutex_);

            // Check connection limit
            if (connections_.size() >= config_.max_connections) {
                LogWarning("Connection rejected: maximum connections reached ("
                          + std::to_string(config_.max_connections) + ")");
                return false;
            }

            auto con = ws_server_.get_con_from_hdl(hdl);

            // Check subprotocols (MCP should use 'mcp' subprotocol)
            const auto& subprotocols = con->get_requested_subprotocols();
            bool mcp_supported = false;
            for (const auto& protocol : subprotocols) {
                if (protocol == "mcp" || protocol == "model-context-protocol") {
                    con->select_subprotocol(protocol);
                    mcp_supported = true;
                    break;
                }
            }

            if (!mcp_supported && !subprotocols.empty()) {
                LogWarning("Connection rejected: unsupported subprotocol");
                return false;
            }

            return true;
        } catch (const std::exception& e) {
            LogError("Error in OnValidate: " + std::string(e.what()));
            return false;
        }
    }

    void OnOpen(connection_hdl hdl) {
        try {
            std::string connection_id = GenerateConnectionId();
            ConnectionMetadata metadata(connection_id, hdl);

            auto con = ws_server_.get_con_from_hdl(hdl);
            metadata.client_info = con->get_request_header("User-Agent");

            {
                std::lock_guard<std::mutex> lock(connections_mutex_);
                connections_[connection_id] = std::move(metadata);
            }

            total_connections_++;

            LogInfo("New connection: " + connection_id + " (total: "
                   + std::to_string(connections_.size()) + ")");

            // Send MCP initialization message
            SendMCPInitialization(connection_id);

        } catch (const std::exception& e) {
            LogError("Error in OnOpen: " + std::string(e.what()));
        }
    }

    void OnClose(connection_hdl hdl) {
        try {
            std::string connection_id;

            {
                std::lock_guard<std::mutex> lock(connections_mutex_);
                for (auto it = connections_.begin(); it != connections_.end(); ++it) {
                    if (it->second.hdl.lock() == hdl.lock()) {
                        connection_id = it->first;
                        connections_.erase(it);
                        break;
                    }
                }
            }

            if (!connection_id.empty()) {
                LogInfo("Connection closed: " + connection_id + " (remaining: "
                       + std::to_string(connections_.size()) + ")");
            }

        } catch (const std::exception& e) {
            LogError("Error in OnClose: " + std::string(e.what()));
        }
    }

    void OnMessage(connection_hdl hdl, message_ptr msg) {
        try {
            std::string connection_id = GetConnectionId(hdl);
            if (connection_id.empty()) {
                LogError("Received message from unknown connection");
                return;
            }

            // Update connection activity
            UpdateConnectionActivity(connection_id);

            // Parse and handle MCP message
            auto payload = msg->get_payload();
            ProcessMCPMessage(connection_id, payload);

            total_messages_++;

        } catch (const std::exception& e) {
            LogError("Error in OnMessage: " + std::string(e.what()));
            total_errors_++;
        }
    }

    void OnFail(connection_hdl hdl) {
        try {
            auto con = ws_server_.get_con_from_hdl(hdl);
            LogError("Connection failed: " + con->get_ec().message());
            total_errors_++;
        } catch (const std::exception& e) {
            LogError("Error in OnFail: " + std::string(e.what()));
        }
    }

    // MCP Protocol Implementation
    void SendMCPInitialization(const std::string& connection_id) {
        nlohmann::json init_msg = {
            {"jsonrpc", "2.0"},
            {"method", "initialize"},
            {"params", {
                {"protocolVersion", "2024-11-05"},
                {"capabilities", {
                    {"tools", {}},
                    {"logging", {}},
                    {"prompts", {}}
                }},
                {"serverInfo", {
                    {"name", "gemma-mcp-server"},
                    {"version", "1.0.0"}
                }}
            }}
        };

        SendMessageToConnection(connection_id, init_msg);
    }

    void ProcessMCPMessage(const std::string& connection_id, const std::string& payload) {
        try {
            auto json_msg = nlohmann::json::parse(payload);

            // Validate JSON-RPC structure
            if (!json_msg.contains("jsonrpc") || json_msg["jsonrpc"] != "2.0") {
                SendErrorResponse(connection_id, "", -32600, "Invalid Request", "Missing or invalid jsonrpc field");
                return;
            }

            std::string request_id = json_msg.value("id", "");
            std::string method = json_msg.value("method", "");

            if (method.empty()) {
                SendErrorResponse(connection_id, request_id, -32600, "Invalid Request", "Missing method field");
                return;
            }

            // Handle MCP methods
            if (method == "initialize") {
                HandleInitialize(connection_id, json_msg);
            } else if (method == "initialized") {
                HandleInitialized(connection_id, json_msg);
            } else if (method == "tools/list") {
                HandleToolsList(connection_id, json_msg);
            } else if (method == "tools/call") {
                HandleToolsCall(connection_id, json_msg);
            } else if (method == "ping") {
                HandlePing(connection_id, json_msg);
            } else {
                SendErrorResponse(connection_id, request_id, -32601, "Method not found", "Unknown method: " + method);
            }

        } catch (const nlohmann::json::parse_error& e) {
            LogError("JSON parse error: " + std::string(e.what()));
            SendErrorResponse(connection_id, "", -32700, "Parse error", "Invalid JSON");
        } catch (const std::exception& e) {
            LogError("Error processing MCP message: " + std::string(e.what()));
            SendErrorResponse(connection_id, "", -32603, "Internal error", e.what());
        }
    }

    void HandleInitialize(const std::string& connection_id, const nlohmann::json& request) {
        try {
            std::string request_id = request.value("id", "");

            // Mark handshake as complete
            {
                std::lock_guard<std::mutex> lock(connections_mutex_);
                auto it = connections_.find(connection_id);
                if (it != connections_.end()) {
                    it->second.mcp_handshake_complete = true;
                    if (request.contains("params") && request["params"].contains("clientInfo")) {
                        auto client_info = request["params"]["clientInfo"];
                        it->second.client_version = client_info.value("version", "unknown");
                    }
                }
            }

            nlohmann::json response = {
                {"jsonrpc", "2.0"},
                {"id", request_id},
                {"result", {
                    {"protocolVersion", "2024-11-05"},
                    {"capabilities", {
                        {"tools", {}},
                        {"logging", {}},
                        {"prompts", {}}
                    }},
                    {"serverInfo", {
                        {"name", "gemma-mcp-server"},
                        {"version", "1.0.0"},
                        {"description", "Gemma.cpp MCP Server for AI inference"}
                    }}
                }}
            };

            SendMessageToConnection(connection_id, response);
            LogInfo("MCP handshake completed for connection: " + connection_id);

        } catch (const std::exception& e) {
            LogError("Error in HandleInitialize: " + std::string(e.what()));
            SendErrorResponse(connection_id, request.value("id", ""), -32603, "Internal error", e.what());
        }
    }

    void HandleInitialized(const std::string& connection_id, const nlohmann::json& request) {
        // This is a notification, no response needed
        LogInfo("Client initialized notification received from: " + connection_id);
    }

    void HandleToolsList(const std::string& connection_id, const nlohmann::json& request) {
        try {
            std::string request_id = request.value("id", "");
            nlohmann::json tools = GetToolsListResponse();

            nlohmann::json response = {
                {"jsonrpc", "2.0"},
                {"id", request_id},
                {"result", {
                    {"tools", tools}
                }}
            };

            SendMessageToConnection(connection_id, response);

        } catch (const std::exception& e) {
            LogError("Error in HandleToolsList: " + std::string(e.what()));
            SendErrorResponse(connection_id, request.value("id", ""), -32603, "Internal error", e.what());
        }
    }

    void HandleToolsCall(const std::string& connection_id, const nlohmann::json& request) {
        try {
            std::string request_id = request.value("id", "");

            if (!request.contains("params") || !request["params"].contains("name")) {
                SendErrorResponse(connection_id, request_id, -32602, "Invalid params", "Missing tool name");
                return;
            }

            std::string tool_name = request["params"]["name"];
            nlohmann::json arguments = request["params"].value("arguments", nlohmann::json::object());

            // Execute tool
            auto result = ExecuteTool(tool_name, arguments);

            nlohmann::json response = {
                {"jsonrpc", "2.0"},
                {"id", request_id},
                {"result", {
                    {"content", {
                        {
                            {"type", "text"},
                            {"text", result.dump()}
                        }
                    }}
                }}
            };

            SendMessageToConnection(connection_id, response);

        } catch (const std::exception& e) {
            LogError("Error in HandleToolsCall: " + std::string(e.what()));
            SendErrorResponse(connection_id, request.value("id", ""), -32603, "Internal error", e.what());
        }
    }

    void HandlePing(const std::string& connection_id, const nlohmann::json& request) {
        std::string request_id = request.value("id", "");
        nlohmann::json response = {
            {"jsonrpc", "2.0"},
            {"id", request_id},
            {"result", {
                {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()}
            }}
        };
        SendMessageToConnection(connection_id, response);
    }

    nlohmann::json HandleToolCall(const nlohmann::json& params) {
        if (!params.contains("name")) {
            throw std::invalid_argument("Tool name not specified");
        }

        std::string tool_name = params["name"];
        auto it = tool_handlers_.find(tool_name);
        if (it == tool_handlers_.end()) {
            throw std::invalid_argument("Unknown tool: " + tool_name);
        }

        nlohmann::json tool_params = params.value("arguments", nlohmann::json::object());
        return it->second(tool_params);
    }

    nlohmann::json HandleToolList(const nlohmann::json& params) {
        nlohmann::json tools = nlohmann::json::array();
        
        for (const auto& [name, handler] : tool_handlers_) {
            nlohmann::json tool;
            tool["name"] = name;
            tool["description"] = GetToolDescription(name);
            tool["inputSchema"] = GetToolInputSchema(name);
            tools.push_back(tool);
        }

        return {{"tools", tools}};
    }

    // Utility methods for WebSocket and MCP functionality
    std::string GenerateConnectionId() {
        return "conn_" + std::to_string(++next_connection_id_) + "_" +
               std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch()).count());
    }

    std::string GetConnectionId(connection_hdl hdl) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        for (const auto& conn : connections_) {
            if (conn.second.hdl.lock() == hdl.lock()) {
                return conn.first;
            }
        }
        return "";
    }

    void UpdateConnectionActivity(const std::string& connection_id) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = connections_.find(connection_id);
        if (it != connections_.end()) {
            it->second.last_activity = std::chrono::steady_clock::now();
            it->second.messages_received++;
        }
    }

    void SendMessageToConnection(const std::string& connection_id, const nlohmann::json& message) {
        try {
            std::lock_guard<std::mutex> lock(connections_mutex_);
            auto it = connections_.find(connection_id);
            if (it != connections_.end()) {
                auto payload = message.dump();
                ws_server_.send(it->second.hdl, payload, websocketpp::frame::opcode::text);
                it->second.messages_sent++;
            }
        } catch (const std::exception& e) {
            LogError("Error sending message: " + std::string(e.what()));
        }
    }

    void SendErrorResponse(const std::string& connection_id, const std::string& request_id,
                          int error_code, const std::string& error_message, const std::string& error_data = "") {
        nlohmann::json error_response = {
            {"jsonrpc", "2.0"},
            {"id", request_id},
            {"error", {
                {"code", error_code},
                {"message", error_message}
            }}
        };

        if (!error_data.empty()) {
            error_response["error"]["data"] = error_data;
        }

        SendMessageToConnection(connection_id, error_response);
    }

    nlohmann::json ExecuteTool(const std::string& tool_name, const nlohmann::json& arguments) {
        auto it = tool_handlers_.find(tool_name);
        if (it == tool_handlers_.end()) {
            throw std::invalid_argument("Unknown tool: " + tool_name);
        }
        return it->second(arguments);
    }

    nlohmann::json GetToolsListResponse() {
        nlohmann::json tools = nlohmann::json::array();

        for (const auto& [name, handler] : tool_handlers_) {
            nlohmann::json tool;
            tool["name"] = name;
            tool["description"] = GetToolDescription(name);
            tool["inputSchema"] = GetToolInputSchema(name);
            tools.push_back(tool);
        }

        return tools;
    }

    void RunHeartbeatLoop() {
        while (running_) {
            try {
                std::this_thread::sleep_for(std::chrono::seconds(config_.heartbeat_interval_seconds));

                if (!running_) break;

                // Check for inactive connections
                auto now = std::chrono::steady_clock::now();
                std::vector<std::string> inactive_connections;

                {
                    std::lock_guard<std::mutex> lock(connections_mutex_);
                    for (const auto& conn : connections_) {
                        auto idle_time = std::chrono::duration_cast<std::chrono::seconds>(
                            now - conn.second.last_activity).count();

                        if (idle_time > config_.websocket_timeout_seconds) {
                            inactive_connections.push_back(conn.first);
                        }
                    }
                }

                // Close inactive connections
                for (const auto& conn_id : inactive_connections) {
                    LogInfo("Closing inactive connection: " + conn_id);
                    CloseConnection(conn_id, "Connection timeout");
                }

                // Send ping to active connections
                {
                    std::lock_guard<std::mutex> lock(connections_mutex_);
                    for (const auto& conn : connections_) {
                        if (conn.second.mcp_handshake_complete) {
                            nlohmann::json ping = {
                                {"jsonrpc", "2.0"},
                                {"method", "ping"},
                                {"id", "heartbeat_" + std::to_string(std::time(nullptr))}
                            };
                            SendMessageToConnection(conn.first, ping);
                        }
                    }
                }

            } catch (const std::exception& e) {
                LogError("Heartbeat error: " + std::string(e.what()));
            }
        }
    }

    void CloseConnection(const std::string& connection_id, const std::string& reason) {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        auto it = connections_.find(connection_id);
        if (it != connections_.end()) {
            try {
                ws_server_.close(it->second.hdl, websocketpp::close::status::normal, reason);
            } catch (const std::exception& e) {
                LogError("Error closing connection: " + std::string(e.what()));
            }
            connections_.erase(it);
        }
    }

    nlohmann::json GetServerStatusInternal() {
        auto now = std::chrono::steady_clock::now();
        auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - server_start_time_).count();

        std::lock_guard<std::mutex> lock(connections_mutex_);
        nlohmann::json status;
        status["running"] = running_.load();
        status["model_loaded"] = model_manager_->IsModelLoaded();
        status["uptime_seconds"] = uptime;
        status["active_connections"] = connections_.size();
        status["total_connections"] = total_connections_.load();
        status["total_messages"] = total_messages_.load();
        status["total_errors"] = total_errors_.load();
        status["config"] = {
            {"host", config_.host},
            {"port", config_.port},
            {"max_concurrent_requests", config_.max_concurrent_requests},
            {"max_connections", config_.max_connections},
            {"websocket_timeout_seconds", config_.websocket_timeout_seconds}
        };

        return status;
    }

    // Logging methods
    void LogInfo(const std::string& message) {
        if (config_.enable_logging) {
            auto timestamp = GetTimestamp();
            std::cout << "[" << timestamp << "] [INFO] " << message << std::endl;
        }
    }

    void LogWarning(const std::string& message) {
        if (config_.enable_logging) {
            auto timestamp = GetTimestamp();
            std::cout << "[" << timestamp << "] [WARN] " << message << std::endl;
        }
    }

    void LogError(const std::string& message) {
        if (config_.enable_logging) {
            auto timestamp = GetTimestamp();
            std::cerr << "[" << timestamp << "] [ERROR] " << message << std::endl;
        }
    }

    std::string GetTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

    std::string GetToolDescription(const std::string& tool_name) {
        static const std::unordered_map<std::string, std::string> descriptions = {
            {"generate_text", "Generate text using the Gemma model with customizable parameters"},
            {"get_model_info", "Get comprehensive information about the loaded Gemma model"},
            {"tokenize_text", "Tokenize input text using the model's tokenizer"},
            {"set_generation_params", "Configure text generation parameters (temperature, max_tokens, etc.)"},
            {"get_server_status", "Get detailed server status, performance metrics, and connection information"}
        };

        auto it = descriptions.find(tool_name);
        return (it != descriptions.end()) ? it->second : "Custom tool - no description available";
    }

    nlohmann::json GetToolInputSchema(const std::string& tool_name) {
        static const std::unordered_map<std::string, nlohmann::json> schemas = {
            {"generate_text", {
                {"type", "object"},
                {"properties", {
                    {"prompt", {
                        {"type", "string"},
                        {"description", "The input prompt for text generation"}
                    }},
                    {"temperature", {
                        {"type", "number"},
                        {"description", "Controls randomness (0.0-2.0, default: 0.7)"},
                        {"minimum", 0.0},
                        {"maximum", 2.0}
                    }},
                    {"max_tokens", {
                        {"type", "integer"},
                        {"description", "Maximum tokens to generate"},
                        {"minimum", 1},
                        {"maximum", 8192}
                    }},
                    {"top_k", {
                        {"type", "integer"},
                        {"description", "Top-K sampling parameter"},
                        {"minimum", 1}
                    }},
                    {"top_p", {
                        {"type", "number"},
                        {"description", "Top-P sampling parameter (0.0-1.0)"},
                        {"minimum", 0.0},
                        {"maximum", 1.0}
                    }},
                    {"stop_sequence", {
                        {"type", "string"},
                        {"description", "Stop generation at this sequence"}
                    }}
                }},
                {"required", nlohmann::json::array({"prompt"})}
            }},
            {"get_model_info", {
                {"type", "object"},
                {"properties", {}},
                {"description", "No parameters required"}
            }},
            {"tokenize_text", {
                {"type", "object"},
                {"properties", {
                    {"text", {
                        {"type", "string"},
                        {"description", "Text to tokenize"}
                    }}
                }},
                {"required", nlohmann::json::array({"text"})}
            }},
            {"set_generation_params", {
                {"type", "object"},
                {"properties", {
                    {"temperature", {
                        {"type", "number"},
                        {"minimum", 0.0},
                        {"maximum", 2.0}
                    }},
                    {"max_tokens", {
                        {"type", "integer"},
                        {"minimum", 1},
                        {"maximum", 8192}
                    }},
                    {"top_k", {
                        {"type", "integer"},
                        {"minimum", 1}
                    }},
                    {"top_p", {
                        {"type", "number"},
                        {"minimum", 0.0},
                        {"maximum", 1.0}
                    }}
                }}
            }},
            {"get_server_status", {
                {"type", "object"},
                {"properties", {}},
                {"description", "No parameters required"}
            }}
        };

        auto it = schemas.find(tool_name);
        if (it != schemas.end()) {
            return it->second;
        }

        // Default schema for custom tools
        return {
            {"type", "object"},
            {"properties", {}}
        };
    }

    // Public interface implementations
    void RegisterCustomTool(const std::string& tool_name,
                           std::function<nlohmann::json(const nlohmann::json&)> handler) {
        if (tool_name.empty()) {
            throw std::invalid_argument("Tool name cannot be empty");
        }

        tool_handlers_[tool_name] = handler;
        LogInfo("Registered custom tool: " + tool_name);
    }

    size_t GetActiveConnectionCount() const {
        std::lock_guard<std::mutex> lock(connections_mutex_);
        return connections_.size();
    }

    nlohmann::json GetServerStats() const {
        return GetServerStatusInternal();
    }

    size_t BroadcastMessage(const nlohmann::json& message) {
        size_t sent_count = 0;
        std::lock_guard<std::mutex> lock(connections_mutex_);

        for (const auto& conn : connections_) {
            if (conn.second.mcp_handshake_complete) {
                try {
                    SendMessageToConnection(conn.first, message);
                    sent_count++;
                } catch (const std::exception& e) {
                    LogError("Error broadcasting to " + conn.first + ": " + std::string(e.what()));
                }
            }
        }

        return sent_count;
    }

    bool SendMessage(const std::string& connection_id, const nlohmann::json& message) {
        try {
            SendMessageToConnection(connection_id, message);
            return true;
        } catch (const std::exception& e) {
            LogError("Error sending message to " + connection_id + ": " + std::string(e.what()));
            return false;
        }
    }
};

// MCPServer implementation
MCPServer::MCPServer(const Config& config) 
    : impl_(std::make_unique<Impl>(config)) {
}

MCPServer::~MCPServer() = default;

bool MCPServer::Initialize() {
    return impl_->Initialize();
}

bool MCPServer::Start() {
    return impl_->Start();
}

void MCPServer::Stop() {
    impl_->Stop();
}

bool MCPServer::IsRunning() const {
    return impl_->IsRunning();
}

MCPServer::MCPResponse MCPServer::HandleRequest(const MCPRequest& request) {
    return impl_->HandleRequest(request);
}

void MCPServer::RegisterTool(const std::string& tool_name,
                           std::function<nlohmann::json(const nlohmann::json&)> handler) {
    impl_->RegisterCustomTool(tool_name, handler);
}

size_t MCPServer::GetActiveConnectionCount() const {
    return impl_->GetActiveConnectionCount();
}

nlohmann::json MCPServer::GetServerStats() const {
    return impl_->GetServerStats();
}

size_t MCPServer::BroadcastMessage(const nlohmann::json& message) {
    return impl_->BroadcastMessage(message);
}

bool MCPServer::SendMessage(const std::string& connection_id, const nlohmann::json& message) {
    return impl_->SendMessage(connection_id, message);
}

} // namespace mcp
} // namespace gemma