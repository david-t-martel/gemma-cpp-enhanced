#pragma once

/**
 * @file mcp_server.h
 * @brief MCP (Model Context Protocol) Server for Gemma.cpp
 * 
 * This header defines the MCP server interface for exposing Gemma.cpp
 * inference capabilities through the Model Context Protocol.
 */

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <nlohmann/json.hpp>

#include "gemma/gemma.h"
#include "inference_handler.h"
#include "model_manager.h"

namespace gemma {
namespace mcp {

/**
 * @brief Main MCP server class
 * 
 * Handles MCP protocol communication and coordinates with Gemma inference engine
 */
class MCPServer {
public:
    /**
     * @brief Configuration structure for MCP server
     */
    struct Config {
        std::string host = "localhost";
        int port = 8080;
        std::string model_path;
        std::string tokenizer_path;
        int max_concurrent_requests = 4;
        float temperature = 0.7f;
        int max_tokens = 1024;
        bool enable_logging = true;
        std::string log_level = "INFO";
    };

    /**
     * @brief MCP request structure
     */
    struct MCPRequest {
        std::string id;
        std::string method;
        nlohmann::json params;
        std::string jsonrpc = "2.0";
    };

    /**
     * @brief MCP response structure
     */
    struct MCPResponse {
        std::string id;
        nlohmann::json result;
        nlohmann::json error;
        std::string jsonrpc = "2.0";
    };

    /**
     * @brief Constructor
     * @param config Server configuration
     */
    explicit MCPServer(const Config& config);

    /**
     * @brief Destructor
     */
    ~MCPServer();

    /**
     * @brief Initialize the MCP server
     * @return true if initialization successful
     */
    bool Initialize();

    /**
     * @brief Start the MCP server
     * @return true if server started successfully
     */
    bool Start();

    /**
     * @brief Stop the MCP server
     */
    void Stop();

    /**
     * @brief Check if server is running
     * @return true if server is running
     */
    bool IsRunning() const;

    /**
     * @brief Handle MCP request
     * @param request The MCP request to handle
     * @return MCP response
     */
    MCPResponse HandleRequest(const MCPRequest& request);

    /**
     * @brief Register a custom tool handler
     * @param tool_name Name of the tool
     * @param handler Function to handle the tool
     */
    void RegisterTool(const std::string& tool_name, 
                     std::function<nlohmann::json(const nlohmann::json&)> handler);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Available MCP tools for Gemma.cpp
 */
namespace tools {
    /**
     * @brief Generate text using Gemma model
     */
    nlohmann::json GenerateText(const nlohmann::json& params);

    /**
     * @brief Get model information
     */
    nlohmann::json GetModelInfo(const nlohmann::json& params);

    /**
     * @brief Tokenize text
     */
    nlohmann::json TokenizeText(const nlohmann::json& params);

    /**
     * @brief Set generation parameters
     */
    nlohmann::json SetGenerationParams(const nlohmann::json& params);

    /**
     * @brief Get server status
     */
    nlohmann::json GetServerStatus(const nlohmann::json& params);
}

} // namespace mcp
} // namespace gemma