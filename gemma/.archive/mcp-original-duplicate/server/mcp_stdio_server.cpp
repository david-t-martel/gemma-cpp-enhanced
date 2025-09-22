/**
 * @file mcp_stdio_server.cpp
 * @brief MCP Server with stdio transport for initial testing
 */

#include <iostream>
#include <string>
#include <sstream>
#include <memory>
#include <thread>
#include <atomic>

#include <nlohmann/json.hpp>
#include "mcp_server.h"

using namespace gemma::mcp;
using json = nlohmann::json;

class MCPStdioServer {
public:
    explicit MCPStdioServer(const MCPServer::Config& config)
        : config_(config), running_(false) {
        mcp_server_ = std::make_unique<MCPServer>(config);
    }

    bool Initialize() {
        if (!mcp_server_->Initialize()) {
            std::cerr << "Failed to initialize MCP server" << std::endl;
            return false;
        }
        return true;
    }

    void Run() {
        running_ = true;

        std::string line;
        while (running_ && std::getline(std::cin, line)) {
            if (line.empty()) {
                continue;
            }

            try {
                ProcessMessage(line);
            } catch (const std::exception& e) {
                SendError("", -32603, "Internal error", e.what());
            }
        }
    }

    void Stop() {
        running_ = false;
        if (mcp_server_) {
            mcp_server_->Stop();
        }
    }

private:
    MCPServer::Config config_;
    std::unique_ptr<MCPServer> mcp_server_;
    std::atomic<bool> running_;

    void ProcessMessage(const std::string& message) {
        json request;

        try {
            request = json::parse(message);
        } catch (const json::parse_error& e) {
            SendError("", -32700, "Parse error", e.what());
            return;
        }

        // Validate JSON-RPC 2.0 structure
        if (!request.contains("jsonrpc") || request["jsonrpc"] != "2.0") {
            SendError("", -32600, "Invalid Request", "Missing or invalid jsonrpc field");
            return;
        }

        std::string request_id = request.value("id", "");

        if (!request.contains("method")) {
            SendError(request_id, -32600, "Invalid Request", "Missing method field");
            return;
        }

        std::string method = request["method"];
        json params = request.value("params", json::object());

        // Handle different MCP methods
        if (method == "initialize") {
            HandleInitialize(request_id, params);
        } else if (method == "tools/list") {
            HandleToolsList(request_id, params);
        } else if (method == "tools/call") {
            HandleToolsCall(request_id, params);
        } else if (method == "ping") {
            HandlePing(request_id, params);
        } else {
            SendError(request_id, -32601, "Method not found", "Unknown method: " + method);
        }
    }

    void HandleInitialize(const std::string& request_id, const json& params) {
        json response = {
            {"jsonrpc", "2.0"},
            {"id", request_id},
            {"result", {
                {"protocolVersion", "2024-11-05"},
                {"capabilities", {
                    {"tools", json::object()}
                }},
                {"serverInfo", {
                    {"name", "gemma-mcp-server"},
                    {"version", "1.0.0"}
                }}
            }}
        };

        SendResponse(response);
    }

    void HandleToolsList(const std::string& request_id, const json& params) {
        json tools = json::array();

        // Define available tools with proper schemas
        tools.push_back({
            {"name", "generate_text"},
            {"description", "Generate text using the Gemma model"},
            {"inputSchema", {
                {"type", "object"},
                {"properties", {
                    {"prompt", {
                        {"type", "string"},
                        {"description", "The input prompt for text generation"}
                    }},
                    {"temperature", {
                        {"type", "number"},
                        {"description", "Sampling temperature (0.0 to 2.0)"},
                        {"minimum", 0.0},
                        {"maximum", 2.0}
                    }},
                    {"max_tokens", {
                        {"type", "integer"},
                        {"description", "Maximum number of tokens to generate"},
                        {"minimum", 1},
                        {"maximum", 4096}
                    }},
                    {"top_k", {
                        {"type", "integer"},
                        {"description", "Top-k sampling parameter"},
                        {"minimum", 1}
                    }},
                    {"top_p", {
                        {"type", "number"},
                        {"description", "Top-p (nucleus) sampling parameter"},
                        {"minimum", 0.0},
                        {"maximum", 1.0}
                    }}
                }},
                {"required", json::array({"prompt"})}
            }}
        });

        tools.push_back({
            {"name", "count_tokens"},
            {"description", "Count tokens in the given text"},
            {"inputSchema", {
                {"type", "object"},
                {"properties", {
                    {"text", {
                        {"type", "string"},
                        {"description", "The text to tokenize and count"}
                    }},
                    {"include_details", {
                        {"type", "boolean"},
                        {"description", "Include detailed token information"},
                        {"default", false}
                    }}
                }},
                {"required", json::array({"text"})}
            }}
        });

        tools.push_back({
            {"name", "get_model_info"},
            {"description", "Get information about the loaded Gemma model"},
            {"inputSchema", {
                {"type", "object"},
                {"properties", {}},
                {"additionalProperties", false}
            }}
        });

        json response = {
            {"jsonrpc", "2.0"},
            {"id", request_id},
            {"result", {
                {"tools", tools}
            }}
        };

        SendResponse(response);
    }

    void HandleToolsCall(const std::string& request_id, const json& params) {
        if (!params.contains("name")) {
            SendError(request_id, -32602, "Invalid params", "Missing tool name");
            return;
        }

        std::string tool_name = params["name"];
        json arguments = params.value("arguments", json::object());

        try {
            json result;

            if (tool_name == "generate_text") {
                result = HandleGenerateText(arguments);
            } else if (tool_name == "count_tokens") {
                result = HandleCountTokens(arguments);
            } else if (tool_name == "get_model_info") {
                result = HandleGetModelInfo(arguments);
            } else {
                SendError(request_id, -32602, "Invalid params", "Unknown tool: " + tool_name);
                return;
            }

            json response = {
                {"jsonrpc", "2.0"},
                {"id", request_id},
                {"result", {
                    {"content", json::array({
                        {
                            {"type", "text"},
                            {"text", result.dump(2)}
                        }
                    })}
                }}
            };

            SendResponse(response);

        } catch (const std::exception& e) {
            SendError(request_id, -32603, "Internal error", e.what());
        }
    }

    void HandlePing(const std::string& request_id, const json& params) {
        json response = {
            {"jsonrpc", "2.0"},
            {"id", request_id},
            {"result", {{"status", "pong"}}}
        };

        SendResponse(response);
    }

    json HandleGenerateText(const json& arguments) {
        // Create MCP request for the server
        MCPServer::MCPRequest request;
        request.id = "internal";
        request.method = "tools/call";
        request.params = {
            {"name", "generate_text"},
            {"arguments", arguments}
        };

        auto response = mcp_server_->HandleRequest(request);

        if (!response.error.is_null()) {
            throw std::runtime_error(response.error.dump());
        }

        return response.result;
    }

    json HandleCountTokens(const json& arguments) {
        // Create MCP request for tokenization
        MCPServer::MCPRequest request;
        request.id = "internal";
        request.method = "tools/call";
        request.params = {
            {"name", "tokenize_text"},
            {"arguments", arguments}
        };

        auto response = mcp_server_->HandleRequest(request);

        if (!response.error.is_null()) {
            throw std::runtime_error(response.error.dump());
        }

        return response.result;
    }

    json HandleGetModelInfo(const json& arguments) {
        // Create MCP request for model info
        MCPServer::MCPRequest request;
        request.id = "internal";
        request.method = "tools/call";
        request.params = {
            {"name", "get_model_info"},
            {"arguments", arguments}
        };

        auto response = mcp_server_->HandleRequest(request);

        if (!response.error.is_null()) {
            throw std::runtime_error(response.error.dump());
        }

        return response.result;
    }

    void SendResponse(const json& response) {
        std::cout << response.dump() << std::endl;
        std::cout.flush();
    }

    void SendError(const std::string& request_id, int error_code, const std::string& message, const std::string& data = "") {
        json error_response = {
            {"jsonrpc", "2.0"},
            {"id", request_id},
            {"error", {
                {"code", error_code},
                {"message", message}
            }}
        };

        if (!data.empty()) {
            error_response["error"]["data"] = data;
        }

        SendResponse(error_response);
    }
};

void PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " --model PATH [options]\n"
              << "MCP Server with stdio transport for Gemma.cpp\n\n"
              << "Required:\n"
              << "  --model PATH          Path to model weights file\n\n"
              << "Optional:\n"
              << "  --tokenizer PATH      Path to tokenizer (for separate tokenizer files)\n"
              << "  --temperature F       Default temperature (default: 0.7)\n"
              << "  --max-tokens N        Default max tokens (default: 1024)\n"
              << "  --help               Show this help message\n"
              << std::endl;
}

MCPServer::Config ParseArguments(int argc, char* argv[]) {
    MCPServer::Config config;
    config.enable_logging = false; // Disable logging for stdio mode

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help") {
            PrintUsage(argv[0]);
            exit(0);
        } else if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (arg == "--tokenizer" && i + 1 < argc) {
            config.tokenizer_path = argv[++i];
        } else if (arg == "--temperature" && i + 1 < argc) {
            config.temperature = std::stof(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            config.max_tokens = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            exit(1);
        }
    }

    return config;
}

bool ValidateConfig(const MCPServer::Config& config) {
    if (config.model_path.empty()) {
        std::cerr << "Error: Model path is required (--model)" << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    MCPServer::Config config = ParseArguments(argc, argv);

    // Validate configuration
    if (!ValidateConfig(config)) {
        return 1;
    }

    try {
        // Create and initialize stdio server
        MCPStdioServer server(config);

        if (!server.Initialize()) {
            std::cerr << "Failed to initialize MCP server" << std::endl;
            return 1;
        }

        // Run the server (will read from stdin and write to stdout)
        server.Run();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}