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

namespace gemma {
namespace mcp {

/**
 * @brief Private implementation of MCPServer
 */
class MCPServer::Impl {
public:
    explicit Impl(const Config& config) 
        : config_(config), running_(false) {
        // Initialize inference handler and model manager
        inference_handler_ = std::make_unique<InferenceHandler>(config);
        model_manager_ = std::make_unique<ModelManager>(config);
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
            // TODO: Initialize WebSocket server here
            // For now, just set running flag
            running_ = true;
            
            if (config_.enable_logging) {
                std::cout << "MCP Server started on " << config_.host 
                         << ":" << config_.port << std::endl;
            }

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error starting server: " << e.what() << std::endl;
            return false;
        }
    }

    void Stop() {
        if (!running_) {
            return;
        }

        running_ = false;
        
        if (config_.enable_logging) {
            std::cout << "MCP Server stopped" << std::endl;
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
    bool running_;
    std::unique_ptr<InferenceHandler> inference_handler_;
    std::unique_ptr<ModelManager> model_manager_;
    std::map<std::string, std::function<nlohmann::json(const nlohmann::json&)>> tool_handlers_;

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
            nlohmann::json status;
            status["running"] = running_;
            status["model_loaded"] = model_manager_->IsModelLoaded();
            status["config"] = {
                {"host", config_.host},
                {"port", config_.port},
                {"max_concurrent_requests", config_.max_concurrent_requests}
            };
            return status;
        };
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

    std::string GetToolDescription(const std::string& tool_name) {
        // TODO: Implement proper tool descriptions
        if (tool_name == "generate_text") {
            return "Generate text using the Gemma model";
        } else if (tool_name == "get_model_info") {
            return "Get information about the loaded model";
        } else if (tool_name == "tokenize_text") {
            return "Tokenize input text";
        } else if (tool_name == "set_generation_params") {
            return "Set generation parameters";
        } else if (tool_name == "get_server_status") {
            return "Get server status information";
        }
        return "No description available";
    }

    nlohmann::json GetToolInputSchema(const std::string& tool_name) {
        // TODO: Implement proper JSON schemas for tool inputs
        return {
            {"type", "object"},
            {"properties", nlohmann::json::object()}
        };
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
    // TODO: Implement custom tool registration
}

} // namespace mcp
} // namespace gemma