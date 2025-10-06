// Shared mock system interfaces extracted from monolithic end-to-end test
#pragma once

#include <gmock/gmock.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <cstdint>

using json = nlohmann::json;

struct SystemConfig {
    std::string model_weights_path;
    std::string tokenizer_path;
    std::string backend_name = "cpu";
    std::string session_storage_path;
    bool enable_mcp_server = true;
    uint16_t mcp_server_port = 8080;
    size_t max_context_tokens = 8192;
    bool enable_metrics = true;
};

// Segregated interfaces (Interface Segregation Principle) – could be expanded later
struct IModelOperations { virtual ~IModelOperations() = default; };
struct ITextGeneration { virtual ~ITextGeneration() = default; };
struct ISessionOperations { virtual ~ISessionOperations() = default; };
struct IBackendOperations { virtual ~IBackendOperations() = default; };
struct IMCPServerOperations { virtual ~IMCPServerOperations() = default; };
struct ISystemStatus { virtual ~ISystemStatus() = default; };

// Composite mock satisfying multiple responsibilities but kept lean per test usage
class MockGemmaSystem : public IModelOperations,
                        public ITextGeneration,
                        public ISessionOperations,
                        public IBackendOperations,
                        public IMCPServerOperations,
                        public ISystemStatus {
public:
    // System lifecycle
    MOCK_METHOD(bool, initialize, (const SystemConfig& config), ());
    MOCK_METHOD(void, shutdown, (), ());
    MOCK_METHOD(bool, is_initialized, (), (const));

    // Model operations
    MOCK_METHOD(bool, load_model, (const std::string& weights_path, const std::string& tokenizer_path), ());
    MOCK_METHOD(bool, unload_model, (), ());
    MOCK_METHOD(bool, is_model_loaded, (), (const));
    MOCK_METHOD(json, get_model_info, (), (const));

    // Text generation
    MOCK_METHOD(std::string, generate_text, (const std::string& prompt, const json& options), ());
    MOCK_METHOD(std::vector<std::string>, generate_batch, (const std::vector<std::string>& prompts, const json& options), ());
    MOCK_METHOD(int, count_tokens, (const std::string& text), (const));

    // Session management
    MOCK_METHOD(std::string, create_session, (const json& options), ());
    MOCK_METHOD(bool, delete_session, (const std::string& session_id), ());
    MOCK_METHOD(bool, add_message_to_session, (const std::string& session_id, const std::string& role, const std::string& content), ());
    MOCK_METHOD(json, get_session_history, (const std::string& session_id), ());
    MOCK_METHOD(std::vector<std::string>, list_sessions, (), ());

    // Backend management
    MOCK_METHOD(bool, set_backend, (const std::string& backend_name), ());
    MOCK_METHOD(std::string, get_current_backend, (), (const));
    MOCK_METHOD(std::vector<std::string>, list_available_backends, (), (const));
    MOCK_METHOD(json, get_backend_status, (), (const));

    // MCP server operations
    MOCK_METHOD(bool, start_mcp_server, (uint16_t port), ());
    MOCK_METHOD(bool, stop_mcp_server, (), ());
    MOCK_METHOD(bool, is_mcp_server_running, (), (const));
    MOCK_METHOD(json, handle_mcp_request, (const json& request), ());

    // System status & metrics
    MOCK_METHOD(json, get_system_status, (), (const));
    MOCK_METHOD(json, get_metrics, (), (const));
    MOCK_METHOD(void, reset_metrics, (), ());
    MOCK_METHOD(SystemConfig, get_config, (), (const));
    MOCK_METHOD(bool, update_config, (const SystemConfig& new_config), ());
};

// Matcher for SystemConfig equality (Single Responsibility – matching only)
MATCHER_P(MatchesConfig, expected_config, "SystemConfig fields differ") {
    return arg.model_weights_path == expected_config.model_weights_path &&
           arg.tokenizer_path == expected_config.tokenizer_path &&
           arg.backend_name == expected_config.backend_name &&
           arg.max_context_tokens == expected_config.max_context_tokens;
}
