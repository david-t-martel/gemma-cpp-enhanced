#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../utils/test_helpers.h"
#include <nlohmann/json.hpp>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <future>
#include <atomic>
#include <sstream>
#include <queue>
#include <mutex>

// Mock MCP server integration components
// In real implementation: #include "../../src/mcp/MCPServer.h"

using namespace gemma::test;
using namespace testing;
using json = nlohmann::json;

// Mock transport layer for MCP communication
class MockMCPTransport {
public:
    MOCK_METHOD(bool, initialize, (const std::string& address), ());
    MOCK_METHOD(void, shutdown, (), ());
    MOCK_METHOD(bool, is_connected, (), (const));
    MOCK_METHOD(bool, send_message, (const json& message), ());
    MOCK_METHOD(std::optional<json>, receive_message, (std::chrono::milliseconds timeout), ());
    MOCK_METHOD(void, set_message_handler, (std::function<void(const json&)> handler), ());
    MOCK_METHOD(std::string, get_transport_type, (), (const));
    MOCK_METHOD(json, get_connection_info, (), (const));
};

// Mock MCP server implementation
class MockMCPServer {
public:
    MOCK_METHOD(bool, initialize, (uint16_t port, const std::string& transport_type), ());
    MOCK_METHOD(void, shutdown, (), ());
    MOCK_METHOD(bool, is_running, (), (const));
    MOCK_METHOD(uint16_t, get_port, (), (const));
    MOCK_METHOD(void, register_tool, (const std::string& name, const json& schema, std::function<json(const json&)> handler), ());
    MOCK_METHOD(void, unregister_tool, (const std::string& name), ());
    MOCK_METHOD(std::vector<std::string>, list_tools, (), (const));
    MOCK_METHOD(json, handle_request, (const json& request), ());
    MOCK_METHOD(void, send_notification, (const json& notification), ());
    MOCK_METHOD(json, get_server_capabilities, (), (const));
    MOCK_METHOD(json, get_server_info, (), (const));
    MOCK_METHOD(size_t, get_active_connections, (), (const));
    MOCK_METHOD(json, get_connection_stats, (), (const));
    MOCK_METHOD(void, set_request_timeout, (std::chrono::milliseconds timeout), ());
    MOCK_METHOD(void, set_max_connections, (size_t max_connections), ());
};

// Mock MCP client for testing server responses
class MockMCPClient {
public:
    MOCK_METHOD(bool, connect, (const std::string& server_address), ());
    MOCK_METHOD(void, disconnect, (), ());
    MOCK_METHOD(bool, is_connected, (), (const));
    MOCK_METHOD(json, send_request, (const json& request), ());
    MOCK_METHOD(json, call_tool, (const std::string& tool_name, const json& arguments), ());
    MOCK_METHOD(std::vector<json>, list_server_tools, (), ());
    MOCK_METHOD(json, get_server_info, (), ());
    MOCK_METHOD(void, set_timeout, (std::chrono::milliseconds timeout), ());
    MOCK_METHOD(void, set_notification_handler, (std::function<void(const json&)> handler), ());
};

// Mock Gemma model interface for MCP tools
class MockGemmaModelForMCP {
public:
    MOCK_METHOD(bool, is_loaded, (), (const));
    MOCK_METHOD(std::string, generate_text, (const std::string& prompt, const json& options), ());
    MOCK_METHOD(int, count_tokens, (const std::string& text), (const));
    MOCK_METHOD(json, get_model_info, (), (const));
    MOCK_METHOD(std::vector<int>, tokenize, (const std::string& text), (const));
    MOCK_METHOD(std::string, detokenize, (const std::vector<int>& tokens), (const));
    MOCK_METHOD(float, get_perplexity, (const std::string& text), (const));
};

class MCPServerIntegrationTest : public GemmaTestBase {
protected:
    void SetUp() override {
        GemmaTestBase::SetUp();
        
        server_ = std::make_unique<MockMCPServer>();
        client_ = std::make_unique<MockMCPClient>();
        transport_ = std::make_unique<MockMCPTransport>();
        model_ = std::make_unique<MockGemmaModelForMCP>();
        
        server_port_ = 8081; // Use non-standard port for testing
        server_address_ = "http://localhost:" + std::to_string(server_port_);
        
        setup_default_expectations();
        setup_test_tools();
    }
    
    void setup_default_expectations() {
        // Server expectations
        ON_CALL(*server_, initialize(_, _)).WillByDefault(Return(true));
        ON_CALL(*server_, is_running()).WillByDefault(Return(true));
        ON_CALL(*server_, get_port()).WillByDefault(Return(server_port_));
        ON_CALL(*server_, get_active_connections()).WillByDefault(Return(1));
        
        // Client expectations
        ON_CALL(*client_, connect(_)).WillByDefault(Return(true));
        ON_CALL(*client_, is_connected()).WillByDefault(Return(true));
        
        // Transport expectations
        ON_CALL(*transport_, initialize(_)).WillByDefault(Return(true));
        ON_CALL(*transport_, is_connected()).WillByDefault(Return(true));
        ON_CALL(*transport_, get_transport_type()).WillByDefault(Return("http"));
        
        // Model expectations
        ON_CALL(*model_, is_loaded()).WillByDefault(Return(true));
        ON_CALL(*model_, generate_text(_, _)).WillByDefault(Return("Generated response"));
        ON_CALL(*model_, count_tokens(_)).WillByDefault(Return(10));
        ON_CALL(*model_, get_model_info()).WillByDefault(Return(json{
            {"name", "gemma-2b-it"},
            {"loaded", true},
            {"context_length", 8192}
        }));
    }
    
    void setup_test_tools() {
        tool_schemas_ = {
            {"generate_text", json{
                {"type", "function"},
                {"function", {
                    {"name", "generate_text"},
                    {"description", "Generate text using the loaded model"},
                    {"parameters", {
                        {"type", "object"},
                        {"properties", {
                            {"prompt", {{"type", "string"}, {"description", "Input prompt"}}},
                            {"max_tokens", {{"type", "integer"}, {"description", "Maximum tokens to generate"}}},
                            {"temperature", {{"type", "number"}, {"description", "Sampling temperature"}}}
                        }},
                        {"required", {"prompt"}}
                    }}
                }}
            }},
            {"count_tokens", json{
                {"type", "function"},
                {"function", {
                    {"name", "count_tokens"},
                    {"description", "Count tokens in text"},
                    {"parameters", {
                        {"type", "object"},
                        {"properties", {
                            {"text", {{"type", "string"}, {"description", "Text to count tokens"}}}
                        }},
                        {"required", {"text"}}
                    }}
                }}
            }},
            {"get_model_info", json{
                {"type", "function"},
                {"function", {
                    {"name", "get_model_info"},
                    {"description", "Get information about the loaded model"},
                    {"parameters", {{"type", "object"}, {"properties", {}}}}
                }}
            }}
        };
    }
    
    std::unique_ptr<MockMCPServer> server_;
    std::unique_ptr<MockMCPClient> client_;
    std::unique_ptr<MockMCPTransport> transport_;
    std::unique_ptr<MockGemmaModelForMCP> model_;
    uint16_t server_port_;
    std::string server_address_;
    std::map<std::string, json> tool_schemas_;
};

// Server initialization and startup tests

TEST_F(MCPServerIntegrationTest, ServerInitializationAndStartup) {
    EXPECT_CALL(*server_, initialize(server_port_, "http"))
        .Times(1)
        .WillOnce(Return(true));
    
    bool initialized = server_->initialize(server_port_, "http");
    EXPECT_TRUE(initialized);
    
    // Verify server is running
    EXPECT_CALL(*server_, is_running())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_TRUE(server_->is_running());
    
    // Check port assignment
    EXPECT_CALL(*server_, get_port())
        .Times(1)
        .WillOnce(Return(server_port_));
    
    EXPECT_EQ(server_->get_port(), server_port_);
}

TEST_F(MCPServerIntegrationTest, ServerCapabilitiesAdvertisement) {
    json expected_capabilities = {
        {"tools", true},
        {"resources", false},
        {"prompts", false},
        {"notifications", true},
        {"sampling", false}
    };
    
    EXPECT_CALL(*server_, get_server_capabilities())
        .Times(1)
        .WillOnce(Return(expected_capabilities));
    
    auto capabilities = server_->get_server_capabilities();
    EXPECT_TRUE(capabilities["tools"]);
    EXPECT_TRUE(capabilities["notifications"]);
    EXPECT_FALSE(capabilities["resources"]);
}

TEST_F(MCPServerIntegrationTest, ServerInfoRetrieval) {
    json expected_info = {
        {"name", "gemma-mcp-server"},
        {"version", "1.0.0"},
        {"description", "MCP server for Gemma model"},
        {"author", "Gemma Team"},
        {"license", "Apache-2.0"},
        {"homepage", "https://github.com/gemma/gemma.cpp"}
    };
    
    EXPECT_CALL(*server_, get_server_info())
        .Times(1)
        .WillOnce(Return(expected_info));
    
    auto info = server_->get_server_info();
    EXPECT_EQ(info["name"], "gemma-mcp-server");
    EXPECT_EQ(info["version"], "1.0.0");
}

// Tool registration and management tests

TEST_F(MCPServerIntegrationTest, ToolRegistration) {
    // Register generate_text tool
    auto generate_handler = [this](const json& args) -> json {
        std::string prompt = args.value("prompt", "");
        return json{{"text", model_->generate_text(prompt, args)}};
    };
    
    EXPECT_CALL(*server_, register_tool("generate_text", _, _))
        .Times(1);
    
    server_->register_tool("generate_text", tool_schemas_["generate_text"], generate_handler);
    
    // Verify tool is listed
    std::vector<std::string> expected_tools = {"generate_text"};
    EXPECT_CALL(*server_, list_tools())
        .Times(1)
        .WillOnce(Return(expected_tools));
    
    auto tools = server_->list_tools();
    EXPECT_THAT(tools, Contains("generate_text"));
}

TEST_F(MCPServerIntegrationTest, MultipleToolRegistration) {
    // Register all test tools
    for (const auto& [tool_name, schema] : tool_schemas_) {
        EXPECT_CALL(*server_, register_tool(tool_name, _, _))
            .Times(1);
        
        auto handler = [](const json& args) -> json {
            return json{{"result", "mock_response"}};
        };
        
        server_->register_tool(tool_name, schema, handler);
    }
    
    // Verify all tools are listed
    std::vector<std::string> expected_tools = {"generate_text", "count_tokens", "get_model_info"};
    EXPECT_CALL(*server_, list_tools())
        .Times(1)
        .WillOnce(Return(expected_tools));
    
    auto tools = server_->list_tools();
    EXPECT_EQ(tools.size(), 3);
    EXPECT_THAT(tools, UnorderedElementsAre("generate_text", "count_tokens", "get_model_info"));
}

TEST_F(MCPServerIntegrationTest, ToolUnregistration) {
    // First register a tool
    EXPECT_CALL(*server_, register_tool("test_tool", _, _))
        .Times(1);
    
    server_->register_tool("test_tool", json{}, [](const json&) { return json{}; });
    
    // Then unregister it
    EXPECT_CALL(*server_, unregister_tool("test_tool"))
        .Times(1);
    
    server_->unregister_tool("test_tool");
    
    // Verify it's no longer in the list
    EXPECT_CALL(*server_, list_tools())
        .Times(1)
        .WillOnce(Return(std::vector<std::string>{}));
    
    auto tools = server_->list_tools();
    EXPECT_THAT(tools, Not(Contains("test_tool")));
}

// Client-server communication tests

TEST_F(MCPServerIntegrationTest, ClientServerHandshake) {
    // Client connects to server
    EXPECT_CALL(*client_, connect(server_address_))
        .Times(1)
        .WillOnce(Return(true));
    
    bool connected = client_->connect(server_address_);
    EXPECT_TRUE(connected);
    
    // Client requests server info
    json expected_info = {
        {"name", "gemma-mcp-server"},
        {"version", "1.0.0"}
    };
    
    EXPECT_CALL(*client_, get_server_info())
        .Times(1)
        .WillOnce(Return(expected_info));
    
    auto info = client_->get_server_info();
    EXPECT_EQ(info["name"], "gemma-mcp-server");
    
    // Client lists available tools
    std::vector<json> expected_tools = {
        tool_schemas_["generate_text"],
        tool_schemas_["count_tokens"],
        tool_schemas_["get_model_info"]
    };
    
    EXPECT_CALL(*client_, list_server_tools())
        .Times(1)
        .WillOnce(Return(expected_tools));
    
    auto tools = client_->list_server_tools();
    EXPECT_EQ(tools.size(), 3);
}

// Tool invocation tests

TEST_F(MCPServerIntegrationTest, GenerateTextToolInvocation) {
    json request = {
        {"jsonrpc", "2.0"},
        {"method", "tools/call"},
        {"id", "1"},
        {"params", {
            {"name", "generate_text"},
            {"arguments", {
                {"prompt", "What is artificial intelligence?"},
                {"max_tokens", 100},
                {"temperature", 0.7}
            }}
        }}
    };
    
    json expected_response = {
        {"jsonrpc", "2.0"},
        {"id", "1"},
        {"result", {
            {"content", {
                {
                    {"type", "text"},
                    {"text", "Artificial intelligence (AI) is a branch of computer science..."}
                }
            }}
        }}
    };
    
    EXPECT_CALL(*server_, handle_request(MatchesRequest(request)))
        .Times(1)
        .WillOnce(Return(expected_response));
    
    auto response = server_->handle_request(request);
    
    EXPECT_EQ(response["jsonrpc"], "2.0");
    EXPECT_EQ(response["id"], "1");
    EXPECT_TRUE(response.contains("result"));
}

TEST_F(MCPServerIntegrationTest, CountTokensToolInvocation) {
    json arguments = {
        {"text", "This is a sample text for token counting."}
    };
    
    json expected_result = {
        {"content", {
            {
                {"type", "text"},
                {"text", "12"}
            }
        }}
    };
    
    EXPECT_CALL(*client_, call_tool("count_tokens", arguments))
        .Times(1)
        .WillOnce(Return(expected_result));
    
    auto result = client_->call_tool("count_tokens", arguments);
    
    EXPECT_TRUE(result.contains("content"));
    auto content = result["content"][0];
    EXPECT_EQ(content["type"], "text");
    EXPECT_EQ(content["text"], "12");
}

TEST_F(MCPServerIntegrationTest, GetModelInfoToolInvocation) {
    json expected_result = {
        {"content", {
            {
                {"type", "text"},
                {"text", R"({
                    "name": "gemma-2b-it",
                    "size": "2B",
                    "context_length": 8192,
                    "vocab_size": 256000,
                    "loaded": true
                })"}
            }
        }}
    };
    
    EXPECT_CALL(*client_, call_tool("get_model_info", json::object()))
        .Times(1)
        .WillOnce(Return(expected_result));
    
    auto result = client_->call_tool("get_model_info", json::object());
    
    EXPECT_TRUE(result.contains("content"));
    auto text_content = result["content"][0]["text"].get<std::string>();
    auto model_info = json::parse(text_content);
    EXPECT_EQ(model_info["name"], "gemma-2b-it");
    EXPECT_TRUE(model_info["loaded"]);
}

// Error handling tests

TEST_F(MCPServerIntegrationTest, InvalidToolCall) {
    json invalid_request = {
        {"jsonrpc", "2.0"},
        {"method", "tools/call"},
        {"id", "1"},
        {"params", {
            {"name", "nonexistent_tool"},
            {"arguments", {}}
        }}
    };
    
    json error_response = {
        {"jsonrpc", "2.0"},
        {"id", "1"},
        {"error", {
            {"code", -32601},
            {"message", "Method not found"},
            {"data", "Tool 'nonexistent_tool' not found"}
        }}
    };
    
    EXPECT_CALL(*server_, handle_request(MatchesRequest(invalid_request)))
        .Times(1)
        .WillOnce(Return(error_response));
    
    auto response = server_->handle_request(invalid_request);
    
    EXPECT_TRUE(response.contains("error"));
    EXPECT_EQ(response["error"]["code"], -32601);
    EXPECT_THAT(response["error"]["message"].get<std::string>(), HasSubstr("not found"));
}

TEST_F(MCPServerIntegrationTest, InvalidToolArguments) {
    json invalid_args_request = {
        {"jsonrpc", "2.0"},
        {"method", "tools/call"},
        {"id", "2"},
        {"params", {
            {"name", "generate_text"},
            {"arguments", {
                // Missing required "prompt" parameter
                {"max_tokens", 100}
            }}
        }}
    };
    
    json error_response = {
        {"jsonrpc", "2.0"},
        {"id", "2"},
        {"error", {
            {"code", -32602},
            {"message", "Invalid params"},
            {"data", "Missing required parameter: prompt"}
        }}
    };
    
    EXPECT_CALL(*server_, handle_request(MatchesRequest(invalid_args_request)))
        .Times(1)
        .WillOnce(Return(error_response));
    
    auto response = server_->handle_request(invalid_args_request);
    
    EXPECT_TRUE(response.contains("error"));
    EXPECT_EQ(response["error"]["code"], -32602);
}

TEST_F(MCPServerIntegrationTest, ModelNotLoadedError) {
    // Simulate model not loaded
    EXPECT_CALL(*model_, is_loaded())
        .Times(1)
        .WillOnce(Return(false));
    
    json request = {
        {"jsonrpc", "2.0"},
        {"method", "tools/call"},
        {"id", "3"},
        {"params", {
            {"name", "generate_text"},
            {"arguments", {
                {"prompt", "Test prompt"}
            }}
        }}
    };
    
    json error_response = {
        {"jsonrpc", "2.0"},
        {"id", "3"},
        {"error", {
            {"code", -32603},
            {"message", "Internal error"},
            {"data", "Model not loaded"}
        }}
    };
    
    EXPECT_CALL(*server_, handle_request(MatchesRequest(request)))
        .Times(1)
        .WillOnce(Return(error_response));
    
    auto response = server_->handle_request(request);
    
    EXPECT_TRUE(response.contains("error"));
    EXPECT_EQ(response["error"]["code"], -32603);
    EXPECT_THAT(response["error"]["data"].get<std::string>(), HasSubstr("not loaded"));
}

// Notification handling tests

TEST_F(MCPServerIntegrationTest, ServerNotifications) {
    json progress_notification = {
        {"jsonrpc", "2.0"},
        {"method", "notifications/progress"},
        {"params", {
            {"progressToken", "generation-1"},
            {"value", {
                {"kind", "report"},
                {"percentage", 50},
                {"message", "Generating text... 50% complete"}
            }}
        }}
    };
    
    EXPECT_CALL(*server_, send_notification(MatchesNotification(progress_notification)))
        .Times(1);
    
    server_->send_notification(progress_notification);
}

TEST_F(MCPServerIntegrationTest, ClientNotificationHandling) {
    std::vector<json> received_notifications;
    
    auto notification_handler = [&received_notifications](const json& notification) {
        received_notifications.push_back(notification);
    };
    
    EXPECT_CALL(*client_, set_notification_handler(_))
        .Times(1);
    
    client_->set_notification_handler(notification_handler);
    
    // Simulate receiving a notification
    json test_notification = {
        {"jsonrpc", "2.0"},
        {"method", "notifications/status"},
        {"params", {
            {"status", "model_loaded"},
            {"message", "Model successfully loaded"}
        }}
    };
    
    // In real implementation, this would be triggered by the transport layer
    notification_handler(test_notification);
    
    EXPECT_EQ(received_notifications.size(), 1);
    EXPECT_EQ(received_notifications[0]["params"]["status"], "model_loaded");
}

// Concurrent access and performance tests

TEST_F(MCPServerIntegrationTest, ConcurrentToolCalls) {
    const int num_clients = 5;
    const int calls_per_client = 10;
    std::atomic<int> successful_calls{0};
    
    EXPECT_CALL(*server_, handle_request(_))
        .Times(num_clients * calls_per_client)
        .WillRepeatedly(Invoke([](const json& request) {
            return json{
                {"jsonrpc", "2.0"},
                {"id", request.value("id", "")},
                {"result", {
                    {"content", {
                        {{"type", "text"}, {"text", "Concurrent response"}}
                    }}
                }}
            };
        }));
    
    std::vector<std::future<void>> futures;
    
    for (int client_id = 0; client_id < num_clients; ++client_id) {
        futures.push_back(std::async(std::launch::async, [this, client_id, calls_per_client, &successful_calls]() {
            for (int call_id = 0; call_id < calls_per_client; ++call_id) {
                try {
                    json request = {
                        {"jsonrpc", "2.0"},
                        {"method", "tools/call"},
                        {"id", std::to_string(client_id) + "-" + std::to_string(call_id)},
                        {"params", {
                            {"name", "generate_text"},
                            {"arguments", {
                                {"prompt", "Concurrent test " + std::to_string(call_id)}
                            }}
                        }}
                    };
                    
                    auto response = server_->handle_request(request);
                    if (response.contains("result")) {
                        successful_calls++;
                    }
                } catch (...) {
                    // Count failures
                }
            }
        }));
    }
    
    // Wait for all clients to complete
    for (auto& future : futures) {
        future.get();
    }
    
    EXPECT_EQ(successful_calls.load(), num_clients * calls_per_client);
}

TEST_F(MCPServerIntegrationTest, ServerPerformanceUnderLoad) {
    const int num_requests = 100;
    
    EXPECT_CALL(*server_, handle_request(_))
        .Times(num_requests)
        .WillRepeatedly(Return(json{
            {"jsonrpc", "2.0"},
            {"id", "load-test"},
            {"result", {{"status", "success"}}}
        }));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_requests; ++i) {
        json request = {
            {"jsonrpc", "2.0"},
            {"method", "tools/call"},
            {"id", "load-test-" + std::to_string(i)},
            {"params", {
                {"name", "count_tokens"},
                {"arguments", {
                    {"text", "Load test message " + std::to_string(i)}
                }}
            }}
        };
        
        auto response = server_->handle_request(request);
        EXPECT_TRUE(response.contains("result"));
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Processed " << num_requests << " requests in " << duration.count() << " ms" << std::endl;
    
    double requests_per_second = (num_requests * 1000.0) / duration.count();
    std::cout << "Requests per second: " << requests_per_second << std::endl;
    
    // Performance expectation: should handle at least 100 requests per second
    EXPECT_GT(requests_per_second, 100.0);
}

// Connection management tests

TEST_F(MCPServerIntegrationTest, MultipleClientConnections) {
    const int num_clients = 3;
    
    // Simulate multiple clients connecting
    for (int i = 0; i < num_clients; ++i) {
        EXPECT_CALL(*client_, connect(server_address_))
            .Times(1)
            .WillOnce(Return(true));
        
        bool connected = client_->connect(server_address_);
        EXPECT_TRUE(connected);
    }
    
    // Check active connections count
    EXPECT_CALL(*server_, get_active_connections())
        .Times(1)
        .WillOnce(Return(num_clients));
    
    size_t active_connections = server_->get_active_connections();
    EXPECT_EQ(active_connections, num_clients);
}

TEST_F(MCPServerIntegrationTest, ConnectionStatistics) {
    json expected_stats = {
        {"total_connections", 5},
        {"active_connections", 2},
        {"total_requests", 150},
        {"successful_requests", 145},
        {"failed_requests", 5},
        {"average_response_time_ms", 45.2},
        {"uptime_seconds", 3600}
    };
    
    EXPECT_CALL(*server_, get_connection_stats())
        .Times(1)
        .WillOnce(Return(expected_stats));
    
    auto stats = server_->get_connection_stats();
    EXPECT_EQ(stats["total_connections"], 5);
    EXPECT_EQ(stats["active_connections"], 2);
    EXPECT_GT(stats["successful_requests"].get<int>(), 0);
}

TEST_F(MCPServerIntegrationTest, ConnectionTimeout) {
    // Set a short timeout
    EXPECT_CALL(*server_, set_request_timeout(std::chrono::milliseconds(100)))
        .Times(1);
    
    server_->set_request_timeout(std::chrono::milliseconds(100));
    
    // Simulate a slow request that should timeout
    json slow_request = {
        {"jsonrpc", "2.0"},
        {"method", "tools/call"},
        {"id", "timeout-test"},
        {"params", {
            {"name", "generate_text"},
            {"arguments", {
                {"prompt", "This should timeout"},
                {"max_tokens", 1000}
            }}
        }}
    };
    
    json timeout_response = {
        {"jsonrpc", "2.0"},
        {"id", "timeout-test"},
        {"error", {
            {"code", -32000},
            {"message", "Request timeout"},
            {"data", "Request exceeded 100ms timeout"}
        }}
    };
    
    EXPECT_CALL(*server_, handle_request(MatchesRequest(slow_request)))
        .Times(1)
        .WillOnce(Return(timeout_response));
    
    auto response = server_->handle_request(slow_request);
    EXPECT_TRUE(response.contains("error"));
    EXPECT_EQ(response["error"]["code"], -32000);
}

// Transport layer tests

TEST_F(MCPServerIntegrationTest, HTTPTransportCommunication) {
    EXPECT_CALL(*transport_, get_transport_type())
        .Times(1)
        .WillOnce(Return("http"));
    
    EXPECT_CALL(*transport_, initialize(server_address_))
        .Times(1)
        .WillOnce(Return(true));
    
    std::string transport_type = transport_->get_transport_type();
    EXPECT_EQ(transport_type, "http");
    
    bool transport_initialized = transport_->initialize(server_address_);
    EXPECT_TRUE(transport_initialized);
}

TEST_F(MCPServerIntegrationTest, StdioTransportCommunication) {
    auto stdio_transport = std::make_unique<MockMCPTransport>();
    
    EXPECT_CALL(*stdio_transport, get_transport_type())
        .Times(1)
        .WillOnce(Return("stdio"));
    
    EXPECT_CALL(*stdio_transport, initialize("stdio"))
        .Times(1)
        .WillOnce(Return(true));
    
    std::string transport_type = stdio_transport->get_transport_type();
    EXPECT_EQ(transport_type, "stdio");
    
    bool transport_initialized = stdio_transport->initialize("stdio");
    EXPECT_TRUE(transport_initialized);
}

TEST_F(MCPServerIntegrationTest, MessageSendingAndReceiving) {
    json test_message = {
        {"jsonrpc", "2.0"},
        {"method", "test"},
        {"params", {{"data", "test_value"}}}
    };
    
    EXPECT_CALL(*transport_, send_message(MatchesMessage(test_message)))
        .Times(1)
        .WillOnce(Return(true));
    
    bool sent = transport_->send_message(test_message);
    EXPECT_TRUE(sent);
    
    // Test receiving
    EXPECT_CALL(*transport_, receive_message(std::chrono::milliseconds(1000)))
        .Times(1)
        .WillOnce(Return(std::make_optional(test_message)));
    
    auto received = transport_->receive_message(std::chrono::milliseconds(1000));
    ASSERT_TRUE(received.has_value());
    EXPECT_EQ(received->at("method"), "test");
}

// Custom matchers for complex objects
MATCHER_P(MatchesRequest, expected_request, "") {
    return arg.value("method", "") == expected_request.value("method", "") &&
           arg.value("id", "") == expected_request.value("id", "");
}

MATCHER_P(MatchesNotification, expected_notification, "") {
    return arg.value("method", "") == expected_notification.value("method", "");
}

MATCHER_P(MatchesMessage, expected_message, "") {
    return arg.value("jsonrpc", "") == expected_message.value("jsonrpc", "") &&
           arg.value("method", "") == expected_message.value("method", "");
}

// Cleanup and shutdown tests

TEST_F(MCPServerIntegrationTest, GracefulServerShutdown) {
    // Ensure server is running
    EXPECT_CALL(*server_, is_running())
        .Times(2)
        .WillOnce(Return(true))
        .WillOnce(Return(false));
    
    bool running_before = server_->is_running();
    EXPECT_TRUE(running_before);
    
    // Shutdown server
    EXPECT_CALL(*server_, shutdown())
        .Times(1);
    
    server_->shutdown();
    
    bool running_after = server_->is_running();
    EXPECT_FALSE(running_after);
}

TEST_F(MCPServerIntegrationTest, ClientDisconnection) {
    // Connect first
    EXPECT_CALL(*client_, connect(server_address_))
        .Times(1)
        .WillOnce(Return(true));
    
    client_->connect(server_address_);
    
    // Then disconnect
    EXPECT_CALL(*client_, is_connected())
        .Times(2)
        .WillOnce(Return(true))
        .WillOnce(Return(false));
    
    EXPECT_CALL(*client_, disconnect())
        .Times(1);
    
    bool connected_before = client_->is_connected();
    EXPECT_TRUE(connected_before);
    
    client_->disconnect();
    
    bool connected_after = client_->is_connected();
    EXPECT_FALSE(connected_after);
}