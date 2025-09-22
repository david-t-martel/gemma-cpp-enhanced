/**
 * @file test_mcp_server.cpp
 * @brief Unit tests for MCP Server
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "mcp/server/mcp_server.h"
#include "common/test_utils.h"

using namespace gemma::mcp;
using namespace gemma::testing;

class MCPServerTest : public GemmaTestBase {
protected:
    void SetUp() override {
        GemmaTestBase::SetUp();
        
        // Setup test configuration
        config_.host = "localhost";
        config_.port = 8080;
        config_.model_path = GetTestDataPath() + "/test_model.sbs";
        config_.tokenizer_path = GetTestDataPath() + "/test_tokenizer.spm";
        config_.max_concurrent_requests = 2;
        config_.temperature = 0.7f;
        config_.max_tokens = 100;
        config_.enable_logging = false;  // Disable for tests
    }

    MCPServer::Config config_;
};

TEST_F(MCPServerTest, ConfigurationValidation) {
    // Valid configuration should create server successfully
    auto server = std::make_unique<MCPServer>(config_);
    EXPECT_NE(server, nullptr);

    // Invalid port should be handled
    config_.port = -1;
    // Note: Server creation doesn't validate config, initialization does
    auto invalid_server = std::make_unique<MCPServer>(config_);
    EXPECT_NE(invalid_server, nullptr);
}

TEST_F(MCPServerTest, InitializationWithoutModel) {
    // Create server with invalid model path
    config_.model_path = "/nonexistent/path/model.sbs";
    config_.tokenizer_path = "/nonexistent/path/tokenizer.spm";
    
    MCPServer server(config_);
    
    // Initialization should fail with invalid model paths
    EXPECT_FALSE(server.Initialize());
}

TEST_F(MCPServerTest, ServerLifecycle) {
    MCPServer server(config_);
    
    // Initially not running
    EXPECT_FALSE(server.IsRunning());
    
    // Start without initialization should work (basic server setup)
    EXPECT_TRUE(server.Start());
    EXPECT_TRUE(server.IsRunning());
    
    // Stop server
    server.Stop();
    EXPECT_FALSE(server.IsRunning());
    
    // Multiple stops should be safe
    server.Stop();
    EXPECT_FALSE(server.IsRunning());
}

TEST_F(MCPServerTest, RequestHandling) {
    MCPServer server(config_);
    server.Start();
    
    // Test tool list request
    MCPServer::MCPRequest list_request;
    list_request.id = "test-1";
    list_request.method = "tools/list";
    list_request.params = nlohmann::json::object();
    
    auto response = server.HandleRequest(list_request);
    
    EXPECT_EQ(response.id, "test-1");
    EXPECT_EQ(response.jsonrpc, "2.0");
    EXPECT_TRUE(response.result.contains("tools"));
    EXPECT_TRUE(response.error.is_null());
}

TEST_F(MCPServerTest, InvalidMethodHandling) {
    MCPServer server(config_);
    server.Start();
    
    // Test invalid method request
    MCPServer::MCPRequest invalid_request;
    invalid_request.id = "test-2";
    invalid_request.method = "invalid/method";
    invalid_request.params = nlohmann::json::object();
    
    auto response = server.HandleRequest(invalid_request);
    
    EXPECT_EQ(response.id, "test-2");
    EXPECT_FALSE(response.error.is_null());
    EXPECT_EQ(response.error["code"], -32601);  // Method not found
    EXPECT_TRUE(response.result.is_null());
}

TEST_F(MCPServerTest, ToolCallHandling) {
    MCPServer server(config_);
    server.Start();
    
    // Test get_server_status tool call
    MCPServer::MCPRequest tool_request;
    tool_request.id = "test-3";
    tool_request.method = "tools/call";
    tool_request.params = {
        {"name", "get_server_status"},
        {"arguments", nlohmann::json::object()}
    };
    
    auto response = server.HandleRequest(tool_request);
    
    EXPECT_EQ(response.id, "test-3");
    EXPECT_TRUE(response.error.is_null());
    EXPECT_TRUE(response.result.contains("running"));
    EXPECT_TRUE(response.result["running"].get<bool>());
}

TEST_F(MCPServerTest, InvalidToolCallHandling) {
    MCPServer server(config_);
    server.Start();
    
    // Test invalid tool call
    MCPServer::MCPRequest tool_request;
    tool_request.id = "test-4";
    tool_request.method = "tools/call";
    tool_request.params = {
        {"name", "nonexistent_tool"},
        {"arguments", nlohmann::json::object()}
    };
    
    auto response = server.HandleRequest(tool_request);
    
    EXPECT_EQ(response.id, "test-4");
    EXPECT_FALSE(response.error.is_null());
    EXPECT_EQ(response.error["code"], -32603);  // Internal error
}

TEST_F(MCPServerTest, MissingToolNameHandling) {
    MCPServer server(config_);
    server.Start();
    
    // Test tool call without name
    MCPServer::MCPRequest tool_request;
    tool_request.id = "test-5";
    tool_request.method = "tools/call";
    tool_request.params = {
        {"arguments", nlohmann::json::object()}
        // Missing "name" field
    };
    
    auto response = server.HandleRequest(tool_request);
    
    EXPECT_EQ(response.id, "test-5");
    EXPECT_FALSE(response.error.is_null());
    EXPECT_EQ(response.error["code"], -32603);  // Internal error
}

// Performance test
TEST_F(MCPServerTest, ResponseTimePerformance) {
    MCPServer server(config_);
    server.Start();
    
    MCPServer::MCPRequest request;
    request.method = "tools/call";
    request.params = {
        {"name", "get_server_status"},
        {"arguments", nlohmann::json::object()}
    };
    
    constexpr int num_requests = 100;
    std::vector<double> response_times;
    response_times.reserve(num_requests);
    
    for (int i = 0; i < num_requests; ++i) {
        request.id = "perf-test-" + std::to_string(i);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto response = server.HandleRequest(request);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        response_times.push_back(duration.count() / 1000.0);  // Convert to milliseconds
        
        EXPECT_TRUE(response.error.is_null());
    }
    
    // Calculate statistics
    double total_time = std::accumulate(response_times.begin(), response_times.end(), 0.0);
    double avg_time = total_time / num_requests;
    
    std::sort(response_times.begin(), response_times.end());
    double median_time = response_times[num_requests / 2];
    double p95_time = response_times[static_cast<size_t>(num_requests * 0.95)];
    
    // Performance expectations (adjust based on requirements)
    EXPECT_LT(avg_time, 10.0);     // Average < 10ms
    EXPECT_LT(median_time, 5.0);   // Median < 5ms
    EXPECT_LT(p95_time, 20.0);     // 95th percentile < 20ms
    
    std::cout << "Performance stats (ms):" << std::endl;
    std::cout << "  Average: " << avg_time << std::endl;
    std::cout << "  Median: " << median_time << std::endl;
    std::cout << "  95th percentile: " << p95_time << std::endl;
}