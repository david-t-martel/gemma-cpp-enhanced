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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <string>
#include <memory>
#include <vector>
#include <nlohmann/json.hpp>

#include "../utils/test_common.h"
#include "gemma/gemma.h"
#include "gemma/configs.h"
#include "util/threading_context.h"

using json = nlohmann::json;

namespace gcpp {
namespace {

using namespace test_utils;

// Mock MCP server interface for testing
class MockMCPServer {
 public:
  struct MCPRequest {
    std::string method;
    json params;
    std::string id;
  };

  struct MCPResponse {
    json result;
    json error;
    std::string id;
    bool success = true;
  };

  virtual ~MockMCPServer() = default;
  virtual MCPResponse HandleRequest(const MCPRequest& request) = 0;
  virtual bool IsConnected() const = 0;
  virtual void Connect() = 0;
  virtual void Disconnect() = 0;
};

// Mock Gemma inference server implementing MCP protocol
class MockGemmaMCPServer : public MockMCPServer {
 public:
  explicit MockGemmaMCPServer(std::shared_ptr<Gemma> model)
      : model_(model), connected_(false) {}

  MCPResponse HandleRequest(const MCPRequest& request) override {
    MCPResponse response;
    response.id = request.id;

    if (!connected_) {
      response.success = false;
      response.error = json{{"code", -1}, {"message", "Server not connected"}};
      return response;
    }

    try {
      if (request.method == "initialize") {
        return HandleInitialize(request);
      } else if (request.method == "generate") {
        return HandleGenerate(request);
      } else if (request.method == "tokenize") {
        return HandleTokenize(request);
      } else if (request.method == "capabilities") {
        return HandleCapabilities(request);
      } else if (request.method == "model_info") {
        return HandleModelInfo(request);
      } else {
        response.success = false;
        response.error = json{{"code", -32601}, {"message", "Method not found"}};
      }
    } catch (const std::exception& e) {
      response.success = false;
      response.error = json{{"code", -1}, {"message", e.what()}};
    }

    return response;
  }

  bool IsConnected() const override {
    return connected_;
  }

  void Connect() override {
    connected_ = true;
  }

  void Disconnect() override {
    connected_ = false;
  }

 private:
  MCPResponse HandleInitialize(const MCPRequest& request) {
    MCPResponse response;
    response.id = request.id;
    response.result = json{
        {"protocolVersion", "1.0"},
        {"serverInfo", json{
            {"name", "gemma-mcp-server"},
            {"version", "1.0.0"}
        }},
        {"capabilities", json{
            {"generation", true},
            {"tokenization", true},
            {"model_info", true}
        }}
    };
    return response;
  }

  MCPResponse HandleGenerate(const MCPRequest& request) {
    MCPResponse response;
    response.id = request.id;

    if (!request.params.contains("prompt")) {
      response.success = false;
      response.error = json{{"code", -32602}, {"message", "Missing prompt parameter"}};
      return response;
    }

    std::string prompt = request.params["prompt"];
    int max_tokens = request.params.value("max_tokens", 100);
    float temperature = request.params.value("temperature", 1.0f);

    // Mock generation response
    response.result = json{
        {"text", "Generated response for: " + prompt},
        {"tokens_used", max_tokens / 2},
        {"finish_reason", "length"},
        {"model", model_ ? "gemma-2b" : "mock"}
    };

    return response;
  }

  MCPResponse HandleTokenize(const MCPRequest& request) {
    MCPResponse response;
    response.id = request.id;

    if (!request.params.contains("text")) {
      response.success = false;
      response.error = json{{"code", -32602}, {"message", "Missing text parameter"}};
      return response;
    }

    std::string text = request.params["text"];

    // Mock tokenization
    std::vector<int> tokens;
    for (size_t i = 0; i < text.length(); i += 4) {  // Rough approximation
      tokens.push_back(static_cast<int>(i / 4));
    }

    response.result = json{
        {"tokens", tokens},
        {"count", tokens.size()}
    };

    return response;
  }

  MCPResponse HandleCapabilities(const MCPRequest& request) {
    MCPResponse response;
    response.id = request.id;
    response.result = json{
        {"generation", json{
            {"max_tokens", 2048},
            {"temperature_range", json{{"min", 0.0}, {"max", 2.0}}},
            {"supports_streaming", false}
        }},
        {"tokenization", json{
            {"vocab_size", 256000},
            {"supports_decode", false}
        }},
        {"model", json{
            {"name", "Gemma-2B"},
            {"version", "1.0"},
            {"parameters", "2B"}
        }}
    };
    return response;
  }

  MCPResponse HandleModelInfo(const MCPRequest& request) {
    MCPResponse response;
    response.id = request.id;

    if (model_) {
      const auto& config = model_->Config();
      response.result = json{
          {"model_name", config.Specifier()},
          {"vocab_size", config.vocab_size},
          {"sequence_length", config.seq_len},
          {"num_layers", config.num_layers},
          {"num_heads", config.num_heads},
          {"model_dim", config.model_dim}
      };
    } else {
      response.result = json{
          {"model_name", "mock-model"},
          {"vocab_size", 256000},
          {"sequence_length", 2048},
          {"num_layers", 18},
          {"num_heads", 8},
          {"model_dim", 2048}
      };
    }

    return response;
  }

  std::shared_ptr<Gemma> model_;
  bool connected_;
};

// Test fixture for MCP server tests
class MCPServerTest : public GemmaTestBase {
 protected:
  void SetUp() override {
    GemmaTestBase::SetUp();
    // Create mock MCP server without actual model for faster testing
    mcp_server_ = std::make_unique<MockGemmaMCPServer>(nullptr);
  }

  void TearDown() override {
    if (mcp_server_) {
      mcp_server_->Disconnect();
    }
    GemmaTestBase::TearDown();
  }

  std::unique_ptr<MockGemmaMCPServer> mcp_server_;
};

// Test MCP server initialization
TEST_F(MCPServerTest, ServerInitialization) {
  EXPECT_FALSE(mcp_server_->IsConnected());

  mcp_server_->Connect();
  EXPECT_TRUE(mcp_server_->IsConnected());

  MockMCPServer::MCPRequest init_request;
  init_request.method = "initialize";
  init_request.id = "test-1";
  init_request.params = json{{"protocolVersion", "1.0"}};

  auto response = mcp_server_->HandleRequest(init_request);

  EXPECT_TRUE(response.success);
  EXPECT_EQ(response.id, "test-1");
  EXPECT_TRUE(response.result.contains("protocolVersion"));
  EXPECT_TRUE(response.result.contains("serverInfo"));
  EXPECT_TRUE(response.result.contains("capabilities"));
}

// Test text generation endpoint
TEST_F(MCPServerTest, TextGeneration) {
  mcp_server_->Connect();

  MockMCPServer::MCPRequest gen_request;
  gen_request.method = "generate";
  gen_request.id = "gen-1";
  gen_request.params = json{
      {"prompt", "What is the capital of France?"},
      {"max_tokens", 50},
      {"temperature", 0.7}
  };

  auto response = mcp_server_->HandleRequest(gen_request);

  EXPECT_TRUE(response.success);
  EXPECT_EQ(response.id, "gen-1");
  EXPECT_TRUE(response.result.contains("text"));
  EXPECT_TRUE(response.result.contains("tokens_used"));
  EXPECT_TRUE(response.result.contains("finish_reason"));

  std::string generated_text = response.result["text"];
  EXPECT_FALSE(generated_text.empty());
}

// Test generation with missing parameters
TEST_F(MCPServerTest, GenerationMissingPrompt) {
  mcp_server_->Connect();

  MockMCPServer::MCPRequest gen_request;
  gen_request.method = "generate";
  gen_request.id = "gen-2";
  gen_request.params = json{{"max_tokens", 50}};  // Missing prompt

  auto response = mcp_server_->HandleRequest(gen_request);

  EXPECT_FALSE(response.success);
  EXPECT_TRUE(response.error.contains("code"));
  EXPECT_TRUE(response.error.contains("message"));
  EXPECT_EQ(response.error["code"], -32602);  // Invalid params
}

// Test tokenization endpoint
TEST_F(MCPServerTest, Tokenization) {
  mcp_server_->Connect();

  MockMCPServer::MCPRequest tok_request;
  tok_request.method = "tokenize";
  tok_request.id = "tok-1";
  tok_request.params = json{{"text", "Hello, world! How are you today?"}};

  auto response = mcp_server_->HandleRequest(tok_request);

  EXPECT_TRUE(response.success);
  EXPECT_EQ(response.id, "tok-1");
  EXPECT_TRUE(response.result.contains("tokens"));
  EXPECT_TRUE(response.result.contains("count"));

  std::vector<int> tokens = response.result["tokens"];
  int count = response.result["count"];
  EXPECT_EQ(tokens.size(), count);
  EXPECT_GT(count, 0);
}

// Test capabilities endpoint
TEST_F(MCPServerTest, Capabilities) {
  mcp_server_->Connect();

  MockMCPServer::MCPRequest cap_request;
  cap_request.method = "capabilities";
  cap_request.id = "cap-1";
  cap_request.params = json::object();

  auto response = mcp_server_->HandleRequest(cap_request);

  EXPECT_TRUE(response.success);
  EXPECT_TRUE(response.result.contains("generation"));
  EXPECT_TRUE(response.result.contains("tokenization"));
  EXPECT_TRUE(response.result.contains("model"));

  auto generation_caps = response.result["generation"];
  EXPECT_TRUE(generation_caps.contains("max_tokens"));
  EXPECT_TRUE(generation_caps.contains("temperature_range"));
}

// Test model info endpoint
TEST_F(MCPServerTest, ModelInfo) {
  mcp_server_->Connect();

  MockMCPServer::MCPRequest info_request;
  info_request.method = "model_info";
  info_request.id = "info-1";
  info_request.params = json::object();

  auto response = mcp_server_->HandleRequest(info_request);

  EXPECT_TRUE(response.success);
  EXPECT_TRUE(response.result.contains("model_name"));
  EXPECT_TRUE(response.result.contains("vocab_size"));
  EXPECT_TRUE(response.result.contains("sequence_length"));
  EXPECT_TRUE(response.result.contains("num_layers"));
  EXPECT_TRUE(response.result.contains("num_heads"));
  EXPECT_TRUE(response.result.contains("model_dim"));
}

// Test unknown method handling
TEST_F(MCPServerTest, UnknownMethod) {
  mcp_server_->Connect();

  MockMCPServer::MCPRequest unknown_request;
  unknown_request.method = "unknown_method";
  unknown_request.id = "unknown-1";
  unknown_request.params = json::object();

  auto response = mcp_server_->HandleRequest(unknown_request);

  EXPECT_FALSE(response.success);
  EXPECT_TRUE(response.error.contains("code"));
  EXPECT_EQ(response.error["code"], -32601);  // Method not found
}

// Test server disconnection handling
TEST_F(MCPServerTest, DisconnectedServer) {
  // Don't connect the server
  EXPECT_FALSE(mcp_server_->IsConnected());

  MockMCPServer::MCPRequest request;
  request.method = "generate";
  request.id = "disconnected-1";
  request.params = json{{"prompt", "test"}};

  auto response = mcp_server_->HandleRequest(request);

  EXPECT_FALSE(response.success);
  EXPECT_TRUE(response.error.contains("message"));
  EXPECT_THAT(response.error["message"], ::testing::HasSubstr("not connected"));
}

// Test concurrent request handling (stress test)
TEST_F(MCPServerTest, ConcurrentRequests) {
  mcp_server_->Connect();

  const int num_requests = 10;
  std::vector<std::future<MockMCPServer::MCPResponse>> futures;

  // Launch concurrent requests
  for (int i = 0; i < num_requests; ++i) {
    auto future = std::async(std::launch::async, [this, i]() {
      MockMCPServer::MCPRequest request;
      request.method = "model_info";
      request.id = "concurrent-" + std::to_string(i);
      request.params = json::object();
      return mcp_server_->HandleRequest(request);
    });
    futures.push_back(std::move(future));
  }

  // Collect results
  for (auto& future : futures) {
    auto response = future.get();
    EXPECT_TRUE(response.success);
    EXPECT_TRUE(response.result.contains("model_name"));
  }
}

// Test parameter validation
class MCPParameterValidationTest : public MCPServerTest {
 protected:
  void TestInvalidParameter(const std::string& method, const json& params, int expected_error_code) {
    mcp_server_->Connect();

    MockMCPServer::MCPRequest request;
    request.method = method;
    request.id = "validation-test";
    request.params = params;

    auto response = mcp_server_->HandleRequest(request);

    EXPECT_FALSE(response.success);
    EXPECT_TRUE(response.error.contains("code"));
    EXPECT_EQ(response.error["code"], expected_error_code);
  }
};

TEST_F(MCPParameterValidationTest, GenerationParameters) {
  // Test various invalid parameter combinations
  TestInvalidParameter("generate", json{}, -32602);  // Missing prompt
  TestInvalidParameter("tokenize", json{}, -32602);  // Missing text
}

// Performance tests
class MCPPerformanceTest : public MCPServerTest {
 protected:
  void BenchmarkMethod(const std::string& method, const json& params, int iterations = 1000) {
    mcp_server_->Connect();

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
      MockMCPServer::MCPRequest request;
      request.method = method;
      request.id = "perf-" + std::to_string(i);
      request.params = params;

      auto response = mcp_server_->HandleRequest(request);
      EXPECT_TRUE(response.success);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    double requests_per_second = (iterations * 1000.0) / duration.count();
    std::cout << method << " performance: " << requests_per_second << " requests/second" << std::endl;

    // Basic performance expectations (adjust based on requirements)
    EXPECT_GT(requests_per_second, 100.0) << "Method " << method << " too slow";
  }
};

TEST_F(MCPPerformanceTest, ModelInfoPerformance) {
  BenchmarkMethod("model_info", json::object());
}

TEST_F(MCPPerformanceTest, TokenizationPerformance) {
  BenchmarkMethod("tokenize", json{{"text", "Short test text"}});
}

}  // namespace
}  // namespace gcpp