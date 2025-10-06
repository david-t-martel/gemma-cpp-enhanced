// End-to-end integration test (progressively decomposed – remaining complex & multi-stage flows)
// NOTE: Simple workflows (model inference, backend switching, MCP server basics) have been
// extracted into dedicated *workflow* tests for Single Responsibility clarity.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <chrono>
#include <thread>
#include <future>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <iostream>

#include "mock_system.h"            // SystemConfig / MockGemmaSystem
#include "../utils/test_helpers.h"  // GemmaTestBase fixture utilities
// #include "../utils/mock_backend.h" // (Unused in remaining sections – kept commented for reference)

using namespace gemma::test;  // NOLINT(build/namespaces) test convenience
using namespace testing;      // gMock matchers
using json = nlohmann::json;

class EndToEndTest : public GemmaTestBase {
protected:
    void SetUp() override {
        GemmaTestBase::SetUp();
        
        system_ = std::make_unique<MockGemmaSystem>();
        setup_test_config();
        setup_test_files();
        setup_default_expectations();
    }
    
    void setup_test_config() {
        config_.model_weights_path = (test_dir_ / "test_model.sbs").string();
        config_.tokenizer_path = (test_dir_ / "tokenizer.spm").string();
        config_.backend_name = "cpu";
        config_.session_storage_path = (test_dir_ / "sessions").string();
        config_.enable_mcp_server = true;
        config_.mcp_server_port = 8081; // Use non-standard port for testing
        config_.max_context_tokens = 4096; // Smaller for testing
        config_.enable_metrics = true;
    }
    
    void setup_test_files() {
        // Create mock model files
        std::filesystem::create_directories(test_dir_);
        
        // Create dummy model weights file
        std::ofstream weights_file(config_.model_weights_path, std::ios::binary);
        weights_file << "dummy_weights_data_for_testing";
        weights_file.close();
        
        // Create dummy tokenizer file
        std::ofstream tokenizer_file(config_.tokenizer_path, std::ios::binary);
        tokenizer_file << "dummy_tokenizer_data_for_testing";
        tokenizer_file.close();
        
        // Create session storage directory
        std::filesystem::create_directories(config_.session_storage_path);
    }
    
    void setup_default_expectations() {
        // System initialization
        ON_CALL(*system_, initialize(_)).WillByDefault(Return(true));
        ON_CALL(*system_, is_initialized()).WillByDefault(Return(true));
        
        // Model operations
        ON_CALL(*system_, load_model(_, _)).WillByDefault(Return(true));
        ON_CALL(*system_, is_model_loaded()).WillByDefault(Return(true));
        ON_CALL(*system_, get_model_info()).WillByDefault(Return(json{
            {"name", "test-model"},
            {"size", "2B"},
            {"context_length", 4096},
            {"vocab_size", 32000},
            {"loaded", true}
        }));
        
        // Text generation
        ON_CALL(*system_, generate_text(_, _)).WillByDefault(Return("Generated response"));
        ON_CALL(*system_, count_tokens(_)).WillByDefault(Return(10));
        
        // Backend operations
        ON_CALL(*system_, get_current_backend()).WillByDefault(Return("cpu"));
        ON_CALL(*system_, list_available_backends()).WillByDefault(Return(std::vector<std::string>{"cpu", "intel", "cuda"}));
        
        // Session operations
        ON_CALL(*system_, create_session(_)).WillByDefault(Return("test-session-id"));
        ON_CALL(*system_, list_sessions()).WillByDefault(Return(std::vector<std::string>{}));
        
        // MCP server
        ON_CALL(*system_, start_mcp_server(_)).WillByDefault(Return(true));
        ON_CALL(*system_, is_mcp_server_running()).WillByDefault(Return(true));
    }
    
    std::unique_ptr<MockGemmaSystem> system_;
    SystemConfig config_;
};

// (Lifecycle tests moved to test_system_lifecycle.cpp)

// (Model loading/inference, backend switching, and MCP server basic workflows extracted
//  into dedicated test_*_workflow.cpp files.)

// (Batch, error recovery, performance stress, metrics/monitoring, and configuration workflows
//  extracted into dedicated workflow test files.)

// (Config matcher now in mock_system.h)

// Full integration test combining all components

TEST_F(EndToEndTest, CompleteIntegrationWorkflow) {
    // This test exercises the entire system pipeline
    
    // Phase 1: System startup
    EXPECT_CALL(*system_, initialize(_)).WillOnce(Return(true));
    EXPECT_CALL(*system_, start_mcp_server(_)).WillOnce(Return(true));
    
    system_->initialize(config_);
    system_->start_mcp_server(config_.mcp_server_port);
    
    // Phase 2: Model operations
    EXPECT_CALL(*system_, load_model(_, _)).WillOnce(Return(true));
    EXPECT_CALL(*system_, get_model_info()).WillOnce(Return(json{{"loaded", true}}));
    
    system_->load_model(config_.model_weights_path, config_.tokenizer_path);
    auto info = system_->get_model_info();
    EXPECT_TRUE(info["loaded"]);
    
    // Phase 3: Session management
    EXPECT_CALL(*system_, create_session(_)).WillOnce(Return("session-1"));
    EXPECT_CALL(*system_, add_message_to_session(_, _, _)).Times(2).WillRepeatedly(Return(true));
    
    auto session_id = system_->create_session(json{});
    system_->add_message_to_session(session_id, "user", "Hello");
    system_->add_message_to_session(session_id, "assistant", "Hi there!");
    
    // Phase 4: Text generation and processing
    EXPECT_CALL(*system_, generate_text(_, _)).WillOnce(Return("Generated response"));
    EXPECT_CALL(*system_, count_tokens(_)).WillOnce(Return(15));
    
    auto response = system_->generate_text("Test prompt", json{});
    auto token_count = system_->count_tokens(response);
    EXPECT_EQ(token_count, 15);
    
    // Phase 5: Backend switching
    EXPECT_CALL(*system_, set_backend("intel")).WillOnce(Return(true));
    EXPECT_CALL(*system_, get_current_backend()).WillOnce(Return("intel"));
    
    system_->set_backend("intel");
    auto backend = system_->get_current_backend();
    EXPECT_EQ(backend, "intel");
    
    // Phase 6: MCP operations
    json mcp_request = {{"method", "generate_text"}, {"params", {{"prompt", "MCP test"}}}};
    json mcp_response = {{"result", "MCP response"}};
    
    EXPECT_CALL(*system_, handle_mcp_request(_)).WillOnce(Return(mcp_response));
    
    auto mcp_result = system_->handle_mcp_request(mcp_request);
    EXPECT_TRUE(mcp_result.contains("result"));
    
    // Phase 7: Cleanup
    EXPECT_CALL(*system_, delete_session(_)).WillOnce(Return(true));
    EXPECT_CALL(*system_, stop_mcp_server()).WillOnce(Return(true));
    EXPECT_CALL(*system_, unload_model()).WillOnce(Return(true));
    EXPECT_CALL(*system_, shutdown()).Times(1);
    
    system_->delete_session(session_id);
    system_->stop_mcp_server();
    system_->unload_model();
    system_->shutdown();
}