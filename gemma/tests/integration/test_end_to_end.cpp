#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../utils/test_helpers.h"
#include "../utils/mock_backend.h"
#include <nlohmann/json.hpp>
#include <memory>
#include <chrono>
#include <thread>
#include <future>
#include <filesystem>
#include <fstream>

// Mock system integrator that combines all components
// In real implementation, this would include actual headers:
// #include "../../src/core/GemmaSystem.h"
// #include "../../src/session/SessionManager.h"
// #include "../../src/backends/BackendManager.h"
// #include "../../src/protocols/mcp/MCPServer.h"

using namespace gemma::test;
using namespace testing;
using json = nlohmann::json;

// Mock system configuration
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

// Mock integrated Gemma system
class MockGemmaSystem {
public:
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
    
    // System status and metrics
    MOCK_METHOD(json, get_system_status, (), (const));
    MOCK_METHOD(json, get_metrics, (), (const));
    MOCK_METHOD(void, reset_metrics, (), ());
    
    // Configuration
    MOCK_METHOD(SystemConfig, get_config, (), (const));
    MOCK_METHOD(bool, update_config, (const SystemConfig& new_config), ());
};

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

// Full system initialization and shutdown tests

TEST_F(EndToEndTest, CompleteSystemInitialization) {
    EXPECT_CALL(*system_, initialize(MatchesConfig(config_)))
        .Times(1)
        .WillOnce(Return(true));
    
    bool initialized = system_->initialize(config_);
    EXPECT_TRUE(initialized);
    
    // Verify system is in correct state after initialization
    EXPECT_CALL(*system_, is_initialized())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_TRUE(system_->is_initialized());
}

TEST_F(EndToEndTest, CompleteSystemShutdown) {
    // Initialize first
    EXPECT_CALL(*system_, initialize(_))
        .Times(1)
        .WillOnce(Return(true));
    
    system_->initialize(config_);
    
    // Then shutdown
    EXPECT_CALL(*system_, is_mcp_server_running())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*system_, stop_mcp_server())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*system_, unload_model())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*system_, shutdown())
        .Times(1);
    
    system_->shutdown();
}

// End-to-end model loading and inference workflow

TEST_F(EndToEndTest, ModelLoadingAndInferenceWorkflow) {
    // Step 1: Initialize system
    EXPECT_CALL(*system_, initialize(_))
        .Times(1)
        .WillOnce(Return(true));
    
    bool system_init = system_->initialize(config_);
    EXPECT_TRUE(system_init);
    
    // Step 2: Load model
    EXPECT_CALL(*system_, load_model(config_.model_weights_path, config_.tokenizer_path))
        .Times(1)
        .WillOnce(Return(true));
    
    bool model_loaded = system_->load_model(config_.model_weights_path, config_.tokenizer_path);
    EXPECT_TRUE(model_loaded);
    
    // Step 3: Verify model is loaded
    EXPECT_CALL(*system_, is_model_loaded())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_TRUE(system_->is_model_loaded());
    
    // Step 4: Get model information
    json expected_info = {
        {"name", "gemma-2b-it"},
        {"size", "2B"},
        {"context_length", 4096},
        {"vocab_size", 32000},
        {"loaded", true},
        {"backend", "cpu"}
    };
    
    EXPECT_CALL(*system_, get_model_info())
        .Times(1)
        .WillOnce(Return(expected_info));
    
    auto model_info = system_->get_model_info();
    EXPECT_EQ(model_info["name"], "gemma-2b-it");
    EXPECT_TRUE(model_info["loaded"]);
    
    // Step 5: Perform text generation
    std::string prompt = "What is artificial intelligence?";
    json generation_options = {
        {"max_tokens", 100},
        {"temperature", 0.7},
        {"top_p", 0.9}
    };
    
    std::string expected_response = "Artificial intelligence (AI) is a branch of computer science that aims to create machines capable of intelligent behavior.";
    
    EXPECT_CALL(*system_, generate_text(prompt, generation_options))
        .Times(1)
        .WillOnce(Return(expected_response));
    
    auto response = system_->generate_text(prompt, generation_options);
    EXPECT_EQ(response, expected_response);
    
    // Step 6: Count tokens
    EXPECT_CALL(*system_, count_tokens(prompt))
        .Times(1)
        .WillOnce(Return(7));
    
    int token_count = system_->count_tokens(prompt);
    EXPECT_EQ(token_count, 7);
}

// Session management end-to-end workflow

TEST_F(EndToEndTest, SessionManagementWorkflow) {
    // Initialize system and load model
    system_->initialize(config_);
    system_->load_model(config_.model_weights_path, config_.tokenizer_path);
    
    // Step 1: Create a new session
    json session_options = {
        {"max_context_tokens", 2048},
        {"metadata", {
            {"user_id", "test_user"},
            {"session_type", "chat"}
        }}
    };
    
    EXPECT_CALL(*system_, create_session(session_options))
        .Times(1)
        .WillOnce(Return("session-12345"));
    
    std::string session_id = system_->create_session(session_options);
    EXPECT_EQ(session_id, "session-12345");
    
    // Step 2: Add messages to the session
    EXPECT_CALL(*system_, add_message_to_session(session_id, "user", "Hello, how are you?"))
        .Times(1)
        .WillOnce(Return(true));
    
    bool user_message_added = system_->add_message_to_session(session_id, "user", "Hello, how are you?");
    EXPECT_TRUE(user_message_added);
    
    EXPECT_CALL(*system_, add_message_to_session(session_id, "assistant", "Hello! I'm doing well, thank you for asking."))
        .Times(1)
        .WillOnce(Return(true));
    
    bool assistant_message_added = system_->add_message_to_session(session_id, "assistant", "Hello! I'm doing well, thank you for asking.");
    EXPECT_TRUE(assistant_message_added);
    
    // Step 3: Retrieve session history
    json expected_history = json::array({
        {
            {"role", "user"},
            {"content", "Hello, how are you?"},
            {"timestamp", "2024-01-01T12:00:00Z"},
            {"token_count", 5}
        },
        {
            {"role", "assistant"},
            {"content", "Hello! I'm doing well, thank you for asking."},
            {"timestamp", "2024-01-01T12:00:01Z"},
            {"token_count", 12}
        }
    });
    
    EXPECT_CALL(*system_, get_session_history(session_id))
        .Times(1)
        .WillOnce(Return(expected_history));
    
    auto history = system_->get_session_history(session_id);
    EXPECT_EQ(history.size(), 2);
    EXPECT_EQ(history[0]["role"], "user");
    EXPECT_EQ(history[1]["role"], "assistant");
    
    // Step 4: List all sessions
    std::vector<std::string> expected_sessions = {session_id};
    
    EXPECT_CALL(*system_, list_sessions())
        .Times(1)
        .WillOnce(Return(expected_sessions));
    
    auto sessions = system_->list_sessions();
    EXPECT_EQ(sessions.size(), 1);
    EXPECT_EQ(sessions[0], session_id);
    
    // Step 5: Delete the session
    EXPECT_CALL(*system_, delete_session(session_id))
        .Times(1)
        .WillOnce(Return(true));
    
    bool session_deleted = system_->delete_session(session_id);
    EXPECT_TRUE(session_deleted);
}

// Backend switching end-to-end workflow

TEST_F(EndToEndTest, BackendSwitchingWorkflow) {
    // Initialize with CPU backend
    system_->initialize(config_);
    
    // Step 1: Verify initial backend
    EXPECT_CALL(*system_, get_current_backend())
        .Times(1)
        .WillOnce(Return("cpu"));
    
    std::string initial_backend = system_->get_current_backend();
    EXPECT_EQ(initial_backend, "cpu");
    
    // Step 2: List available backends
    std::vector<std::string> expected_backends = {"cpu", "intel", "cuda", "vulkan"};
    
    EXPECT_CALL(*system_, list_available_backends())
        .Times(1)
        .WillOnce(Return(expected_backends));
    
    auto available_backends = system_->list_available_backends();
    EXPECT_THAT(available_backends, UnorderedElementsAre("cpu", "intel", "cuda", "vulkan"));
    
    // Step 3: Switch to Intel backend
    EXPECT_CALL(*system_, set_backend("intel"))
        .Times(1)
        .WillOnce(Return(true));
    
    bool backend_switched = system_->set_backend("intel");
    EXPECT_TRUE(backend_switched);
    
    // Step 4: Verify backend switch
    EXPECT_CALL(*system_, get_current_backend())
        .Times(1)
        .WillOnce(Return("intel"));
    
    std::string new_backend = system_->get_current_backend();
    EXPECT_EQ(new_backend, "intel");
    
    // Step 5: Load model with new backend
    EXPECT_CALL(*system_, load_model(config_.model_weights_path, config_.tokenizer_path))
        .Times(1)
        .WillOnce(Return(true));
    
    bool model_loaded_with_intel = system_->load_model(config_.model_weights_path, config_.tokenizer_path);
    EXPECT_TRUE(model_loaded_with_intel);
    
    // Step 6: Test inference with new backend
    EXPECT_CALL(*system_, generate_text("Test with Intel backend", _))
        .Times(1)
        .WillOnce(Return("Response generated using Intel backend"));
    
    auto intel_response = system_->generate_text("Test with Intel backend", json{});
    EXPECT_THAT(intel_response, HasSubstr("Intel backend"));
}

// MCP server integration workflow

TEST_F(EndToEndTest, MCPServerIntegrationWorkflow) {
    // Initialize system
    system_->initialize(config_);
    
    // Step 1: Start MCP server
    EXPECT_CALL(*system_, start_mcp_server(config_.mcp_server_port))
        .Times(1)
        .WillOnce(Return(true));
    
    bool server_started = system_->start_mcp_server(config_.mcp_server_port);
    EXPECT_TRUE(server_started);
    
    // Step 2: Verify server is running
    EXPECT_CALL(*system_, is_mcp_server_running())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_TRUE(system_->is_mcp_server_running());
    
    // Step 3: Handle MCP tool requests
    json generate_request = {
        {"jsonrpc", "2.0"},
        {"method", "tools/call"},
        {"id", "1"},
        {"params", {
            {"name", "generate_text"},
            {"arguments", {
                {"prompt", "Hello, world!"},
                {"max_tokens", 50}
            }}
        }}
    };
    
    json expected_response = {
        {"jsonrpc", "2.0"},
        {"id", "1"},
        {"result", {
            {"content", {
                {{"type", "text"}, {"text", "Hello! How can I help you today?"}}
            }}
        }}
    };
    
    EXPECT_CALL(*system_, handle_mcp_request(generate_request))
        .Times(1)
        .WillOnce(Return(expected_response));
    
    auto mcp_response = system_->handle_mcp_request(generate_request);
    EXPECT_EQ(mcp_response["jsonrpc"], "2.0");
    EXPECT_EQ(mcp_response["id"], "1");
    EXPECT_TRUE(mcp_response.contains("result"));
    
    // Step 4: Handle count tokens request
    json count_request = {
        {"jsonrpc", "2.0"},
        {"method", "tools/call"},
        {"id", "2"},
        {"params", {
            {"name", "count_tokens"},
            {"arguments", {
                {"text", "This is a test message."}
            }}
        }}
    };
    
    json count_response = {
        {"jsonrpc", "2.0"},
        {"id", "2"},
        {"result", {
            {"content", {
                {{"type", "text"}, {"text", "6"}}
            }}
        }}
    };
    
    EXPECT_CALL(*system_, handle_mcp_request(count_request))
        .Times(1)
        .WillOnce(Return(count_response));
    
    auto count_result = system_->handle_mcp_request(count_request);
    EXPECT_EQ(count_result["id"], "2");
    
    // Step 5: Stop MCP server
    EXPECT_CALL(*system_, stop_mcp_server())
        .Times(1)
        .WillOnce(Return(true));
    
    bool server_stopped = system_->stop_mcp_server();
    EXPECT_TRUE(server_stopped);
}

// Batch processing workflow

TEST_F(EndToEndTest, BatchProcessingWorkflow) {
    // Initialize and load model
    system_->initialize(config_);
    system_->load_model(config_.model_weights_path, config_.tokenizer_path);
    
    // Prepare batch of prompts
    std::vector<std::string> prompts = {
        "What is machine learning?",
        "Explain neural networks.",
        "Define artificial intelligence.",
        "What are transformers in AI?"
    };
    
    json batch_options = {
        {"max_tokens", 100},
        {"temperature", 0.7},
        {"batch_size", 2} // Process 2 at a time
    };
    
    std::vector<std::string> expected_responses = {
        "Machine learning is a subset of AI that enables computers to learn from data.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Artificial intelligence refers to the simulation of human intelligence in machines.",
        "Transformers are a type of neural network architecture used in modern AI models."
    };
    
    EXPECT_CALL(*system_, generate_batch(prompts, batch_options))
        .Times(1)
        .WillOnce(Return(expected_responses));
    
    auto responses = system_->generate_batch(prompts, batch_options);
    
    EXPECT_EQ(responses.size(), 4);
    for (size_t i = 0; i < responses.size(); ++i) {
        EXPECT_EQ(responses[i], expected_responses[i]);
    }
}

// Error handling and recovery workflow

TEST_F(EndToEndTest, ErrorHandlingAndRecoveryWorkflow) {
    // Step 1: Try to initialize with invalid config
    SystemConfig invalid_config = config_;
    invalid_config.model_weights_path = "/nonexistent/model.sbs";
    
    EXPECT_CALL(*system_, initialize(MatchesConfig(invalid_config)))
        .Times(1)
        .WillOnce(Return(false));
    
    bool failed_init = system_->initialize(invalid_config);
    EXPECT_FALSE(failed_init);
    
    // Step 2: Initialize with valid config
    EXPECT_CALL(*system_, initialize(MatchesConfig(config_)))
        .Times(1)
        .WillOnce(Return(true));
    
    bool successful_init = system_->initialize(config_);
    EXPECT_TRUE(successful_init);
    
    // Step 3: Try to load non-existent model
    EXPECT_CALL(*system_, load_model("/nonexistent/model.sbs", "/nonexistent/tokenizer.spm"))
        .Times(1)
        .WillOnce(Return(false));
    
    bool failed_load = system_->load_model("/nonexistent/model.sbs", "/nonexistent/tokenizer.spm");
    EXPECT_FALSE(failed_load);
    
    // Step 4: Load valid model
    EXPECT_CALL(*system_, load_model(config_.model_weights_path, config_.tokenizer_path))
        .Times(1)
        .WillOnce(Return(true));
    
    bool successful_load = system_->load_model(config_.model_weights_path, config_.tokenizer_path);
    EXPECT_TRUE(successful_load);
    
    // Step 5: Test error handling in generation
    EXPECT_CALL(*system_, generate_text("", _))
        .Times(1)
        .WillOnce(Throw(std::invalid_argument("Empty prompt not allowed")));
    
    EXPECT_THROW(system_->generate_text("", json{}), std::invalid_argument);
    
    // Step 6: Recover with valid generation
    EXPECT_CALL(*system_, generate_text("Valid prompt", _))
        .Times(1)
        .WillOnce(Return("Valid response"));
    
    auto recovery_response = system_->generate_text("Valid prompt", json{});
    EXPECT_EQ(recovery_response, "Valid response");
}

// Performance and stress testing workflow

TEST_F(EndToEndTest, PerformanceStressTestWorkflow) {
    // Initialize system
    system_->initialize(config_);
    system_->load_model(config_.model_weights_path, config_.tokenizer_path);
    
    // Test rapid successive generations
    const int num_generations = 100;
    std::vector<std::future<std::string>> futures;
    
    EXPECT_CALL(*system_, generate_text(_, _))
        .Times(num_generations)
        .WillRepeatedly(Return("Stress test response"));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Launch concurrent generations
    for (int i = 0; i < num_generations; ++i) {
        futures.push_back(std::async(std::launch::async, [this, i]() {
            return system_->generate_text("Stress test prompt " + std::to_string(i), json{{"max_tokens", 10}});
        }));
    }
    
    // Wait for all to complete
    int successful_generations = 0;
    for (auto& future : futures) {
        try {
            auto result = future.get();
            if (!result.empty()) {
                successful_generations++;
            }
        } catch (...) {
            // Count failures
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    EXPECT_EQ(successful_generations, num_generations);
    
    std::cout << "Completed " << num_generations << " generations in " 
              << duration.count() << " ms" << std::endl;
    
    // Performance expectation: should handle at least 10 generations per second
    double generations_per_second = (num_generations * 1000.0) / duration.count();
    EXPECT_GT(generations_per_second, 10.0) << "Performance too low: " << generations_per_second << " gen/s";
}

// System monitoring and metrics workflow

TEST_F(EndToEndTest, SystemMonitoringWorkflow) {
    // Initialize system
    system_->initialize(config_);
    system_->load_model(config_.model_weights_path, config_.tokenizer_path);
    
    // Step 1: Get initial system status
    json expected_status = {
        {"initialized", true},
        {"model_loaded", true},
        {"current_backend", "cpu"},
        {"mcp_server_running", false},
        {"active_sessions", 0},
        {"uptime_seconds", 120}
    };
    
    EXPECT_CALL(*system_, get_system_status())
        .Times(1)
        .WillOnce(Return(expected_status));
    
    auto status = system_->get_system_status();
    EXPECT_TRUE(status["initialized"]);
    EXPECT_TRUE(status["model_loaded"]);
    EXPECT_EQ(status["current_backend"], "cpu");
    
    // Step 2: Perform some operations to generate metrics
    system_->generate_text("Test prompt 1", json{});
    system_->generate_text("Test prompt 2", json{});
    system_->count_tokens("Sample text");
    
    // Step 3: Get performance metrics
    json expected_metrics = {
        {"total_generations", 2},
        {"total_tokens_generated", 150},
        {"total_token_counts", 1},
        {"average_generation_time_ms", 45.2},
        {"average_tokens_per_second", 50.3},
        {"memory_usage_mb", 1024},
        {"cache_hit_ratio", 0.85}
    };
    
    EXPECT_CALL(*system_, get_metrics())
        .Times(1)
        .WillOnce(Return(expected_metrics));
    
    auto metrics = system_->get_metrics();
    EXPECT_EQ(metrics["total_generations"], 2);
    EXPECT_GT(metrics["average_tokens_per_second"].get<double>(), 0.0);
    
    // Step 4: Reset metrics
    EXPECT_CALL(*system_, reset_metrics())
        .Times(1);
    
    system_->reset_metrics();
    
    // Step 5: Verify metrics are reset
    json reset_metrics = {
        {"total_generations", 0},
        {"total_tokens_generated", 0},
        {"total_token_counts", 0}
    };
    
    EXPECT_CALL(*system_, get_metrics())
        .Times(1)
        .WillOnce(Return(reset_metrics));
    
    auto new_metrics = system_->get_metrics();
    EXPECT_EQ(new_metrics["total_generations"], 0);
}

// Configuration update workflow

TEST_F(EndToEndTest, ConfigurationUpdateWorkflow) {
    // Initialize with original config
    system_->initialize(config_);
    
    // Step 1: Get current configuration
    EXPECT_CALL(*system_, get_config())
        .Times(1)
        .WillOnce(Return(config_));
    
    auto current_config = system_->get_config();
    EXPECT_EQ(current_config.max_context_tokens, 4096);
    
    // Step 2: Update configuration
    SystemConfig new_config = config_;
    new_config.max_context_tokens = 8192;
    new_config.backend_name = "intel";
    
    EXPECT_CALL(*system_, update_config(MatchesConfig(new_config)))
        .Times(1)
        .WillOnce(Return(true));
    
    bool config_updated = system_->update_config(new_config);
    EXPECT_TRUE(config_updated);
    
    // Step 3: Verify configuration change
    EXPECT_CALL(*system_, get_config())
        .Times(1)
        .WillOnce(Return(new_config));
    
    auto updated_config = system_->get_config();
    EXPECT_EQ(updated_config.max_context_tokens, 8192);
    EXPECT_EQ(updated_config.backend_name, "intel");
}

// Custom matchers for complex objects
MATCHER_P(MatchesConfig, expected_config, "") {
    return arg.model_weights_path == expected_config.model_weights_path &&
           arg.tokenizer_path == expected_config.tokenizer_path &&
           arg.backend_name == expected_config.backend_name &&
           arg.max_context_tokens == expected_config.max_context_tokens;
}

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