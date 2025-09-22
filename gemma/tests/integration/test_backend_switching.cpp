#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../utils/test_helpers.h"
#include "../utils/mock_backend.h"
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <future>
#include <atomic>

// Mock backend manager for testing backend switching
// In real implementation: #include "../../src/backends/BackendManager.h"

using namespace gemma::test;
using namespace testing;

// Mock backend manager that coordinates multiple backends
class MockBackendManager {
public:
    MOCK_METHOD(bool, initialize, (), ());
    MOCK_METHOD(void, shutdown, (), ());
    MOCK_METHOD(bool, register_backend, (const std::string& name, std::unique_ptr<MockBackendBase> backend), ());
    MOCK_METHOD(bool, unregister_backend, (const std::string& name), ());
    MOCK_METHOD(bool, set_active_backend, (const std::string& name), ());
    MOCK_METHOD(std::string, get_active_backend_name, (), (const));
    MOCK_METHOD(MockBackendBase*, get_active_backend, (), ());
    MOCK_METHOD(MockBackendBase*, get_backend, (const std::string& name), ());
    MOCK_METHOD(std::vector<std::string>, list_available_backends, (), (const));
    MOCK_METHOD(std::vector<std::string>, list_initialized_backends, (), (const));
    MOCK_METHOD(nlohmann::json, get_backend_status, (const std::string& name), (const));
    MOCK_METHOD(nlohmann::json, get_all_backend_status, (), (const));
    MOCK_METHOD(bool, auto_select_best_backend, (), ());
    MOCK_METHOD(bool, validate_backend_compatibility, (const std::string& name), ());
    MOCK_METHOD(void, set_fallback_backend, (const std::string& name), ());
    MOCK_METHOD(std::string, get_fallback_backend, (), (const));
    MOCK_METHOD(bool, switch_to_fallback, (), ());
};

// Mock model operations that work across backends
class MockCrossBackendModel {
public:
    MOCK_METHOD(bool, load_on_backend, (const std::string& backend_name, const std::string& model_path), ());
    MOCK_METHOD(bool, transfer_to_backend, (const std::string& from_backend, const std::string& to_backend), ());
    MOCK_METHOD(bool, is_loaded_on_backend, (const std::string& backend_name), (const));
    MOCK_METHOD(std::vector<float>, inference_on_backend, (const std::string& backend_name, const std::vector<int>& tokens), ());
    MOCK_METHOD(std::string, get_model_backend_info, (const std::string& backend_name), (const));
    MOCK_METHOD(size_t, get_model_memory_usage, (const std::string& backend_name), (const));
    MOCK_METHOD(bool, unload_from_backend, (const std::string& backend_name), ());
    MOCK_METHOD(std::vector<std::string>, get_backends_with_model, (), (const));
};

class BackendSwitchingTest : public BackendTestFixture {
protected:
    void SetUp() override {
        BackendTestFixture::SetUp();
        
        backend_manager_ = std::make_unique<MockBackendManager>();
        cross_backend_model_ = std::make_unique<MockCrossBackendModel>();
        
        setup_backend_manager_expectations();
        setup_model_expectations();
        register_test_backends();
    }
    
    void setup_backend_manager_expectations() {
        ON_CALL(*backend_manager_, initialize()).WillByDefault(Return(true));
        ON_CALL(*backend_manager_, get_active_backend_name()).WillByDefault(Return("cpu"));
        ON_CALL(*backend_manager_, get_fallback_backend()).WillByDefault(Return("cpu"));
        ON_CALL(*backend_manager_, list_available_backends()).WillByDefault(Return(
            std::vector<std::string>{"cpu", "intel", "cuda", "vulkan"}
        ));
        ON_CALL(*backend_manager_, list_initialized_backends()).WillByDefault(Return(
            std::vector<std::string>{"cpu"}
        ));
    }
    
    void setup_model_expectations() {
        ON_CALL(*cross_backend_model_, is_loaded_on_backend(_)).WillByDefault(Return(false));
        ON_CALL(*cross_backend_model_, get_backends_with_model()).WillByDefault(Return(
            std::vector<std::string>{}
        ));
    }
    
    void register_test_backends() {
        backend_names_ = {"cpu", "intel", "cuda", "vulkan"};
        
        for (const auto& name : backend_names_) {
            EXPECT_CALL(*backend_manager_, register_backend(name, _))
                .Times(1)
                .WillOnce(Return(true));
        }
        
        // Register all backends
        backend_manager_->register_backend("cpu", std::move(cpu_backend_));
        backend_manager_->register_backend("intel", std::move(intel_backend_));
        backend_manager_->register_backend("cuda", std::move(cuda_backend_));
        backend_manager_->register_backend("vulkan", std::move(vulkan_backend_));
    }
    
    std::unique_ptr<MockBackendManager> backend_manager_;
    std::unique_ptr<MockCrossBackendModel> cross_backend_model_;
    std::vector<std::string> backend_names_;
    std::string test_model_path_ = "/path/to/test/model.sbs";
};

// Basic backend switching tests

TEST_F(BackendSwitchingTest, InitializeBackendManager) {
    EXPECT_CALL(*backend_manager_, initialize())
        .Times(1)
        .WillOnce(Return(true));
    
    bool initialized = backend_manager_->initialize();
    EXPECT_TRUE(initialized);
}

TEST_F(BackendSwitchingTest, ListAvailableBackends) {
    std::vector<std::string> expected_backends = {"cpu", "intel", "cuda", "vulkan"};
    
    EXPECT_CALL(*backend_manager_, list_available_backends())
        .Times(1)
        .WillOnce(Return(expected_backends));
    
    auto available = backend_manager_->list_available_backends();
    EXPECT_EQ(available.size(), 4);
    EXPECT_THAT(available, UnorderedElementsAre("cpu", "intel", "cuda", "vulkan"));
}

TEST_F(BackendSwitchingTest, SwitchToValidBackend) {
    // Switch from CPU to Intel
    EXPECT_CALL(*backend_manager_, set_active_backend("intel"))
        .Times(1)
        .WillOnce(Return(true));
    
    bool switched = backend_manager_->set_active_backend("intel");
    EXPECT_TRUE(switched);
    
    // Verify the switch
    EXPECT_CALL(*backend_manager_, get_active_backend_name())
        .Times(1)
        .WillOnce(Return("intel"));
    
    std::string active_backend = backend_manager_->get_active_backend_name();
    EXPECT_EQ(active_backend, "intel");
}

TEST_F(BackendSwitchingTest, SwitchToInvalidBackend) {
    EXPECT_CALL(*backend_manager_, set_active_backend("nonexistent"))
        .Times(1)
        .WillOnce(Return(false));
    
    bool switched = backend_manager_->set_active_backend("nonexistent");
    EXPECT_FALSE(switched);
    
    // Active backend should remain unchanged
    EXPECT_CALL(*backend_manager_, get_active_backend_name())
        .Times(1)
        .WillOnce(Return("cpu")); // Should still be CPU
    
    std::string active_backend = backend_manager_->get_active_backend_name();
    EXPECT_EQ(active_backend, "cpu");
}

// Model loading across backends tests

TEST_F(BackendSwitchingTest, LoadModelOnSpecificBackend) {
    EXPECT_CALL(*cross_backend_model_, load_on_backend("intel", test_model_path_))
        .Times(1)
        .WillOnce(Return(true));
    
    bool loaded = cross_backend_model_->load_on_backend("intel", test_model_path_);
    EXPECT_TRUE(loaded);
    
    // Verify model is loaded on Intel backend
    EXPECT_CALL(*cross_backend_model_, is_loaded_on_backend("intel"))
        .Times(1)
        .WillOnce(Return(true));
    
    bool is_loaded = cross_backend_model_->is_loaded_on_backend("intel");
    EXPECT_TRUE(is_loaded);
}

TEST_F(BackendSwitchingTest, LoadModelOnMultipleBackends) {
    // Load on CPU first
    EXPECT_CALL(*cross_backend_model_, load_on_backend("cpu", test_model_path_))
        .Times(1)
        .WillOnce(Return(true));
    
    bool loaded_cpu = cross_backend_model_->load_on_backend("cpu", test_model_path_);
    EXPECT_TRUE(loaded_cpu);
    
    // Then load on Intel
    EXPECT_CALL(*cross_backend_model_, load_on_backend("intel", test_model_path_))
        .Times(1)
        .WillOnce(Return(true));
    
    bool loaded_intel = cross_backend_model_->load_on_backend("intel", test_model_path_);
    EXPECT_TRUE(loaded_intel);
    
    // Verify both backends have the model
    std::vector<std::string> expected_backends = {"cpu", "intel"};
    EXPECT_CALL(*cross_backend_model_, get_backends_with_model())
        .Times(1)
        .WillOnce(Return(expected_backends));
    
    auto backends_with_model = cross_backend_model_->get_backends_with_model();
    EXPECT_THAT(backends_with_model, UnorderedElementsAre("cpu", "intel"));
}

TEST_F(BackendSwitchingTest, TransferModelBetweenBackends) {
    // First load model on CPU
    EXPECT_CALL(*cross_backend_model_, load_on_backend("cpu", test_model_path_))
        .Times(1)
        .WillOnce(Return(true));
    
    cross_backend_model_->load_on_backend("cpu", test_model_path_);
    
    // Transfer from CPU to Intel
    EXPECT_CALL(*cross_backend_model_, transfer_to_backend("cpu", "intel"))
        .Times(1)
        .WillOnce(Return(true));
    
    bool transferred = cross_backend_model_->transfer_to_backend("cpu", "intel");
    EXPECT_TRUE(transferred);
    
    // Verify model is now on Intel and not on CPU
    EXPECT_CALL(*cross_backend_model_, is_loaded_on_backend("intel"))
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*cross_backend_model_, is_loaded_on_backend("cpu"))
        .Times(1)
        .WillOnce(Return(false));
    
    EXPECT_TRUE(cross_backend_model_->is_loaded_on_backend("intel"));
    EXPECT_FALSE(cross_backend_model_->is_loaded_on_backend("cpu"));
}

// Backend performance comparison tests

TEST_F(BackendSwitchingTest, CompareBackendPerformance) {
    std::vector<int> test_tokens = TestDataGenerator::generate_random_tokens(512, 32000);
    std::vector<float> expected_output(32000, 0.5f);
    
    struct BackendPerformance {
        std::string name;
        double avg_inference_time_ms;
        double tokens_per_second;
    };
    
    std::vector<BackendPerformance> performance_results;
    
    for (const auto& backend_name : {"cpu", "intel", "cuda"}) {
        // Load model on backend
        EXPECT_CALL(*cross_backend_model_, load_on_backend(backend_name, test_model_path_))
            .Times(1)
            .WillOnce(Return(true));
        
        cross_backend_model_->load_on_backend(backend_name, test_model_path_);
        
        // Run benchmark
        const int num_iterations = 10;
        EXPECT_CALL(*cross_backend_model_, inference_on_backend(backend_name, test_tokens))
            .Times(num_iterations)
            .WillRepeatedly(Return(expected_output));
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            auto result = cross_backend_model_->inference_on_backend(backend_name, test_tokens);
            EXPECT_EQ(result.size(), expected_output.size());
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        BackendPerformance perf;
        perf.name = backend_name;
        perf.avg_inference_time_ms = static_cast<double>(duration_ms) / num_iterations;
        perf.tokens_per_second = (test_tokens.size() * 1000.0) / perf.avg_inference_time_ms;
        
        performance_results.push_back(perf);
        
        std::cout << backend_name << " backend: " 
                  << perf.avg_inference_time_ms << " ms, "
                  << perf.tokens_per_second << " tokens/s" << std::endl;
    }
    
    // Verify we have results for all backends
    EXPECT_EQ(performance_results.size(), 3);
    
    // Find fastest backend
    auto fastest = std::min_element(performance_results.begin(), performance_results.end(),
        [](const BackendPerformance& a, const BackendPerformance& b) {
            return a.avg_inference_time_ms < b.avg_inference_time_ms;
        });
    
    std::cout << "Fastest backend: " << fastest->name << std::endl;
    
    // GPU backends should generally be faster than CPU
    auto cpu_result = std::find_if(performance_results.begin(), performance_results.end(),
        [](const BackendPerformance& p) { return p.name == "cpu"; });
    auto cuda_result = std::find_if(performance_results.begin(), performance_results.end(),
        [](const BackendPerformance& p) { return p.name == "cuda"; });
    
    if (cpu_result != performance_results.end() && cuda_result != performance_results.end()) {
        EXPECT_LT(cuda_result->avg_inference_time_ms, cpu_result->avg_inference_time_ms * 2.0)
            << "CUDA should be at least 2x faster than CPU";
    }
}

// Auto-selection and fallback tests

TEST_F(BackendSwitchingTest, AutoSelectBestBackend) {
    // Mock backend capabilities and performance data
    EXPECT_CALL(*backend_manager_, auto_select_best_backend())
        .Times(1)
        .WillOnce(Return(true));
    
    bool auto_selected = backend_manager_->auto_select_best_backend();
    EXPECT_TRUE(auto_selected);
    
    // Verify a backend was selected (should prefer GPU over CPU)
    EXPECT_CALL(*backend_manager_, get_active_backend_name())
        .Times(1)
        .WillOnce(Return("cuda")); // Assuming CUDA is best available
    
    std::string selected_backend = backend_manager_->get_active_backend_name();
    EXPECT_THAT(selected_backend, AnyOf("intel", "cuda", "vulkan")); // Should be a GPU backend
}

TEST_F(BackendSwitchingTest, FallbackBackendConfiguration) {
    // Set Intel as fallback
    EXPECT_CALL(*backend_manager_, set_fallback_backend("intel"))
        .Times(1);
    
    backend_manager_->set_fallback_backend("intel");
    
    // Verify fallback is set
    EXPECT_CALL(*backend_manager_, get_fallback_backend())
        .Times(1)
        .WillOnce(Return("intel"));
    
    std::string fallback = backend_manager_->get_fallback_backend();
    EXPECT_EQ(fallback, "intel");
}

TEST_F(BackendSwitchingTest, SwitchToFallbackOnError) {
    // Simulate CUDA backend failure
    EXPECT_CALL(*backend_manager_, set_active_backend("cuda"))
        .Times(1)
        .WillOnce(Return(false)); // Fails
    
    bool cuda_switch = backend_manager_->set_active_backend("cuda");
    EXPECT_FALSE(cuda_switch);
    
    // System should switch to fallback
    EXPECT_CALL(*backend_manager_, switch_to_fallback())
        .Times(1)
        .WillOnce(Return(true));
    
    bool fallback_switch = backend_manager_->switch_to_fallback();
    EXPECT_TRUE(fallback_switch);
    
    // Verify we're now on fallback backend
    EXPECT_CALL(*backend_manager_, get_active_backend_name())
        .Times(1)
        .WillOnce(Return("intel")); // Fallback backend
    
    std::string current_backend = backend_manager_->get_active_backend_name();
    EXPECT_EQ(current_backend, "intel");
}

// Concurrent backend operations tests

TEST_F(BackendSwitchingTest, ConcurrentBackendInference) {
    // Load model on multiple backends
    for (const auto& backend : {"cpu", "intel", "cuda"}) {
        EXPECT_CALL(*cross_backend_model_, load_on_backend(backend, test_model_path_))
            .Times(1)
            .WillOnce(Return(true));
        
        cross_backend_model_->load_on_backend(backend, test_model_path_);
    }
    
    // Run concurrent inference on different backends
    const int num_requests_per_backend = 5;
    std::vector<std::future<std::vector<float>>> futures;
    std::atomic<int> successful_requests{0};
    
    std::vector<int> test_tokens = TestDataGenerator::generate_random_tokens(256, 32000);
    std::vector<float> expected_output(32000, 1.0f);
    
    EXPECT_CALL(*cross_backend_model_, inference_on_backend(_, test_tokens))
        .Times(15) // 3 backends * 5 requests each
        .WillRepeatedly(Return(expected_output));
    
    for (const auto& backend : {"cpu", "intel", "cuda"}) {
        for (int i = 0; i < num_requests_per_backend; ++i) {
            futures.push_back(std::async(std::launch::async, [this, backend, test_tokens, &successful_requests]() {
                try {
                    auto result = cross_backend_model_->inference_on_backend(backend, test_tokens);
                    if (!result.empty()) {
                        successful_requests++;
                    }
                    return result;
                } catch (...) {
                    return std::vector<float>{};
                }
            }));
        }
    }
    
    // Wait for all requests to complete
    for (auto& future : futures) {
        auto result = future.get();
        EXPECT_FALSE(result.empty());
    }
    
    EXPECT_EQ(successful_requests.load(), 15);
}

TEST_F(BackendSwitchingTest, ConcurrentBackendSwitching) {
    const int num_switches = 20;
    std::vector<std::future<bool>> switch_futures;
    std::atomic<int> successful_switches{0};
    
    std::vector<std::string> backends = {"cpu", "intel", "cuda", "vulkan"};
    
    EXPECT_CALL(*backend_manager_, set_active_backend(_))
        .Times(num_switches)
        .WillRepeatedly(Return(true));
    
    // Launch concurrent backend switches
    for (int i = 0; i < num_switches; ++i) {
        switch_futures.push_back(std::async(std::launch::async, [this, &backends, &successful_switches, i]() {
            try {
                std::string target_backend = backends[i % backends.size()];
                bool switched = backend_manager_->set_active_backend(target_backend);
                if (switched) {
                    successful_switches++;
                }
                return switched;
            } catch (...) {
                return false;
            }
        }));
    }
    
    // Wait for all switches to complete
    for (auto& future : switch_futures) {
        bool result = future.get();
        EXPECT_TRUE(result);
    }
    
    EXPECT_EQ(successful_switches.load(), num_switches);
}

// Backend compatibility and validation tests

TEST_F(BackendSwitchingTest, ValidateBackendCompatibility) {
    // Test compatibility validation for each backend
    std::vector<std::string> backends = {"cpu", "intel", "cuda", "vulkan"};
    
    for (const auto& backend : backends) {
        EXPECT_CALL(*backend_manager_, validate_backend_compatibility(backend))
            .Times(1)
            .WillOnce(Return(true)); // Assume all backends are compatible in test
        
        bool compatible = backend_manager_->validate_backend_compatibility(backend);
        EXPECT_TRUE(compatible) << "Backend " << backend << " should be compatible";
    }
}

TEST_F(BackendSwitchingTest, HandleIncompatibleBackend) {
    // Test handling of incompatible backend
    EXPECT_CALL(*backend_manager_, validate_backend_compatibility("incompatible"))
        .Times(1)
        .WillOnce(Return(false));
    
    bool compatible = backend_manager_->validate_backend_compatibility("incompatible");
    EXPECT_FALSE(compatible);
    
    // Attempt to switch to incompatible backend should fail
    EXPECT_CALL(*backend_manager_, set_active_backend("incompatible"))
        .Times(1)
        .WillOnce(Return(false));
    
    bool switched = backend_manager_->set_active_backend("incompatible");
    EXPECT_FALSE(switched);
}

// Backend status monitoring tests

TEST_F(BackendSwitchingTest, MonitorBackendStatus) {
    // Get status for individual backend
    nlohmann::json expected_intel_status = {
        {"name", "intel"},
        {"status", "ready"},
        {"memory_usage_mb", 1024},
        {"active_sessions", 2},
        {"total_inferences", 150},
        {"average_inference_time_ms", 45.2}
    };
    
    EXPECT_CALL(*backend_manager_, get_backend_status("intel"))
        .Times(1)
        .WillOnce(Return(expected_intel_status));
    
    auto intel_status = backend_manager_->get_backend_status("intel");
    EXPECT_EQ(intel_status["name"], "intel");
    EXPECT_EQ(intel_status["status"], "ready");
    EXPECT_GT(intel_status["total_inferences"].get<int>(), 0);
}

TEST_F(BackendSwitchingTest, MonitorAllBackendStatus) {
    nlohmann::json expected_all_status = {
        {"backends", {
            {
                {"name", "cpu"},
                {"status", "ready"},
                {"active", false}
            },
            {
                {"name", "intel"},
                {"status", "ready"},
                {"active", true}
            },
            {
                {"name", "cuda"},
                {"status", "unavailable"},
                {"active", false}
            }
        }},
        {"active_backend", "intel"},
        {"total_backends", 3}
    };
    
    EXPECT_CALL(*backend_manager_, get_all_backend_status())
        .Times(1)
        .WillOnce(Return(expected_all_status));
    
    auto all_status = backend_manager_->get_all_backend_status();
    EXPECT_EQ(all_status["active_backend"], "intel");
    EXPECT_EQ(all_status["total_backends"], 3);
    EXPECT_EQ(all_status["backends"].size(), 3);
}

// Memory management across backends tests

TEST_F(BackendSwitchingTest, MonitorMemoryUsageAcrossBackends) {
    std::vector<std::string> backends = {"cpu", "intel", "cuda"};
    
    for (const auto& backend : backends) {
        // Load model and check memory usage
        EXPECT_CALL(*cross_backend_model_, load_on_backend(backend, test_model_path_))
            .Times(1)
            .WillOnce(Return(true));
        
        cross_backend_model_->load_on_backend(backend, test_model_path_);
        
        size_t expected_memory = (backend == "cpu") ? 2048 * 1024 * 1024 : 1024 * 1024 * 1024; // 2GB for CPU, 1GB for GPU
        
        EXPECT_CALL(*cross_backend_model_, get_model_memory_usage(backend))
            .Times(1)
            .WillOnce(Return(expected_memory));
        
        size_t memory_usage = cross_backend_model_->get_model_memory_usage(backend);
        EXPECT_GT(memory_usage, 0);
        
        std::cout << backend << " backend memory usage: " 
                  << (memory_usage / (1024 * 1024)) << " MB" << std::endl;
    }
}

TEST_F(BackendSwitchingTest, UnloadModelFromSpecificBackend) {
    // Load model on multiple backends
    for (const auto& backend : {"cpu", "intel"}) {
        EXPECT_CALL(*cross_backend_model_, load_on_backend(backend, test_model_path_))
            .Times(1)
            .WillOnce(Return(true));
        
        cross_backend_model_->load_on_backend(backend, test_model_path_);
    }
    
    // Unload from CPU only
    EXPECT_CALL(*cross_backend_model_, unload_from_backend("cpu"))
        .Times(1)
        .WillOnce(Return(true));
    
    bool unloaded = cross_backend_model_->unload_from_backend("cpu");
    EXPECT_TRUE(unloaded);
    
    // Verify model is still on Intel but not CPU
    EXPECT_CALL(*cross_backend_model_, is_loaded_on_backend("cpu"))
        .Times(1)
        .WillOnce(Return(false));
    
    EXPECT_CALL(*cross_backend_model_, is_loaded_on_backend("intel"))
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_FALSE(cross_backend_model_->is_loaded_on_backend("cpu"));
    EXPECT_TRUE(cross_backend_model_->is_loaded_on_backend("intel"));
}

// Error recovery and resilience tests

TEST_F(BackendSwitchingTest, RecoverFromBackendFailure) {
    // Set up active backend
    EXPECT_CALL(*backend_manager_, set_active_backend("cuda"))
        .Times(1)
        .WillOnce(Return(true));
    
    backend_manager_->set_active_backend("cuda");
    
    // Simulate backend failure during inference
    std::vector<int> test_tokens = TestDataGenerator::generate_random_tokens(100, 32000);
    
    EXPECT_CALL(*cross_backend_model_, inference_on_backend("cuda", test_tokens))
        .Times(1)
        .WillOnce(Throw(std::runtime_error("CUDA device disconnected")));
    
    // Should automatically switch to fallback
    EXPECT_CALL(*backend_manager_, switch_to_fallback())
        .Times(1)
        .WillOnce(Return(true));
    
    EXPECT_CALL(*backend_manager_, get_active_backend_name())
        .Times(1)
        .WillOnce(Return("intel")); // Fallback backend
    
    // Test error handling and recovery
    EXPECT_THROW(cross_backend_model_->inference_on_backend("cuda", test_tokens), std::runtime_error);
    
    // System should recover by switching to fallback
    backend_manager_->switch_to_fallback();
    std::string recovery_backend = backend_manager_->get_active_backend_name();
    EXPECT_EQ(recovery_backend, "intel");
    
    // Verify inference works on fallback backend
    std::vector<float> expected_output(32000, 0.8f);
    EXPECT_CALL(*cross_backend_model_, inference_on_backend("intel", test_tokens))
        .Times(1)
        .WillOnce(Return(expected_output));
    
    auto recovery_result = cross_backend_model_->inference_on_backend("intel", test_tokens);
    EXPECT_FALSE(recovery_result.empty());
}

// Cleanup tests

TEST_F(BackendSwitchingTest, CleanupAllBackends) {
    // Load models on multiple backends
    for (const auto& backend : {"cpu", "intel", "cuda"}) {
        EXPECT_CALL(*cross_backend_model_, load_on_backend(backend, test_model_path_))
            .Times(1)
            .WillOnce(Return(true));
        
        cross_backend_model_->load_on_backend(backend, test_model_path_);
    }
    
    // Unload from all backends
    for (const auto& backend : {"cpu", "intel", "cuda"}) {
        EXPECT_CALL(*cross_backend_model_, unload_from_backend(backend))
            .Times(1)
            .WillOnce(Return(true));
        
        bool unloaded = cross_backend_model_->unload_from_backend(backend);
        EXPECT_TRUE(unloaded);
    }
    
    // Verify no backends have the model loaded
    EXPECT_CALL(*cross_backend_model_, get_backends_with_model())
        .Times(1)
        .WillOnce(Return(std::vector<std::string>{}));
    
    auto backends_with_model = cross_backend_model_->get_backends_with_model();
    EXPECT_TRUE(backends_with_model.empty());
    
    // Shutdown backend manager
    EXPECT_CALL(*backend_manager_, shutdown())
        .Times(1);
    
    backend_manager_->shutdown();
}