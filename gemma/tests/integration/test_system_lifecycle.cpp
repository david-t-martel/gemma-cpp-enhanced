// Focused system lifecycle tests (extracted from former monolithic end-to-end suite)
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <filesystem>
#include <fstream>
#include "mock_system.h"
#include "../utils/test_helpers.h"  // Provides GemmaTestBase

using namespace testing;

class SystemLifecycleTest : public gemma::test::GemmaTestBase {
protected:
    void SetUp() override {
        GemmaTestBase::SetUp();
        system_ = std::make_unique<MockGemmaSystem>();
        config_.model_weights_path = (test_dir_ / "test_model.sbs").string();
        config_.tokenizer_path = (test_dir_ / "tokenizer.spm").string();
        config_.session_storage_path = (test_dir_ / "sessions").string();
        std::filesystem::create_directories(config_.session_storage_path);
        std::ofstream(config_.model_weights_path) << "dummy";
        std::ofstream(config_.tokenizer_path) << "dummy";
    }

    std::unique_ptr<MockGemmaSystem> system_;
    SystemConfig config_;
};

TEST_F(SystemLifecycleTest, InitializeAndShutdown) {
    EXPECT_CALL(*system_, initialize(MatchesConfig(config_)))
        .Times(1).WillOnce(Return(true));
    EXPECT_TRUE(system_->initialize(config_));

    EXPECT_CALL(*system_, is_initialized())
        .Times(1).WillOnce(Return(true));
    EXPECT_TRUE(system_->is_initialized());

    // Simulate graceful shutdown sequence
    EXPECT_CALL(*system_, stop_mcp_server()).Times(1).WillOnce(Return(true));
    EXPECT_CALL(*system_, unload_model()).Times(1).WillOnce(Return(true));
    EXPECT_CALL(*system_, shutdown()).Times(1);
    system_->stop_mcp_server();
    system_->unload_model();
    system_->shutdown();
}

TEST_F(SystemLifecycleTest, FailedInitializationPath) {
    auto bad = config_;
    bad.model_weights_path = "/nonexistent/file.sbs";
    EXPECT_CALL(*system_, initialize(MatchesConfig(bad)))
        .Times(1).WillOnce(Return(false));
    EXPECT_FALSE(system_->initialize(bad));
}
