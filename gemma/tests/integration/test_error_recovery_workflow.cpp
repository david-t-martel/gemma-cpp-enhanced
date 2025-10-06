// Focused error handling & recovery workflow test
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <memory>

#include "mock_system.h"
#include "../utils/test_helpers.h"
#include "../utils/system_config_builder.h"

using namespace testing;
using json = nlohmann::json;

class ErrorRecoveryWorkflowTest : public gemma::test::GemmaTestBase {
protected:
    void SetUp() override {
        GemmaTestBase::SetUp();
    using gemma::test::SystemConfigBuilder; 
    system_ = std::make_unique<MockGemmaSystem>();
    config_ = SystemConfigBuilder::WithDefaults(test_dir_).Build();
    SystemConfigBuilder::EnsureArtifacts(config_);
    }
    std::unique_ptr<MockGemmaSystem> system_;
    SystemConfig config_;
};

TEST_F(ErrorRecoveryWorkflowTest, InvalidThenValidPathsAndPrompt) {
    SystemConfig bad = config_;
    bad.model_weights_path = "/nonexistent/file.sbs";
    EXPECT_CALL(*system_, initialize(MatchesConfig(bad))).WillOnce(Return(false));
    EXPECT_FALSE(system_->initialize(bad));

    EXPECT_CALL(*system_, initialize(MatchesConfig(config_))).WillOnce(Return(true));
    EXPECT_TRUE(system_->initialize(config_));

    EXPECT_CALL(*system_, load_model("/nonexistent/model.sbs", "/nonexistent/tokenizer.spm")).WillOnce(Return(false));
    EXPECT_FALSE(system_->load_model("/nonexistent/model.sbs", "/nonexistent/tokenizer.spm"));

    EXPECT_CALL(*system_, load_model(config_.model_weights_path, config_.tokenizer_path)).WillOnce(Return(true));
    EXPECT_TRUE(system_->load_model(config_.model_weights_path, config_.tokenizer_path));

    EXPECT_CALL(*system_, generate_text("", _)).WillOnce(Throw(std::invalid_argument("Empty prompt not allowed")));
    EXPECT_THROW(system_->generate_text("", json{}), std::invalid_argument);

    EXPECT_CALL(*system_, generate_text("Valid prompt", _)).WillOnce(Return("Valid response"));
    EXPECT_EQ(system_->generate_text("Valid prompt", json{}), "Valid response");
}
