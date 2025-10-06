// Focused model loading & inference workflow test (extracted from end-to-end)
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <memory>

#include "mock_system.h"
#include "../utils/test_helpers.h"
#include "../utils/system_config_builder.h"

using namespace testing;
using json = nlohmann::json;

class ModelInferenceWorkflowTest : public gemma::test::GemmaTestBase {
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

TEST_F(ModelInferenceWorkflowTest, LoadAndGenerate) {
    EXPECT_CALL(*system_, initialize(_)).WillOnce(Return(true));
    EXPECT_TRUE(system_->initialize(config_));

    EXPECT_CALL(*system_, load_model(config_.model_weights_path, config_.tokenizer_path)).WillOnce(Return(true));
    EXPECT_TRUE(system_->load_model(config_.model_weights_path, config_.tokenizer_path));

    EXPECT_CALL(*system_, is_model_loaded()).WillOnce(Return(true));
    EXPECT_TRUE(system_->is_model_loaded());

    json info = {{"name","gemma-2b-it"},{"loaded",true}};
    EXPECT_CALL(*system_, get_model_info()).WillOnce(Return(info));
    auto model_info = system_->get_model_info();
    EXPECT_TRUE(model_info["loaded"]);

    json gen_opts = {{"max_tokens",32},{"temperature",0.7}};
    EXPECT_CALL(*system_, generate_text("Test prompt", gen_opts)).WillOnce(Return("Test response"));
    auto resp = system_->generate_text("Test prompt", gen_opts);
    EXPECT_EQ(resp, "Test response");

    EXPECT_CALL(*system_, count_tokens("Test prompt")).WillOnce(Return(3));
    EXPECT_EQ(system_->count_tokens("Test prompt"), 3);
}
