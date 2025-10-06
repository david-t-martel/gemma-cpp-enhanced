// Focused system monitoring & metrics workflow
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

class SystemMetricsWorkflowTest : public gemma::test::GemmaTestBase {
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

TEST_F(SystemMetricsWorkflowTest, MetricsLifecycle) {
    system_->initialize(config_);
    system_->load_model(config_.model_weights_path, config_.tokenizer_path);

    json status = {{"initialized",true},{"model_loaded",true},{"current_backend","cpu"}};
    EXPECT_CALL(*system_, get_system_status()).WillOnce(Return(status));
    auto s = system_->get_system_status();
    EXPECT_TRUE(s["initialized"]);

    system_->generate_text("A", json{});
    system_->generate_text("B", json{});
    system_->count_tokens("B");

    json metrics = {{"total_generations",2},{"average_tokens_per_second",42.0}};
    EXPECT_CALL(*system_, get_metrics()).WillOnce(Return(metrics));
    auto m = system_->get_metrics();
    EXPECT_EQ(m["total_generations"], 2);

    EXPECT_CALL(*system_, reset_metrics()).Times(1);
    system_->reset_metrics();

    json reset = {{"total_generations",0}};
    EXPECT_CALL(*system_, get_metrics()).WillOnce(Return(reset));
    auto post = system_->get_metrics();
    EXPECT_EQ(post["total_generations"], 0);
}
