// Focused configuration update workflow
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

class ConfigurationUpdateWorkflowTest : public gemma::test::GemmaTestBase {
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

TEST_F(ConfigurationUpdateWorkflowTest, UpdateContextAndBackend) {
    system_->initialize(config_);
    EXPECT_CALL(*system_, get_config()).WillOnce(Return(config_));
    auto cur = system_->get_config();
    EXPECT_EQ(cur.max_context_tokens, 4096);

    SystemConfig updated = config_;
    updated.max_context_tokens = 8192;
    updated.backend_name = "intel";
    EXPECT_CALL(*system_, update_config(MatchesConfig(updated))).WillOnce(Return(true));
    EXPECT_TRUE(system_->update_config(updated));

    EXPECT_CALL(*system_, get_config()).WillOnce(Return(updated));
    auto after = system_->get_config();
    EXPECT_EQ(after.max_context_tokens, 8192);
    EXPECT_EQ(after.backend_name, "intel");
}
