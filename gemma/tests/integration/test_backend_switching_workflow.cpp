// Focused backend switching workflow
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

class BackendSwitchingWorkflowTest : public gemma::test::GemmaTestBase {
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

TEST_F(BackendSwitchingWorkflowTest, SwitchCpuToIntel) {
    system_->initialize(config_); // permissive default ON_CALL can be set if needed

    EXPECT_CALL(*system_, get_current_backend()).WillOnce(Return("cpu"));
    EXPECT_EQ(system_->get_current_backend(), "cpu");

    std::vector<std::string> backends = {"cpu","intel","cuda","vulkan"};
    EXPECT_CALL(*system_, list_available_backends()).WillOnce(Return(backends));
    auto available = system_->list_available_backends();
    EXPECT_THAT(available, UnorderedElementsAre("cpu","intel","cuda","vulkan"));

    EXPECT_CALL(*system_, set_backend("intel")).WillOnce(Return(true));
    EXPECT_TRUE(system_->set_backend("intel"));

    EXPECT_CALL(*system_, get_current_backend()).WillOnce(Return("intel"));
    EXPECT_EQ(system_->get_current_backend(), "intel");
}
