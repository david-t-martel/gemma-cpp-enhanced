// Focused batch processing workflow test
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

class BatchProcessingWorkflowTest : public gemma::test::GemmaTestBase {
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

TEST_F(BatchProcessingWorkflowTest, GenerateBatch) {
    system_->initialize(config_);
    system_->load_model(config_.model_weights_path, config_.tokenizer_path);

    std::vector<std::string> prompts = {"A","B","C","D"};
    json opts = {{"max_tokens",64},{"batch_size",2}};
    std::vector<std::string> expected = {"RA","RB","RC","RD"};

    EXPECT_CALL(*system_, generate_batch(prompts, opts)).WillOnce(Return(expected));
    auto out = system_->generate_batch(prompts, opts);
    EXPECT_EQ(out.size(), expected.size());
}
