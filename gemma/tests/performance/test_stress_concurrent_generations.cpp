// Performance stress workflow (moved from integration to performance suite)
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <future>
#include <vector>
#include <nlohmann/json.hpp>

#include "../integration/mock_system.h"
#include "../utils/test_helpers.h"
#include "../utils/system_config_builder.h"

using namespace testing;
using json = nlohmann::json;

class StressConcurrentGenerationsTest : public gemma::test::GemmaTestBase {
protected:
    void SetUp() override {
        GemmaTestBase::SetUp();
        using gemma::test::SystemConfigBuilder; using gemma::test::SystemConfigBuilder;
        system_ = std::make_unique<MockGemmaSystem>();
        config_ = SystemConfigBuilder::WithDefaults(test_dir_).Context(4096).Build();
        SystemConfigBuilder::EnsureArtifacts(config_);
    }
    std::unique_ptr<MockGemmaSystem> system_;
    SystemConfig config_{};
};

TEST_F(StressConcurrentGenerationsTest, ConcurrentGenerations) {
    system_->initialize(config_);
    system_->load_model(config_.model_weights_path, config_.tokenizer_path);
    const int N = 50; // moderate
    EXPECT_CALL(*system_, generate_text(_, _)).Times(N).WillRepeatedly(Return("Stress"));
    std::vector<std::future<std::string>> futures; futures.reserve(N);
    for(int i=0;i<N;++i) {
        futures.push_back(std::async(std::launch::async, [this,i]{ return system_->generate_text("P"+std::to_string(i), json{{"max_tokens",8}}); }));
    }
    int ok = 0; for(auto &f : futures) { try { if(!f.get().empty()) ++ok; } catch(...) {} }
    EXPECT_EQ(ok, N);
}
