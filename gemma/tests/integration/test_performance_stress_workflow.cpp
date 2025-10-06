// Focused performance stress workflow (can be reclassified to performance category later)
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <nlohmann/json.hpp>
#include <future>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <iostream>

#include "mock_system.h"
#include "../utils/test_helpers.h"

using namespace testing;
using json = nlohmann::json;

class PerformanceStressWorkflowTest : public gemma::test::GemmaTestBase {
protected:
    void SetUp() override {
        GemmaTestBase::SetUp();
        system_ = std::make_unique<MockGemmaSystem>();
        config_.model_weights_path = (test_dir_ / "test_model.sbs").string();
        config_.tokenizer_path = (test_dir_ / "tokenizer.spm").string();
        std::ofstream(config_.model_weights_path) << "dummy";
        std::ofstream(config_.tokenizer_path) << "dummy";
    }
    std::unique_ptr<MockGemmaSystem> system_;
    SystemConfig config_;
};

TEST_F(PerformanceStressWorkflowTest, ConcurrentGenerations) {
    system_->initialize(config_);
    system_->load_model(config_.model_weights_path, config_.tokenizer_path);
    const int N = 50; // Reduced vs original 100 to keep runtime modest
    EXPECT_CALL(*system_, generate_text(_, _)).Times(N).WillRepeatedly(Return("Stress"));
    std::vector<std::future<std::string>> futures;
    for(int i=0;i<N;++i){
        futures.push_back(std::async(std::launch::async, [this,i]{ return system_->generate_text("P"+std::to_string(i), json{{"max_tokens",8}}); }));
    }
    int ok=0; for(auto &f: futures){ try { if(!f.get().empty()) ++ok; } catch(...){} }
    EXPECT_EQ(ok, N);
}
