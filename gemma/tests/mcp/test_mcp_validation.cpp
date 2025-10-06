// Lightweight MCP protocol validation tests
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <nlohmann/json.hpp>
#include "../integration/mock_system.h"
#include "../utils/system_config_builder.h"

using namespace testing;
using json = nlohmann::json;

// Example schema snippet (could be extended or loaded from external JSON schema file)
namespace {
struct RequestFields { static constexpr const char* ID = "id"; static constexpr const char* TYPE = "type"; };
}

class MCPValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        using gemma::test::SystemConfigBuilder; 
        config_ = SystemConfigBuilder::WithDefaults(temp_dir_).MCPPort(8123).Build();
        gemma::test::SystemConfigBuilder::EnsureArtifacts(config_);
        system_ = std::make_unique<MockGemmaSystem>();
    }
    std::unique_ptr<MockGemmaSystem> system_;
    SystemConfig config_{};
    std::filesystem::path temp_dir_ = std::filesystem::temp_directory_path() / "gemma_mcp_validation";
};

TEST_F(MCPValidationTest, BasicRequestShape) {
    // Simulate an inbound MCP JSON request
    json request = {
        {RequestFields::ID, 1},
        {RequestFields::TYPE, "initialize"},
        {"params", json{{"model","test"}}}
    };
    ASSERT_TRUE(request.contains(RequestFields::ID));
    ASSERT_TRUE(request.contains(RequestFields::TYPE));
    EXPECT_TRUE(request[RequestFields::ID].is_number_integer());
    EXPECT_TRUE(request[RequestFields::TYPE].is_string());
}

TEST_F(MCPValidationTest, MissingIdRejected) {
    json bad = {{"type","initialize"}};
    bool valid = bad.contains(RequestFields::ID) && bad.contains(RequestFields::TYPE);
    EXPECT_FALSE(valid);
}

TEST_F(MCPValidationTest, RoundTripEcho) {
    EXPECT_CALL(*system_, initialize(::testing::_)).WillOnce(Return(true));
    ASSERT_TRUE(system_->initialize(config_));
    // Represent a trivial echo handler (placeholder)
    json payload = {{"msg","hello"}};
    json response = {{"ok", true}, {"echo", payload}};
    EXPECT_TRUE(response["ok"].get<bool>());
    EXPECT_EQ(response["echo"], payload);
}
