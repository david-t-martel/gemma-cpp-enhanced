// Focused MCP server interaction workflow
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

class MCPServerWorkflowTest : public gemma::test::GemmaTestBase {
protected:
    void SetUp() override {
        GemmaTestBase::SetUp();
    using gemma::test::SystemConfigBuilder; 
    system_ = std::make_unique<MockGemmaSystem>();
    config_ = SystemConfigBuilder::WithDefaults(test_dir_).MCPPort(9099).Build();
    SystemConfigBuilder::EnsureArtifacts(config_);
    }
    std::unique_ptr<MockGemmaSystem> system_;
    SystemConfig config_;
};

TEST_F(MCPServerWorkflowTest, StartHandleRequestsStop) {
    system_->initialize(config_);

    EXPECT_CALL(*system_, start_mcp_server(config_.mcp_server_port)).WillOnce(Return(true));
    EXPECT_TRUE(system_->start_mcp_server(config_.mcp_server_port));

    EXPECT_CALL(*system_, is_mcp_server_running()).WillOnce(Return(true));
    EXPECT_TRUE(system_->is_mcp_server_running());

    json gen_req = {{"jsonrpc","2.0"},{"method","tools/call"},{"id","1"},{"params",{{"name","generate_text"},{"arguments",{{"prompt","Hi"},{"max_tokens",8}}}}}};
    json gen_resp = {{"jsonrpc","2.0"},{"id","1"},{"result",{{"content",{{ {"type","text" }, {"text","Hello"}}}}}}};
    EXPECT_CALL(*system_, handle_mcp_request(gen_req)).WillOnce(Return(gen_resp));
    auto r = system_->handle_mcp_request(gen_req);
    EXPECT_EQ(r["id"], "1");

    EXPECT_CALL(*system_, stop_mcp_server()).WillOnce(Return(true));
    EXPECT_TRUE(system_->stop_mcp_server());
}
