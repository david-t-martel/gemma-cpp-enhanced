// Focused session workflow tests extracted from monolithic file
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <filesystem>
#include <fstream>
#include "mock_system.h"
#include "../utils/test_helpers.h"

using namespace testing;

class SessionWorkflowTest : public gemma::test::GemmaTestBase {
protected:
    void SetUp() override {
        GemmaTestBase::SetUp();
        system_ = std::make_unique<MockGemmaSystem>();
        config_.model_weights_path = (test_dir_ / "test_model.sbs").string();
        config_.tokenizer_path = (test_dir_ / "tokenizer.spm").string();
        config_.session_storage_path = (test_dir_ / "sessions").string();
        std::filesystem::create_directories(config_.session_storage_path);
        std::ofstream(config_.model_weights_path) << "dummy";
        std::ofstream(config_.tokenizer_path) << "dummy";

        ON_CALL(*system_, initialize(::testing::_)).WillByDefault(Return(true));
        ON_CALL(*system_, load_model(::testing::_, ::testing::_)).WillByDefault(Return(true));
        system_->initialize(config_);
        system_->load_model(config_.model_weights_path, config_.tokenizer_path);
    }

    std::unique_ptr<MockGemmaSystem> system_;
    SystemConfig config_;
};

TEST_F(SessionWorkflowTest, CreateAddMessagesListAndDelete) {
    nlohmann::json session_opts = {{"max_context_tokens", 2048}};
    EXPECT_CALL(*system_, create_session(session_opts))
        .Times(1).WillOnce(Return("session-123"));
    auto sid = system_->create_session(session_opts);
    EXPECT_EQ(sid, "session-123");

    EXPECT_CALL(*system_, add_message_to_session(sid, "user", "Hello"))
        .Times(1).WillOnce(Return(true));
    EXPECT_TRUE(system_->add_message_to_session(sid, "user", "Hello"));

    EXPECT_CALL(*system_, add_message_to_session(sid, "assistant", ::testing::_))
        .Times(1).WillOnce(Return(true));
    EXPECT_TRUE(system_->add_message_to_session(sid, "assistant", "Hi!"));

    nlohmann::json history = nlohmann::json::array({
        {{"role","user"},{"content","Hello"}},
        {{"role","assistant"},{"content","Hi!"}}
    });
    EXPECT_CALL(*system_, get_session_history(sid))
        .Times(1).WillOnce(Return(history));
    auto h = system_->get_session_history(sid);
    EXPECT_EQ(h.size(), 2);

    EXPECT_CALL(*system_, list_sessions())
        .Times(1).WillOnce(Return(std::vector<std::string>{sid}));
    auto sessions = system_->list_sessions();
    EXPECT_EQ(sessions.size(), 1);

    EXPECT_CALL(*system_, delete_session(sid))
        .Times(1).WillOnce(Return(true));
    EXPECT_TRUE(system_->delete_session(sid));
}
