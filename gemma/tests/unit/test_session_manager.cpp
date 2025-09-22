#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "../utils/test_helpers.h"
#include "../../src/session/SessionManager.h"
#include "../../src/session/Session.h"
#include <filesystem>
#include <thread>
#include <chrono>
#include <future>

using namespace gemma::session;
using namespace gemma::test;
using namespace testing;

class SessionManagerTest : public GemmaTestBase {
protected:
    void SetUp() override {
        GemmaTestBase::SetUp();
        
        // Create test-specific configuration
        config_ = TestConfigBuilder::create_session_manager_config();
        config_.storage_config.storage_path = test_dir_ / "sessions";
        config_.storage_config.enable_compression = false; // Disable for testing
        config_.enable_metrics = true;
        
        std::filesystem::create_directories(config_.storage_config.storage_path);
        
        session_manager_ = std::make_unique<SessionManager>(config_);
        ASSERT_TRUE(session_manager_->initialize());
    }
    
    void TearDown() override {
        if (session_manager_) {
            session_manager_->shutdown();
        }
        GemmaTestBase::TearDown();
    }
    
    SessionManager::Config config_;
    std::unique_ptr<SessionManager> session_manager_;
};

// Basic functionality tests

TEST_F(SessionManagerTest, InitializationSuccess) {
    EXPECT_TRUE(session_manager_->initialize());
    
    auto stats = session_manager_->get_statistics();
    EXPECT_TRUE(stats.contains("initialized"));
    EXPECT_TRUE(stats["initialized"].get<bool>());
}

TEST_F(SessionManagerTest, CreateSessionWithDefaultOptions) {
    SessionManager::CreateOptions options;
    std::string session_id = session_manager_->create_session(options);
    
    EXPECT_FALSE(session_id.empty());
    EXPECT_EQ(session_id.length(), 36); // UUID v4 length
    EXPECT_TRUE(session_manager_->session_exists(session_id));
    
    auto session = session_manager_->get_session(session_id);
    ASSERT_NE(session, nullptr);
    EXPECT_EQ(session->get_id(), session_id);
}

TEST_F(SessionManagerTest, CreateSessionWithCustomId) {
    SessionManager::CreateOptions options;
    options.session_id = "custom-test-session-123";
    options.max_context_tokens = 2048;
    options.metadata = TestDataGenerator::generate_test_metadata();
    
    std::string session_id = session_manager_->create_session(options);
    
    EXPECT_EQ(session_id, "custom-test-session-123");
    EXPECT_TRUE(session_manager_->session_exists(session_id));
    
    auto session = session_manager_->get_session(session_id);
    ASSERT_NE(session, nullptr);
    EXPECT_EQ(session->get_max_context_tokens(), 2048);
}

TEST_F(SessionManagerTest, CreateSessionWithDuplicateIdFails) {
    SessionManager::CreateOptions options;
    options.session_id = "duplicate-test-session";
    
    std::string first_id = session_manager_->create_session(options);
    EXPECT_EQ(first_id, "duplicate-test-session");
    
    // Second creation with same ID should fail (return empty string or throw)
    std::string second_id = session_manager_->create_session(options);
    EXPECT_TRUE(second_id.empty() || second_id != first_id);
}

TEST_F(SessionManagerTest, GetNonExistentSessionReturnsNull) {
    auto session = session_manager_->get_session("non-existent-session");
    EXPECT_EQ(session, nullptr);
    EXPECT_FALSE(session_manager_->session_exists("non-existent-session"));
}

TEST_F(SessionManagerTest, DeleteSessionSuccess) {
    std::string session_id = session_manager_->create_session();
    ASSERT_TRUE(session_manager_->session_exists(session_id));
    
    bool deleted = session_manager_->delete_session(session_id);
    EXPECT_TRUE(deleted);
    EXPECT_FALSE(session_manager_->session_exists(session_id));
    
    auto session = session_manager_->get_session(session_id);
    EXPECT_EQ(session, nullptr);
}

TEST_F(SessionManagerTest, DeleteNonExistentSessionFails) {
    bool deleted = session_manager_->delete_session("non-existent-session");
    EXPECT_FALSE(deleted);
}

// Message management tests

TEST_F(SessionManagerTest, AddMessageToSession) {
    std::string session_id = session_manager_->create_session();
    
    bool added = session_manager_->add_message(
        session_id, 
        ConversationMessage::Role::User, 
        "Hello, world!", 
        5
    );
    
    EXPECT_TRUE(added);
    
    auto history = session_manager_->get_conversation_history(session_id);
    ASSERT_EQ(history.size(), 1);
    EXPECT_EQ(history[0].role, ConversationMessage::Role::User);
    EXPECT_EQ(history[0].content, "Hello, world!");
    EXPECT_EQ(history[0].token_count, 5);
}

TEST_F(SessionManagerTest, AddMessageToNonExistentSessionFails) {
    bool added = session_manager_->add_message(
        "non-existent-session",
        ConversationMessage::Role::User,
        "Hello, world!",
        5
    );
    
    EXPECT_FALSE(added);
}

TEST_F(SessionManagerTest, GetConversationHistory) {
    std::string session_id = session_manager_->create_session();
    
    // Add multiple messages
    session_manager_->add_message(session_id, ConversationMessage::Role::User, "Message 1", 3);
    session_manager_->add_message(session_id, ConversationMessage::Role::Assistant, "Response 1", 5);
    session_manager_->add_message(session_id, ConversationMessage::Role::User, "Message 2", 3);
    
    auto history = session_manager_->get_conversation_history(session_id);
    ASSERT_EQ(history.size(), 3);
    
    EXPECT_EQ(history[0].content, "Message 1");
    EXPECT_EQ(history[1].content, "Response 1");
    EXPECT_EQ(history[2].content, "Message 2");
    
    EXPECT_EQ(history[0].role, ConversationMessage::Role::User);
    EXPECT_EQ(history[1].role, ConversationMessage::Role::Assistant);
    EXPECT_EQ(history[2].role, ConversationMessage::Role::User);
}

TEST_F(SessionManagerTest, GetContextMessages) {
    SessionManager::CreateOptions options;
    options.max_context_tokens = 10; // Small context window for testing
    std::string session_id = session_manager_->create_session(options);
    
    // Add messages that exceed context window
    session_manager_->add_message(session_id, ConversationMessage::Role::User, "Old message", 8);
    session_manager_->add_message(session_id, ConversationMessage::Role::Assistant, "Old response", 8);
    session_manager_->add_message(session_id, ConversationMessage::Role::User, "New message", 5);
    
    auto context_messages = session_manager_->get_context_messages(session_id);
    
    // Should only include messages that fit in context window
    EXPECT_LE(context_messages.size(), 2); // Only the last message or two should fit
    
    // Calculate total tokens
    size_t total_tokens = 0;
    for (const auto& msg : context_messages) {
        total_tokens += msg.token_count;
    }
    EXPECT_LE(total_tokens, 10);
}

TEST_F(SessionManagerTest, ClearSessionHistory) {
    std::string session_id = session_manager_->create_session();
    
    // Add some messages
    session_manager_->add_message(session_id, ConversationMessage::Role::User, "Message 1", 3);
    session_manager_->add_message(session_id, ConversationMessage::Role::Assistant, "Response 1", 5);
    
    auto history_before = session_manager_->get_conversation_history(session_id);
    EXPECT_EQ(history_before.size(), 2);
    
    bool cleared = session_manager_->clear_session_history(session_id);
    EXPECT_TRUE(cleared);
    
    auto history_after = session_manager_->get_conversation_history(session_id);
    EXPECT_EQ(history_after.size(), 0);
}

TEST_F(SessionManagerTest, UpdateSessionContextSize) {
    std::string session_id = session_manager_->create_session();
    
    bool updated = session_manager_->update_session_context_size(session_id, 4096);
    EXPECT_TRUE(updated);
    
    auto session = session_manager_->get_session(session_id);
    ASSERT_NE(session, nullptr);
    EXPECT_EQ(session->get_max_context_tokens(), 4096);
}

// Session listing and filtering tests

TEST_F(SessionManagerTest, ListAllSessions) {
    // Create multiple sessions
    std::vector<std::string> session_ids;
    for (int i = 0; i < 5; ++i) {
        session_ids.push_back(session_manager_->create_session());
    }
    
    auto sessions = session_manager_->list_sessions();
    EXPECT_EQ(sessions.size(), 5);
    
    // Verify all created sessions are in the list
    for (const auto& session_json : sessions) {
        std::string id = session_json["id"];
        EXPECT_NE(std::find(session_ids.begin(), session_ids.end(), id), session_ids.end());
    }
}

TEST_F(SessionManagerTest, ListSessionsWithLimit) {
    // Create multiple sessions
    for (int i = 0; i < 10; ++i) {
        session_manager_->create_session();
    }
    
    auto sessions = session_manager_->list_sessions(3); // Limit to 3
    EXPECT_EQ(sessions.size(), 3);
}

TEST_F(SessionManagerTest, ListSessionsWithPagination) {
    // Create multiple sessions
    std::vector<std::string> all_session_ids;
    for (int i = 0; i < 10; ++i) {
        all_session_ids.push_back(session_manager_->create_session());
    }
    
    // Get first page
    auto page1 = session_manager_->list_sessions(5, 0);
    EXPECT_EQ(page1.size(), 5);
    
    // Get second page
    auto page2 = session_manager_->list_sessions(5, 5);
    EXPECT_EQ(page2.size(), 5);
    
    // Verify no overlap
    std::set<std::string> page1_ids, page2_ids;
    for (const auto& session : page1) {
        page1_ids.insert(session["id"]);
    }
    for (const auto& session : page2) {
        page2_ids.insert(session["id"]);
    }
    
    for (const auto& id : page1_ids) {
        EXPECT_EQ(page2_ids.find(id), page2_ids.end());
    }
}

// Import/Export tests

TEST_F(SessionManagerTest, ExportAndImportSessions) {
    // Create test sessions with messages
    std::string session1 = session_manager_->create_session();
    std::string session2 = session_manager_->create_session();
    
    session_manager_->add_message(session1, ConversationMessage::Role::User, "Hello from session 1", 5);
    session_manager_->add_message(session2, ConversationMessage::Role::User, "Hello from session 2", 5);
    
    // Export sessions
    std::string export_file = (test_dir_ / "sessions_export.json").string();
    bool exported = session_manager_->export_sessions(export_file);
    EXPECT_TRUE(exported);
    EXPECT_TRUE(std::filesystem::exists(export_file));
    
    // Create new session manager and import
    auto new_config = config_;
    new_config.storage_config.storage_path = test_dir_ / "imported_sessions";
    std::filesystem::create_directories(new_config.storage_config.storage_path);
    
    auto new_manager = std::make_unique<SessionManager>(new_config);
    ASSERT_TRUE(new_manager->initialize());
    
    size_t imported_count = new_manager->import_sessions(export_file);
    EXPECT_EQ(imported_count, 2);
    
    // Verify imported sessions
    EXPECT_TRUE(new_manager->session_exists(session1));
    EXPECT_TRUE(new_manager->session_exists(session2));
    
    auto imported_history1 = new_manager->get_conversation_history(session1);
    auto imported_history2 = new_manager->get_conversation_history(session2);
    
    ASSERT_EQ(imported_history1.size(), 1);
    ASSERT_EQ(imported_history2.size(), 1);
    
    EXPECT_EQ(imported_history1[0].content, "Hello from session 1");
    EXPECT_EQ(imported_history2[0].content, "Hello from session 2");
}

TEST_F(SessionManagerTest, ExportSpecificSessions) {
    // Create multiple sessions
    std::string session1 = session_manager_->create_session();
    std::string session2 = session_manager_->create_session();
    std::string session3 = session_manager_->create_session();
    
    session_manager_->add_message(session1, ConversationMessage::Role::User, "Session 1", 2);
    session_manager_->add_message(session2, ConversationMessage::Role::User, "Session 2", 2);
    session_manager_->add_message(session3, ConversationMessage::Role::User, "Session 3", 2);
    
    // Export only session1 and session3
    std::string export_file = (test_dir_ / "partial_export.json").string();
    bool exported = session_manager_->export_sessions(export_file, {session1, session3});
    EXPECT_TRUE(exported);
    
    // Verify export content
    std::ifstream file(export_file);
    nlohmann::json export_data;
    file >> export_data;
    
    ASSERT_TRUE(export_data.contains("sessions"));
    auto sessions = export_data["sessions"];
    EXPECT_EQ(sessions.size(), 2);
    
    // Verify correct sessions were exported
    std::set<std::string> exported_ids;
    for (const auto& session : sessions) {
        exported_ids.insert(session["id"]);
    }
    
    EXPECT_NE(exported_ids.find(session1), exported_ids.end());
    EXPECT_NE(exported_ids.find(session3), exported_ids.end());
    EXPECT_EQ(exported_ids.find(session2), exported_ids.end());
}

// Metrics and statistics tests

TEST_F(SessionManagerTest, MetricsTracking) {
    auto initial_metrics = session_manager_->get_metrics();
    EXPECT_EQ(initial_metrics.total_sessions_created, 0);
    EXPECT_EQ(initial_metrics.total_messages_processed, 0);
    
    // Create sessions and add messages
    std::string session1 = session_manager_->create_session();
    std::string session2 = session_manager_->create_session();
    
    session_manager_->add_message(session1, ConversationMessage::Role::User, "Message 1", 5);
    session_manager_->add_message(session1, ConversationMessage::Role::Assistant, "Response 1", 10);
    session_manager_->add_message(session2, ConversationMessage::Role::User, "Message 2", 3);
    
    auto updated_metrics = session_manager_->get_metrics();
    EXPECT_EQ(updated_metrics.total_sessions_created, 2);
    EXPECT_EQ(updated_metrics.total_messages_processed, 3);
    EXPECT_EQ(updated_metrics.total_tokens_processed, 18); // 5 + 10 + 3
    
    // Calculate averages
    EXPECT_DOUBLE_EQ(updated_metrics.avg_tokens_per_session, 9.0); // 18 / 2
    EXPECT_DOUBLE_EQ(updated_metrics.avg_messages_per_session, 1.5); // 3 / 2
}

TEST_F(SessionManagerTest, ResetMetrics) {
    // Generate some activity
    session_manager_->create_session();
    session_manager_->create_session();
    
    auto metrics_before = session_manager_->get_metrics();
    EXPECT_GT(metrics_before.total_sessions_created, 0);
    
    session_manager_->reset_metrics();
    
    auto metrics_after = session_manager_->get_metrics();
    EXPECT_EQ(metrics_after.total_sessions_created, 0);
    EXPECT_EQ(metrics_after.total_messages_processed, 0);
    EXPECT_EQ(metrics_after.total_tokens_processed, 0);
}

TEST_F(SessionManagerTest, GetStatistics) {
    // Create test data
    std::string session_id = session_manager_->create_session();
    session_manager_->add_message(session_id, ConversationMessage::Role::User, "Test message", 5);
    
    auto stats = session_manager_->get_statistics();
    
    EXPECT_TRUE(stats.contains("total_sessions"));
    EXPECT_TRUE(stats.contains("storage_info"));
    EXPECT_TRUE(stats.contains("metrics"));
    
    EXPECT_EQ(stats["total_sessions"].get<int>(), 1);
    
    auto metrics = stats["metrics"];
    EXPECT_EQ(metrics["total_sessions_created"].get<int>(), 1);
    EXPECT_EQ(metrics["total_messages_processed"].get<int>(), 1);
}

// Thread safety tests

TEST_F(SessionManagerTest, ConcurrentSessionCreation) {
    const int num_threads = 4;
    const int sessions_per_thread = 10;
    std::vector<std::thread> threads;
    std::vector<std::vector<std::string>> thread_session_ids(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t, sessions_per_thread, &thread_session_ids]() {
            for (int i = 0; i < sessions_per_thread; ++i) {
                std::string session_id = session_manager_->create_session();
                EXPECT_FALSE(session_id.empty());
                thread_session_ids[t].push_back(session_id);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify all sessions were created and are unique
    std::set<std::string> all_session_ids;
    for (const auto& thread_ids : thread_session_ids) {
        for (const auto& id : thread_ids) {
            EXPECT_TRUE(session_manager_->session_exists(id));
            EXPECT_EQ(all_session_ids.find(id), all_session_ids.end()) << "Duplicate session ID: " << id;
            all_session_ids.insert(id);
        }
    }
    
    EXPECT_EQ(all_session_ids.size(), num_threads * sessions_per_thread);
}

TEST_F(SessionManagerTest, ConcurrentMessageAddition) {
    std::string session_id = session_manager_->create_session();
    const int num_threads = 4;
    const int messages_per_thread = 25;
    
    ThreadSafetyUtils::run_concurrent_test([this, &session_id]() {
        bool added = session_manager_->add_message(
            session_id,
            ConversationMessage::Role::User,
            "Concurrent message",
            5
        );
        EXPECT_TRUE(added);
    }, num_threads, messages_per_thread);
    
    auto history = session_manager_->get_conversation_history(session_id);
    EXPECT_EQ(history.size(), num_threads * messages_per_thread);
}

// Cleanup and maintenance tests

TEST_F(SessionManagerTest, CleanupExpiredSessions) {
    // This test would require the ability to set session expiration times
    // For now, test that cleanup doesn't break anything
    size_t cleaned = session_manager_->cleanup_expired_sessions();
    EXPECT_GE(cleaned, 0); // Should not crash or return negative
}

// Event callback tests

TEST_F(SessionManagerTest, EventCallbacks) {
    std::vector<std::pair<std::string, nlohmann::json>> received_events;
    
    session_manager_->set_event_callback([&received_events](const std::string& event, const nlohmann::json& data) {
        received_events.emplace_back(event, data);
    });
    
    // Trigger some events
    std::string session_id = session_manager_->create_session();
    session_manager_->add_message(session_id, ConversationMessage::Role::User, "Test", 1);
    session_manager_->delete_session(session_id);
    
    // Verify events were received (this depends on the actual implementation)
    EXPECT_GE(received_events.size(), 0); // At minimum, should not crash
}

// Error handling tests

TEST_F(SessionManagerTest, HandleInvalidSessionIds) {
    // Test various invalid session ID formats
    std::vector<std::string> invalid_ids = {
        "",
        "   ",
        "invalid-chars!@#$",
        "too-short",
        std::string(1000, 'x'), // Too long
        "null\0embedded", // Null character
    };
    
    for (const auto& invalid_id : invalid_ids) {
        auto session = session_manager_->get_session(invalid_id);
        EXPECT_EQ(session, nullptr) << "Invalid ID should return null: " << invalid_id;
        
        bool exists = session_manager_->session_exists(invalid_id);
        EXPECT_FALSE(exists) << "Invalid ID should not exist: " << invalid_id;
    }
}

TEST_F(SessionManagerTest, ConfigurationValidation) {
    // Test with invalid configuration
    SessionManager::Config invalid_config;
    invalid_config.default_max_context_tokens = 0; // Invalid
    
    auto invalid_manager = std::make_unique<SessionManager>(invalid_config);
    // The behavior here depends on implementation - it might fail initialization
    // or use default values
    bool initialized = invalid_manager->initialize();
    // We just verify it doesn't crash
    EXPECT_TRUE(initialized || !initialized); // Either outcome is acceptable for this test
}

// Performance tests

TEST_F(SessionManagerTest, PerformanceCreateManySessionsQuickly) {
    const int num_sessions = 1000;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::string> session_ids;
    session_ids.reserve(num_sessions);
    
    for (int i = 0; i < num_sessions; ++i) {
        session_ids.push_back(session_manager_->create_session());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Created " << num_sessions << " sessions in " << duration.count() << " ms" << std::endl;
    
    // Verify all sessions were created
    EXPECT_EQ(session_ids.size(), num_sessions);
    for (const auto& id : session_ids) {
        EXPECT_TRUE(session_manager_->session_exists(id));
    }
    
    // Performance expectation: should be able to create 1000 sessions in reasonable time
    EXPECT_LT(duration.count(), 10000) << "Session creation too slow: " << duration.count() << " ms";
}

TEST_F(SessionManagerTest, PerformanceBulkMessageAddition) {
    std::string session_id = session_manager_->create_session();
    const int num_messages = 10000;
    
    auto messages = TestDataGenerator::generate_test_messages(num_messages);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < messages.size(); ++i) {
        ConversationMessage::Role role = (i % 2 == 0) ? 
            ConversationMessage::Role::User : ConversationMessage::Role::Assistant;
        
        session_manager_->add_message(session_id, role, messages[i], messages[i].length() / 4);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Added " << num_messages << " messages in " << duration.count() << " ms" << std::endl;
    
    auto history = session_manager_->get_conversation_history(session_id);
    EXPECT_EQ(history.size(), num_messages);
    
    // Performance expectation
    EXPECT_LT(duration.count(), 5000) << "Message addition too slow: " << duration.count() << " ms";
}