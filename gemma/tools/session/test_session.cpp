#include "SessionManager.h"
#include <cassert>
#include <iostream>
#include <filesystem>
#include <thread>
#include <chrono>

using namespace gemma::session;

// Test helper macros
#define ASSERT_TRUE(condition) \
    do { \
        if (!(condition)) { \
            std::cerr << "ASSERTION FAILED: " << #condition << " at line " << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

#define ASSERT_FALSE(condition) ASSERT_TRUE(!(condition))
#define ASSERT_EQ(a, b) ASSERT_TRUE((a) == (b))
#define ASSERT_NE(a, b) ASSERT_TRUE((a) != (b))

/**
 * @brief Simple test suite for the session management system
 */
class SessionManagerTest {
private:
    std::string test_db_path_;
    std::unique_ptr<SessionManager> manager_;
    
public:
    SessionManagerTest() : test_db_path_("test_sessions.db") {
        // Clean up any previous test database
        std::filesystem::remove(test_db_path_);
    }
    
    ~SessionManagerTest() {
        // Clean up test database
        std::filesystem::remove(test_db_path_);
        std::filesystem::remove("exported_test_sessions.json");
    }
    
    bool run_all_tests() {
        std::cout << "Running Session Management Tests..." << std::endl;
        
        bool all_passed = true;
        
        all_passed &= test_initialization();
        all_passed &= test_session_creation();
        all_passed &= test_message_handling();
        all_passed &= test_session_persistence();
        all_passed &= test_session_deletion();
        all_passed &= test_context_management();
        all_passed &= test_export_import();
        all_passed &= test_statistics();
        all_passed &= test_thread_safety();
        
        if (all_passed) {
            std::cout << "All tests PASSED!" << std::endl;
        } else {
            std::cout << "Some tests FAILED!" << std::endl;
        }
        
        return all_passed;
    }
    
private:
    bool test_initialization() {
        std::cout << "Testing initialization..." << std::endl;
        
        SessionManager::Config config;
        config.storage_config.db_path = test_db_path_;
        config.storage_config.cache_capacity = 10;
        
        manager_ = std::make_unique<SessionManager>(config);
        ASSERT_TRUE(manager_->initialize());
        
        std::cout << "✓ Initialization test passed" << std::endl;
        return true;
    }
    
    bool test_session_creation() {
        std::cout << "Testing session creation..." << std::endl;
        
        // Test auto-generated session ID
        std::string session_id1 = manager_->create_session();
        ASSERT_FALSE(session_id1.empty());
        ASSERT_TRUE(manager_->session_exists(session_id1));
        
        // Test custom session ID
        SessionManager::CreateOptions opts;
        opts.session_id = "custom-test-session";
        opts.max_context_tokens = 1024;
        
        std::string session_id2 = manager_->create_session(opts);
        ASSERT_EQ(session_id2, "custom-test-session");
        ASSERT_TRUE(manager_->session_exists(session_id2));
        
        // Test duplicate session ID should throw
        try {
            manager_->create_session(opts);
            ASSERT_TRUE(false); // Should not reach here
        } catch (const std::runtime_error&) {
            // Expected exception
        }
        
        std::cout << "✓ Session creation test passed" << std::endl;
        return true;
    }
    
    bool test_message_handling() {
        std::cout << "Testing message handling..." << std::endl;
        
        std::string session_id = manager_->create_session();
        
        // Add messages
        ASSERT_TRUE(manager_->add_message(session_id, ConversationMessage::Role::USER, 
                                        "Hello", 3));
        ASSERT_TRUE(manager_->add_message(session_id, ConversationMessage::Role::ASSISTANT, 
                                        "Hi there!", 4));
        ASSERT_TRUE(manager_->add_message(session_id, ConversationMessage::Role::USER, 
                                        "How are you?", 5));
        
        // Get conversation history
        auto history = manager_->get_conversation_history(session_id);
        ASSERT_EQ(history.size(), 3);
        
        ASSERT_EQ(history[0].role, ConversationMessage::Role::USER);
        ASSERT_EQ(history[0].content, "Hello");
        ASSERT_EQ(history[0].token_count, 3);
        
        ASSERT_EQ(history[1].role, ConversationMessage::Role::ASSISTANT);
        ASSERT_EQ(history[1].content, "Hi there!");
        ASSERT_EQ(history[1].token_count, 4);
        
        // Test context messages
        auto context = manager_->get_context_messages(session_id);
        ASSERT_EQ(context.size(), 3); // All messages should fit in default context
        
        std::cout << "✓ Message handling test passed" << std::endl;
        return true;
    }
    
    bool test_session_persistence() {
        std::cout << "Testing session persistence..." << std::endl;
        
        std::string session_id = "persistence-test";
        SessionManager::CreateOptions opts;
        opts.session_id = session_id;
        
        manager_->create_session(opts);
        manager_->add_message(session_id, ConversationMessage::Role::USER, 
                            "Test persistence", 10);
        
        // Create new manager instance (simulate restart)
        SessionManager::Config config;
        config.storage_config.db_path = test_db_path_;
        
        SessionManager manager2(config);
        ASSERT_TRUE(manager2.initialize());
        
        // Load session
        auto session = manager2.get_session(session_id);
        ASSERT_TRUE(session != nullptr);
        ASSERT_EQ(session->get_session_id(), session_id);
        ASSERT_EQ(session->get_conversation_history().size(), 1);
        ASSERT_EQ(session->get_total_tokens(), 10);
        
        std::cout << "✓ Session persistence test passed" << std::endl;
        return true;
    }
    
    bool test_session_deletion() {
        std::cout << "Testing session deletion..." << std::endl;
        
        std::string session_id = manager_->create_session();
        ASSERT_TRUE(manager_->session_exists(session_id));
        
        ASSERT_TRUE(manager_->delete_session(session_id));
        ASSERT_FALSE(manager_->session_exists(session_id));
        
        // Deleting non-existent session should return false
        ASSERT_FALSE(manager_->delete_session("non-existent"));
        
        std::cout << "✓ Session deletion test passed" << std::endl;
        return true;
    }
    
    bool test_context_management() {
        std::cout << "Testing context management..." << std::endl;
        
        SessionManager::CreateOptions opts;
        opts.max_context_tokens = 10; // Very small context for testing
        std::string session_id = manager_->create_session(opts);
        
        // Add messages that exceed context
        manager_->add_message(session_id, ConversationMessage::Role::USER, "First", 5);
        manager_->add_message(session_id, ConversationMessage::Role::USER, "Second", 5);
        manager_->add_message(session_id, ConversationMessage::Role::USER, "Third", 5);
        
        auto history = manager_->get_conversation_history(session_id);
        ASSERT_EQ(history.size(), 3);
        
        auto context = manager_->get_context_messages(session_id);
        ASSERT_EQ(context.size(), 2); // Only last 2 messages should fit
        
        // Update context size
        ASSERT_TRUE(manager_->update_session_context_size(session_id, 20));
        context = manager_->get_context_messages(session_id);
        ASSERT_EQ(context.size(), 3); // Now all messages should fit
        
        std::cout << "✓ Context management test passed" << std::endl;
        return true;
    }
    
    bool test_export_import() {
        std::cout << "Testing export/import..." << std::endl;
        
        // Create session with data
        std::string session_id = "export-test";
        SessionManager::CreateOptions opts;
        opts.session_id = session_id;
        
        manager_->create_session(opts);
        manager_->add_message(session_id, ConversationMessage::Role::USER, "Export test", 8);
        
        // Export
        ASSERT_TRUE(manager_->export_sessions("exported_test_sessions.json"));
        
        // Delete session
        manager_->delete_session(session_id);
        ASSERT_FALSE(manager_->session_exists(session_id));
        
        // Import
        size_t imported = manager_->import_sessions("exported_test_sessions.json");
        ASSERT_EQ(imported, 1);
        ASSERT_TRUE(manager_->session_exists(session_id));
        
        // Verify imported data
        auto history = manager_->get_conversation_history(session_id);
        ASSERT_EQ(history.size(), 1);
        ASSERT_EQ(history[0].content, "Export test");
        
        std::cout << "✓ Export/import test passed" << std::endl;
        return true;
    }
    
    bool test_statistics() {
        std::cout << "Testing statistics..." << std::endl;
        
        auto initial_stats = manager_->get_statistics();
        size_t initial_sessions = initial_stats["metrics"]["total_sessions_created"].get<size_t>();
        
        // Create session and add messages
        std::string session_id = manager_->create_session();
        manager_->add_message(session_id, ConversationMessage::Role::USER, "Stats test", 5);
        
        auto stats = manager_->get_statistics();
        ASSERT_EQ(stats["metrics"]["total_sessions_created"].get<size_t>(), initial_sessions + 1);
        ASSERT_EQ(stats["metrics"]["total_messages_processed"].get<size_t>(), 1);
        ASSERT_EQ(stats["metrics"]["total_tokens_processed"].get<size_t>(), 5);
        
        std::cout << "✓ Statistics test passed" << std::endl;
        return true;
    }
    
    bool test_thread_safety() {
        std::cout << "Testing thread safety..." << std::endl;
        
        const int num_threads = 4;
        const int messages_per_thread = 10;
        std::vector<std::thread> threads;
        std::vector<std::string> session_ids;
        
        // Create sessions for each thread
        for (int i = 0; i < num_threads; ++i) {
            session_ids.push_back(manager_->create_session());
        }
        
        // Launch threads that add messages concurrently
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([this, &session_ids, i, messages_per_thread]() {
                for (int j = 0; j < messages_per_thread; ++j) {
                    std::string content = "Thread " + std::to_string(i) + " Message " + std::to_string(j);
                    manager_->add_message(session_ids[i], ConversationMessage::Role::USER, content, 5);
                    
                    // Small delay to increase chance of race conditions
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Verify all messages were added correctly
        for (int i = 0; i < num_threads; ++i) {
            auto history = manager_->get_conversation_history(session_ids[i]);
            ASSERT_EQ(history.size(), messages_per_thread);
            
            for (int j = 0; j < messages_per_thread; ++j) {
                std::string expected = "Thread " + std::to_string(i) + " Message " + std::to_string(j);
                ASSERT_EQ(history[j].content, expected);
            }
        }
        
        std::cout << "✓ Thread safety test passed" << std::endl;
        return true;
    }
};

int main() {
    SessionManagerTest test;
    bool success = test.run_all_tests();
    return success ? 0 : 1;
}