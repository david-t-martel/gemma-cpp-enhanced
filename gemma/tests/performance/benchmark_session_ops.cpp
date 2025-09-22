#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <benchmark/benchmark.h>
#include "../utils/test_helpers.h"
#include <nlohmann/json.hpp>
#include <memory>
#include <vector>
#include <chrono>
#include <string>
#include <random>
#include <algorithm>
#include <thread>
#include <future>
#include <atomic>

// Performance benchmarks for session operations
// Tests session creation, message handling, persistence, and cleanup

using namespace gemma::test;
using namespace testing;
using json = nlohmann::json;

// Mock session data for benchmarking
struct BenchmarkSessionData {
    std::string id;
    std::string created_at;
    std::string last_activity;
    size_t max_context_tokens;
    json metadata;
    std::vector<json> messages;
    
    static BenchmarkSessionData create_test_session(const std::string& id) {
        BenchmarkSessionData session;
        session.id = id;
        session.created_at = "2024-01-01T10:00:00Z";
        session.last_activity = "2024-01-01T10:30:00Z";
        session.max_context_tokens = 8192;
        session.metadata = json{
            {"user_id", "user_" + std::to_string(std::hash<std::string>{}(id) % 1000)},
            {"session_type", "chat"},
            {"priority", "normal"}
        };
        return session;
    }
    
    void add_messages(size_t count) {
        messages.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            json user_message = {
                {"role", "user"},
                {"content", "User message " + std::to_string(i)},
                {"timestamp", "2024-01-01T10:15:" + std::to_string(i % 60) + "Z"},
                {"token_count", 10 + (i % 20)}
            };
            
            json assistant_message = {
                {"role", "assistant"},
                {"content", "Assistant response " + std::to_string(i)},
                {"timestamp", "2024-01-01T10:15:" + std::to_string((i + 1) % 60) + "Z"},
                {"token_count", 20 + (i % 30)}
            };
            
            messages.push_back(user_message);
            messages.push_back(assistant_message);
        }
    }
};

// Mock session manager for benchmarking
class MockSessionManagerForBenchmark {
public:
    MOCK_METHOD(std::string, create_session, (const json& options), ());
    MOCK_METHOD(bool, delete_session, (const std::string& session_id), ());
    MOCK_METHOD(bool, add_message, (const std::string& session_id, const std::string& role, const std::string& content), ());
    MOCK_METHOD(std::vector<json>, get_conversation_history, (const std::string& session_id), ());
    MOCK_METHOD(std::vector<json>, get_context_messages, (const std::string& session_id), ());
    MOCK_METHOD(bool, clear_session_history, (const std::string& session_id), ());
    MOCK_METHOD(std::vector<std::string>, list_sessions, (), ());
    MOCK_METHOD(json, get_session_metadata, (const std::string& session_id), ());
    MOCK_METHOD(bool, update_session_metadata, (const std::string& session_id, const json& metadata), ());
    MOCK_METHOD(bool, save_session_to_storage, (const std::string& session_id), ());
    MOCK_METHOD(bool, load_session_from_storage, (const std::string& session_id), ());
    MOCK_METHOD(size_t, cleanup_expired_sessions, (std::chrono::hours max_age), ());
    MOCK_METHOD(json, get_session_statistics, (), ());
};

// Session benchmark test fixture
class SessionBenchmarkTest : public GemmaTestBase {
protected:
    void SetUp() override {
        GemmaTestBase::SetUp();
        
        session_manager_ = std::make_unique<MockSessionManagerForBenchmark>();
        setup_session_manager_expectations();
        
        // Pre-generate test data
        generate_test_sessions();
        generate_test_messages();
    }
    
    void setup_session_manager_expectations() {
        // Default successful operations
        ON_CALL(*session_manager_, create_session(_))
            .WillByDefault(Invoke([this](const json& options) {
                std::string id = "session_" + std::to_string(session_counter_++);
                active_sessions_.insert(id);
                return id;
            }));
        
        ON_CALL(*session_manager_, delete_session(_))
            .WillByDefault(Invoke([this](const std::string& session_id) {
                active_sessions_.erase(session_id);
                return true;
            }));
        
        ON_CALL(*session_manager_, add_message(_, _, _))
            .WillByDefault(Return(true));
        
        ON_CALL(*session_manager_, get_conversation_history(_))
            .WillByDefault(Return(test_conversation_history_));
        
        ON_CALL(*session_manager_, list_sessions())
            .WillByDefault(Invoke([this]() {
                return std::vector<std::string>(active_sessions_.begin(), active_sessions_.end());
            }));
        
        ON_CALL(*session_manager_, save_session_to_storage(_))
            .WillByDefault(Return(true));
        
        ON_CALL(*session_manager_, load_session_from_storage(_))
            .WillByDefault(Return(true));
    }
    
    void generate_test_sessions() {
        test_sessions_.clear();
        for (int i = 0; i < 1000; ++i) {
            auto session = BenchmarkSessionData::create_test_session("bench_session_" + std::to_string(i));
            test_sessions_.push_back(session);
        }
    }
    
    void generate_test_messages() {
        test_conversation_history_.clear();
        for (int i = 0; i < 100; ++i) {
            test_conversation_history_.push_back(json{
                {"role", (i % 2 == 0) ? "user" : "assistant"},
                {"content", "Benchmark message " + std::to_string(i)},
                {"timestamp", "2024-01-01T10:00:" + std::to_string(i) + "Z"},
                {"token_count", 10 + (i % 20)}
            });
        }
        
        test_messages_.clear();
        for (int i = 0; i < 10000; ++i) {
            test_messages_.push_back("Test message " + std::to_string(i) + " for performance testing.");
        }
    }
    
    std::unique_ptr<MockSessionManagerForBenchmark> session_manager_;
    std::vector<BenchmarkSessionData> test_sessions_;
    std::vector<json> test_conversation_history_;
    std::vector<std::string> test_messages_;
    std::set<std::string> active_sessions_;
    std::atomic<int> session_counter_{0};
};

// Session creation performance benchmark
TEST_F(SessionBenchmarkTest, SessionCreationPerformance) {
    const int num_sessions = 1000;
    
    EXPECT_CALL(*session_manager_, create_session(_))
        .Times(num_sessions)
        .WillRepeatedly(Invoke([this](const json& options) {
            return "session_" + std::to_string(session_counter_++);
        }));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::string> created_sessions;
    created_sessions.reserve(num_sessions);
    
    for (int i = 0; i < num_sessions; ++i) {
        json options = {
            {"max_context_tokens", 4096},
            {"metadata", {
                {"user_id", "user_" + std::to_string(i)},
                {"session_type", "benchmark"}
            }}
        };
        
        std::string session_id = session_manager_->create_session(options);
        created_sessions.push_back(session_id);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    double sessions_per_second = (num_sessions * 1000.0) / duration.count();
    double avg_time_per_session = static_cast<double>(duration.count()) / num_sessions;
    
    std::cout << "Session creation performance:" << std::endl;
    std::cout << "  Created " << num_sessions << " sessions in " << duration.count() << " ms" << std::endl;
    std::cout << "  Average time per session: " << avg_time_per_session << " ms" << std::endl;
    std::cout << "  Sessions per second: " << sessions_per_second << std::endl;
    
    EXPECT_EQ(created_sessions.size(), num_sessions);
    EXPECT_GT(sessions_per_second, 100.0) << "Session creation rate too slow";
    EXPECT_LT(avg_time_per_session, 10.0) << "Individual session creation too slow";
}

// Message addition performance benchmark
TEST_F(SessionBenchmarkTest, MessageAdditionPerformance) {
    const int num_messages = 10000;
    const std::string session_id = "benchmark_session";
    
    EXPECT_CALL(*session_manager_, add_message(session_id, _, _))
        .Times(num_messages)
        .WillRepeatedly(Return(true));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_messages; ++i) {
        std::string role = (i % 2 == 0) ? "user" : "assistant";
        std::string content = test_messages_[i % test_messages_.size()];
        
        bool added = session_manager_->add_message(session_id, role, content);
        EXPECT_TRUE(added);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    double messages_per_second = (num_messages * 1000.0) / duration.count();
    double avg_time_per_message = static_cast<double>(duration.count()) / num_messages;
    
    std::cout << "Message addition performance:" << std::endl;
    std::cout << "  Added " << num_messages << " messages in " << duration.count() << " ms" << std::endl;
    std::cout << "  Average time per message: " << avg_time_per_message << " ms" << std::endl;
    std::cout << "  Messages per second: " << messages_per_second << std::endl;
    
    EXPECT_GT(messages_per_second, 1000.0) << "Message addition rate too slow";
    EXPECT_LT(avg_time_per_message, 1.0) << "Individual message addition too slow";
}

// Conversation history retrieval benchmark
TEST_F(SessionBenchmarkTest, ConversationHistoryRetrievalPerformance) {
    const int num_retrievals = 1000;
    const std::string session_id = "benchmark_session";
    
    // Create a large conversation history for testing
    std::vector<json> large_history;
    for (int i = 0; i < 5000; ++i) {
        large_history.push_back(json{
            {"role", (i % 2 == 0) ? "user" : "assistant"},
            {"content", "Large history message " + std::to_string(i)},
            {"timestamp", "2024-01-01T10:00:00Z"},
            {"token_count", 15}
        });
    }
    
    EXPECT_CALL(*session_manager_, get_conversation_history(session_id))
        .Times(num_retrievals)
        .WillRepeatedly(Return(large_history));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_retrievals; ++i) {
        auto history = session_manager_->get_conversation_history(session_id);
        EXPECT_EQ(history.size(), 5000);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    double retrievals_per_second = (num_retrievals * 1000.0) / duration.count();
    double avg_time_per_retrieval = static_cast<double>(duration.count()) / num_retrievals;
    
    std::cout << "History retrieval performance (5000 messages):" << std::endl;
    std::cout << "  " << num_retrievals << " retrievals in " << duration.count() << " ms" << std::endl;
    std::cout << "  Average time per retrieval: " << avg_time_per_retrieval << " ms" << std::endl;
    std::cout << "  Retrievals per second: " << retrievals_per_second << std::endl;
    
    EXPECT_GT(retrievals_per_second, 50.0) << "History retrieval rate too slow";
    EXPECT_LT(avg_time_per_retrieval, 20.0) << "Individual history retrieval too slow";
}

// Context window filtering performance benchmark
TEST_F(SessionBenchmarkTest, ContextWindowFilteringPerformance) {
    const int num_filterings = 1000;
    const std::string session_id = "benchmark_session";
    
    // Create conversation that exceeds context window
    std::vector<json> context_messages;
    for (int i = 0; i < 50; ++i) { // Simulate messages that fit in context
        context_messages.push_back(json{
            {"role", (i % 2 == 0) ? "user" : "assistant"},
            {"content", "Context message " + std::to_string(i)},
            {"token_count", 25}
        });
    }
    
    EXPECT_CALL(*session_manager_, get_context_messages(session_id))
        .Times(num_filterings)
        .WillRepeatedly(Return(context_messages));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_filterings; ++i) {
        auto context = session_manager_->get_context_messages(session_id);
        EXPECT_LE(context.size(), 50); // Should fit in context window
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    double filterings_per_second = (num_filterings * 1000.0) / duration.count();
    
    std::cout << "Context filtering performance:" << std::endl;
    std::cout << "  " << num_filterings << " filterings in " << duration.count() << " ms" << std::endl;
    std::cout << "  Filterings per second: " << filterings_per_second << std::endl;
    
    EXPECT_GT(filterings_per_second, 200.0) << "Context filtering rate too slow";
}

// Session listing performance benchmark
TEST_F(SessionBenchmarkTest, SessionListingPerformance) {
    const int num_listings = 1000;
    const int num_active_sessions = 10000;
    
    // Create a list of many session IDs
    std::vector<std::string> many_sessions;
    for (int i = 0; i < num_active_sessions; ++i) {
        many_sessions.push_back("session_" + std::to_string(i));
    }
    
    EXPECT_CALL(*session_manager_, list_sessions())
        .Times(num_listings)
        .WillRepeatedly(Return(many_sessions));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_listings; ++i) {
        auto sessions = session_manager_->list_sessions();
        EXPECT_EQ(sessions.size(), num_active_sessions);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    double listings_per_second = (num_listings * 1000.0) / duration.count();
    
    std::cout << "Session listing performance (" << num_active_sessions << " sessions):" << std::endl;
    std::cout << "  " << num_listings << " listings in " << duration.count() << " ms" << std::endl;
    std::cout << "  Listings per second: " << listings_per_second << std::endl;
    
    EXPECT_GT(listings_per_second, 100.0) << "Session listing rate too slow";
}

// Session persistence performance benchmark
TEST_F(SessionBenchmarkTest, SessionPersistencePerformance) {
    const int num_save_operations = 1000;
    const int num_load_operations = 1000;
    
    std::vector<std::string> session_ids;
    for (int i = 0; i < num_save_operations; ++i) {
        session_ids.push_back("persist_session_" + std::to_string(i));
    }
    
    // Benchmark session saving
    EXPECT_CALL(*session_manager_, save_session_to_storage(_))
        .Times(num_save_operations)
        .WillRepeatedly(Return(true));
    
    auto save_start = std::chrono::high_resolution_clock::now();
    
    for (const auto& session_id : session_ids) {
        bool saved = session_manager_->save_session_to_storage(session_id);
        EXPECT_TRUE(saved);
    }
    
    auto save_end = std::chrono::high_resolution_clock::now();
    auto save_duration = std::chrono::duration_cast<std::chrono::milliseconds>(save_end - save_start);
    
    // Benchmark session loading
    EXPECT_CALL(*session_manager_, load_session_from_storage(_))
        .Times(num_load_operations)
        .WillRepeatedly(Return(true));
    
    auto load_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_load_operations; ++i) {
        std::string session_id = session_ids[i % session_ids.size()];
        bool loaded = session_manager_->load_session_from_storage(session_id);
        EXPECT_TRUE(loaded);
    }
    
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);
    
    double saves_per_second = (num_save_operations * 1000.0) / save_duration.count();
    double loads_per_second = (num_load_operations * 1000.0) / load_duration.count();
    
    std::cout << "Session persistence performance:" << std::endl;
    std::cout << "  Save operations: " << saves_per_second << " sessions/s" << std::endl;
    std::cout << "  Load operations: " << loads_per_second << " sessions/s" << std::endl;
    
    EXPECT_GT(saves_per_second, 50.0) << "Session save rate too slow";
    EXPECT_GT(loads_per_second, 100.0) << "Session load rate too slow";
}

// Concurrent session operations benchmark
TEST_F(SessionBenchmarkTest, ConcurrentSessionOperationsPerformance) {
    const int num_threads = 8;
    const int operations_per_thread = 100;
    
    EXPECT_CALL(*session_manager_, create_session(_))
        .Times(num_threads * operations_per_thread)
        .WillRepeatedly(Invoke([this](const json& options) {
            return "concurrent_session_" + std::to_string(session_counter_++);
        }));
    
    EXPECT_CALL(*session_manager_, add_message(_, _, _))
        .Times(num_threads * operations_per_thread * 2) // 2 messages per session
        .WillRepeatedly(Return(true));
    
    EXPECT_CALL(*session_manager_, delete_session(_))
        .Times(num_threads * operations_per_thread)
        .WillRepeatedly(Return(true));
    
    std::atomic<int> successful_operations{0};
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < operations_per_thread; ++i) {
                try {
                    // Create session
                    json options = {{"max_context_tokens", 4096}};
                    std::string session_id = session_manager_->create_session(options);
                    
                    // Add messages
                    session_manager_->add_message(session_id, "user", "Test message 1");
                    session_manager_->add_message(session_id, "assistant", "Test response 1");
                    
                    // Delete session
                    session_manager_->delete_session(session_id);
                    
                    successful_operations++;
                } catch (...) {
                    // Count failures
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    double operations_per_second = (successful_operations * 1000.0) / duration.count();
    
    std::cout << "Concurrent operations performance:" << std::endl;
    std::cout << "  Successful operations: " << successful_operations << "/" << (num_threads * operations_per_thread) << std::endl;
    std::cout << "  Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Operations per second: " << operations_per_second << std::endl;
    
    EXPECT_EQ(successful_operations, num_threads * operations_per_thread)
        << "Some concurrent operations failed";
    EXPECT_GT(operations_per_second, 100.0) << "Concurrent operation rate too slow";
}

// Memory usage under load benchmark
TEST_F(SessionBenchmarkTest, MemoryUsageUnderLoadBenchmark) {
    const int num_sessions = 1000;
    const int messages_per_session = 100;
    
    // Simulate creating many sessions with many messages
    EXPECT_CALL(*session_manager_, create_session(_))
        .Times(num_sessions)
        .WillRepeatedly(Invoke([this](const json& options) {
            return "memory_session_" + std::to_string(session_counter_++);
        }));
    
    EXPECT_CALL(*session_manager_, add_message(_, _, _))
        .Times(num_sessions * messages_per_session)
        .WillRepeatedly(Return(true));
    
    size_t initial_memory = MemoryTestUtils::get_peak_memory_usage();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::string> session_ids;
    for (int i = 0; i < num_sessions; ++i) {
        json options = {{"max_context_tokens", 8192}};
        std::string session_id = session_manager_->create_session(options);
        session_ids.push_back(session_id);
        
        // Add messages to session
        for (int j = 0; j < messages_per_session; ++j) {
            std::string role = (j % 2 == 0) ? "user" : "assistant";
            std::string content = "Load test message " + std::to_string(j) + " with some content to simulate real usage.";
            session_manager_->add_message(session_id, role, content);
        }
        
        // Check memory usage every 100 sessions
        if (i % 100 == 0) {
            size_t current_memory = MemoryTestUtils::get_peak_memory_usage();
            size_t memory_growth = current_memory - initial_memory;
            
            std::cout << "After " << (i + 1) << " sessions: " 
                      << (memory_growth / (1024 * 1024)) << " MB additional memory" << std::endl;
            
            // Memory growth should be reasonable
            EXPECT_LT(memory_growth, 1ULL * 1024 * 1024 * 1024) // < 1GB growth
                << "Memory usage growing too quickly";
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    size_t final_memory = MemoryTestUtils::get_peak_memory_usage();
    size_t total_memory_growth = final_memory - initial_memory;
    
    std::cout << "Memory usage under load:" << std::endl;
    std::cout << "  Created " << num_sessions << " sessions with " << messages_per_session << " messages each" << std::endl;
    std::cout << "  Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Memory growth: " << (total_memory_growth / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "  Memory per session: " << (total_memory_growth / num_sessions) << " bytes" << std::endl;
    
    EXPECT_LT(total_memory_growth / num_sessions, 10 * 1024) // < 10KB per session
        << "Memory usage per session too high";
}

// Session cleanup performance benchmark
TEST_F(SessionBenchmarkTest, SessionCleanupPerformance) {
    const int num_sessions_to_cleanup = 5000;
    
    EXPECT_CALL(*session_manager_, cleanup_expired_sessions(std::chrono::hours{24}))
        .Times(1)
        .WillOnce(Return(num_sessions_to_cleanup));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    size_t cleaned_count = session_manager_->cleanup_expired_sessions(std::chrono::hours{24});
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    double cleanup_rate = (cleaned_count * 1000.0) / duration.count();
    
    std::cout << "Session cleanup performance:" << std::endl;
    std::cout << "  Cleaned " << cleaned_count << " sessions in " << duration.count() << " ms" << std::endl;
    std::cout << "  Cleanup rate: " << cleanup_rate << " sessions/s" << std::endl;
    
    EXPECT_EQ(cleaned_count, num_sessions_to_cleanup);
    EXPECT_GT(cleanup_rate, 500.0) << "Session cleanup rate too slow";
}

// Session statistics collection benchmark
TEST_F(SessionBenchmarkTest, SessionStatisticsPerformance) {
    const int num_stat_requests = 1000;
    
    json mock_statistics = {
        {"total_sessions", 10000},
        {"active_sessions", 2500},
        {"total_messages", 150000},
        {"average_session_length", 15.5},
        {"memory_usage_mb", 512},
        {"storage_size_mb", 1024}
    };
    
    EXPECT_CALL(*session_manager_, get_session_statistics())
        .Times(num_stat_requests)
        .WillRepeatedly(Return(mock_statistics));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_stat_requests; ++i) {
        auto stats = session_manager_->get_session_statistics();
        EXPECT_TRUE(stats.contains("total_sessions"));
        EXPECT_EQ(stats["total_sessions"], 10000);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    double stats_per_second = (num_stat_requests * 1000.0) / duration.count();
    
    std::cout << "Statistics collection performance:" << std::endl;
    std::cout << "  " << num_stat_requests << " requests in " << duration.count() << " ms" << std::endl;
    std::cout << "  Statistics requests per second: " << stats_per_second << std::endl;
    
    EXPECT_GT(stats_per_second, 1000.0) << "Statistics collection rate too slow";
}

// Stress test with realistic usage patterns
TEST_F(SessionBenchmarkTest, RealisticUsagePatternStressTest) {
    const int simulation_duration_seconds = 10;
    const double sessions_per_second = 5.0;
    const double messages_per_session_per_second = 1.0;
    
    std::atomic<bool> stop_simulation{false};
    std::atomic<int> operations_completed{0};
    
    // Setup expectations for realistic patterns
    EXPECT_CALL(*session_manager_, create_session(_))
        .WillRepeatedly(Invoke([this](const json& options) {
            return "realistic_session_" + std::to_string(session_counter_++);
        }));
    
    EXPECT_CALL(*session_manager_, add_message(_, _, _))
        .WillRepeatedly(Return(true));
    
    EXPECT_CALL(*session_manager_, get_conversation_history(_))
        .WillRepeatedly(Return(test_conversation_history_));
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Session creation thread
    std::thread session_creator([&]() {
        while (!stop_simulation) {
            session_manager_->create_session(json{{"type", "realistic_test"}});
            operations_completed++;
            std::this_thread::sleep_for(std::chrono::milliseconds(
                static_cast<int>(1000.0 / sessions_per_second)));
        }
    });
    
    // Message addition thread
    std::thread message_adder([&]() {
        int message_counter = 0;
        while (!stop_simulation) {
            std::string session_id = "realistic_session_1"; // Use existing session
            std::string role = (message_counter % 2 == 0) ? "user" : "assistant";
            std::string content = test_messages_[message_counter % test_messages_.size()];
            
            session_manager_->add_message(session_id, role, content);
            operations_completed++;
            message_counter++;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(
                static_cast<int>(1000.0 / messages_per_session_per_second)));
        }
    });
    
    // History retrieval thread (periodic)
    std::thread history_retriever([&]() {
        while (!stop_simulation) {
            session_manager_->get_conversation_history("realistic_session_1");
            operations_completed++;
            std::this_thread::sleep_for(std::chrono::milliseconds(2000)); // Every 2 seconds
        }
    });
    
    // Run simulation
    std::this_thread::sleep_for(std::chrono::seconds(simulation_duration_seconds));
    stop_simulation = true;
    
    // Wait for threads to finish
    session_creator.join();
    message_adder.join();
    history_retriever.join();
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    double operations_per_second = (operations_completed * 1000.0) / duration.count();
    
    std::cout << "Realistic usage pattern stress test:" << std::endl;
    std::cout << "  Simulation duration: " << simulation_duration_seconds << " seconds" << std::endl;
    std::cout << "  Total operations: " << operations_completed << std::endl;
    std::cout << "  Operations per second: " << operations_per_second << std::endl;
    
    EXPECT_GT(operations_completed, simulation_duration_seconds * 2)
        << "Not enough operations completed during stress test";
    EXPECT_GT(operations_per_second, 5.0)
        << "Overall operation rate too low during stress test";
}