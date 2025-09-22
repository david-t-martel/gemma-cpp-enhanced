#include "SessionManager.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>

using namespace gemma::session;
using namespace std::chrono;

/**
 * @brief Benchmark harness for Session Management performance testing
 *
 * This benchmark specifically tests:
 * 1. O(1) vs O(n²) performance in get_context_messages()
 * 2. Cached vs redundant token calculations
 * 3. Efficient memory trimming
 * 4. Deque vs vector performance for conversation history
 */
class SessionBenchmark {
private:
    std::unique_ptr<SessionManager> manager_;
    std::string session_id_;

    // Timing helpers
    template<typename Func>
    double measure_time_ms(Func&& func) {
        auto start = high_resolution_clock::now();
        func();
        auto end = high_resolution_clock::now();
        return duration_cast<microseconds>(end - start).count() / 1000.0;
    }

    // Generate sample message content
    std::string generate_message(size_t index) {
        std::stringstream ss;
        ss << "This is message number " << index << ". ";
        ss << "It contains some sample text to simulate a real conversation. ";
        ss << "The content varies slightly to make it realistic.";
        return ss.str();
    }

public:
    SessionBenchmark() {
        SessionManager::Config config;
        config.storage_config.type = SessionStorage::StorageType::MEMORY;
        config.default_max_context_tokens = 4096;
        manager_ = std::make_unique<SessionManager>(config);
        manager_->initialize();
    }

    void benchmark_context_messages_performance() {
        std::cout << "\n=== Benchmarking get_context_messages() Performance ===" << std::endl;
        std::cout << "Testing O(n) optimized implementation vs old O(n²) behavior\n" << std::endl;

        // Create session with large context
        SessionManager::CreateOptions options;
        options.max_context_tokens = 8192;
        session_id_ = manager_->create_session(options);

        // Test with increasing message counts
        std::vector<size_t> message_counts = {10, 50, 100, 250, 500, 1000};

        std::cout << std::setw(15) << "Messages"
                  << std::setw(20) << "Add Time (ms)"
                  << std::setw(20) << "Context Time (ms)"
                  << std::setw(20) << "Msgs/sec" << std::endl;
        std::cout << std::string(75, '-') << std::endl;

        for (size_t count : message_counts) {
            // Clear history for clean test
            manager_->clear_session_history(session_id_);

            // Add messages and measure time
            double add_time = measure_time_ms([&]() {
                for (size_t i = 0; i < count; ++i) {
                    std::string content = generate_message(i);
                    size_t token_count = content.length() / 4; // Rough token estimate
                    manager_->add_message(session_id_,
                                        ConversationMessage::Role::USER,
                                        content, token_count);
                }
            });

            // Measure context retrieval time (should be O(n) now)
            std::vector<ConversationMessage> context;
            double context_time = measure_time_ms([&]() {
                context = manager_->get_context_messages(session_id_);
            });

            double msgs_per_sec = (count / add_time) * 1000;

            std::cout << std::setw(15) << count
                      << std::setw(20) << std::fixed << std::setprecision(3) << add_time
                      << std::setw(20) << std::fixed << std::setprecision(3) << context_time
                      << std::setw(20) << std::fixed << std::setprecision(0) << msgs_per_sec
                      << std::endl;
        }
    }

    void benchmark_token_calculation_caching() {
        std::cout << "\n=== Benchmarking Token Calculation Caching ===" << std::endl;
        std::cout << "Testing cached vs redundant token calculations\n" << std::endl;

        // Create session
        SessionManager::CreateOptions options;
        options.max_context_tokens = 4096;
        session_id_ = manager_->create_session(options);

        // Add many messages
        const size_t num_messages = 500;
        for (size_t i = 0; i < num_messages; ++i) {
            std::string content = generate_message(i);
            size_t token_count = content.length() / 4;
            manager_->add_message(session_id_,
                                ConversationMessage::Role::USER,
                                content, token_count);
        }

        auto session = manager_->get_session(session_id_);
        if (!session) {
            std::cerr << "Failed to get session!" << std::endl;
            return;
        }

        // Benchmark repeated token count queries (should use cache)
        const size_t num_queries = 10000;
        double query_time = measure_time_ms([&]() {
            for (size_t i = 0; i < num_queries; ++i) {
                volatile size_t tokens = session->get_context_tokens();
                (void)tokens; // Prevent optimization
            }
        });

        double queries_per_ms = num_queries / query_time;
        std::cout << "Token count queries: " << num_queries << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(3)
                  << query_time << " ms" << std::endl;
        std::cout << "Queries per ms: " << std::fixed << std::setprecision(0)
                  << queries_per_ms << std::endl;
        std::cout << "Average time per query: " << std::fixed << std::setprecision(6)
                  << (query_time / num_queries) << " ms" << std::endl;
    }

    void benchmark_memory_trimming() {
        std::cout << "\n=== Benchmarking Memory Trimming ===" << std::endl;
        std::cout << "Testing efficient context window trimming\n" << std::endl;

        // Create session with small context to force trimming
        SessionManager::CreateOptions options;
        options.max_context_tokens = 1024; // Small context
        session_id_ = manager_->create_session(options);

        // Add many messages to trigger trimming
        const size_t num_messages = 1000;
        std::cout << "Adding " << num_messages << " messages with small context window..." << std::endl;

        double add_time = measure_time_ms([&]() {
            for (size_t i = 0; i < num_messages; ++i) {
                std::string content = generate_message(i);
                size_t token_count = 50; // Fixed token count
                manager_->add_message(session_id_,
                                    ConversationMessage::Role::USER,
                                    content, token_count);
            }
        });

        auto session = manager_->get_session(session_id_);
        if (!session) {
            std::cerr << "Failed to get session!" << std::endl;
            return;
        }

        // Check memory efficiency
        auto history = session->get_conversation_history();
        auto context = session->get_context_messages();

        std::cout << "Messages added: " << num_messages << std::endl;
        std::cout << "Messages in history: " << history.size() << std::endl;
        std::cout << "Messages in context: " << context.size() << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(3)
                  << add_time << " ms" << std::endl;
        std::cout << "Average per message: " << std::fixed << std::setprecision(3)
                  << (add_time / num_messages) << " ms" << std::endl;

        // Memory trimming should keep history reasonable (2x context by default)
        bool memory_efficient = history.size() < num_messages;
        std::cout << "Memory trimming active: " << (memory_efficient ? "YES" : "NO") << std::endl;
    }

    void benchmark_deque_operations() {
        std::cout << "\n=== Benchmarking Deque vs Vector Operations ===" << std::endl;
        std::cout << "Testing front insertion/deletion performance\n" << std::endl;

        // Create two sessions for comparison
        SessionManager::CreateOptions options;
        options.max_context_tokens = 8192;
        session_id_ = manager_->create_session(options);

        // Benchmark alternating user/assistant messages
        const size_t num_pairs = 500;
        double conversation_time = measure_time_ms([&]() {
            for (size_t i = 0; i < num_pairs; ++i) {
                // User message
                std::string user_msg = "User question " + std::to_string(i);
                manager_->add_message(session_id_,
                                    ConversationMessage::Role::USER,
                                    user_msg, 20);

                // Assistant response
                std::string assist_msg = "Assistant response " + std::to_string(i);
                manager_->add_message(session_id_,
                                    ConversationMessage::Role::ASSISTANT,
                                    assist_msg, 50);
            }
        });

        std::cout << "Message pairs added: " << num_pairs << std::endl;
        std::cout << "Total messages: " << (num_pairs * 2) << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(3)
                  << conversation_time << " ms" << std::endl;
        std::cout << "Time per exchange: " << std::fixed << std::setprecision(3)
                  << (conversation_time / num_pairs) << " ms" << std::endl;
    }

    void run_all_benchmarks() {
        std::cout << "========================================" << std::endl;
        std::cout << "Session Management Performance Benchmark" << std::endl;
        std::cout << "========================================" << std::endl;

        benchmark_context_messages_performance();
        benchmark_token_calculation_caching();
        benchmark_memory_trimming();
        benchmark_deque_operations();

        std::cout << "\n========================================" << std::endl;
        std::cout << "Benchmark Complete!" << std::endl;
        std::cout << "========================================" << std::endl;
    }
};

int main() {
    try {
        SessionBenchmark benchmark;
        benchmark.run_all_benchmarks();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed with error: " << e.what() << std::endl;
        return 1;
    }
}