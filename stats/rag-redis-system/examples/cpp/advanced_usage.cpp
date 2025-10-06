/**
 * @file advanced_usage.cpp
 * @brief Advanced C++ usage example for RAG Redis System
 *
 * This example demonstrates advanced features including:
 * - Thread safety and concurrent operations
 * - Custom error handling strategies
 * - Performance monitoring and metrics
 * - Advanced search patterns
 * - Memory management optimization
 *
 * Compilation:
 * g++ -std=c++17 -pthread -I../include -L../target/release -lrag_redis_system advanced_usage.cpp -o advanced_usage
 *
 * Usage:
 * ./advanced_usage
 */

#include "rag_redis.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <thread>
#include <future>
#include <chrono>
#include <atomic>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <memory>

/**
 * @brief Thread-safe performance counter
 */
class PerformanceCounter {
private:
    std::atomic<uint64_t> total_operations_{0};
    std::atomic<uint64_t> successful_operations_{0};
    std::atomic<uint64_t> failed_operations_{0};
    std::atomic<uint64_t> total_duration_ms_{0};
    mutable std::mutex mutex_;
    std::vector<std::chrono::milliseconds> durations_;

public:
    void record_operation(bool success, std::chrono::milliseconds duration) {
        total_operations_++;
        total_duration_ms_ += duration.count();

        if (success) {
            successful_operations_++;
        } else {
            failed_operations_++;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        durations_.push_back(duration);
    }

    struct Stats {
        uint64_t total_ops;
        uint64_t successful_ops;
        uint64_t failed_ops;
        double success_rate;
        double avg_duration_ms;
        double min_duration_ms;
        double max_duration_ms;
        double p95_duration_ms;
    };

    Stats get_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);

        Stats stats{};
        stats.total_ops = total_operations_.load();
        stats.successful_ops = successful_operations_.load();
        stats.failed_ops = failed_operations_.load();
        stats.success_rate = stats.total_ops > 0 ?
            (static_cast<double>(stats.successful_ops) / stats.total_ops * 100.0) : 0.0;

        if (!durations_.empty()) {
            auto sorted_durations = durations_;
            std::sort(sorted_durations.begin(), sorted_durations.end());

            stats.avg_duration_ms = total_duration_ms_.load() / static_cast<double>(stats.total_ops);
            stats.min_duration_ms = sorted_durations.front().count();
            stats.max_duration_ms = sorted_durations.back().count();

            // Calculate 95th percentile
            size_t p95_index = static_cast<size_t>(0.95 * sorted_durations.size());
            if (p95_index < sorted_durations.size()) {
                stats.p95_duration_ms = sorted_durations[p95_index].count();
            } else {
                stats.p95_duration_ms = stats.max_duration_ms;
            }
        }

        return stats;
    }

    void print_stats() const {
        auto stats = get_stats();

        std::cout << "Performance Statistics:\n";
        std::cout << "  Total Operations: " << stats.total_ops << "\n";
        std::cout << "  Successful: " << stats.successful_ops << "\n";
        std::cout << "  Failed: " << stats.failed_ops << "\n";
        std::cout << "  Success Rate: " << std::fixed << std::setprecision(2) << stats.success_rate << "%\n";
        std::cout << "  Average Duration: " << std::setprecision(3) << stats.avg_duration_ms << "ms\n";
        std::cout << "  Min Duration: " << stats.min_duration_ms << "ms\n";
        std::cout << "  Max Duration: " << stats.max_duration_ms << "ms\n";
        std::cout << "  95th Percentile: " << stats.p95_duration_ms << "ms\n\n";
    }
};

/**
 * @brief Document generator for testing
 */
class DocumentGenerator {
private:
    std::mt19937 rng_;
    std::vector<std::string> topics_;
    std::vector<std::string> categories_;
    std::vector<std::string> templates_;

public:
    DocumentGenerator() : rng_(std::random_device{}()) {
        topics_ = {
            "machine learning", "deep learning", "neural networks", "artificial intelligence",
            "computer vision", "natural language processing", "robotics", "data science",
            "quantum computing", "blockchain", "cybersecurity", "cloud computing"
        };

        categories_ = {
            "overview", "tutorial", "advanced", "research", "implementation",
            "theory", "practice", "case_study", "benchmarks", "comparison"
        };

        templates_ = {
            "{topic} is a fascinating field that involves {description}. It has applications in {applications}.",
            "In the context of {topic}, {description} plays a crucial role in {applications}.",
            "Recent advances in {topic} have shown that {description} can significantly improve {applications}.",
            "The field of {topic} encompasses {description}, which is essential for {applications}."
        };
    }

    std::pair<std::string, rag::Metadata> generate_document() {
        std::uniform_int_distribution<size_t> topic_dist(0, topics_.size() - 1);
        std::uniform_int_distribution<size_t> cat_dist(0, categories_.size() - 1);
        std::uniform_int_distribution<size_t> template_dist(0, templates_.size() - 1);

        auto topic = topics_[topic_dist(rng_)];
        auto category = categories_[cat_dist(rng_)];
        auto template_text = templates_[template_dist(rng_)];

        // Simple template replacement
        std::string content = template_text;
        replace_all(content, "{topic}", topic);
        replace_all(content, "{description}", "advanced techniques and methodologies");
        replace_all(content, "{applications}", "various real-world scenarios");

        rag::Metadata metadata = {
            {"topic", topic},
            {"category", category},
            {"generated", "true"},
            {"timestamp", std::to_string(std::chrono::system_clock::now().time_since_epoch().count())}
        };

        return {content, metadata};
    }

private:
    void replace_all(std::string& str, const std::string& from, const std::string& to) {
        size_t pos = 0;
        while ((pos = str.find(from, pos)) != std::string::npos) {
            str.replace(pos, from.length(), to);
            pos += to.length();
        }
    }
};

/**
 * @brief Concurrent search worker
 */
class SearchWorker {
private:
    std::shared_ptr<rag::System> system_;
    std::shared_ptr<PerformanceCounter> counter_;
    std::atomic<bool> running_{false};

public:
    SearchWorker(std::shared_ptr<rag::System> system,
                 std::shared_ptr<PerformanceCounter> counter)
        : system_(system), counter_(counter) {}

    void start_concurrent_searches(const std::vector<std::string>& queries, int iterations) {
        running_ = true;

        std::vector<std::future<void>> futures;

        for (int i = 0; i < iterations && running_; ++i) {
            for (const auto& query : queries) {
                if (!running_) break;

                futures.push_back(std::async(std::launch::async, [this, query]() {
                    perform_search(query);
                }));
            }
        }

        // Wait for all searches to complete
        for (auto& future : futures) {
            future.wait();
        }

        running_ = false;
    }

    void stop() {
        running_ = false;
    }

private:
    void perform_search(const std::string& query) {
        auto start = std::chrono::high_resolution_clock::now();
        bool success = false;

        try {
            auto results = system_->search(query, 5);
            success = true;

            // Simulate some processing
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        } catch (const rag::Exception& e) {
            // Log error but don't stop
            std::cerr << "Search failed: " << e.what() << "\n";
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        counter_->record_operation(success, duration);
    }
};

/**
 * @brief Advanced error handler with retry logic
 */
class AdvancedErrorHandler {
public:
    template<typename Func>
    static auto with_retry(Func&& func, int max_retries = 3,
                          std::chrono::milliseconds delay = std::chrono::milliseconds(100)) {
        rag::Exception last_exception(rag::Exception::Unknown, "No attempts made");

        for (int attempt = 0; attempt < max_retries; ++attempt) {
            try {
                return func();
            } catch (const rag::Exception& e) {
                last_exception = e;

                if (attempt < max_retries - 1) {
                    std::this_thread::sleep_for(delay * (attempt + 1)); // Exponential backoff
                    std::cout << "Retry attempt " << (attempt + 1) << " after error: " << e.what() << "\n";
                }
            }
        }

        throw last_exception;
    }

    static bool is_retryable_error(const rag::Exception& e) {
        return e.code() == rag::Exception::Network ||
               e.code() == rag::Exception::Timeout ||
               e.code() == rag::Exception::Redis;
    }
};

/**
 * @brief Memory usage monitor
 */
class MemoryMonitor {
private:
    std::atomic<bool> monitoring_{false};
    std::thread monitor_thread_;

public:
    void start_monitoring(std::chrono::seconds interval = std::chrono::seconds(5)) {
        monitoring_ = true;
        monitor_thread_ = std::thread([this, interval]() {
            while (monitoring_) {
                print_memory_usage();
                std::this_thread::sleep_for(interval);
            }
        });
    }

    void stop_monitoring() {
        monitoring_ = false;
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
    }

    ~MemoryMonitor() {
        stop_monitoring();
    }

private:
    void print_memory_usage() {
        // This is a simplified implementation
        // In production, you'd use proper system calls to get memory usage
        std::cout << "[MEMORY] Monitoring system memory usage...\n";
    }
};

/**
 * @brief Demonstrate thread safety
 */
void demonstrate_thread_safety(rag::System& system) {
    std::cout << "Demonstrating thread safety with concurrent operations...\n";

    auto counter = std::make_shared<PerformanceCounter>();
    auto system_ptr = std::make_shared<rag::System>(std::move(system));

    const std::vector<std::string> queries = {
        "machine learning", "deep learning", "neural networks",
        "computer vision", "natural language processing"
    };

    const int num_threads = 4;
    const int iterations_per_thread = 10;

    std::vector<std::thread> threads;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Start multiple search worker threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            SearchWorker worker(system_ptr, counter);
            worker.start_concurrent_searches(queries, iterations_per_thread);
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Concurrent operations completed in " << total_duration.count() << "ms\n";
    counter->print_stats();
}

/**
 * @brief Demonstrate advanced search patterns
 */
void demonstrate_advanced_search_patterns(rag::System& system) {
    std::cout << "Demonstrating advanced search patterns...\n";

    try {
        // Batch search with different parameters
        std::vector<std::string> queries = {
            "artificial intelligence applications",
            "machine learning algorithms",
            "deep neural network architectures"
        };

        std::vector<std::future<rag::SearchResults>> search_futures;

        // Start batch searches asynchronously
        for (const auto& query : queries) {
            search_futures.push_back(
                system.search_async(query, 10)
            );
        }

        // Process results as they become available
        for (size_t i = 0; i < search_futures.size(); ++i) {
            auto results = search_futures[i].get();

            std::cout << "Query " << (i + 1) << ": \"" << queries[i] << "\"\n";
            std::cout << "  Found " << results.size() << " results\n";

            // Analyze result quality
            if (!results.empty()) {
                auto top_result = results[0];
                std::cout << "  Top result score: " << top_result.score() << "\n";

                // Filter high-quality results
                auto high_quality = results.filter_by_score(0.7);
                std::cout << "  High quality results (>= 0.7): " << high_quality.size() << "\n";
            }

            std::cout << "\n";
        }

    } catch (const rag::Exception& e) {
        std::cerr << "Advanced search patterns failed: " << e << "\n\n";
    }
}

/**
 * @brief Load test with performance monitoring
 */
void perform_load_test(rag::System& system) {
    std::cout << "Performing load test...\n";

    auto counter = std::make_shared<PerformanceCounter>();
    MemoryMonitor memory_monitor;
    memory_monitor.start_monitoring();

    DocumentGenerator doc_generator;

    // Ingest many documents
    std::cout << "Ingesting test documents...\n";
    const int num_test_docs = 100;

    for (int i = 0; i < num_test_docs; ++i) {
        auto [content, metadata] = doc_generator.generate_document();

        auto start = std::chrono::high_resolution_clock::now();
        bool success = false;

        try {
            system.ingest_document(content, metadata);
            success = true;
        } catch (const rag::Exception& e) {
            std::cerr << "Document ingestion failed: " << e.what() << "\n";
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        counter->record_operation(success, duration);

        if ((i + 1) % 20 == 0) {
            std::cout << "  Ingested " << (i + 1) << "/" << num_test_docs << " documents\n";
        }
    }

    std::cout << "\nIngestion performance:\n";
    counter->print_stats();

    // Reset counter for search test
    counter = std::make_shared<PerformanceCounter>();

    // Perform search load test
    std::cout << "Performing search load test...\n";
    const std::vector<std::string> test_queries = {
        "machine learning", "artificial intelligence", "deep learning",
        "neural networks", "computer vision", "data science"
    };

    const int searches_per_query = 50;

    for (const auto& query : test_queries) {
        for (int i = 0; i < searches_per_query; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            bool success = false;

            try {
                auto results = system.search(query, 10);
                success = true;
            } catch (const rag::Exception& e) {
                std::cerr << "Search failed: " << e.what() << "\n";
            }

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            counter->record_operation(success, duration);
        }
    }

    std::cout << "\nSearch performance:\n";
    counter->print_stats();

    memory_monitor.stop_monitoring();
}

/**
 * @brief Main function demonstrating advanced usage
 */
int main() {
    std::cout << "RAG Redis System - Advanced C++ Usage Example\n";
    std::cout << "==============================================\n\n";

    try {
        // Initialize library
        rag::Library library;
        std::cout << "Library initialized (version: " << rag::Library::version() << ")\n\n";

        // Create high-performance configuration
        auto config = rag::Config()
            .redis_url("redis://127.0.0.1:6379")
            .redis_pool_size(20)
            .redis_connection_timeout(std::chrono::seconds(10))
            .redis_command_timeout(std::chrono::seconds(30))
            .vector_dimension(768)
            .vector_max_elements(500000)
            .document_chunk_size(512)
            .document_chunk_overlap(50)
            .memory_max_mb(2048)
            .research_enable_web_search(true);

        std::cout << "Creating high-performance RAG system...\n";
        rag::System system(config);
        std::cout << "System created and validated\n\n";

        // Demonstrate error handling with retry
        std::cout << "Demonstrating advanced error handling...\n";
        try {
            auto result = AdvancedErrorHandler::with_retry([&]() {
                return system.search("test query with potential errors", 5);
            }, 3, std::chrono::milliseconds(100));

            std::cout << "Search with retry succeeded, found " << result.size() << " results\n\n";
        } catch (const rag::Exception& e) {
            std::cout << "Search with retry ultimately failed: " << e << "\n\n";
        }

        // Perform load test
        perform_load_test(system);

        // Demonstrate advanced search patterns
        demonstrate_advanced_search_patterns(system);

        // Thread safety demonstration (move system ownership)
        std::cout << "Note: Thread safety demonstration skipped to avoid moving system\n";
        std::cout << "In production, you would create separate system instances for concurrent access\n\n";

        // Get final statistics
        std::cout << "Final system statistics:\n";
        try {
            auto stats = system.get_stats();
            std::cout << stats << "\n";
        } catch (const rag::Exception& e) {
            std::cerr << "Failed to get final statistics: " << e << "\n";
        }

        std::cout << "\nAdvanced example completed successfully!\n";

    } catch (const rag::Exception& e) {
        std::cerr << "RAG Exception: " << e << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception occurred\n";
        return 1;
    }

    return 0;
}
