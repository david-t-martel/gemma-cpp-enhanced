/**
 * @file basic_usage.cpp
 * @brief Basic C++ usage example for RAG Redis System
 *
 * This example demonstrates basic usage of the RAG Redis System C++ API,
 * showcasing RAII resource management, exception handling, and modern C++ features.
 *
 * Compilation:
 * g++ -std=c++17 -I../include -L../target/release -lrag_redis_system basic_usage.cpp -o basic_usage
 *
 * Usage:
 * ./basic_usage
 */

#include "rag_redis.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>

/**
 * @brief Sample documents for ingestion
 */
const std::vector<std::pair<std::string, rag::Metadata>> sample_documents = {
    {
        "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        {{"topic", "machine_learning"}, {"category", "overview"}, {"difficulty", "beginner"}}
    },
    {
        "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
        {{"topic", "deep_learning"}, {"category", "neural_networks"}, {"difficulty", "intermediate"}}
    },
    {
        "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language.",
        {{"topic", "nlp"}, {"category", "language"}, {"difficulty", "intermediate"}}
    },
    {
        "Computer vision is an interdisciplinary field that deals with how computers can gain high-level understanding from digital images or videos.",
        {{"topic", "computer_vision"}, {"category", "image_processing"}, {"difficulty", "intermediate"}}
    },
    {
        "Reinforcement learning is an area of machine learning where an agent learns to behave in an environment by performing actions and seeing the results.",
        {{"topic", "reinforcement_learning"}, {"category", "learning_algorithms"}, {"difficulty", "advanced"}}
    }
};

/**
 * @brief Print search results with formatting
 */
void print_results(const rag::SearchResults& results, const std::string& query_title = "") {
    if (!query_title.empty()) {
        std::cout << "Results for: \"" << query_title << "\"\n";
        std::cout << std::string(80, '=') << "\n";
    }

    if (results.empty()) {
        std::cout << "No results found.\n\n";
        return;
    }

    std::cout << "Found " << results.size() << " results:\n";
    std::cout << std::left
              << std::setw(8) << "Rank"
              << std::setw(8) << "Score"
              << std::setw(15) << "Source"
              << std::setw(49) << "Text Preview"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];

        // Truncate text for preview
        std::string text_preview = result.text();
        if (text_preview.length() > 48) {
            text_preview = text_preview.substr(0, 45) + "...";
        }

        std::cout << std::left
                  << std::setw(8) << (i + 1)
                  << std::setw(8) << std::fixed << std::setprecision(3) << result.score()
                  << std::setw(15) << result.source()
                  << std::setw(49) << text_preview
                  << "\n";

        if (result.has_url()) {
            std::cout << "    URL: " << result.url().value() << "\n";
        }

        // Print metadata if available
        const auto& metadata = result.metadata();
        if (!metadata.empty()) {
            std::cout << "    Metadata: ";
            bool first = true;
            for (const auto& [key, value] : metadata) {
                if (!first) std::cout << ", ";
                std::cout << key << "=" << value;
                first = false;
            }
            std::cout << "\n";
        }
    }
    std::cout << "\n";
}

/**
 * @brief Demonstrate async operations
 */
void demonstrate_async_operations(rag::System& system) {
    std::cout << "Demonstrating asynchronous operations...\n";

    // Start async search
    auto search_future = system.search_async("artificial intelligence", 5);

    // Start async document ingestion
    auto ingest_future = system.ingest_document_async(
        "Quantum computing leverages quantum mechanical phenomena to process information in fundamentally new ways.",
        {{"topic", "quantum_computing"}, {"category", "advanced_computing"}, {"difficulty", "expert"}}
    );

    // Do some work while operations are running
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "  Async operations started, doing other work...\n";

    // Wait for results
    try {
        auto search_results = search_future.get();
        std::cout << "  Async search completed with " << search_results.size() << " results\n";

        auto doc_id = ingest_future.get();
        std::cout << "  Async ingestion completed with document ID: " << doc_id << "\n";
    } catch (const rag::Exception& e) {
        std::cerr << "  Async operation failed: " << e << "\n";
    }

    std::cout << "\n";
}

/**
 * @brief Demonstrate result filtering and manipulation
 */
void demonstrate_result_operations(rag::System& system) {
    std::cout << "Demonstrating result operations...\n";

    try {
        auto results = system.search("learning algorithms", 10);

        std::cout << "Original results: " << results.size() << "\n";

        // Filter by score
        auto high_score_results = results.filter_by_score(0.5);
        std::cout << "High score results (>= 0.5): " << high_score_results.size() << "\n";

        // Get top 3 results
        auto top_results = results.top(3);
        std::cout << "Top 3 results: " << top_results.size() << "\n";

        // Use range-based for loop
        std::cout << "Iterating through results:\n";
        for (const auto& result : top_results) {
            std::cout << "  - " << result.id() << " (score: " << result.score() << ")\n";
        }

        std::cout << "\n";
    } catch (const rag::Exception& e) {
        std::cerr << "Result operations failed: " << e << "\n\n";
    }
}

/**
 * @brief Demonstrate configuration variations
 */
void demonstrate_configurations() {
    std::cout << "Demonstrating different configuration patterns...\n";

    try {
        // Development configuration
        auto dev_config = rag::create_dev_config("redis://127.0.0.1:6379");
        std::cout << "Development configuration created\n";

        // Production configuration
        auto prod_config = rag::create_production_config("redis://127.0.0.1:6379");
        std::cout << "Production configuration created\n";

        // Custom configuration with fluent interface
        auto custom_config = rag::Config()
            .redis_url("redis://127.0.0.1:6379")
            .redis_pool_size(15)
            .vector_dimension(512)
            .document_chunk_size(256)
            .document_chunk_overlap(25)
            .memory_max_mb(2048)
            .research_enable_web_search(true)
            .research_max_results(15);

        std::cout << "Custom configuration created with fluent interface\n";

        // Validate configurations
        dev_config.validate();
        prod_config.validate();
        custom_config.validate();
        std::cout << "All configurations validated successfully\n";

    } catch (const rag::Exception& e) {
        std::cerr << "Configuration error: " << e << "\n";
    }

    std::cout << "\n";
}

/**
 * @brief Main function demonstrating comprehensive usage
 */
int main() {
    std::cout << "RAG Redis System C++ API Example\n";
    std::cout << "=================================\n\n";

    try {
        // RAII library initialization
        rag::Library library;
        std::cout << "Library initialized (version: " << rag::Library::version() << ")\n\n";

        // Demonstrate different configuration patterns
        demonstrate_configurations();

        // Create system with development configuration
        std::cout << "Creating RAG system...\n";
        auto config = rag::create_dev_config("redis://127.0.0.1:6379");
        rag::System system(config);

        std::cout << "System created successfully\n";
        std::cout << "Health check: " << (system.health_check() ? "PASS" : "FAIL") << "\n\n";

        // Ingest sample documents
        std::cout << "Ingesting sample documents...\n";
        std::vector<std::string> doc_ids;

        for (size_t i = 0; i < sample_documents.size(); ++i) {
            const auto& [content, metadata] = sample_documents[i];
            auto doc_id = system.ingest_document(content, metadata);
            doc_ids.push_back(doc_id);
            std::cout << "  Document " << (i + 1) << " ingested: " << doc_id << "\n";
        }
        std::cout << "\n";

        // Perform searches
        const std::vector<std::string> queries = {
            "artificial intelligence learning",
            "neural networks deep learning",
            "computer language processing",
            "image recognition vision",
            "reinforcement learning algorithms"
        };

        std::cout << "Performing searches...\n";
        for (const auto& query : queries) {
            try {
                auto results = system.search(query, 3);
                print_results(results, query);
            } catch (const rag::Exception& e) {
                std::cerr << "Search failed for \"" << query << "\": " << e << "\n\n";
            }
        }

        // Demonstrate research functionality
        std::cout << "Demonstrating research functionality...\n";
        try {
            rag::Sources sources = {"wikipedia", "arxiv", "github"};
            auto research_results = system.research("machine learning algorithms", sources);
            print_results(research_results, "Research: machine learning algorithms");
        } catch (const rag::Exception& e) {
            std::cerr << "Research failed: " << e << "\n\n";
        }

        // Demonstrate async operations
        demonstrate_async_operations(system);

        // Demonstrate result operations
        demonstrate_result_operations(system);

        // Get system statistics
        std::cout << "System statistics:\n";
        try {
            auto stats = system.get_stats();
            std::cout << stats << "\n\n";
        } catch (const rag::Exception& e) {
            std::cerr << "Failed to get statistics: " << e << "\n\n";
        }

        // Performance test
        std::cout << "Performance test...\n";
        auto start_time = std::chrono::high_resolution_clock::now();

        const int num_searches = 10;
        for (int i = 0; i < num_searches; ++i) {
            system.search("test query " + std::to_string(i), 5);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Performed " << num_searches << " searches in "
                  << duration.count() << "ms ("
                  << (duration.count() / static_cast<double>(num_searches)) << "ms avg)\n\n";

        std::cout << "Example completed successfully!\n";

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

    // Library cleanup happens automatically via RAII
    return 0;
}
