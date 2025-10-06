/**
 * @file basic_usage.c
 * @brief Basic C usage example for RAG Redis System
 *
 * This example demonstrates basic usage of the RAG Redis System C API,
 * including initialization, document ingestion, searching, and cleanup.
 *
 * Compilation:
 * gcc -std=c11 -I../include -L../target/release -lrag_redis_system basic_usage.c -o basic_usage
 *
 * Usage:
 * ./basic_usage
 */

#include "rag_redis.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Print error information and exit if error occurred
 */
void check_error(RagErrorInfo* error, const char* operation) {
    if (RAG_HAS_ERROR(error)) {
        fprintf(stderr, "Error during %s: [%d] %s\n",
                operation, error->code, RAG_ERROR_MESSAGE(error));
        if (error->message) {
            rag_free_error_message(error->message);
        }
        exit(1);
    }
    if (error->message) {
        rag_free_error_message(error->message);
    }
}

/**
 * @brief Print search results
 */
void print_results(RagSearchResults* results) {
    if (!results) {
        printf("No results found.\n");
        return;
    }

    printf("Found %u results:\n", results->count);
    printf("%-40s %-8s %-20s %s\n", "ID", "Score", "Source", "Text Preview");
    printf("%-40s %-8s %-20s %s\n", "----", "-----", "------", "------------");

    for (unsigned int i = 0; i < results->count; i++) {
        RagSearchResult* result = &results->results[i];

        // Truncate text for preview
        char text_preview[61];
        if (strlen(result->text) > 60) {
            strncpy(text_preview, result->text, 57);
            text_preview[57] = '.';
            text_preview[58] = '.';
            text_preview[59] = '.';
            text_preview[60] = '\0';
        } else {
            strcpy(text_preview, result->text);
        }

        printf("%-40s %-8.3f %-20s %s\n",
               result->id, result->score, result->source, text_preview);

        if (result->url) {
            printf("    URL: %s\n", result->url);
        }
        if (result->metadata_json && strlen(result->metadata_json) > 2) {
            printf("    Metadata: %s\n", result->metadata_json);
        }
        printf("\n");
    }
}

/**
 * @brief Main function demonstrating basic usage
 */
int main(void) {
    printf("RAG Redis System C API Example\n");
    printf("==============================\n\n");

    // Initialize the library
    printf("Initializing RAG library...\n");
    if (!rag_init()) {
        fprintf(stderr, "Failed to initialize RAG library\n");
        return 1;
    }
    printf("Library version: %s\n\n", rag_version());

    // Create configuration
    RagConfig config;
    rag_config_default(&config);

    // Override some defaults
    config.redis_url = "redis://127.0.0.1:6379";
    config.vector_dimension = 768;
    config.doc_chunk_size = 512;
    config.research_enable_web_search = 0; // Disable for this example

    printf("Configuration:\n");
    printf("  Redis URL: %s\n", config.redis_url);
    printf("  Vector Dimension: %u\n", config.vector_dimension);
    printf("  Chunk Size: %u\n", config.doc_chunk_size);
    printf("  Web Search: %s\n\n", config.research_enable_web_search ? "Enabled" : "Disabled");

    // Create RAG system
    printf("Creating RAG system...\n");
    RagErrorInfo error = {0};
    RagHandle handle = rag_create(&config, &error);
    check_error(&error, "system creation");
    printf("System created with handle: %lu\n\n", handle);

    // Health check
    if (rag_health_check(handle)) {
        printf("System health check: PASS\n\n");
    } else {
        printf("System health check: FAIL\n\n");
    }

    // Sample documents to ingest
    const char* documents[] = {
        "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
        "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
        "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language.",
        "Computer vision is an interdisciplinary field that deals with how computers can gain high-level understanding from digital images or videos.",
        "Reinforcement learning is an area of machine learning where an agent learns to behave in an environment by performing actions and seeing the results."
    };

    const char* metadata[] = {
        "{\"topic\": \"machine_learning\", \"category\": \"overview\", \"difficulty\": \"beginner\"}",
        "{\"topic\": \"deep_learning\", \"category\": \"neural_networks\", \"difficulty\": \"intermediate\"}",
        "{\"topic\": \"nlp\", \"category\": \"language\", \"difficulty\": \"intermediate\"}",
        "{\"topic\": \"computer_vision\", \"category\": \"image_processing\", \"difficulty\": \"intermediate\"}",
        "{\"topic\": \"reinforcement_learning\", \"category\": \"learning_algorithms\", \"difficulty\": \"advanced\"}"
    };

    // Ingest documents
    printf("Ingesting documents...\n");
    const int num_docs = sizeof(documents) / sizeof(documents[0]);
    char* doc_ids[num_docs];

    for (int i = 0; i < num_docs; i++) {
        char* doc_id = NULL;
        memset(&error, 0, sizeof(error));

        int result = rag_ingest_document(handle, documents[i], metadata[i], &doc_id, &error);
        check_error(&error, "document ingestion");

        if (result && doc_id) {
            doc_ids[i] = doc_id;
            printf("  Document %d ingested with ID: %s\n", i + 1, doc_id);
        } else {
            fprintf(stderr, "Failed to ingest document %d\n", i + 1);
            doc_ids[i] = NULL;
        }
    }
    printf("\n");

    // Search examples
    const char* queries[] = {
        "artificial intelligence learning",
        "neural networks deep learning",
        "computer language processing",
        "image recognition vision"
    };

    const int num_queries = sizeof(queries) / sizeof(queries[0]);

    printf("Performing searches...\n");
    for (int i = 0; i < num_queries; i++) {
        printf("Query %d: \"%s\"\n", i + 1, queries[i]);
        printf("%-80s\n", "================================================================================");

        memset(&error, 0, sizeof(error));
        RagSearchResults* results = rag_search(handle, queries[i], 3, &error);
        check_error(&error, "search");

        print_results(results);
        rag_free_search_results(results);
        printf("\n");
    }

    // Get system statistics
    printf("Getting system statistics...\n");
    memset(&error, 0, sizeof(error));
    char* stats = rag_get_stats(handle, &error);
    check_error(&error, "get statistics");

    if (stats) {
        printf("System Statistics:\n%s\n\n", stats);
        rag_free_string(stats);
    }

    // Research example (if web search was enabled)
    if (config.research_enable_web_search) {
        printf("Performing research...\n");
        const char* sources[] = {"wikipedia", "arxiv"};
        const int num_sources = sizeof(sources) / sizeof(sources[0]);

        memset(&error, 0, sizeof(error));
        RagSearchResults* research_results = rag_research(handle, "machine learning algorithms",
                                                         sources, num_sources, &error);
        check_error(&error, "research");

        printf("Research Results:\n");
        print_results(research_results);
        rag_free_search_results(research_results);
    }

    // Cleanup
    printf("Cleaning up...\n");

    // Free document IDs
    for (int i = 0; i < num_docs; i++) {
        if (doc_ids[i]) {
            rag_free_string(doc_ids[i]);
        }
    }

    // Destroy system
    if (rag_destroy(handle)) {
        printf("System destroyed successfully\n");
    } else {
        printf("Failed to destroy system\n");
    }

    // Cleanup library
    if (rag_cleanup()) {
        printf("Library cleanup successful\n");
    } else {
        printf("Library cleanup failed\n");
    }

    printf("\nExample completed successfully!\n");
    return 0;
}
