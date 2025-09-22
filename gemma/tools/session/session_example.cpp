#include "SessionManager.h"
#include <iostream>
#include <thread>
#include <chrono>

using namespace gemma::session;

/**
 * @brief Example demonstrating the session management system
 * 
 * This example shows how to:
 * - Initialize the session manager
 * - Create and manage sessions
 * - Add messages to conversations
 * - Handle session persistence
 * - Monitor session statistics
 */
int main() {
    try {
        // Configure the session manager
        SessionManager::Config config;
        config.default_max_context_tokens = 4096;
        config.storage_config.db_path = "example_sessions.db";
        config.storage_config.cache_capacity = 50;
        config.storage_config.session_ttl = std::chrono::hours(24);
        config.enable_metrics = true;
        
        // Create and initialize session manager
        SessionManager manager(config);
        
        if (!manager.initialize()) {
            std::cerr << "Failed to initialize session manager!" << std::endl;
            return 1;
        }
        
        std::cout << "=== Gemma Session Management Example ===" << std::endl;
        
        // Set up event callback for monitoring
        manager.set_event_callback([](const std::string& event, const nlohmann::json& data) {
            std::cout << "[EVENT] " << event << ": " << data.dump() << std::endl;
        });
        
        // Create a new session
        SessionManager::CreateOptions create_opts;
        create_opts.max_context_tokens = 2048;
        create_opts.metadata = nlohmann::json{
            {"user_id", "example_user"},
            {"purpose", "demonstration"}
        };
        
        std::string session_id = manager.create_session(create_opts);
        std::cout << "\nCreated session: " << session_id << std::endl;
        
        // Add some conversation messages
        std::cout << "\nAdding conversation messages..." << std::endl;
        
        manager.add_message(session_id, ConversationMessage::Role::USER, 
                          "Hello! Can you help me understand C++ session management?", 
                          15);
        
        manager.add_message(session_id, ConversationMessage::Role::ASSISTANT,
                          "Certainly! C++ session management involves maintaining state "
                          "across multiple interactions. In this system, we use SQLite "
                          "for persistence, LRU caching for performance, and thread-safe "
                          "operations for concurrent access.",
                          42);
        
        manager.add_message(session_id, ConversationMessage::Role::USER,
                          "That's great! How does the token counting work?",
                          12);
        
        manager.add_message(session_id, ConversationMessage::Role::ASSISTANT,
                          "Token counting tracks the computational cost of processing "
                          "text. Each message stores its token count, and we maintain "
                          "both total tokens for the entire session and context tokens "
                          "for the current working window.",
                          35);
        
        // Retrieve and display conversation history
        std::cout << "\nConversation History:" << std::endl;
        auto history = manager.get_conversation_history(session_id);
        
        for (size_t i = 0; i < history.size(); ++i) {
            const auto& msg = history[i];
            std::string role_str;
            switch (msg.role) {
                case ConversationMessage::Role::USER: role_str = "USER"; break;
                case ConversationMessage::Role::ASSISTANT: role_str = "ASSISTANT"; break;
                case ConversationMessage::Role::SYSTEM: role_str = "SYSTEM"; break;
            }
            
            std::cout << "[" << (i + 1) << "] " << role_str 
                     << " (" << msg.token_count << " tokens): " 
                     << msg.content.substr(0, 80);
            if (msg.content.length() > 80) std::cout << "...";
            std::cout << std::endl;
        }
        
        // Show context messages (within the context window)
        std::cout << "\nContext Messages:" << std::endl;
        auto context = manager.get_context_messages(session_id);
        std::cout << "Messages in context: " << context.size() << std::endl;
        
        // Create another session to demonstrate multiple sessions
        std::string session_id2 = manager.create_session();
        std::cout << "\nCreated second session: " << session_id2 << std::endl;
        
        manager.add_message(session_id2, ConversationMessage::Role::USER,
                          "This is a different conversation.", 7);
        
        // List all sessions
        std::cout << "\nAll Sessions:" << std::endl;
        auto sessions = manager.list_sessions();
        for (const auto& session_meta : sessions) {
            std::cout << "  Session ID: " << session_meta["session_id"].get<std::string>()
                     << ", Total Tokens: " << session_meta["total_tokens"].get<size_t>()
                     << std::endl;
        }
        
        // Export sessions to JSON
        std::cout << "\nExporting sessions to JSON..." << std::endl;
        if (manager.export_sessions("exported_sessions.json")) {
            std::cout << "Sessions exported successfully!" << std::endl;
        }
        
        // Show statistics
        std::cout << "\nSession Manager Statistics:" << std::endl;
        auto stats = manager.get_statistics();
        std::cout << "Total sessions created: " 
                 << stats["metrics"]["total_sessions_created"].get<size_t>() << std::endl;
        std::cout << "Total messages processed: " 
                 << stats["metrics"]["total_messages_processed"].get<size_t>() << std::endl;
        std::cout << "Total tokens processed: " 
                 << stats["metrics"]["total_tokens_processed"].get<size_t>() << std::endl;
        std::cout << "Cache size: " 
                 << stats["storage"]["cache_size"].get<size_t>() << std::endl;
        std::cout << "Cache capacity: " 
                 << stats["storage"]["cache_capacity"].get<size_t>() << std::endl;
        
        // Demonstrate session modification
        std::cout << "\nUpdating session context size..." << std::endl;
        manager.update_session_context_size(session_id, 1024);
        
        auto modified_session = manager.get_session(session_id);
        if (modified_session) {
            std::cout << "New context size: " << modified_session->get_max_context_tokens() << std::endl;
        }
        
        // Test session persistence by simulating restart
        std::cout << "\nTesting session persistence..." << std::endl;
        
        // Create a new manager instance (simulating application restart)
        SessionManager manager2(config);
        manager2.initialize();
        
        // Try to load the first session
        auto loaded_session = manager2.get_session(session_id);
        if (loaded_session) {
            std::cout << "Successfully loaded session from storage!" << std::endl;
            std::cout << "Loaded session has " << loaded_session->get_conversation_history().size() 
                     << " messages" << std::endl;
        } else {
            std::cout << "Failed to load session from storage!" << std::endl;
        }
        
        // Cleanup - delete one session
        std::cout << "\nCleaning up..." << std::endl;
        if (manager.delete_session(session_id2)) {
            std::cout << "Deleted session: " << session_id2 << std::endl;
        }
        
        // Final statistics
        std::cout << "\nFinal Statistics:" << std::endl;
        auto final_stats = manager.get_statistics();
        std::cout << "Sessions deleted: " 
                 << final_stats["metrics"]["total_sessions_deleted"].get<size_t>() << std::endl;
        
        std::cout << "\n=== Session Management Example Complete ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}