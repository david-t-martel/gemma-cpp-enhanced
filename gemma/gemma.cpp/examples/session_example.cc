// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Example demonstrating the session management framework for Gemma.

#include <iostream>
#include <memory>
#include <string>

#include "gemma/gemma.h"
#include "gemma/session.h"
#include "gemma/gemma_args.h"
#include "gemma/configs.h"
#include "util/threading_context.h"
#include "ops/matmul.h"

namespace gcpp {

void PrintSessionStats(const SessionStats& stats) {
  std::cout << "Session Statistics:\n"
            << "  Total turns: " << stats.total_turns.load() << "\n"
            << "  Input tokens: " << stats.total_input_tokens.load() << "\n"
            << "  Output tokens: " << stats.total_output_tokens.load() << "\n"
            << "  Cache hits: " << stats.cache_hits.load() << "\n"
            << "  Cache misses: " << stats.cache_misses.load() << "\n"
            << "  Avg response time: " << stats.avg_response_time_ms.load() << "ms\n";
}

int SessionExample() {
  // Setup paths - update these to point to your model files
  const std::string tokenizer_path = "c:/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/tokenizer.spm";
  const std::string weights_path = "c:/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs";

  try {
    // Initialize loader and inference arguments
    LoaderArgs loader(tokenizer_path, weights_path);
    InferenceArgs inference;
    inference.seq_len = 4096;

    // Setup threading context
    ThreadingContext ctx;

    // Load the Gemma model
    std::cout << "Loading Gemma model...\n";
    Gemma gemma(loader, inference, ctx);

    // Create session manager
    std::cout << "Creating session manager...\n";
    SessionManager manager(gemma.Config(), inference, ctx);

    // Create a session with custom configuration
    SessionConfig session_config = session_utils::CreateInteractiveConfig();
    session_config.max_generation_tokens = 512;
    session_config.enable_streaming = true;

    auto session = manager.CreateSession("demo_session", session_config);
    if (!session) {
      std::cerr << "Failed to create session\n";
      return 1;
    }

    // Initialize the session
    std::cout << "Initializing session...\n";
    if (!session->Initialize(gemma)) {
      std::cerr << "Failed to initialize session\n";
      return 1;
    }

    // Setup runtime configuration
    RuntimeConfig runtime_config;
    runtime_config.max_generated_tokens = session_config.max_generation_tokens;
    runtime_config.temperature = 0.7f;
    runtime_config.top_k = 40;

    // Create matrix multiplication environment
    MatMulEnv env;

    // Interactive conversation loop
    std::cout << "\n=== Interactive Session Demo ===\n";
    std::cout << "Type 'quit' to exit, 'stats' to see session statistics\n\n";

    std::string user_input;
    while (true) {
      std::cout << "User: ";
      std::getline(std::cin, user_input);

      if (user_input == "quit") {
        break;
      }

      if (user_input == "stats") {
        PrintSessionStats(session->GetStats());
        continue;
      }

      if (user_input.empty()) {
        continue;
      }

      // Add user message to session
      session->AddUserMessage(user_input);

      std::cout << "Assistant: ";

      // Generate response with streaming
      session->GenerateResponseStream(
          gemma, env, runtime_config,
          [](const std::string& token, bool is_final) -> bool {
            if (!is_final) {
              std::cout << token << std::flush;
            } else {
              std::cout << "\n\n";
            }
            return true;  // Continue generation
          });
    }

    // Demonstrate session persistence
    std::cout << "\n=== Session Persistence Demo ===\n";
    Path save_path("demo_session.json");

    if (session->SaveToFile(save_path)) {
      std::cout << "Session saved to " << save_path.path << "\n";

      // Create a new session and load from file
      auto loaded_session = manager.CreateSession("loaded_session");
      if (loaded_session && loaded_session->LoadFromFile(save_path, gemma)) {
        std::cout << "Session loaded successfully\n";
        std::cout << "Conversation length: " << loaded_session->GetConversationLength() << "\n";

        // Show conversation history
        auto history = loaded_session->GetHistory(5); // Last 5 messages
        std::cout << "\nRecent conversation history:\n";
        for (const auto& msg : history) {
          std::cout << "  " << ((msg.role == MessageRole::USER) ? "User" : "Assistant")
                    << ": " << msg.content << "\n";
        }
      } else {
        std::cout << "Failed to load session\n";
      }
    } else {
      std::cout << "Failed to save session\n";
    }

    // Demonstrate session manager capabilities
    std::cout << "\n=== Session Manager Demo ===\n";
    auto manager_stats = manager.GetStats();
    std::cout << "Manager Statistics:\n"
              << "  Total sessions created: " << manager_stats.total_sessions_created << "\n"
              << "  Current active sessions: " << manager_stats.current_active_sessions << "\n"
              << "  Total conversations: " << manager_stats.total_conversations << "\n"
              << "  Total tokens processed: " << manager_stats.total_tokens_processed << "\n";

    // List all sessions
    auto session_list = manager.ListSessions();
    std::cout << "\nActive sessions:\n";
    for (const auto& id : session_list) {
      std::cout << "  - " << id << "\n";
    }

    std::cout << "\nSession demo completed successfully!\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}

} // namespace gcpp

int main() {
  return gcpp::SessionExample();
}