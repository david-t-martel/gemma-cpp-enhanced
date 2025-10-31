// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Command line text interface to gemma.

#include <stdio.h>

#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "compression/types.h"  // PromptWrapping
#include "evals/benchmark_helper.h"
#include "gemma/gemma.h"  // Gemma
#include "gemma/gemma_args.h"
#include "gemma/session.h"  // Session management
#include "gemma/tokenizer.h"  // WrapAndTokenize
#include "ops/matmul.h"       // MatMulEnv
#include "paligemma/image.h"
#include "util/args.h"  // HasHelp
#include "hwy/base.h"
#include "hwy/highway.h"
#include "hwy/profiler.h"

#if (!defined(HWY_VERSION_LT) || HWY_VERSION_LT(1, 2)) && !HWY_IDE
#error "Please update to version 1.2 of github.com/google/highway."
#endif
#if HWY_CXX_LANG < 201703L
#error "Gemma.cpp requires C++17, please pass -std=c++17."
#endif

static constexpr bool kVerboseLogTokens = false;

namespace gcpp {

static constexpr std::string_view kAsciiArtBanner = R""(
  __ _  ___ _ __ ___  _ __ ___   __ _   ___ _ __  _ __
 / _` |/ _ \ '_ ` _ \| '_ ` _ \ / _` | / __| '_ \| '_ \
| (_| |  __/ | | | | | | | | | | (_| || (__| |_) | |_) |
 \__, |\___|_| |_| |_|_| |_| |_|\__,_(_)___| .__/| .__/
  __/ |                                    | |   | |
 |___/                                     |_|   |_|
)"";

std::string GetPromptFromStream(std::istream& input, int verbosity,
                                std::string_view eot_line) {
  if (verbosity >= 1) {
    std::cout << "> " << std::flush;
  }

  std::string prompt_string;
  if (eot_line.empty()) {
    std::getline(input, prompt_string);
  } else {
    std::string line;
    while (std::getline(input, line)) {
      if (line == eot_line) {
        break;
      }
      prompt_string += line + "\n";
    }
  }
  return prompt_string;
}

// Get prompt either from interactive input or command line
std::string GetPrompt(const InferenceArgs& inference) {
  // If prompt is provided via command line, use that
  if (!inference.prompt.empty()) return inference.prompt;
  if (!inference.prompt_file.Empty()) {
    PROFILER_ZONE("Gen.ReadPrompt");
    return ReadFileToString(inference.prompt_file);
  }

  PROFILER_ZONE("Gen.input");
  return GetPromptFromStream(std::cin, inference.verbosity, inference.eot_line);
}

// The main Read-Eval-Print Loop.
void ReplGemma(const ThreadingArgs& threading, const InferenceArgs& inference,
               const Gemma& gemma, KVCache& kv_cache, MatMulEnv& env) {
  PROFILER_ZONE("Gen.misc");
  size_t abs_pos = 0;                     // across turns
  size_t tokens_generated_this_turn = 0;  // differentiates prefill from reply
  size_t prompt_size = 0;
  const ModelConfig& config = gemma.Config();

  std::mt19937 gen;
  InitGenerator(inference, gen);

  const bool have_image = !inference.image_file.path.empty();
  Image image;
  const size_t pool_dim = config.vit_config.pool_dim;
  ImageTokens image_tokens(
      "image_tokens",
      have_image ? Extents2D(config.vit_config.seq_len / (pool_dim * pool_dim),
                             config.model_dim)
                 : Extents2D(0, 0),
      env.ctx.allocator, MatPadding::kOdd);
  image_tokens.AllocateAndAttachRowPtrs(env.row_ptrs);
  if (have_image) {
    HWY_ASSERT(config.wrapping == PromptWrapping::PALIGEMMA ||
               config.wrapping == PromptWrapping::GEMMA_VLM);
    HWY_ASSERT(image.ReadPPM(inference.image_file.path));
    const size_t image_size = config.vit_config.image_size;
    image.Resize(image_size, image_size);
    RuntimeConfig runtime_config = {.gen = &gen,
                                    .verbosity = inference.verbosity,
                                    .use_spinning = threading.spin};
    double image_tokens_start = hwy::platform::Now();
    gemma.GenerateImageTokens(runtime_config, kv_cache.SeqLen(), image,
                              image_tokens, env);
    if (inference.verbosity >= 1) {
      double image_tokens_duration = hwy::platform::Now() - image_tokens_start;
      fprintf(stderr,
              "\n\n[ Timing info ] Image token generation took: %d ms\n",
              static_cast<int>(image_tokens_duration * 1000));
    }
  }

  // callback function invoked for each generated token.
  auto stream_token = [&](int token, float) {
    ++abs_pos;
    const bool in_prompt = tokens_generated_this_turn < prompt_size;
    const bool first_response_token = tokens_generated_this_turn == prompt_size;
    ++tokens_generated_this_turn;
    if (in_prompt) {
      if (inference.verbosity >= 1) {
        std::cout << "." << std::flush;
      }
      return true;
    } else if (config.IsEOS(token)) {
      if (inference.verbosity >= 2) {
        std::cout << "\n[ End ]\n";
      }
      return true;
    }
    std::string token_text;
    HWY_ASSERT(gemma.Tokenizer().Decode(std::vector<int>{token}, &token_text));
    if (first_response_token) {
      token_text.erase(0, token_text.find_first_not_of(" \t\n"));
      if (inference.verbosity >= 1) {
        std::cout << "\n\n";
      }
    }
    std::cout << token_text << std::flush;
    return true;
  };

  while (true) {  // Loop until user quits.
    tokens_generated_this_turn = 0;

    const std::string prompt_string = GetPrompt(inference);
    const bool is_interactive = inference.IsInteractive();
    if (is_interactive) {  // handle special commands:
      if (!std::cin) return;

      // If !eot_line.empty(), we append \n, so only look at the first 2 chars.
      if (prompt_string.size() >= 2 && prompt_string[0] == '%') {
        if (prompt_string[1] == 'q' || prompt_string[1] == 'Q') return;
        if (prompt_string[1] == 'c' || prompt_string[1] == 'C') {
          abs_pos = 0;
          continue;
        }
      }

      if (prompt_string.empty()) {
        std::cout << "Use '%q' to quit.\n";
        continue;
      }
    }

    // Set up runtime config.
    TimingInfo timing_info = {.verbosity = inference.verbosity};
    RuntimeConfig runtime_config = {.gen = &gen,
                                    .verbosity = inference.verbosity,
                                    .stream_token = stream_token,
                                    .use_spinning = threading.spin};
    inference.CopyTo(runtime_config);
    std::vector<int> prompt;
    size_t prefix_end = 0;
    if (have_image) {
      prompt = WrapAndTokenize(gemma.Tokenizer(), gemma.ChatTemplate(),
                               config.wrapping, abs_pos, prompt_string,
                               image_tokens.Rows());
      runtime_config.image_tokens = &image_tokens;
      prompt_size = prompt.size();
      if (config.wrapping == PromptWrapping::PALIGEMMA) {
        // The end of the prefix for prefix-LM style attention in Paligemma.
        // See Figure 2 of https://arxiv.org/abs/2407.07726.
        prefix_end = prompt_size;
        // We need to look at all the tokens for the prefix.
        // NOTE: Online softmax is on the roadmap, after which this requirement
        // can be lifted.
        runtime_config.prefill_tbatch_size = prompt_size;
      }
    } else {
      prompt = WrapAndTokenize(gemma.Tokenizer(), gemma.ChatTemplate(),
                               config.wrapping, abs_pos, prompt_string);
      prompt_size = prompt.size();
    }

    if constexpr (kVerboseLogTokens) {
      for (int i = 0; i < prompt_size; ++i) {
        fprintf(stderr, "DDD TOKEN %3d: %6d\n", i, prompt[i]);
      }
    }

    // Generate until EOS or max_generated_tokens.
    if (inference.verbosity >= 1) {
      std::cerr << "\n[ Reading prompt ] " << std::flush;
    }
    gemma.Generate(runtime_config, prompt, abs_pos, prefix_end, kv_cache, env,
                   timing_info);
    std::cout << "\n\n";

    // In non-interactive mode, we only process one prompt/turn.
    if (!is_interactive) break;

    // Prepare for the next turn. Works only for PaliGemma.
    if (!inference.multiturn || config.wrapping == PromptWrapping::PALIGEMMA) {
      abs_pos = 0;  // Start a new turn at position 0.
      InitGenerator(inference, gen);
    } else {
      // The last token was either EOS, then it should be ignored because it is
      // never part of the dialog, see Table 5 in the Gemma-2 paper:
      // https://arxiv.org/pdf/2408.00118
      // Or we have hit max_generated_tokens, then the last token will be lost.
      // (We could store it in stream_token, and then prepend to the next turn,
      // but it's not worth the complexity, as multi-turn with max_generated is
      // not a common use case.)
      // In either case, we need to rewind abs_pos by one.
      HWY_ASSERT(abs_pos > 0);
      abs_pos--;
    }
  }
}

// Session-enabled REPL that uses GemmaSession for conversation management
void ReplGemmaWithSession(const ThreadingArgs& threading,
                          const InferenceArgs& inference,
                          const Gemma& gemma,
                          std::shared_ptr<GemmaSession> session,
                          SessionManager* session_manager,
                          MatMulEnv& env) {
  PROFILER_ZONE("Gen.misc");
  const ModelConfig& config = gemma.Config();

  std::mt19937 gen;
  InitGenerator(inference, gen);

  // Handle image tokens if needed
  const bool have_image = !inference.image_file.path.empty();
  Image image;
  const size_t pool_dim = config.vit_config.pool_dim;
  ImageTokens image_tokens(
      "image_tokens",
      have_image ? Extents2D(config.vit_config.seq_len / (pool_dim * pool_dim),
                             config.model_dim)
                 : Extents2D(0, 0),
      env.ctx.allocator, MatPadding::kOdd);
  image_tokens.AllocateAndAttachRowPtrs(env.row_ptrs);
  if (have_image) {
    HWY_ASSERT(config.wrapping == PromptWrapping::PALIGEMMA ||
               config.wrapping == PromptWrapping::GEMMA_VLM);
    HWY_ASSERT(image.ReadPPM(inference.image_file.path));
    const size_t image_size = config.vit_config.image_size;
    image.Resize(image_size, image_size);
    RuntimeConfig runtime_config = {.gen = &gen,
                                    .verbosity = inference.verbosity,
                                    .use_spinning = threading.spin};
    double image_tokens_start = hwy::platform::Now();
    gemma.GenerateImageTokens(runtime_config, session->GetKVCacheSize(), image,
                              image_tokens, env);
    if (inference.verbosity >= 1) {
      double image_tokens_duration = hwy::platform::Now() - image_tokens_start;
      fprintf(stderr,
              "\n\n[ Timing info ] Image token generation took: %d ms\n",
              static_cast<int>(image_tokens_duration * 1000));
    }
  }

  while (true) {  // Loop until user quits.
    const std::string prompt_string = GetPrompt(inference);
    const bool is_interactive = inference.IsInteractive();

    if (is_interactive) {  // handle special commands:
      if (!std::cin) return;

      if (prompt_string.size() >= 2 && prompt_string[0] == '%') {
        char cmd = prompt_string[1];

        // %q or %Q - quit
        if (cmd == 'q' || cmd == 'Q') return;

        // %c or %C - clear session
        if (cmd == 'c' || cmd == 'C') {
          session->Reset();
          std::cout << "Session cleared.\n";
          continue;
        }

        // %s or %S - save session
        if (cmd == 's' || cmd == 'S') {
          std::string filename = "session_" + inference.session_id + ".json";
          if (prompt_string.size() > 3) {
            filename = prompt_string.substr(3);  // Use custom filename
          }
          Path session_file(filename);
          if (session->SaveToFile(session_file)) {
            std::cout << "Session saved to: " << filename << "\n";
          } else {
            std::cout << "Failed to save session.\n";
          }
          continue;
        }

        // %l or %L - load session
        if (cmd == 'l' || cmd == 'L') {
          std::string filename = "session_" + inference.session_id + ".json";
          if (prompt_string.size() > 3) {
            filename = prompt_string.substr(3);  // Use custom filename
          }
          Path session_file(filename);
          if (session->LoadFromFile(session_file, gemma)) {
            std::cout << "Session loaded from: " << filename << "\n";
          } else {
            std::cout << "Failed to load session.\n";
          }
          continue;
        }

        // %h or %H - show history
        if (cmd == 'h' || cmd == 'H') {
          size_t max_messages = 10;  // Default: show last 10 messages
          if (prompt_string.size() > 3) {
            try {
              max_messages = std::stoul(prompt_string.substr(3));
            } catch (...) {
              max_messages = 10;
            }
          }

          auto history = session->GetHistory(max_messages);
          std::cout << "\n=== Session History (last " << history.size()
                    << " messages) ===\n";

          for (const auto& msg : history) {
            std::string role_str;
            switch (msg.role) {
              case MessageRole::USER: role_str = "USER"; break;
              case MessageRole::ASSISTANT: role_str = "ASSISTANT"; break;
              case MessageRole::SYSTEM: role_str = "SYSTEM"; break;
            }
            std::cout << "\n[" << role_str << "]: " << msg.content << "\n";
          }
          std::cout << "=========================\n\n";
          continue;
        }

        // %i or %I - show statistics
        if (cmd == 'i' || cmd == 'I') {
          auto stats = session->GetStats();
          std::cout << "\n=== Session Statistics ===\n";
          std::cout << "Total turns: " << stats.total_turns.load() << "\n";
          std::cout << "Total input tokens: " << stats.total_input_tokens.load() << "\n";
          std::cout << "Total output tokens: " << stats.total_output_tokens.load() << "\n";
          std::cout << "Session state: " << static_cast<int>(session->GetState()) << "\n";
          std::cout << "Current token count: " << session->GetConversationLength() << " messages\n";
          std::cout << "KV cache size: " << session->GetKVCacheSize() << " tokens\n";
          std::cout << "==========================\n\n";
          continue;
        }

        // %m or %M - list all managed sessions
        if (cmd == 'm' || cmd == 'M') {
          if (session_manager) {
            auto session_list = session_manager->ListSessions();
            auto manager_stats = session_manager->GetStats();

            std::cout << "\n=== Managed Sessions ===\n";
            std::cout << "Active sessions: " << session_manager->GetActiveSessionCount() << "\n";
            std::cout << "Total created: " << manager_stats.total_sessions_created << "\n\n";

            if (session_list.empty()) {
              std::cout << "No sessions found.\n";
            } else {
              std::cout << "Session IDs:\n";
              for (const auto& id : session_list) {
                std::cout << "  - " << id;
                if (id == inference.session_id) {
                  std::cout << " (current)";
                }
                std::cout << "\n";
              }
            }
            std::cout << "========================\n\n";
          } else {
            std::cout << "Session manager not available.\n";
          }
          continue;
        }
      }

      if (prompt_string.empty()) {
        std::cout << "Session commands:\n";
        std::cout << "  %q - quit\n";
        std::cout << "  %c - clear session\n";
        std::cout << "  %s [filename] - save session (default: session_<id>.json)\n";
        std::cout << "  %l [filename] - load session (default: session_<id>.json)\n";
        std::cout << "  %h [N] - show last N messages (default: 10)\n";
        std::cout << "  %i - show session statistics\n";
        std::cout << "  %m - list all managed sessions\n";
        continue;
      }
    }

    // Add user message to session
    session->AddUserMessage(prompt_string);

    // Set up runtime config for generation
    RuntimeConfig runtime_config = {.gen = &gen,
                                    .verbosity = inference.verbosity,
                                    .use_spinning = threading.spin};
    inference.CopyTo(runtime_config);

    if (have_image) {
      runtime_config.image_tokens = &image_tokens;
    }

    // Stream tokens to console
    auto stream_callback = [&](const std::string& token_text, bool is_final) -> bool {
      if (!is_final) {
        std::cout << token_text << std::flush;
      }
      return true;  // Continue generation
    };

    if (inference.verbosity >= 1) {
      std::cerr << "\n[ Generating response ] " << std::flush;
    }

    // Generate response using session
    session->GenerateResponseStream(gemma, env, runtime_config, stream_callback);

    std::cout << "\n\n";

    // In non-interactive mode, we only process one prompt/turn.
    if (!is_interactive) break;
  }
}

void Run(const LoaderArgs& loader, const ThreadingArgs& threading,
         const InferenceArgs& inference) {
  PROFILER_ZONE("Run.misc");

  ThreadingContext ctx(threading);
  MatMulEnv env(ctx);
  if (inference.verbosity >= 2) env.print_best = true;
  const Gemma gemma(loader, inference, ctx);

  // Determine if we should use session management
  const bool use_session = !inference.session_id.empty() ||
                           inference.load_session ||
                           inference.save_on_exit;

  if (use_session) {
    // Create SessionManager and session
    SessionManager session_manager(gemma.Config(), inference, ctx);

    std::shared_ptr<GemmaSession> session;

    if (inference.load_session) {
      // Try to load existing session
      session = session_manager.GetSession(inference.session_id);
      if (!session) {
        if (inference.verbosity >= 1) {
          std::cerr << "Session '" << inference.session_id
                    << "' not found, creating new session.\n";
        }
        session = session_manager.CreateSession(inference.session_id);
      }
    } else {
      // Create new session
      session = session_manager.CreateSession(inference.session_id);
    }

    if (!session) {
      std::cerr << "Failed to create session.\n";
      return;
    }

    // Initialize session
    if (!session->Initialize(gemma)) {
      std::cerr << "Failed to initialize session.\n";
      return;
    }

    if (inference.verbosity >= 1) {
      std::string instructions =
          "*Usage*\n"
          "  Enter an instruction and press enter (%C resets conversation, "
          "%Q quits).\n"
          "  Session: " + inference.session_id + "\n";
      const std::string examples =
          "\n*Examples*\n"
          "  - Write an email to grandma thanking her for the cookies.\n"
          "  - What are some historical attractions to visit around "
          "Massachusetts?\n"
          "  - Compute the nth fibonacci number in javascript.\n"
          "  - Write a standup comedy bit about GPU programming.\n";
      instructions += examples;

      // Skip the banner and instructions in non-interactive mode
      if (inference.IsInteractive()) {
        std::cout << "\033[2J\033[1;1H"  // clear screen
                  << kAsciiArtBanner << "\n\n";
        ShowConfig(loader, threading, inference, gemma.Config(),
                   gemma.WeightReadMode(), ctx);
        std::cout << "\n" << instructions << "\n";
      }
    }

    // Use session-enabled REPL
    ReplGemmaWithSession(threading, inference, gemma, session, &session_manager, env);

    // Save session on exit if requested
    if (inference.save_on_exit) {
      Path session_file("session_" + inference.session_id + ".json");
      if (session->SaveToFile(session_file)) {
        if (inference.verbosity >= 1) {
          std::cout << "\nSession saved to " << session_file.path << "\n";
        }
      }
    }

  } else {
    // Use legacy REPL without session management
    KVCache kv_cache(gemma.Config(), inference, ctx.allocator);

    if (inference.verbosity >= 1) {
      std::string instructions =
          "*Usage*\n"
          "  Enter an instruction and press enter (%C resets conversation, "
          "%Q quits).\n";
      const std::string multiturn =
          inference.multiturn == 0
              ? std::string(
                    "  Since multiturn is set to 0, conversation will "
                    "automatically reset every turn.\n\n")
              : "\n";
      const std::string examples =
          "*Examples*\n"
          "  - Write an email to grandma thanking her for the cookies.\n"
          "  - What are some historical attractions to visit around "
          "Massachusetts?\n"
          "  - Compute the nth fibonacci number in javascript.\n"
          "  - Write a standup comedy bit about GPU programming.\n";
      instructions += multiturn;
      instructions += examples;

      // Skip the banner and instructions in non-interactive mode
      if (inference.IsInteractive()) {
        std::cout << "\033[2J\033[1;1H"  // clear screen
                  << kAsciiArtBanner << "\n\n";
        ShowConfig(loader, threading, inference, gemma.Config(),
                   gemma.WeightReadMode(), ctx);
        std::cout << "\n" << instructions << "\n";
      }
    }

    ReplGemma(threading, inference, gemma, kv_cache, env);
  }
}

}  // namespace gcpp

int main(int argc, char** argv) {
  gcpp::InternalInit();
  {
    // Negligible CPU time.
    gcpp::LoaderArgs loader(argc, argv);
    gcpp::ThreadingArgs threading(argc, argv);
    gcpp::InferenceArgs inference(argc, argv);

    if (gcpp::HasHelp(argc, argv)) {
      std::cerr << gcpp::kAsciiArtBanner;
      gcpp::ShowHelp(loader, threading, inference);
      return 0;
    }

    gcpp::Run(loader, threading, inference);
  }
  PROFILER_PRINT_RESULTS();  // Must call outside the zone above.
  return 0;
}
