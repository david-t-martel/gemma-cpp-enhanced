// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Minimal standalone entrypoint for the Gemma MCP (Model Context Protocol)
// server. This wires together the MCPServer facilities with a lightweight
// command line interface so the server can be launched as a separate
// process and communicate over stdio (default) or any configured transport.
//
// Design goals:
//  - Zero external dependencies (no argparse library) – simple manual parsing.
//  - Non-fatal startup when a model path is not supplied (tools that do not
//    require inference can still respond). A warning is emitted instead.
//  - Graceful shutdown on Ctrl+C / termination signals.
//  - Clear diagnostics for effective configuration.
//  - Keep logic here thin; defer policy & defaults to server_utils.
//
// Example usage:
//   gemma_mcp_server \
//       --model model.sbs --tokenizer tokenizer.spm \
//       --transport stdio: --transport tcp://127.0.0.1:8080 \
//       --max-tokens 512 --temperature 0.8 --log-level 1
//
// Build (from repo root):
//   cmake -S . -B build -DGEMMA_BUILD_MCP_EXECUTABLE=ON
//   cmake --build build -j
//   (binary at build/mcp/gemma_mcp_server[.exe])

#include "MCPServer.h"
#include "MCPTools.h"
#include "MCPTransport.h"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#endif

using gcpp::mcp::MCPServer;
using gcpp::mcp::MCPServerConfig;
namespace server_utils = gcpp::mcp::server_utils;

namespace {

std::atomic<bool> g_shutdown_requested{false};

void PrintUsage(const char* argv0) {
  std::cout << "Gemma MCP Server\n"
            << "Usage: " << argv0 << " [options]\n\n"
            << "Options:\n"
            << "  --model <path>            Path to .sbs (single) weights file\n"
            << "  --tokenizer <path>        Path to sentencepiece tokenizer (optional if embedded)\n"
            << "  --transport <uri>         Transport URI (repeatable). Default: stdio:\n"
            << "                             Supported: stdio: | tcp://host:port | ws://host:port[/path]\n"
            << "  --max-tokens <n>          Default generation max tokens (default 512)\n"
            << "  --temperature <f>         Sampling temperature (default 0.8)\n"
            << "  --top-k <n>               Top-K sampling (default 40)\n"
            << "  --top-p <f>               Nucleus sampling probability (default 0.95)\n"
            << "  --session-timeout <sec>   Session idle timeout seconds (default 3600)\n"
            << "  --log-level <0|1|2>       Logging verbosity (default 1)\n"
            << "  --allow-tool <name>       Restrict allowed tools (repeatable). Empty = all\n"
            << "  --no-stream               Disable streaming capability\n"
            << "  --help                    Show this help and exit\n";
}

bool ParseInt(const char* s, long long& out) {
  char* end = nullptr;
  errno = 0;
  long long v = std::strtoll(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0') return false;
  out = v;
  return true;
}

bool ParseFloat(const char* s, double& out) {
  char* end = nullptr;
  errno = 0;
  double v = std::strtod(s, &end);
  if (errno != 0 || end == s || *end != '\0') return false;
  out = v;
  return true;
}

MCPServerConfig ParseArgs(int argc, char** argv) {
  MCPServerConfig cfg = server_utils::CreateDefaultConfig();

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](const char* name) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        PrintUsage(argv[0]);
        std::exit(2);
      }
    };

    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else if (arg == "--model") {
      need_value("--model");
      cfg.model_path = argv[++i];
    } else if (arg == "--tokenizer") {
      need_value("--tokenizer");
      cfg.tokenizer_path = argv[++i];
    } else if (arg == "--transport") {
      need_value("--transport");
      auto uri = std::string(argv[++i]);
      if (cfg.transport_uris.size() == 1 && cfg.transport_uris[0] == "stdio:" && uri != "stdio:") {
        // Replace default if user supplies custom transport.
        cfg.transport_uris.clear();
      }
      cfg.transport_uris.push_back(uri);
    } else if (arg == "--max-tokens") {
      need_value("--max-tokens");
      long long v; if (!ParseInt(argv[++i], v) || v <= 0) { std::cerr << "Invalid --max-tokens\n"; std::exit(2);} 
      cfg.default_runtime_config.max_tokens = static_cast<int>(v);
    } else if (arg == "--temperature") {
      need_value("--temperature");
      double v; if (!ParseFloat(argv[++i], v) || v < 0.0) { std::cerr << "Invalid --temperature\n"; std::exit(2);} 
      cfg.default_runtime_config.temperature = static_cast<float>(v);
    } else if (arg == "--top-k") {
      need_value("--top-k");
      long long v; if (!ParseInt(argv[++i], v) || v < 0) { std::cerr << "Invalid --top-k\n"; std::exit(2);} 
      cfg.default_runtime_config.top_k = static_cast<int>(v);
    } else if (arg == "--top-p") {
      need_value("--top-p");
      double v; if (!ParseFloat(argv[++i], v) || v <= 0.0 || v > 1.0) { std::cerr << "Invalid --top-p\n"; std::exit(2);} 
      cfg.default_runtime_config.top_p = static_cast<float>(v);
    } else if (arg == "--session-timeout") {
      need_value("--session-timeout");
      long long v; if (!ParseInt(argv[++i], v) || v <= 0) { std::cerr << "Invalid --session-timeout\n"; std::exit(2);} 
      cfg.session_timeout = std::chrono::seconds(static_cast<long long>(v));
    } else if (arg == "--log-level") {
      need_value("--log-level");
      long long v; if (!ParseInt(argv[++i], v) || v < 0 || v > 2) { std::cerr << "Invalid --log-level (0..2)\n"; std::exit(2);} 
      cfg.log_level = static_cast<int>(v);
    } else if (arg == "--allow-tool") {
      need_value("--allow-tool");
      cfg.allowed_tools.push_back(argv[++i]);
    } else if (arg == "--no-stream") {
      cfg.enable_streaming = false;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      PrintUsage(argv[0]);
      std::exit(2);
    }
  }

  return cfg;
}

void InstallSignalHandlers();

MCPServer* g_server_ptr = nullptr; // non-owning raw pointer for signal path

void SignalHandler(int sig) {
  if (g_shutdown_requested.exchange(true)) return; // already handled
  if (g_server_ptr) {
    std::cerr << "\n[signal] Caught signal " << sig << ", initiating shutdown..." << std::endl;
    g_server_ptr->Shutdown();
  }
}

void InstallSignalHandlers() {
#if defined(_WIN32)
  SetConsoleCtrlHandler([](DWORD type) -> BOOL {
    switch (type) {
      case CTRL_C_EVENT:
      case CTRL_BREAK_EVENT:
      case CTRL_CLOSE_EVENT:
        SignalHandler(SIGINT);
        return TRUE;
      default:
        return FALSE;
    }
  }, TRUE);
#else
  std::signal(SIGINT, SignalHandler);
  std::signal(SIGTERM, SignalHandler);
#endif
}

} // namespace

int main(int argc, char** argv) {
  auto config = ParseArgs(argc, argv);

  if (config.model_path.empty()) {
    std::cerr << "[warn] No --model supplied. Generation tools will not function until a model is loaded (future dynamic loading not yet implemented)." << std::endl;
  }

  server_utils::SetupLogging(config.log_level);

  auto server = server_utils::CreateStandardServer(config);
  g_server_ptr = server.get();
  InstallSignalHandlers();

  if (!server->Initialize()) {
    std::cerr << "[error] Server initialization failed." << std::endl;
    return 1;
  }

  if (!server->Start()) {
    std::cerr << "[error] Failed to start transports." << std::endl;
    return 2;
  }

  std::cout << "[info] Gemma MCP server started (status=" << server->GetStatusString() << ")\n";
  std::cout << "[info] Transports: ";
  auto names = server->GetTransportNames();
  for (size_t i = 0; i < names.size(); ++i) {
    std::cout << names[i]; if (i + 1 < names.size()) std::cout << ",";
  }
  std::cout << std::endl;
  if (server->IsModelLoaded()) {
    std::cout << "[info] Model loaded: \n" << server->GetModelInfo() << std::endl;
  }

  // Block until shutdown (signal sets g_shutdown_requested)
  while (!g_shutdown_requested.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  std::cout << "[info] Shutdown complete." << std::endl;
  return 0;
}
