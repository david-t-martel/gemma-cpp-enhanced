// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Enhanced CLI interface for Gemma.cpp with modern REPL functionality

#include <iostream>
#include <exception>
#include <memory>
#include <string>

#include "CLIInterface.h"
#include "gemma/gemma.h"
#include "util/args.h"
#include "hwy/base.h"

#if (!defined(HWY_VERSION_LT) || HWY_VERSION_LT(1, 2)) && !HWY_IDE
#error "Please update to version 1.2 of github.com/google/highway."
#endif
#if HWY_CXX_LANG < 201703L
#error "Gemma.cpp requires C++17, please pass -std=c++17."
#endif

namespace gcpp {

static constexpr std::string_view kBanner = R"(
┌─────────────────────────────────────────────────────────────────┐
│                      Enhanced Gemma CLI                        │
│              Interactive REPL with Advanced Features           │
└─────────────────────────────────────────────────────────────────┘
)";

struct CLIArgs {
    bool interactive = false;
    bool batch_mode = false;
    std::string batch_file;
    std::string history_file;
    bool no_color = false;
    bool verbose = false;
    std::string config_file;
    bool show_help = false;
    
    // Parse command line arguments specific to CLI
    void Parse(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg == "--interactive" || arg == "-i") {
                interactive = true;
            } else if (arg == "--batch" || arg == "-b") {
                batch_mode = true;
                if (i + 1 < argc) {
                    batch_file = argv[++i];
                }
            } else if (arg == "--history" || arg == "-h") {
                if (i + 1 < argc) {
                    history_file = argv[++i];
                }
            } else if (arg == "--no-color") {
                no_color = true;
            } else if (arg == "--verbose" || arg == "-v") {
                verbose = true;
            } else if (arg == "--config" || arg == "-c") {
                if (i + 1 < argc) {
                    config_file = argv[++i];
                }
            } else if (arg == "--help" || arg == "-help" || arg == "/?") {
                show_help = true;
            }
        }
        
        // If no mode specified, default to interactive if no batch file
        if (!interactive && !batch_mode) {
            interactive = batch_file.empty();
        }
    }
};

void ShowCLIHelp() {
    std::cout << "Enhanced Gemma CLI - Interactive REPL Interface\n\n";
    std::cout << "Usage: gemma_cli [OPTIONS] [GEMMA_OPTIONS]\n\n";
    std::cout << "CLI-specific options:\n";
    std::cout << "  --interactive, -i         Force interactive mode (default if no batch)\n";
    std::cout << "  --batch FILE, -b FILE     Run commands from batch file\n";
    std::cout << "  --history FILE, -h FILE   Use custom history file\n";
    std::cout << "  --no-color                Disable colored output\n";
    std::cout << "  --verbose, -v             Enable verbose output\n";
    std::cout << "  --config FILE, -c FILE    Load CLI configuration from file\n";
    std::cout << "  --help                    Show this help\n\n";
    
    std::cout << "Interactive Commands:\n";
    std::cout << "  /help                     Show available commands\n";
    std::cout << "  /model [MODEL]            Load/switch model\n";
    std::cout << "  /backend [BACKEND]        Switch backend (cpu/intel/cuda/vulkan)\n";
    std::cout << "  /session save FILE        Save current session\n";
    std::cout << "  /session load FILE        Load session from file\n";
    std::cout << "  /session clear            Clear current session\n";
    std::cout << "  /config                   Show current configuration\n";
    std::cout << "  /history                  Show command history\n";
    std::cout << "  /clear                    Clear screen\n";
    std::cout << "  /exit, /quit              Exit the CLI\n\n";
    
    std::cout << "Key bindings:\n";
    std::cout << "  Up/Down arrows            Navigate command history\n";
    std::cout << "  Tab                       Command/path completion\n";
    std::cout << "  Ctrl+C                    Interrupt current generation\n";
    std::cout << "  Ctrl+D                    Exit CLI\n";
    std::cout << "  Ctrl+L                    Clear screen\n\n";
}

} // namespace gcpp

int main(int argc, char** argv) {
    gcpp::InternalInit();
    
    try {
        // Parse CLI-specific arguments first
        gcpp::CLIArgs cli_args;
        cli_args.Parse(argc, argv);
        
        if (cli_args.show_help || gcpp::HasHelp(argc, argv)) {
            std::cout << gcpp::kBanner << "\n";
            gcpp::ShowCLIHelp();
            std::cout << "\nGemma Engine Options:\n";
            
            // Show Gemma-specific help
            gcpp::LoaderArgs loader(argc, argv);
            gcpp::ThreadingArgs threading(argc, argv);
            gcpp::InferenceArgs inference(argc, argv);
            gcpp::ShowHelp(loader, threading, inference);
            return 0;
        }
        
        // Initialize Gemma components
        gcpp::LoaderArgs loader(argc, argv);
        gcpp::ThreadingArgs threading(argc, argv);
        gcpp::InferenceArgs inference(argc, argv);
        
        // Create and configure CLI interface
        auto cli = std::make_unique<gcpp::CLIInterface>(loader, threading, inference);
        
        // Configure CLI options
        gcpp::CLIConfig config;
        config.enable_colors = !cli_args.no_color;
        config.verbose = cli_args.verbose;
        config.history_file = cli_args.history_file.empty() ? 
            gcpp::GetDefaultHistoryFile() : cli_args.history_file;
        
        if (!cli_args.config_file.empty()) {
            config.LoadFromFile(cli_args.config_file);
        }
        
        cli->SetConfig(config);
        
        // Initialize the CLI
        if (!cli->Initialize()) {
            std::cerr << "Failed to initialize CLI interface\n";
            return 1;
        }
        
        int result = 0;
        
        if (cli_args.batch_mode && !cli_args.batch_file.empty()) {
            // Batch mode - process commands from file
            result = cli->RunBatch(cli_args.batch_file);
        } else if (cli_args.interactive) {
            // Interactive mode - start REPL
            std::cout << gcpp::kBanner << "\n";
            result = cli->RunInteractive();
        } else {
            // Single prompt mode (for compatibility)
            std::string prompt = gcpp::GetPrompt(inference);
            if (!prompt.empty()) {
                result = cli->ProcessSinglePrompt(prompt);
            } else {
                std::cerr << "No prompt provided. Use --interactive for REPL mode.\n";
                return 1;
            }
        }
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}