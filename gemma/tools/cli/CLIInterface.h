// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Enhanced CLI interface for Gemma.cpp with modern REPL functionality

#ifndef GEMMA_CLI_INTERFACE_H_
#define GEMMA_CLI_INTERFACE_H_

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <fstream>
#include <mutex>

#include "gemma/gemma.h"
#include "gemma/gemma_args.h"
#include "ops/matmul.h"
#include "util/threading_context.h"

namespace gcpp {

// Forward declarations
struct TimingInfo;
class KVCache;

// Color codes for terminal output
namespace Colors {
    extern const char* Reset;
    extern const char* Bold;
    extern const char* Red;
    extern const char* Green;
    extern const char* Yellow;
    extern const char* Blue;
    extern const char* Magenta;
    extern const char* Cyan;
    extern const char* White;
    extern const char* BrightRed;
    extern const char* BrightGreen;
    extern const char* BrightYellow;
    extern const char* BrightBlue;
    extern const char* BrightMagenta;
    extern const char* BrightCyan;
    extern const char* BrightWhite;
}

// Progress bar for model loading and inference
class ProgressBar {
public:
    ProgressBar(const std::string& label, size_t total, bool show_percentage = true);
    ~ProgressBar();
    
    void Update(size_t current);
    void Finish();
    void SetLabel(const std::string& label);
    
private:
    void Render();
    
    std::string label_;
    size_t total_;
    size_t current_;
    bool show_percentage_;
    bool finished_;
    std::mutex mutex_;
};

// Configuration for CLI behavior
struct CLIConfig {
    bool enable_colors = true;
    bool enable_history = true;
    bool enable_completion = true;
    bool verbose = false;
    bool show_timing = true;
    bool show_progress = true;
    std::string history_file;
    size_t max_history_size = 1000;
    std::string prompt_style = "gemma> ";
    std::string multiline_prompt = "  ... ";
    
    // Load configuration from file
    bool LoadFromFile(const std::string& filename);
    bool SaveToFile(const std::string& filename) const;
};

// Session management for save/load functionality
struct Session {
    struct Entry {
        std::string prompt;
        std::string response;
        std::chrono::system_clock::time_point timestamp;
        TimingInfo timing;
    };
    
    std::vector<Entry> entries;
    std::string model_name;
    std::string backend_name;
    
    bool SaveToFile(const std::string& filename) const;
    bool LoadFromFile(const std::string& filename);
    void Clear();
};

// Command handler function type
using CommandHandler = std::function<bool(const std::vector<std::string>& args)>;

// Main CLI interface class
class CLIInterface {
public:
    CLIInterface(const LoaderArgs& loader, const ThreadingArgs& threading, 
                 const InferenceArgs& inference);
    ~CLIInterface();
    
    // Initialize the CLI (load model, setup environment)
    bool Initialize();
    
    // Configuration
    void SetConfig(const CLIConfig& config);
    const CLIConfig& GetConfig() const { return config_; }
    
    // Main execution modes
    int RunInteractive();
    int RunBatch(const std::string& filename);
    int ProcessSinglePrompt(const std::string& prompt);
    
    // Model and backend management
    bool LoadModel(const std::string& model_path);
    bool SwitchBackend(const std::string& backend_name);
    std::vector<std::string> GetAvailableBackends() const;
    
    // Session management
    bool SaveSession(const std::string& filename);
    bool LoadSession(const std::string& filename);
    void ClearSession();
    
    // Utility functions
    std::string FormatResponse(const std::string& text) const;
    void ShowConfig() const;
    void ShowModelInfo() const;
    void ShowTiming(const TimingInfo& timing) const;
    
private:
    // REPL functionality
    std::string ReadInput();
    bool ProcessCommand(const std::string& input);
    bool IsCommand(const std::string& input) const;
    std::vector<std::string> ParseCommand(const std::string& command) const;
    
    // Command handlers
    void RegisterCommands();
    bool HandleHelp(const std::vector<std::string>& args);
    bool HandleModel(const std::vector<std::string>& args);
    bool HandleBackend(const std::vector<std::string>& args);
    bool HandleSession(const std::vector<std::string>& args);
    bool HandleConfig(const std::vector<std::string>& args);
    bool HandleHistory(const std::vector<std::string>& args);
    bool HandleClear(const std::vector<std::string>& args);
    bool HandleExit(const std::vector<std::string>& args);
    bool HandleInfo(const std::vector<std::string>& args);
    bool HandleBenchmark(const std::vector<std::string>& args);
    
    // History management
    void LoadHistory();
    void SaveHistory();
    void AddToHistory(const std::string& command);
    std::vector<std::string> GetHistory() const;
    
    // Tab completion
    std::vector<std::string> GetCompletions(const std::string& partial) const;
    std::vector<std::string> CompleteCommand(const std::string& partial) const;
    std::vector<std::string> CompleteFile(const std::string& partial) const;
    
    // Text generation
    std::string GenerateResponse(const std::string& prompt);
    void StreamToken(int token, float prob, bool& should_continue);
    
    // Utility methods
    std::string Colorize(const std::string& text, const char* color) const;
    void PrintError(const std::string& message) const;
    void PrintWarning(const std::string& message) const;
    void PrintInfo(const std::string& message) const;
    void PrintSuccess(const std::string& message) const;
    void ClearScreen() const;
    
    // Member variables
    CLIConfig config_;
    LoaderArgs loader_args_;
    ThreadingArgs threading_args_;
    InferenceArgs inference_args_;
    
    // Gemma components
    std::unique_ptr<ThreadingContext> ctx_;
    std::unique_ptr<MatMulEnv> env_;
    std::unique_ptr<Gemma> gemma_;
    std::unique_ptr<KVCache> kv_cache_;
    
    // CLI state
    bool initialized_;
    bool should_exit_;
    Session current_session_;
    std::vector<std::string> command_history_;
    std::unordered_map<std::string, CommandHandler> commands_;
    
    // Current conversation state
    size_t conversation_pos_;
    std::string current_backend_;
    std::string current_model_;
    
    // Threading and synchronization
    mutable std::mutex output_mutex_;
    std::atomic<bool> generation_interrupted_;
};

// Utility functions
std::string GetDefaultHistoryFile();
std::string GetDefaultConfigFile();
std::string GetPrompt(const InferenceArgs& inference);
bool IsValidModelFile(const std::string& path);
std::vector<std::string> FindModelFiles(const std::string& directory);

} // namespace gcpp

#endif // GEMMA_CLI_INTERFACE_H_