// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Enhanced CLI interface for Gemma.cpp with modern REPL functionality

#include "CLIInterface.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <filesystem>
#include <random>

#ifdef _WIN32
#include <conio.h>
#include <windows.h>
#else
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>
#endif

#include "gemma/kv_cache.h"
#include "gemma/tokenizer.h"
#include "hwy/timer.h"
#include "hwy/profiler.h"

namespace gcpp {

// Color definitions
namespace Colors {
    const char* Reset = "\033[0m";
    const char* Bold = "\033[1m";
    const char* Red = "\033[31m";
    const char* Green = "\033[32m";
    const char* Yellow = "\033[33m";
    const char* Blue = "\033[34m";
    const char* Magenta = "\033[35m";
    const char* Cyan = "\033[36m";
    const char* White = "\033[37m";
    const char* BrightRed = "\033[91m";
    const char* BrightGreen = "\033[92m";
    const char* BrightYellow = "\033[93m";
    const char* BrightBlue = "\033[94m";
    const char* BrightMagenta = "\033[95m";
    const char* BrightCyan = "\033[96m";
    const char* BrightWhite = "\033[97m";
}

// ProgressBar Implementation
ProgressBar::ProgressBar(const std::string& label, size_t total, bool show_percentage)
    : label_(label), total_(total), current_(0), show_percentage_(show_percentage), finished_(false) {
    Render();
}

ProgressBar::~ProgressBar() {
    if (!finished_) {
        Finish();
    }
}

void ProgressBar::Update(size_t current) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_ = std::min(current, total_);
    if (!finished_) {
        Render();
    }
}

void ProgressBar::Finish() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!finished_) {
        current_ = total_;
        Render();
        std::cout << "\n";
        finished_ = true;
    }
}

void ProgressBar::SetLabel(const std::string& label) {
    std::lock_guard<std::mutex> lock(mutex_);
    label_ = label;
    if (!finished_) {
        Render();
    }
}

void ProgressBar::Render() {
    const int bar_width = 40;
    const float progress = total_ > 0 ? static_cast<float>(current_) / total_ : 0.0f;
    const int filled = static_cast<int>(progress * bar_width);
    
    std::cout << "\r" << label_ << " [";
    for (int i = 0; i < bar_width; ++i) {
        if (i < filled) {
            std::cout << "█";
        } else if (i == filled && current_ < total_) {
            std::cout << "▒";
        } else {
            std::cout << " ";
        }
    }
    std::cout << "] ";
    
    if (show_percentage_) {
        std::cout << static_cast<int>(progress * 100) << "%";
    }
    
    std::cout << std::flush;
}

// CLIConfig Implementation
bool CLIConfig::LoadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        
        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        // Parse configuration values
        if (key == "enable_colors") {
            enable_colors = (value == "true" || value == "1");
        } else if (key == "enable_history") {
            enable_history = (value == "true" || value == "1");
        } else if (key == "enable_completion") {
            enable_completion = (value == "true" || value == "1");
        } else if (key == "verbose") {
            verbose = (value == "true" || value == "1");
        } else if (key == "show_timing") {
            show_timing = (value == "true" || value == "1");
        } else if (key == "show_progress") {
            show_progress = (value == "true" || value == "1");
        } else if (key == "history_file") {
            history_file = value;
        } else if (key == "max_history_size") {
            max_history_size = std::stoull(value);
        } else if (key == "prompt_style") {
            prompt_style = value;
        } else if (key == "multiline_prompt") {
            multiline_prompt = value;
        }
    }
    
    return true;
}

bool CLIConfig::SaveToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << "# Enhanced Gemma CLI Configuration\n";
    file << "enable_colors=" << (enable_colors ? "true" : "false") << "\n";
    file << "enable_history=" << (enable_history ? "true" : "false") << "\n";
    file << "enable_completion=" << (enable_completion ? "true" : "false") << "\n";
    file << "verbose=" << (verbose ? "true" : "false") << "\n";
    file << "show_timing=" << (show_timing ? "true" : "false") << "\n";
    file << "show_progress=" << (show_progress ? "true" : "false") << "\n";
    file << "history_file=" << history_file << "\n";
    file << "max_history_size=" << max_history_size << "\n";
    file << "prompt_style=" << prompt_style << "\n";
    file << "multiline_prompt=" << multiline_prompt << "\n";
    
    return true;
}

// Session Implementation
bool Session::SaveToFile(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    
    // Simple text format for now (could be enhanced with JSON/binary)
    file << "# Enhanced Gemma CLI Session\n";
    file << "model=" << model_name << "\n";
    file << "backend=" << backend_name << "\n";
    file << "entries=" << entries.size() << "\n";
    
    for (const auto& entry : entries) {
        auto time_t = std::chrono::system_clock::to_time_t(entry.timestamp);
        file << "---\n";
        file << "timestamp=" << time_t << "\n";
        file << "prompt_len=" << entry.prompt.length() << "\n";
        file << entry.prompt << "\n";
        file << "response_len=" << entry.response.length() << "\n";
        file << entry.response << "\n";
    }
    
    return true;
}

bool Session::LoadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    Clear();
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        
        if (key == "model") {
            model_name = value;
        } else if (key == "backend") {
            backend_name = value;
        } else if (key == "entries") {
            size_t count = std::stoull(value);
            entries.reserve(count);
        }
        // TODO: Implement full entry parsing
    }
    
    return true;
}

void Session::Clear() {
    entries.clear();
    model_name.clear();
    backend_name.clear();
}

// CLIInterface Implementation
CLIInterface::CLIInterface(const LoaderArgs& loader, const ThreadingArgs& threading, 
                          const InferenceArgs& inference)
    : loader_args_(loader), threading_args_(threading), inference_args_(inference),
      initialized_(false), should_exit_(false), conversation_pos_(0),
      generation_interrupted_(false) {
    
    RegisterCommands();
}

CLIInterface::~CLIInterface() {
    if (config_.enable_history) {
        SaveHistory();
    }
}

bool CLIInterface::Initialize() {
    if (initialized_) {
        return true;
    }
    
    try {
        // Show progress during initialization if enabled
        std::unique_ptr<ProgressBar> progress;
        if (config_.show_progress) {
            progress = std::make_unique<ProgressBar>("Initializing", 4);
        }
        
        // Initialize threading context
        ctx_ = std::make_unique<ThreadingContext>(threading_args_);
        if (progress) progress->Update(1);
        
        // Initialize matrix multiplication environment
        env_ = std::make_unique<MatMulEnv>(*ctx_);
        if (inference_args_.verbosity >= 2) {
            env_->print_best = true;
        }
        if (progress) progress->Update(2);
        
        // Load Gemma model
        gemma_ = std::make_unique<Gemma>(loader_args_, inference_args_, *ctx_);
        current_model_ = loader_args_.weights.path;
        if (progress) progress->Update(3);
        
        // Initialize KV cache
        kv_cache_ = std::make_unique<KVCache>(gemma_->Config(), inference_args_, ctx_->allocator);
        if (progress) progress->Update(4);
        
        if (progress) {
            progress->Finish();
        }
        
        // Initialize session
        current_session_.model_name = current_model_;
        current_session_.backend_name = "cpu"; // Default backend
        
        // Load history if enabled
        if (config_.enable_history) {
            LoadHistory();
        }
        
        initialized_ = true;
        
        if (config_.verbose) {
            PrintSuccess("CLI initialized successfully");
            ShowModelInfo();
        }
        
        return true;
        
    } catch (const std::exception& e) {
        PrintError("Failed to initialize: " + std::string(e.what()));
        return false;
    }
}

void CLIInterface::SetConfig(const CLIConfig& config) {
    config_ = config;
    
    // Set default history file if not specified
    if (config_.history_file.empty()) {
        config_.history_file = GetDefaultHistoryFile();
    }
}

int CLIInterface::RunInteractive() {
    if (!Initialize()) {
        return 1;
    }
    
    PrintInfo("Enhanced Gemma CLI Ready. Type /help for commands or start chatting!");
    PrintInfo("Use Ctrl+C to interrupt generation, Ctrl+D or /exit to quit.");
    
    while (!should_exit_) {
        try {
            std::string input = ReadInput();
            
            if (input.empty()) {
                continue;
            }
            
            // Add to history
            if (config_.enable_history && !IsCommand(input)) {
                AddToHistory(input);
            }
            
            // Process the input
            if (!ProcessCommand(input)) {
                // Not a command, treat as a prompt for generation
                std::string response = GenerateResponse(input);
                
                // Add to session
                Session::Entry entry;
                entry.prompt = input;
                entry.response = response;
                entry.timestamp = std::chrono::system_clock::now();
                current_session_.entries.push_back(entry);
            }
            
        } catch (const std::exception& e) {
            PrintError("Error: " + std::string(e.what()));
        }
    }
    
    return 0;
}

int CLIInterface::RunBatch(const std::string& filename) {
    if (!Initialize()) {
        return 1;
    }
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        PrintError("Cannot open batch file: " + filename);
        return 1;
    }
    
    PrintInfo("Running batch file: " + filename);
    
    std::string line;
    int line_number = 0;
    
    while (std::getline(file, line) && !should_exit_) {
        ++line_number;
        
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        if (config_.verbose) {
            std::cout << Colorize("[" + std::to_string(line_number) + "] ", Colors::Blue) 
                     << line << "\n";
        }
        
        try {
            if (!ProcessCommand(line)) {
                // Not a command, treat as prompt
                GenerateResponse(line);
            }
        } catch (const std::exception& e) {
            PrintError("Line " + std::to_string(line_number) + ": " + e.what());
        }
    }
    
    return should_exit_ ? 1 : 0;
}

int CLIInterface::ProcessSinglePrompt(const std::string& prompt) {
    if (!Initialize()) {
        return 1;
    }
    
    try {
        std::string response = GenerateResponse(prompt);
        std::cout << response << "\n";
        return 0;
    } catch (const std::exception& e) {
        PrintError("Error: " + std::string(e.what()));
        return 1;
    }
}

std::string CLIInterface::ReadInput() {
    std::cout << Colorize(config_.prompt_style, Colors::BrightCyan);
    
    std::string line;
    std::getline(std::cin, line);
    
    // Handle Ctrl+D (EOF)
    if (std::cin.eof()) {
        should_exit_ = true;
        std::cout << "\n";
        return "";
    }
    
    return line;
}

bool CLIInterface::ProcessCommand(const std::string& input) {
    if (!IsCommand(input)) {
        return false;
    }
    
    auto args = ParseCommand(input);
    if (args.empty()) {
        return true; // Empty command
    }
    
    std::string command = args[0];
    args.erase(args.begin()); // Remove command from args
    
    auto it = commands_.find(command);
    if (it != commands_.end()) {
        return it->second(args);
    } else {
        PrintError("Unknown command: " + command);
        PrintInfo("Type /help for available commands");
        return true;
    }
}

bool CLIInterface::IsCommand(const std::string& input) const {
    return !input.empty() && input[0] == '/';
}

std::vector<std::string> CLIInterface::ParseCommand(const std::string& command) const {
    std::vector<std::string> args;
    std::istringstream iss(command.substr(1)); // Remove leading '/'
    std::string arg;
    
    while (iss >> arg) {
        args.push_back(arg);
    }
    
    return args;
}

void CLIInterface::RegisterCommands() {
    commands_["help"] = [this](const std::vector<std::string>& args) { return HandleHelp(args); };
    commands_["model"] = [this](const std::vector<std::string>& args) { return HandleModel(args); };
    commands_["backend"] = [this](const std::vector<std::string>& args) { return HandleBackend(args); };
    commands_["session"] = [this](const std::vector<std::string>& args) { return HandleSession(args); };
    commands_["config"] = [this](const std::vector<std::string>& args) { return HandleConfig(args); };
    commands_["history"] = [this](const std::vector<std::string>& args) { return HandleHistory(args); };
    commands_["clear"] = [this](const std::vector<std::string>& args) { return HandleClear(args); };
    commands_["exit"] = [this](const std::vector<std::string>& args) { return HandleExit(args); };
    commands_["quit"] = [this](const std::vector<std::string>& args) { return HandleExit(args); };
    commands_["info"] = [this](const std::vector<std::string>& args) { return HandleInfo(args); };
    commands_["benchmark"] = [this](const std::vector<std::string>& args) { return HandleBenchmark(args); };
}

std::string CLIInterface::GenerateResponse(const std::string& prompt) {
    if (!initialized_) {
        throw std::runtime_error("CLI not initialized");
    }
    
    generation_interrupted_.store(false);
    
    // Setup random generator
    std::mt19937 gen;
    if (inference_args_.seed != 0) {
        gen.seed(inference_args_.seed);
    } else {
        gen.seed(std::random_device{}());
    }
    
    // Prepare response collection
    std::string response;
    std::mutex response_mutex;
    
    // Token streaming callback
    auto stream_token = [&](int token, float prob) -> bool {
        if (generation_interrupted_.load()) {
            return false;
        }
        
        std::string token_text;
        if (gemma_->Tokenizer().Decode({token}, &token_text)) {
            std::lock_guard<std::mutex> lock(response_mutex);
            response += token_text;
            
            // Real-time output
            std::cout << token_text << std::flush;
        }
        
        return !gemma_->Config().IsEOS(token);
    };
    
    // Setup runtime configuration
    TimingInfo timing_info = {.verbosity = inference_args_.verbosity};
    RuntimeConfig runtime_config = {
        .gen = &gen,
        .verbosity = inference_args_.verbosity,
        .stream_token = stream_token,
        .use_spinning = threading_args_.spin
    };
    inference_args_.CopyTo(runtime_config);
    
    // Tokenize prompt
    std::vector<int> prompt_tokens = WrapAndTokenize(
        gemma_->Tokenizer(), gemma_->ChatTemplate(),
        gemma_->Config().wrapping, conversation_pos_, prompt);
    
    try {
        // Generate response
        std::cout << "\n" << Colorize("Assistant: ", Colors::BrightGreen);
        
        gemma_->Generate(runtime_config, prompt_tokens, conversation_pos_, 
                        /*prefix_end=*/0, *kv_cache_, *env_, timing_info);
        
        std::cout << "\n\n";
        
        // Update conversation position
        conversation_pos_ += prompt_tokens.size() + response.size(); // Approximate
        
        // Show timing if enabled
        if (config_.show_timing) {
            ShowTiming(timing_info);
        }
        
    } catch (const std::exception& e) {
        PrintError("Generation failed: " + std::string(e.what()));
        return "";
    }
    
    return response;
}

// Implementation continues with command handlers...
// (The rest of the implementation would continue with all the command handlers,
// utility functions, history management, etc.)

// Utility function implementations
std::string CLIInterface::Colorize(const std::string& text, const char* color) const {
    if (!config_.enable_colors) {
        return text;
    }
    return std::string(color) + text + Colors::Reset;
}

void CLIInterface::PrintError(const std::string& message) const {
    std::cout << Colorize("Error: ", Colors::BrightRed) << message << "\n";
}

void CLIInterface::PrintWarning(const std::string& message) const {
    std::cout << Colorize("Warning: ", Colors::BrightYellow) << message << "\n";
}

void CLIInterface::PrintInfo(const std::string& message) const {
    std::cout << Colorize("Info: ", Colors::BrightBlue) << message << "\n";
}

void CLIInterface::PrintSuccess(const std::string& message) const {
    std::cout << Colorize("Success: ", Colors::BrightGreen) << message << "\n";
}

void CLIInterface::ClearScreen() const {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

// Command handler implementations
bool CLIInterface::HandleHelp(const std::vector<std::string>& args) {
    std::cout << Colorize("\nAvailable Commands:\n", Colors::Bold);
    
    std::cout << Colorize("  /help", Colors::BrightCyan) << "                     Show this help message\n";
    std::cout << Colorize("  /model [path]", Colors::BrightCyan) << "           Load/switch model or show current\n";
    std::cout << Colorize("  /backend [name]", Colors::BrightCyan) << "         Switch backend (cpu/intel/cuda/vulkan)\n";
    std::cout << Colorize("  /session save <file>", Colors::BrightCyan) << "     Save current session\n";
    std::cout << Colorize("  /session load <file>", Colors::BrightCyan) << "     Load session from file\n";
    std::cout << Colorize("  /session clear", Colors::BrightCyan) << "          Clear current session\n";
    std::cout << Colorize("  /config", Colors::BrightCyan) << "                  Show current configuration\n";
    std::cout << Colorize("  /history", Colors::BrightCyan) << "                 Show command history\n";
    std::cout << Colorize("  /info", Colors::BrightCyan) << "                    Show model and system info\n";
    std::cout << Colorize("  /benchmark", Colors::BrightCyan) << "              Run performance benchmark\n";
    std::cout << Colorize("  /clear", Colors::BrightCyan) << "                   Clear screen\n";
    std::cout << Colorize("  /exit, /quit", Colors::BrightCyan) << "            Exit the CLI\n\n";
    
    std::cout << Colorize("Key Bindings:\n", Colors::Bold);
    std::cout << "  Ctrl+C                    Interrupt current generation\n";
    std::cout << "  Ctrl+D                    Exit CLI\n";
    std::cout << "  Ctrl+L                    Clear screen\n\n";
    
    return true;
}

bool CLIInterface::HandleExit(const std::vector<std::string>& args) {
    PrintInfo("Goodbye!");
    should_exit_ = true;
    return true;
}

bool CLIInterface::HandleClear(const std::vector<std::string>& args) {
    ClearScreen();
    return true;
}

// Placeholder implementations for other handlers
bool CLIInterface::HandleModel(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cout << "Current model: " << Colorize(current_model_, Colors::BrightGreen) << "\n";
    } else {
        PrintInfo("Model switching not implemented yet");
    }
    return true;
}

bool CLIInterface::HandleBackend(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cout << "Current backend: " << Colorize(current_backend_, Colors::BrightGreen) << "\n";
        std::cout << "Available backends: cpu, intel, cuda, vulkan\n";
    } else {
        PrintInfo("Backend switching not implemented yet");
    }
    return true;
}

bool CLIInterface::HandleSession(const std::vector<std::string>& args) {
    PrintInfo("Session management not implemented yet");
    return true;
}

bool CLIInterface::HandleConfig(const std::vector<std::string>& args) {
    ShowConfig();
    return true;
}

bool CLIInterface::HandleHistory(const std::vector<std::string>& args) {
    auto history = GetHistory();
    if (history.empty()) {
        PrintInfo("No command history");
        return true;
    }
    
    std::cout << Colorize("Command History:\n", Colors::Bold);
    for (size_t i = 0; i < history.size(); ++i) {
        std::cout << Colorize(std::to_string(i + 1) + ": ", Colors::Blue) 
                 << history[i] << "\n";
    }
    return true;
}

bool CLIInterface::HandleInfo(const std::vector<std::string>& args) {
    ShowModelInfo();
    return true;
}

bool CLIInterface::HandleBenchmark(const std::vector<std::string>& args) {
    PrintInfo("Benchmark functionality not implemented yet");
    return true;
}

// Utility function implementations
void CLIInterface::ShowConfig() const {
    std::cout << Colorize("\nCurrent Configuration:\n", Colors::Bold);
    std::cout << "  Colors enabled: " << (config_.enable_colors ? "Yes" : "No") << "\n";
    std::cout << "  History enabled: " << (config_.enable_history ? "Yes" : "No") << "\n";
    std::cout << "  Tab completion: " << (config_.enable_completion ? "Yes" : "No") << "\n";
    std::cout << "  Verbose mode: " << (config_.verbose ? "Yes" : "No") << "\n";
    std::cout << "  Show timing: " << (config_.show_timing ? "Yes" : "No") << "\n";
    std::cout << "  Show progress: " << (config_.show_progress ? "Yes" : "No") << "\n";
    std::cout << "  History file: " << config_.history_file << "\n";
    std::cout << "  Max history: " << config_.max_history_size << "\n\n";
}

void CLIInterface::ShowModelInfo() const {
    if (!gemma_) {
        PrintError("No model loaded");
        return;
    }
    
    const auto& config = gemma_->Config();
    std::cout << Colorize("\nModel Information:\n", Colors::Bold);
    std::cout << "  Model: " << current_model_ << "\n";
    std::cout << "  Backend: " << current_backend_ << "\n";
    std::cout << "  Layers: " << config.num_layers << "\n";
    std::cout << "  Model dimension: " << config.model_dim << "\n";
    std::cout << "  Vocabulary size: " << config.vocab_size << "\n";
    std::cout << "  Max sequence length: " << config.max_seq_len << "\n";
    std::cout << "  Conversation position: " << conversation_pos_ << "\n\n";
}

void CLIInterface::ShowTiming(const TimingInfo& timing) const {
    if (timing.tokens_generated > 0) {
        double total_time = timing.prefill_duration + timing.generate_duration;
        double tokens_per_sec = timing.tokens_generated / timing.generate_duration;
        
        std::cout << Colorize("Timing: ", Colors::Yellow);
        std::cout << "Prefill: " << static_cast<int>(timing.prefill_duration * 1000) << "ms, ";
        std::cout << "Generation: " << static_cast<int>(timing.generate_duration * 1000) << "ms, ";
        std::cout << "Speed: " << std::fixed << std::setprecision(1) << tokens_per_sec << " tokens/sec\n";
    }
}

void CLIInterface::LoadHistory() {
    // Simple implementation - could be enhanced
    command_history_.clear();
}

void CLIInterface::SaveHistory() {
    // Simple implementation - could be enhanced
}

void CLIInterface::AddToHistory(const std::string& command) {
    command_history_.push_back(command);
    if (command_history_.size() > config_.max_history_size) {
        command_history_.erase(command_history_.begin());
    }
}

std::vector<std::string> CLIInterface::GetHistory() const {
    return command_history_;
}

// Utility functions
std::string GetDefaultHistoryFile() {
    // Platform-specific default history file location
#ifdef _WIN32
    return std::string(std::getenv("USERPROFILE")) + "\\.gemma_history";
#else
    return std::string(std::getenv("HOME")) + "/.gemma_history";
#endif
}

std::string GetDefaultConfigFile() {
#ifdef _WIN32
    return std::string(std::getenv("USERPROFILE")) + "\\.gemma_config";
#else
    return std::string(std::getenv("HOME")) + "/.gemma_config";
#endif
}

std::string GetPrompt(const InferenceArgs& inference) {
    if (!inference.prompt.empty()) return inference.prompt;
    if (!inference.prompt_file.Empty()) {
        return ReadFileToString(inference.prompt_file);
    }
    return "";
}

bool IsValidModelFile(const std::string& path) {
    return std::filesystem::exists(path) && 
           (path.ends_with(".sbs") || path.ends_with(".safetensors"));
}

std::vector<std::string> FindModelFiles(const std::string& directory) {
    std::vector<std::string> files;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string path = entry.path().string();
                if (IsValidModelFile(path)) {
                    files.push_back(path);
                }
            }
        }
    } catch (const std::exception&) {
        // Directory doesn't exist or can't be read
    }
    return files;
}

} // namespace gcpp