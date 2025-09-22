/**
 * @file main.cpp
 * @brief Main entry point for Gemma MCP Server
 */

#include <iostream>
#include <string>
#include <signal.h>
#include <memory>

#include "mcp_server.h"

using namespace gemma::mcp;

// Global server instance for signal handling
std::unique_ptr<MCPServer> g_server;

void SignalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->Stop();
    }
    exit(0);
}

void PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --host HOST           Server host (default: localhost)\n"
              << "  --port PORT           Server port (default: 8080)\n"
              << "  --model PATH          Path to model weights (required)\n"
              << "  --tokenizer PATH      Path to tokenizer (required)\n"
              << "  --max-requests N      Max concurrent requests (default: 4)\n"
              << "  --temperature F       Default temperature (default: 0.7)\n"
              << "  --max-tokens N        Default max tokens (default: 1024)\n"
              << "  --log-level LEVEL     Log level (DEBUG, INFO, WARN, ERROR)\n"
              << "  --help               Show this help message\n"
              << std::endl;
}

MCPServer::Config ParseArguments(int argc, char* argv[]) {
    MCPServer::Config config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            PrintUsage(argv[0]);
            exit(0);
        } else if (arg == "--host" && i + 1 < argc) {
            config.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = std::stoi(argv[++i]);
        } else if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (arg == "--tokenizer" && i + 1 < argc) {
            config.tokenizer_path = argv[++i];
        } else if (arg == "--max-requests" && i + 1 < argc) {
            config.max_concurrent_requests = std::stoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            config.temperature = std::stof(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            config.max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--log-level" && i + 1 < argc) {
            config.log_level = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            PrintUsage(argv[0]);
            exit(1);
        }
    }
    
    return config;
}

bool ValidateConfig(const MCPServer::Config& config) {
    if (config.model_path.empty()) {
        std::cerr << "Error: Model path is required (--model)" << std::endl;
        return false;
    }
    
    if (config.tokenizer_path.empty()) {
        std::cerr << "Error: Tokenizer path is required (--tokenizer)" << std::endl;
        return false;
    }
    
    if (config.port <= 0 || config.port > 65535) {
        std::cerr << "Error: Invalid port number: " << config.port << std::endl;
        return false;
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "Gemma MCP Server v1.0.0" << std::endl;
    std::cout << "Starting up..." << std::endl;
    
    // Parse command line arguments
    MCPServer::Config config = ParseArguments(argc, argv);
    
    // Validate configuration
    if (!ValidateConfig(config)) {
        return 1;
    }
    
    // Set up signal handlers
    signal(SIGINT, SignalHandler);
    signal(SIGTERM, SignalHandler);
    
    try {
        // Create and initialize server
        g_server = std::make_unique<MCPServer>(config);
        
        if (!g_server->Initialize()) {
            std::cerr << "Failed to initialize MCP server" << std::endl;
            return 1;
        }
        
        // Start server
        if (!g_server->Start()) {
            std::cerr << "Failed to start MCP server" << std::endl;
            return 1;
        }
        
        std::cout << "MCP Server running on " << config.host << ":" << config.port << std::endl;
        std::cout << "Model: " << config.model_path << std::endl;
        std::cout << "Press Ctrl+C to stop..." << std::endl;
        
        // Keep server running
        while (g_server->IsRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "MCP Server shutdown complete" << std::endl;
    return 0;
}