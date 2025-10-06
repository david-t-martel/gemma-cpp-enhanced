// Builder utilities for SystemConfig (kept separate to avoid bloating test_helpers.h)
#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <cstdint>
#include "../integration/mock_system.h" // SystemConfig definition

namespace gemma {
namespace test {

class SystemConfigBuilder {
public:
    static SystemConfigBuilder WithDefaults(const std::filesystem::path& root) {
        SystemConfigBuilder b;
        b.root_ = root;
        b.cfg_.model_weights_path = (root / "test_model.sbs").string();
        b.cfg_.tokenizer_path = (root / "tokenizer.spm").string();
        b.cfg_.backend_name = "cpu";
        b.cfg_.session_storage_path = (root / "sessions").string();
        b.cfg_.enable_mcp_server = true;
        b.cfg_.mcp_server_port = 8081;
        b.cfg_.max_context_tokens = 4096;
        b.cfg_.enable_metrics = true;
        return b;
    }

    SystemConfigBuilder& Backend(std::string v) { cfg_.backend_name = std::move(v); return *this; }
    SystemConfigBuilder& MCPPort(uint16_t p) { cfg_.mcp_server_port = p; return *this; }
    SystemConfigBuilder& Context(size_t tokens) { cfg_.max_context_tokens = tokens; return *this; }
    SystemConfigBuilder& Metrics(bool on) { cfg_.enable_metrics = on; return *this; }
    SystemConfigBuilder& ModelFile(const std::string& name) { cfg_.model_weights_path = (root_ / name).string(); return *this; }
    SystemConfigBuilder& TokenizerFile(const std::string& name) { cfg_.tokenizer_path = (root_ / name).string(); return *this; }
    SystemConfigBuilder& SessionDir(const std::string& name) { cfg_.session_storage_path = (root_ / name).string(); return *this; }

    SystemConfig Build() const { return cfg_; }

    static void EnsureArtifacts(const SystemConfig& cfg, const std::string& weights_content = "dummy", const std::string& tokenizer_content = "dummy") {
        std::filesystem::create_directories(std::filesystem::path(cfg.session_storage_path));
        // weights
        if(!std::filesystem::exists(cfg.model_weights_path)) {
            std::ofstream wf(cfg.model_weights_path, std::ios::binary); wf << weights_content; }
        // tokenizer
        if(!std::filesystem::exists(cfg.tokenizer_path)) {
            std::ofstream tf(cfg.tokenizer_path, std::ios::binary); tf << tokenizer_content; }
    }

private:
    std::filesystem::path root_;
    SystemConfig cfg_{};
};

} // namespace test
} // namespace gemma
