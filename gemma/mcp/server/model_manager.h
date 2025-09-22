#pragma once

/**
 * @file model_manager.h
 * @brief Manages Gemma model loading and lifecycle
 */

#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "gemma/gemma.h"
#include "gemma/configs.h"

namespace gemma {
namespace mcp {

/**
 * @brief Manages Gemma model instances and operations
 */
class ModelManager {
public:
    /**
     * @brief Model information structure
     */
    struct ModelInfo {
        std::string name;
        std::string version;
        std::string architecture;
        size_t parameter_count;
        size_t context_length;
        std::string data_type;
        bool is_loaded;
    };

    /**
     * @brief Constructor
     * @param config Server configuration
     */
    explicit ModelManager(const MCPServer::Config& config);

    /**
     * @brief Destructor
     */
    ~ModelManager();

    /**
     * @brief Load a Gemma model
     * @param model_path Path to model weights
     * @param tokenizer_path Path to tokenizer
     * @return true if loading successful
     */
    bool LoadModel(const std::string& model_path, const std::string& tokenizer_path);

    /**
     * @brief Unload the current model
     */
    void UnloadModel();

    /**
     * @brief Check if a model is currently loaded
     * @return true if model is loaded
     */
    bool IsModelLoaded() const;

    /**
     * @brief Get model information
     * @param params JSON parameters (unused for now)
     * @return JSON object with model information
     */
    nlohmann::json GetModelInfo(const nlohmann::json& params = {}) const;

    /**
     * @brief Tokenize text using the loaded tokenizer
     * @param params JSON parameters containing text to tokenize
     * @return JSON object with token information
     */
    nlohmann::json TokenizeText(const nlohmann::json& params) const;

    /**
     * @brief Get the Gemma instance (for inference)
     * @return Pointer to Gemma instance, nullptr if not loaded
     */
    gcpp::Gemma* GetGemma() const;

    /**
     * @brief Get the tokenizer instance
     * @return Pointer to tokenizer, nullptr if not loaded
     */
    const gcpp::GemmaTokenizer* GetTokenizer() const;

    /**
     * @brief Get model configuration
     * @return Model configuration
     */
    const gcpp::ModelConfig& GetConfig() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mcp
} // namespace gemma