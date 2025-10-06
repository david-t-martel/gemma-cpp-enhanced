#pragma once

/**
 * @file inference_handler.h
 * @brief Handles inference requests for the MCP server
 */

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <nlohmann/json.hpp>

#include "gemma/gemma.h"
#include "model_manager.h"

namespace gemma {
namespace mcp {

/**
 * @brief Handles inference operations for MCP server
 */
class InferenceHandler {
public:
    /**
     * @brief Generation parameters
     */
    struct GenerationParams {
        float temperature = 0.7f;
        int max_tokens = 1024;
        int top_k = 40;
        float top_p = 0.95f;
        std::string stop_sequence;
        bool stream = false;
    };

    /**
     * @brief Constructor
     * @param config Server configuration
     */
    explicit InferenceHandler(const MCPServer::Config& config);

    /**
     * @brief Destructor
     */
    ~InferenceHandler();

    /**
     * @brief Initialize the inference handler
     * @param model_manager Pointer to model manager
     * @return true if successful
     */
    bool Initialize(ModelManager* model_manager);

    /**
     * @brief Generate text based on prompt
     * @param params JSON parameters containing prompt and generation settings
     * @return JSON response with generated text
     */
    nlohmann::json GenerateText(const nlohmann::json& params);

    /**
     * @brief Set generation parameters
     * @param params JSON parameters to update
     * @return JSON response confirming parameter updates
     */
    nlohmann::json SetGenerationParams(const nlohmann::json& params);

    /**
     * @brief Get current generation parameters
     * @return JSON object with current parameters
     */
    nlohmann::json GetGenerationParams() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mcp
} // namespace gemma