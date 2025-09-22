/**
 * @file inference_handler.cpp
 * @brief Implementation of InferenceHandler for MCP server
 */

#include "inference_handler.h"
#include "model_manager.h"

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <thread>

// Gemma includes
#include "gemma/gemma.h"
#include "ops/matmul.h"
#include "util/threading_context.h"

namespace gemma {
namespace mcp {

/**
 * @brief Private implementation of InferenceHandler
 */
class InferenceHandler::Impl {
public:
    explicit Impl(const MCPServer::Config& config)
        : config_(config), model_manager_(nullptr), initialized_(false) {
        // Initialize default generation parameters
        generation_params_.temperature = config.temperature;
        generation_params_.max_tokens = config.max_tokens;
        generation_params_.top_k = 40;
        generation_params_.top_p = 0.95f;
        generation_params_.stream = false;
    }

    bool Initialize(ModelManager* model_manager) {
        if (!model_manager) {
            std::cerr << "Error: ModelManager is null" << std::endl;
            return false;
        }

        model_manager_ = model_manager;

        if (!model_manager_->IsModelLoaded()) {
            std::cerr << "Error: No model loaded in ModelManager" << std::endl;
            return false;
        }

        try {
            // Initialize threading context for Gemma operations
            threading_context_ = std::make_unique<gcpp::ThreadingContext>();

            // Initialize matrix multiplication environment
            matmul_env_ = std::make_unique<gcpp::MatMulEnv>(*threading_context_);

            initialized_ = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error initializing inference handler: " << e.what() << std::endl;
            return false;
        }
    }

    nlohmann::json GenerateText(const nlohmann::json& params) {
        if (!initialized_ || !model_manager_->IsModelLoaded()) {
            throw std::runtime_error("Inference handler not initialized or model not loaded");
        }

        // Parse parameters
        std::string prompt = params.value("prompt", "");
        if (prompt.empty()) {
            throw std::invalid_argument("Prompt is required");
        }

        // Override generation parameters if provided
        GenerationParams gen_params = generation_params_;
        if (params.contains("temperature")) {
            gen_params.temperature = params["temperature"];
        }
        if (params.contains("max_tokens")) {
            gen_params.max_tokens = params["max_tokens"];
        }
        if (params.contains("top_k")) {
            gen_params.top_k = params["top_k"];
        }
        if (params.contains("top_p")) {
            gen_params.top_p = params["top_p"];
        }
        if (params.contains("stream")) {
            gen_params.stream = params["stream"];
        }

        try {
            return GenerateTextInternal(prompt, gen_params);
        } catch (const std::exception& e) {
            throw std::runtime_error("Text generation failed: " + std::string(e.what()));
        }
    }

    nlohmann::json SetGenerationParams(const nlohmann::json& params) {
        nlohmann::json response;
        nlohmann::json updated_params;

        if (params.contains("temperature")) {
            generation_params_.temperature = params["temperature"];
            updated_params["temperature"] = generation_params_.temperature;
        }
        if (params.contains("max_tokens")) {
            generation_params_.max_tokens = params["max_tokens"];
            updated_params["max_tokens"] = generation_params_.max_tokens;
        }
        if (params.contains("top_k")) {
            generation_params_.top_k = params["top_k"];
            updated_params["top_k"] = generation_params_.top_k;
        }
        if (params.contains("top_p")) {
            generation_params_.top_p = params["top_p"];
            updated_params["top_p"] = generation_params_.top_p;
        }
        if (params.contains("stream")) {
            generation_params_.stream = params["stream"];
            updated_params["stream"] = generation_params_.stream;
        }

        response["status"] = "success";
        response["updated_parameters"] = updated_params;
        return response;
    }

    nlohmann::json GetGenerationParams() const {
        nlohmann::json params;
        params["temperature"] = generation_params_.temperature;
        params["max_tokens"] = generation_params_.max_tokens;
        params["top_k"] = generation_params_.top_k;
        params["top_p"] = generation_params_.top_p;
        params["stream"] = generation_params_.stream;
        return params;
    }

private:
    MCPServer::Config config_;
    ModelManager* model_manager_;
    bool initialized_;
    GenerationParams generation_params_;
    std::unique_ptr<gcpp::ThreadingContext> threading_context_;
    std::unique_ptr<gcpp::MatMulEnv> matmul_env_;

    nlohmann::json GenerateTextInternal(const std::string& prompt, const GenerationParams& params) {
        gcpp::Gemma* gemma = model_manager_->GetGemma();
        if (!gemma) {
            throw std::runtime_error("Gemma instance not available");
        }

        // Tokenize the prompt
        std::vector<int> prompt_tokens;
        if (!gemma->Tokenizer().Encode(prompt, &prompt_tokens)) {
            throw std::runtime_error("Failed to tokenize prompt");
        }

        // Create KV cache
        const auto& config = gemma->Config();
        auto kv_cache = gcpp::KVCache::Create(config);

        // Set up runtime configuration
        gcpp::RuntimeConfig runtime_config;
        runtime_config.max_seq_len = config.seq_len;
        runtime_config.temperature = params.temperature;
        runtime_config.decode_qbatch_size = 1;
        runtime_config.prefill_tbatch_size = 1;

        // Configure sampling
        if (params.top_k > 0) {
            runtime_config.top_k = params.top_k;
        }
        if (params.top_p > 0.0f && params.top_p < 1.0f) {
            runtime_config.top_p = params.top_p;
        }

        // Set up timing info
        gcpp::TimingInfo timing_info;
        timing_info.verbosity = 0; // Disable verbose timing output

        // Generate text
        std::string generated_text;
        size_t tokens_generated = 0;

        // Stream callback to collect generated tokens
        auto stream_token = [&](int token, float) -> bool {
            if (tokens_generated >= static_cast<size_t>(params.max_tokens)) {
                return false; // Stop generation
            }

            std::string token_text;
            if (gemma->Tokenizer().Decode({token}, &token_text)) {
                generated_text += token_text;
                tokens_generated++;
            }

            return true; // Continue generation
        };

        // Set the stream callback
        runtime_config.stream_token = stream_token;

        try {
            // Start timing
            timing_info.prefill_start = hwy::platform::Now();
            timing_info.generate_start = hwy::platform::Now();

            // Perform generation
            gemma->Generate(runtime_config, prompt_tokens, 0, *kv_cache, *matmul_env_, timing_info);

            // Prepare response
            nlohmann::json response;
            response["text"] = generated_text;
            response["prompt"] = prompt;
            response["tokens_generated"] = tokens_generated;
            response["prompt_tokens"] = prompt_tokens.size();

            // Add timing information if available
            if (timing_info.tokens_generated > 0) {
                response["timing"] = {
                    {"tokens_per_second", static_cast<double>(timing_info.tokens_generated) / timing_info.generate_duration},
                    {"time_to_first_token_ms", static_cast<int>(timing_info.time_to_first_token * 1000)},
                    {"total_generation_time_ms", static_cast<int>(timing_info.generate_duration * 1000)}
                };
            }

            return response;

        } catch (const std::exception& e) {
            throw std::runtime_error("Generation failed: " + std::string(e.what()));
        }
    }
};

// InferenceHandler implementation
InferenceHandler::InferenceHandler(const MCPServer::Config& config)
    : impl_(std::make_unique<Impl>(config)) {
}

InferenceHandler::~InferenceHandler() = default;

bool InferenceHandler::Initialize(ModelManager* model_manager) {
    return impl_->Initialize(model_manager);
}

nlohmann::json InferenceHandler::GenerateText(const nlohmann::json& params) {
    return impl_->GenerateText(params);
}

nlohmann::json InferenceHandler::SetGenerationParams(const nlohmann::json& params) {
    return impl_->SetGenerationParams(params);
}

nlohmann::json InferenceHandler::GetGenerationParams() const {
    return impl_->GetGenerationParams();
}

} // namespace mcp
} // namespace gemma