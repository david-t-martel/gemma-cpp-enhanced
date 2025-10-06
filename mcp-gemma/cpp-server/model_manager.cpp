/**
 * @file model_manager.cpp
 * @brief Implementation of ModelManager for MCP server
 */

#include "model_manager.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>

// Gemma includes
#include "gemma/gemma.h"
#include "gemma/configs.h"
#include "gemma/model_store.h"
#include "io/blob_store.h"
#include "util/threading_context.h"

namespace gemma {
namespace mcp {

/**
 * @brief Private implementation of ModelManager
 */
class ModelManager::Impl {
public:
    explicit Impl(const MCPServer::Config& config)
        : config_(config), model_loaded_(false), gemma_(nullptr) {
    }

    ~Impl() {
        UnloadModel();
    }

    bool LoadModel(const std::string& model_path, const std::string& tokenizer_path) {
        try {
            // Validate paths
            if (!std::filesystem::exists(model_path)) {
                std::cerr << "Error: Model file not found: " << model_path << std::endl;
                return false;
            }

            // For single-file models, tokenizer might be embedded
            bool has_separate_tokenizer = !tokenizer_path.empty() && std::filesystem::exists(tokenizer_path);

            if (!has_separate_tokenizer && tokenizer_path.empty()) {
                // Try to use single-file model format
                std::cout << "No separate tokenizer provided, assuming single-file model format" << std::endl;
            } else if (!has_separate_tokenizer) {
                std::cerr << "Error: Tokenizer file not found: " << tokenizer_path << std::endl;
                return false;
            }

            // Initialize threading context
            threading_context_ = std::make_unique<gcpp::ThreadingContext>();

            // Set up loader arguments
            gcpp::LoaderArgs loader;
            loader.weights = gcpp::Path(model_path);
            if (has_separate_tokenizer) {
                loader.tokenizer = gcpp::Path(tokenizer_path);
            }

            // Set default model type - this will be auto-detected from the file
            loader.model_type = gcpp::ModelType::GEMMA2_2B;

            // Set up inference arguments
            gcpp::InferenceArgs inference;
            inference.max_seq_len = 8192; // Default sequence length
            inference.prefill_tbatch_size = 1;
            inference.decode_qbatch_size = 1;

            // Load the model
            std::cout << "Loading model from: " << model_path << std::endl;
            if (has_separate_tokenizer) {
                std::cout << "Loading tokenizer from: " << tokenizer_path << std::endl;
            }

            gemma_ = std::make_unique<gcpp::Gemma>(loader, inference, *threading_context_);

            // Store model information
            const auto& config = gemma_->Config();
            model_info_.name = "Gemma";
            model_info_.version = "2.0"; // Default version
            model_info_.architecture = "Transformer";
            model_info_.parameter_count = static_cast<size_t>(config.model_dim * config.ff_hidden_dim * config.heads * config.layers);
            model_info_.context_length = config.seq_len;
            model_info_.data_type = "Mixed (BF16/F32/SFP)";
            model_info_.is_loaded = true;

            // Detect model variant from config
            if (config.layers == 18 && config.model_dim == 2048) {
                model_info_.name = "Gemma-2B";
            } else if (config.layers == 42 && config.model_dim == 3584) {
                model_info_.name = "Gemma-9B";
            } else if (config.layers == 46 && config.model_dim == 4608) {
                model_info_.name = "Gemma-27B";
            }

            model_loaded_ = true;
            std::cout << "Model loaded successfully: " << model_info_.name << std::endl;
            std::cout << "Model dimensions: " << config.model_dim << "x" << config.layers << " layers" << std::endl;
            std::cout << "Vocabulary size: " << config.vocab_size << std::endl;
            std::cout << "Context length: " << config.seq_len << std::endl;

            return true;

        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            UnloadModel();
            return false;
        }
    }

    void UnloadModel() {
        if (gemma_) {
            std::cout << "Unloading model..." << std::endl;
            gemma_.reset();
        }
        threading_context_.reset();
        model_loaded_ = false;
        model_info_.is_loaded = false;
    }

    bool IsModelLoaded() const {
        return model_loaded_ && gemma_ != nullptr;
    }

    nlohmann::json GetModelInfo(const nlohmann::json& params) const {
        nlohmann::json info;

        if (IsModelLoaded()) {
            info["name"] = model_info_.name;
            info["version"] = model_info_.version;
            info["architecture"] = model_info_.architecture;
            info["parameter_count"] = model_info_.parameter_count;
            info["context_length"] = model_info_.context_length;
            info["data_type"] = model_info_.data_type;
            info["is_loaded"] = model_info_.is_loaded;

            // Add configuration details
            const auto& config = gemma_->Config();
            info["config"] = {
                {"model_dim", config.model_dim},
                {"ff_hidden_dim", config.ff_hidden_dim},
                {"heads", config.heads},
                {"layers", config.layers},
                {"vocab_size", config.vocab_size},
                {"seq_len", config.seq_len},
                {"num_kv_heads", config.num_kv_heads}
            };
        } else {
            info["name"] = "No model loaded";
            info["is_loaded"] = false;
        }

        return info;
    }

    nlohmann::json TokenizeText(const nlohmann::json& params) const {
        if (!IsModelLoaded()) {
            throw std::runtime_error("No model loaded");
        }

        std::string text = params.value("text", "");
        if (text.empty()) {
            throw std::invalid_argument("Text parameter is required");
        }

        try {
            // Tokenize the text
            std::vector<int> tokens;
            if (!gemma_->Tokenizer().Encode(text, &tokens)) {
                throw std::runtime_error("Failed to tokenize text");
            }

            nlohmann::json response;
            response["text"] = text;
            response["tokens"] = tokens;
            response["token_count"] = tokens.size();

            // Include token details if requested
            bool include_details = params.value("include_details", false);
            if (include_details) {
                nlohmann::json token_details = nlohmann::json::array();
                for (size_t i = 0; i < tokens.size(); ++i) {
                    // Decode individual token to get text representation
                    std::string token_text;
                    if (gemma_->Tokenizer().Decode({tokens[i]}, &token_text)) {
                        token_details.push_back({
                            {"index", i},
                            {"token_id", tokens[i]},
                            {"text", token_text}
                        });
                    }
                }
                response["token_details"] = token_details;
            }

            return response;

        } catch (const std::exception& e) {
            throw std::runtime_error("Tokenization failed: " + std::string(e.what()));
        }
    }

    gcpp::Gemma* GetGemma() const {
        return gemma_.get();
    }

    const gcpp::GemmaTokenizer* GetTokenizer() const {
        if (!IsModelLoaded()) {
            return nullptr;
        }
        return &gemma_->Tokenizer();
    }

    const gcpp::ModelConfig& GetConfig() const {
        if (!IsModelLoaded()) {
            throw std::runtime_error("No model loaded");
        }
        return gemma_->Config();
    }

private:
    MCPServer::Config config_;
    bool model_loaded_;
    ModelInfo model_info_;
    std::unique_ptr<gcpp::Gemma> gemma_;
    std::unique_ptr<gcpp::ThreadingContext> threading_context_;
};

// ModelManager implementation
ModelManager::ModelManager(const MCPServer::Config& config)
    : impl_(std::make_unique<Impl>(config)) {
}

ModelManager::~ModelManager() = default;

bool ModelManager::LoadModel(const std::string& model_path, const std::string& tokenizer_path) {
    return impl_->LoadModel(model_path, tokenizer_path);
}

void ModelManager::UnloadModel() {
    impl_->UnloadModel();
}

bool ModelManager::IsModelLoaded() const {
    return impl_->IsModelLoaded();
}

nlohmann::json ModelManager::GetModelInfo(const nlohmann::json& params) const {
    return impl_->GetModelInfo(params);
}

nlohmann::json ModelManager::TokenizeText(const nlohmann::json& params) const {
    return impl_->TokenizeText(params);
}

gcpp::Gemma* ModelManager::GetGemma() const {
    return impl_->GetGemma();
}

const gcpp::GemmaTokenizer* ModelManager::GetTokenizer() const {
    return impl_->GetTokenizer();
}

const gcpp::ModelConfig& ModelManager::GetConfig() const {
    return impl_->GetConfig();
}

} // namespace mcp
} // namespace gemma