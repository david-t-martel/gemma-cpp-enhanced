use crate::error::Result;
use crate::config::EmbeddingProvider;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[cfg(feature = "gpu")]
use candle_core::{Device, Tensor, DType};
#[cfg(feature = "gpu")]
use candle_nn::VarBuilder;
#[cfg(feature = "gpu")]
use candle_transformers::models::distilbert::{DistilBertModel, Config as DistilBertConfig};
#[cfg(feature = "gpu")]
use tokenizers::Tokenizer;
#[cfg(feature = "gpu")]
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModel {
    pub name: String,
    pub dimension: usize,
    pub provider: EmbeddingProvider,
}

#[async_trait]
pub trait EmbeddingService: Send + Sync {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
    fn model_name(&self) -> &str;
}

// External Embedding Service implementation
pub struct ExternalEmbeddingService {
    client: reqwest::Client,
    url: String,
    dimension: usize,
    model_name: String,
}

impl ExternalEmbeddingService {
    pub fn new(url: String, dimension: usize, model_name: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            url,
            dimension,
            model_name,
        }
    }
}

#[async_trait]
impl EmbeddingService for ExternalEmbeddingService {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        #[derive(Serialize)]
        struct Request {
            text: String,
        }
        #[derive(Deserialize)]
        struct Response {
            embedding: Vec<f32>,
        }

        let request = Request { text: text.to_string() };
        let response = self.client.post(&self.url).json(&request).send().await?;
        let response: Response = response.json().await?;
        Ok(response.embedding)
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        #[derive(Serialize)]
        struct Request {
            texts: Vec<String>,
        }
        #[derive(Deserialize)]
        struct Response {
            embeddings: Vec<Vec<f32>>,
        }

        let request = Request { texts: texts.to_vec() };
        let response = self.client.post(&self.url).json(&request).send().await?;
        let response: Response = response.json().await?;
        Ok(response.embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

// Candle-based embedding service for local inference
#[cfg(feature = "gpu")]
pub struct CandleEmbeddingService {
    model: DistilBertModel,
    tokenizer: Tokenizer,
    device: Device,
    dimension: usize,
    model_name: String,
}

#[cfg(feature = "gpu")]
impl CandleEmbeddingService {
    pub async fn new(model_path: &Path, tokenizer_path: Option<&Path>, dimension: usize) -> Result<Self> {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        // Load tokenizer
        let tokenizer = if let Some(path) = tokenizer_path {
            Tokenizer::from_file(path)
                .map_err(|e| crate::error::Error::Api(format!("Failed to load tokenizer: {}", e)))?
        } else {
            // Download tokenizer from HuggingFace Hub if not provided
            let repo = hf_hub::api::sync::Api::new()
                .map_err(|e| crate::error::Error::Api(format!("Failed to create HF API: {}", e)))?
                .model("sentence-transformers/all-MiniLM-L6-v2".to_string());
            let tokenizer_path = repo.get("tokenizer.json")
                .map_err(|e| crate::error::Error::Api(format!("Failed to download tokenizer: {}", e)))?;
            Tokenizer::from_file(tokenizer_path)
                .map_err(|e| crate::error::Error::Api(format!("Failed to load tokenizer: {}", e)))?
        };

        // Load model weights
        let model_weights = candle_core::safetensors::load(model_path, &device)
            .map_err(|e| crate::error::Error::Api(format!("Failed to load model weights: {}", e)))?;

        // Create model configuration - use default DistilBERT config
        let config = DistilBertConfig::default();

        let var_builder = VarBuilder::from_tensors(model_weights, DType::F32, &device);
        let model = DistilBertModel::load(var_builder, &config)
            .map_err(|e| crate::error::Error::Api(format!("Failed to load model: {}", e)))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            dimension,
            model_name: model_path.to_string_lossy().to_string(),
        })
    }

    fn mean_pooling(&self, token_embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Apply attention mask and compute mean pooling
        let masked_embeddings = token_embeddings.broadcast_mul(attention_mask)
            .map_err(|e| crate::error::Error::Api(format!("Failed to apply attention mask: {}", e)))?;

        let sum_embeddings = masked_embeddings.sum_keepdim(1)
            .map_err(|e| crate::error::Error::Api(format!("Failed to sum embeddings: {}", e)))?;

        let sum_mask = attention_mask.sum_keepdim(1)
            .map_err(|e| crate::error::Error::Api(format!("Failed to sum mask: {}", e)))?;

        let mean_embeddings = sum_embeddings.broadcast_div(&sum_mask)
            .map_err(|e| crate::error::Error::Api(format!("Failed to compute mean: {}", e)))?;

        Ok(mean_embeddings)
    }
}

#[cfg(feature = "gpu")]
#[async_trait]
impl EmbeddingService for CandleEmbeddingService {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| crate::error::Error::Api(format!("Failed to tokenize text: {}", e)))?;

        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Convert to tensors
        let input_ids_tensor = Tensor::new(
            input_ids.iter().map(|&x| x as u32).collect::<Vec<_>>().as_slice(),
            &self.device
        ).map_err(|e| crate::error::Error::Api(format!("Failed to create input tensor: {}", e)))?
        .unsqueeze(0)
        .map_err(|e| crate::error::Error::Api(format!("Failed to unsqueeze input: {}", e)))?;

        let attention_mask_tensor = Tensor::new(
            attention_mask.iter().map(|&x| x as f32).collect::<Vec<_>>().as_slice(),
            &self.device
        ).map_err(|e| crate::error::Error::Api(format!("Failed to create attention mask: {}", e)))?
        .unsqueeze(0)
        .map_err(|e| crate::error::Error::Api(format!("Failed to unsqueeze mask: {}", e)))?;

        // Run forward pass
        let outputs = self.model.forward(&input_ids_tensor, &attention_mask_tensor)
            .map_err(|e| crate::error::Error::Api(format!("Forward pass failed: {}", e)))?;

        // Apply mean pooling
        let pooled = self.mean_pooling(&outputs, &attention_mask_tensor)?;

        // Convert to CPU and extract values
        let cpu_tensor = pooled.to_device(&Device::Cpu)
            .map_err(|e| crate::error::Error::Api(format!("Failed to move to CPU: {}", e)))?;

        let embedding_vec = cpu_tensor.to_vec1::<f32>()
            .map_err(|e| crate::error::Error::Api(format!("Failed to convert to vec: {}", e)))?;

        Ok(embedding_vec)
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());

        // Process in parallel batches for better performance
        for text in texts {
            results.push(self.embed_text(text).await?);
        }

        Ok(results)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

// Factory for creating embedding services
pub struct EmbeddingFactory;

impl EmbeddingFactory {
    pub async fn create(config: &crate::config::EmbeddingConfig) -> Result<Box<dyn EmbeddingService>> {
        match &config.provider {
            EmbeddingProvider::Custom(url) => {
                Ok(Box::new(ExternalEmbeddingService::new(
                    url.clone(),
                    config.dimension,
                    config.model.clone(),
                )))
            }
            #[cfg(feature = "gpu")]
            EmbeddingProvider::Candle => {
                // Use Candle for local inference
                let model_path = std::env::var("CANDLE_MODEL_PATH")
                    .unwrap_or_else(|_| format!("./models/{}.safetensors", config.model));
                let tokenizer_path = std::env::var("CANDLE_TOKENIZER_PATH")
                    .ok()
                    .map(|p| std::path::PathBuf::from(p));

                let candle_service = CandleEmbeddingService::new(
                    &std::path::Path::new(&model_path),
                    tokenizer_path.as_deref(),
                    config.dimension,
                ).await?;

                Ok(Box::new(candle_service))
            }
            EmbeddingProvider::Local => {
                #[cfg(feature = "gpu")]
                {
                    // Try Candle first if available
                    let model_path = std::env::var("CANDLE_MODEL_PATH")
                        .unwrap_or_else(|_| format!("./models/{}.safetensors", config.model));
                    let tokenizer_path = std::env::var("CANDLE_TOKENIZER_PATH")
                        .ok()
                        .map(|p| std::path::PathBuf::from(p));

                    if std::path::Path::new(&model_path).exists() {
                        match CandleEmbeddingService::new(
                            &std::path::Path::new(&model_path),
                            tokenizer_path.as_deref(),
                            config.dimension,
                        ).await {
                            Ok(service) => return Ok(Box::new(service)),
                            Err(_) => {
                                // Fall back to external service if Candle fails
                                tracing::warn!("Failed to initialize Candle service, falling back to external");
                            }
                        }
                    }
                }

                // Fallback to external service
                Ok(Box::new(ExternalEmbeddingService::new(
                    "http://localhost:8000/embed".to_string(),
                    config.dimension,
                    config.model.clone(),
                )))
            }
            _ => {
                // Other providers not implemented yet
                Ok(Box::new(ExternalEmbeddingService::new(
                    "http://localhost:8000/embed".to_string(),
                    config.dimension,
                    config.model.clone(),
                )))
            }
        }
    }
}

// ONNX-based embedding service (disabled for now due to dependency conflicts)
// #[cfg(feature = "wonnx")]
// use wonnx::{Session, tensor::Tensor, value::Value};
// 
// #[cfg(feature = "wonnx")]
// pub struct WonnxEmbedding {
//     session: Session,
//     tokenizer: tiktoken_rs::CoreBPE,
//     dimension: usize,
//     model_name: String,
// }
// 
// #[cfg(feature = "wonnx")]
// impl WonnxEmbedding {
//     pub async fn new(model_path: &str, dimension: usize) -> Result<Self> {
//         let session = Session::from_path(model_path).await?;
//         let tokenizer = tiktoken_rs::cl100k_base()?;
// 
//         Ok(Self {
//             session,
//             tokenizer,
//             dimension,
//             model_name: model_path.to_string(),
//         })
//     }
// }
// 
// #[cfg(feature = "wonnx")]
// #[async_trait]
// impl EmbeddingService for WonnxEmbedding {
//     async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
//         let tokens = self.tokenizer.encode_with_special_tokens(text);
//         let input_ids: Vec<i64> = tokens.into_iter().map(|t| t as i64).collect();
//         let input_tensor = Tensor::new(&[1, input_ids.len()], &input_ids)?;
// 
//         let inputs = vec![("input_ids", input_tensor.into())];
// 
//         let result = self.session.run(&inputs).await?;
//         let output_tensor = result.get("last_hidden_state").unwrap();
// 
//         if let Value::Tensor(tensor) = output_tensor {
//             Ok(tensor.to_vec())            
//         } else {
//             Err(Error::Api("Invalid output from ONNX model".to_string()))
//         }
//     }
// 
//     async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
//         let mut results = Vec::with_capacity(texts.len());
//         for text in texts {
//             results.push(self.embed_text(text).await?);
//         }
//         Ok(results)
//     }
// 
//     fn dimension(&self) -> usize {
//         self.dimension
//     }
// 
//     fn model_name(&self) -> &str {
//         &self.model_name
//     }
// }