use crate::error::Result;
use crate::config::EmbeddingProvider;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

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
            EmbeddingProvider::Local => {
                // For now, fallback to a default external service
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