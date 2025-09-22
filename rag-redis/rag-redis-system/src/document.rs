//! Document processing and chunking

use crate::{config::DocumentConfig, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub content: String,
    pub metadata: serde_json::Value,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub document_id: String,
    pub text: String,
    pub metadata: serde_json::Value,
    pub start_index: usize,
    pub end_index: usize,
}

pub struct DocumentPipeline {
    config: DocumentConfig,
}

impl DocumentPipeline {
    pub fn new(config: DocumentConfig) -> Self {
        Self { config }
    }

    pub async fn process(&self, content: &str, metadata: serde_json::Value) -> Result<Document> {
        Ok(Document {
            id: Uuid::new_v4().to_string(),
            content: content.to_string(),
            metadata,
            created_at: chrono::Utc::now(),
        })
    }

    pub fn chunk_document(&self, document: &Document) -> Result<Vec<DocumentChunk>> {
        let mut chunks = Vec::new();
        let content = &document.content;
        let chunk_size = self.config.chunking.chunk_size;
        let chunk_overlap = self.config.chunking.chunk_overlap;

        let mut start = 0;
        let mut chunk_index = 0;

        while start < content.len() {
            let end = std::cmp::min(start + chunk_size, content.len());
            let chunk_text = content[start..end].to_string();

            chunks.push(DocumentChunk {
                id: format!("{}-chunk-{}", document.id, chunk_index),
                document_id: document.id.clone(),
                text: chunk_text,
                metadata: document.metadata.clone(),
                start_index: start,
                end_index: end,
            });

            start = if end == content.len() {
                break;
            } else {
                start + chunk_size - chunk_overlap
            };
            chunk_index += 1;
        }

        Ok(chunks)
    }
}
