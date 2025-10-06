use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocumentId(String);

impl DocumentId {
    pub fn new() -> Self {
        DocumentId(Uuid::new_v4().to_string())
    }

    pub fn from_string(id: String) -> Self {
        DocumentId(id)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for DocumentId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub title: Option<String>,
    pub author: Option<String>,
    pub source: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub tags: Vec<String>,
    pub custom: HashMap<String, serde_json::Value>,
}

impl Default for DocumentMetadata {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            title: None,
            author: None,
            source: None,
            created_at: now,
            updated_at: now,
            tags: Vec::new(),
            custom: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: DocumentId,
    pub content: String,
    pub metadata: DocumentMetadata,
    pub chunk_ids: Vec<String>,
}

impl Document {
    pub fn new(content: String, metadata: Option<DocumentMetadata>) -> Self {
        Self {
            id: DocumentId::new(),
            content,
            metadata: metadata.unwrap_or_default(),
            chunk_ids: Vec::new(),
        }
    }

    pub fn with_id(id: DocumentId, content: String, metadata: DocumentMetadata) -> Self {
        Self {
            id,
            content,
            metadata,
            chunk_ids: Vec::new(),
        }
    }

    pub fn add_chunk_id(&mut self, chunk_id: String) {
        self.chunk_ids.push(chunk_id);
    }

    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }

    pub fn char_count(&self) -> usize {
        self.content.len()
    }
}
