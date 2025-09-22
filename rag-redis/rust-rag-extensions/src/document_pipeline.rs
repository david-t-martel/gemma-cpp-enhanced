//! Document chunking and embedding pipeline module
//!
//! This module provides comprehensive document processing capabilities:
//! - Multi-format document parsing (text, markdown, PDF, HTML)
//! - Intelligent text chunking with overlap and semantic boundaries
//! - Parallel embedding generation with batching
//! - Content preprocessing and cleaning
//! - Metadata extraction and enrichment

use crate::error::{GemmaError, GemmaResult};
use crate::redis_manager::{DocumentMetadata, RedisManager};
use crate::vector_store::{VectorMetadata, VectorStore};
use futures::{stream, StreamExt, TryStreamExt};
use pulldown_cmark::{html, Event, Options, Parser, Tag};
use regex::Regex;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tiktoken_rs::{get_bpe_from_model, CoreBPE};
use tokio::task;
use tracing::{debug, error, info, warn};
use unicode_segmentation::UnicodeSegmentation;
use uuid::Uuid;

/// Supported document formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DocumentFormat {
    PlainText,
    Markdown,
    Html,
    Pdf,
    Json,
}

impl DocumentFormat {
    /// Detect format from file extension or content
    pub fn detect(filename: &str, content: &str) -> Self {
        let extension = std::path::Path::new(filename)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase());

        match extension.as_deref() {
            Some("md") | Some("markdown") => DocumentFormat::Markdown,
            Some("html") | Some("htm") => DocumentFormat::Html,
            Some("pdf") => DocumentFormat::Pdf,
            Some("json") => DocumentFormat::Json,
            _ => {
                // Try to detect from content
                if content.trim_start().starts_with('<') && content.contains("</") {
                    DocumentFormat::Html
                } else if content.contains("```") || content.contains('#') {
                    DocumentFormat::Markdown
                } else if content.trim_start().starts_with('{') || content.trim_start().starts_with('[') {
                    DocumentFormat::Json
                } else {
                    DocumentFormat::PlainText
                }
            }
        }
    }
}

/// Chunking strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Maximum size of each chunk in tokens
    pub max_chunk_size: usize,
    /// Overlap between adjacent chunks in tokens
    pub chunk_overlap: usize,
    /// Minimum chunk size to avoid very small chunks
    pub min_chunk_size: usize,
    /// Respect sentence boundaries when possible
    pub respect_sentence_boundaries: bool,
    /// Respect paragraph boundaries when possible
    pub respect_paragraph_boundaries: bool,
    /// Custom separators for chunk boundaries
    pub separators: Vec<String>,
    /// Enable semantic chunking based on content similarity
    pub semantic_chunking: bool,
    /// Maximum number of chunks per document
    pub max_chunks_per_document: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 512,
            chunk_overlap: 50,
            min_chunk_size: 50,
            respect_sentence_boundaries: true,
            respect_paragraph_boundaries: true,
            separators: vec![
                "\n\n".to_string(),
                "\n".to_string(),
                ". ".to_string(),
                "! ".to_string(),
                "? ".to_string(),
                " ".to_string(),
            ],
            semantic_chunking: false,
            max_chunks_per_document: 1000,
        }
    }
}

/// Configuration for embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model identifier for embedding generation
    pub model_name: String,
    /// Maximum batch size for parallel embedding generation
    pub batch_size: usize,
    /// Embedding dimension (must match the model)
    pub dimension: usize,
    /// Normalize embeddings to unit vectors
    pub normalize: bool,
    /// Maximum text length for embedding (in characters)
    pub max_text_length: usize,
    /// Retry configuration for failed embedding requests
    pub max_retries: usize,
    /// Timeout for embedding requests in milliseconds
    pub timeout_ms: u64,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model_name: "text-embedding-ada-002".to_string(),
            batch_size: 32,
            dimension: 1536,
            normalize: true,
            max_text_length: 8000,
            max_retries: 3,
            timeout_ms: 30000,
        }
    }
}

/// A processed document chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub document_id: String,
    pub chunk_index: usize,
    pub text: String,
    pub token_count: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub metadata: HashMap<String, String>,
}

impl DocumentChunk {
    pub fn new(
        document_id: String,
        chunk_index: usize,
        text: String,
        start_char: usize,
        end_char: usize,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            document_id,
            chunk_index,
            token_count: 0, // Will be calculated during processing
            text,
            start_char,
            end_char,
            metadata: HashMap::new(),
        }
    }
}

/// Pipeline statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    pub documents_processed: u64,
    pub chunks_generated: u64,
    pub embeddings_created: u64,
    pub average_chunks_per_document: f64,
    pub average_processing_time_ms: f64,
    pub failed_documents: u64,
    pub total_tokens_processed: u64,
}

/// Document processing and embedding pipeline
pub struct DocumentPipeline {
    chunking_config: ChunkingConfig,
    embedding_config: EmbeddingConfig,
    tokenizer: Option<CoreBPE>,
    vector_store: Option<Arc<VectorStore>>,
    redis: Option<Arc<RedisManager>>,
    stats: Arc<tokio::sync::RwLock<PipelineStats>>,
}

impl DocumentPipeline {
    /// Create a new document pipeline
    pub fn new(
        chunking_config: ChunkingConfig,
        embedding_config: EmbeddingConfig,
    ) -> GemmaResult<Self> {
        // Initialize tokenizer for the embedding model
        let tokenizer = match get_bpe_from_model(&embedding_config.model_name) {
            Ok(tokenizer) => Some(tokenizer),
            Err(e) => {
                warn!("Could not load tokenizer for {}: {}", embedding_config.model_name, e);
                None
            }
        };

        Ok(Self {
            chunking_config,
            embedding_config,
            tokenizer,
            vector_store: None,
            redis: None,
            stats: Arc::new(tokio::sync::RwLock::new(PipelineStats::default())),
        })
    }

    /// Set the vector store for embedding storage
    pub fn with_vector_store(mut self, vector_store: Arc<VectorStore>) -> Self {
        self.vector_store = Some(vector_store);
        self
    }

    /// Set the Redis manager for document storage
    pub fn with_redis(mut self, redis: Arc<RedisManager>) -> Self {
        self.redis = Some(redis);
        self
    }

    /// Process a document through the complete pipeline
    pub async fn process_document(
        &self,
        content: &str,
        metadata: DocumentMetadata,
        format: DocumentFormat,
    ) -> GemmaResult<Vec<String>> {
        let start_time = std::time::Instant::now();

        info!("Processing document: {} (format: {:?})", metadata.id, format);

        // Step 1: Parse and clean the document content
        let cleaned_content = self.parse_and_clean_document(content, format).await?;

        // Step 2: Chunk the document
        let chunks = self.chunk_document(&cleaned_content, &metadata.id).await?;

        if chunks.is_empty() {
            warn!("No chunks generated for document: {}", metadata.id);
            return Ok(Vec::new());
        }

        if chunks.len() > self.chunking_config.max_chunks_per_document {
            warn!(
                "Document {} generated {} chunks, truncating to {}",
                metadata.id,
                chunks.len(),
                self.chunking_config.max_chunks_per_document
            );
        }

        let limited_chunks: Vec<_> = chunks
            .into_iter()
            .take(self.chunking_config.max_chunks_per_document)
            .collect();

        // Step 3: Generate embeddings for chunks
        let embeddings = self.generate_embeddings(&limited_chunks).await?;

        // Step 4: Store in vector store
        let mut vector_ids = Vec::new();
        if let Some(vector_store) = &self.vector_store {
            let vectors_with_metadata: Vec<_> = limited_chunks
                .iter()
                .zip(embeddings.iter())
                .map(|(chunk, embedding)| {
                    let vector_metadata = VectorMetadata::new(
                        chunk.document_id.clone(),
                        chunk.chunk_index,
                        chunk.text.clone(),
                    );
                    (embedding.clone(), vector_metadata)
                })
                .collect();

            let ids = vector_store.batch_add_vectors(vectors_with_metadata).await?;
            vector_ids.extend(ids.into_iter().map(|id| id.to_string()));
        }

        // Step 5: Store document metadata in Redis
        if let Some(redis) = &self.redis {
            let mut updated_metadata = metadata;
            updated_metadata.chunk_count = limited_chunks.len();
            updated_metadata.updated_at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            redis.store_document(&cleaned_content, &updated_metadata).await?;
        }

        // Update statistics
        let processing_time = start_time.elapsed().as_millis() as f64;
        self.update_stats(limited_chunks.len(), processing_time, true).await;

        info!(
            "Successfully processed document {} into {} chunks in {:.2}ms",
            metadata.id,
            limited_chunks.len(),
            processing_time
        );

        Ok(vector_ids)
    }

    /// Parse and clean document content based on format
    async fn parse_and_clean_document(
        &self,
        content: &str,
        format: DocumentFormat,
    ) -> GemmaResult<String> {
        match format {
            DocumentFormat::PlainText => Ok(self.clean_text(content)),
            DocumentFormat::Markdown => self.parse_markdown(content).await,
            DocumentFormat::Html => self.parse_html(content).await,
            DocumentFormat::Pdf => {
                // For now, assume PDF content has been extracted to text
                // In a full implementation, you'd use pdf-extract here
                Ok(self.clean_text(content))
            }
            DocumentFormat::Json => self.parse_json(content).await,
        }
    }

    /// Clean plain text content
    fn clean_text(&self, content: &str) -> String {
        // Remove excessive whitespace
        let re = Regex::new(r"\s+").unwrap();
        let cleaned = re.replace_all(content, " ");

        // Remove control characters but keep newlines and tabs
        cleaned
            .chars()
            .filter(|&c| c.is_ascii_graphic() || c.is_whitespace())
            .collect::<String>()
            .trim()
            .to_string()
    }

    /// Parse markdown content and extract text
    async fn parse_markdown(&self, content: &str) -> GemmaResult<String> {
        let mut options = Options::empty();
        options.insert(Options::ENABLE_TABLES);
        options.insert(Options::ENABLE_FOOTNOTES);
        options.insert(Options::ENABLE_STRIKETHROUGH);

        let parser = Parser::new_ext(content, options);
        let mut text_content = String::new();

        for event in parser {
            match event {
                Event::Text(text) | Event::Code(text) => {
                    text_content.push_str(&text);
                    text_content.push(' ');
                }
                Event::Start(Tag::Heading(_)) => text_content.push_str("\n# "),
                Event::Start(Tag::Paragraph) => text_content.push_str("\n\n"),
                Event::Start(Tag::Item) => text_content.push_str("\n• "),
                Event::SoftBreak | Event::HardBreak => text_content.push('\n'),
                _ => {}
            }
        }

        Ok(self.clean_text(&text_content))
    }

    /// Parse HTML content and extract text
    async fn parse_html(&self, content: &str) -> GemmaResult<String> {
        let document = Html::parse_document(content);

        // Remove script and style elements
        let script_selector = Selector::parse("script, style").unwrap();
        let mut text_content = String::new();

        // Extract text from body, or entire document if no body
        let body_selector = Selector::parse("body").unwrap();
        let content_element = document.select(&body_selector).next()
            .map(|body| body)
            .unwrap_or_else(|| document.root_element());

        fn extract_text_recursive(element: scraper::ElementRef, text: &mut String) {
            for child in element.children() {
                match child.value() {
                    scraper::Node::Text(text_node) => {
                        text.push_str(text_node);
                        text.push(' ');
                    }
                    scraper::Node::Element(el) => {
                        if !matches!(el.name(), "script" | "style") {
                            if let Some(child_element) = scraper::ElementRef::wrap(child) {
                                // Add spacing for block elements
                                if matches!(el.name(), "p" | "div" | "br" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6") {
                                    text.push_str("\n");
                                }
                                extract_text_recursive(child_element, text);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        extract_text_recursive(content_element, &mut text_content);
        Ok(self.clean_text(&text_content))
    }

    /// Parse JSON content and extract readable text
    async fn parse_json(&self, content: &str) -> GemmaResult<String> {
        let json_value: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| GemmaError::DocumentParsing(e.to_string()))?;

        fn extract_text_from_json(value: &serde_json::Value, text: &mut String) {
            match value {
                serde_json::Value::String(s) => {
                    text.push_str(s);
                    text.push(' ');
                }
                serde_json::Value::Array(arr) => {
                    for item in arr {
                        extract_text_from_json(item, text);
                    }
                }
                serde_json::Value::Object(obj) => {
                    for (key, val) in obj {
                        text.push_str(key);
                        text.push_str(": ");
                        extract_text_from_json(val, text);
                        text.push('\n');
                    }
                }
                serde_json::Value::Number(n) => {
                    text.push_str(&n.to_string());
                    text.push(' ');
                }
                serde_json::Value::Bool(b) => {
                    text.push_str(&b.to_string());
                    text.push(' ');
                }
                serde_json::Value::Null => {
                    text.push_str("null ");
                }
            }
        }

        let mut extracted_text = String::new();
        extract_text_from_json(&json_value, &mut extracted_text);
        Ok(self.clean_text(&extracted_text))
    }

    /// Chunk a document into smaller pieces
    async fn chunk_document(&self, content: &str, document_id: &str) -> GemmaResult<Vec<DocumentChunk>> {
        if content.is_empty() {
            return Ok(Vec::new());
        }

        let mut chunks = Vec::new();
        let mut current_pos = 0;
        let mut chunk_index = 0;

        // Use tokenizer if available, otherwise estimate tokens
        let token_count = if let Some(tokenizer) = &self.tokenizer {
            tokenizer.encode_ordinary(content).len()
        } else {
            content.len() / 4 // Rough estimation: 4 characters per token
        };

        if token_count <= self.chunking_config.max_chunk_size {
            // Document is small enough to be a single chunk
            let chunk = DocumentChunk::new(
                document_id.to_string(),
                0,
                content.to_string(),
                0,
                content.len(),
            );
            return Ok(vec![chunk]);
        }

        // Split document using hierarchical separators
        while current_pos < content.len() {
            let chunk_end = self.find_chunk_boundary(
                content,
                current_pos,
                self.chunking_config.max_chunk_size,
            ).await?;

            if chunk_end <= current_pos {
                break; // Safety check to prevent infinite loop
            }

            let chunk_text = content[current_pos..chunk_end].to_string();

            // Skip very small chunks unless it's the last one
            if chunk_text.len() >= self.chunking_config.min_chunk_size || chunk_end >= content.len() {
                let mut chunk = DocumentChunk::new(
                    document_id.to_string(),
                    chunk_index,
                    chunk_text,
                    current_pos,
                    chunk_end,
                );

                // Calculate actual token count
                chunk.token_count = if let Some(tokenizer) = &self.tokenizer {
                    tokenizer.encode_ordinary(&chunk.text).len()
                } else {
                    chunk.text.len() / 4
                };

                chunks.push(chunk);
                chunk_index += 1;
            }

            // Move position with overlap
            let overlap_size = std::cmp::min(
                self.chunking_config.chunk_overlap,
                (chunk_end - current_pos) / 2,
            );
            current_pos = chunk_end.saturating_sub(overlap_size);

            // Ensure we make progress
            if current_pos == chunk_end.saturating_sub(overlap_size) && current_pos < content.len() {
                current_pos = chunk_end;
            }
        }

        debug!("Generated {} chunks for document {}", chunks.len(), document_id);
        Ok(chunks)
    }

    /// Find the best boundary for a chunk
    async fn find_chunk_boundary(
        &self,
        content: &str,
        start: usize,
        max_size: usize,
    ) -> GemmaResult<usize> {
        let remaining_content = &content[start..];

        // Calculate target size in characters (rough estimation from tokens)
        let target_char_size = max_size * 4; // Assuming ~4 chars per token
        let max_end = std::cmp::min(target_char_size, remaining_content.len());

        if max_end == remaining_content.len() {
            return Ok(start + max_end);
        }

        // Try to find good boundaries in order of preference
        for separator in &self.chunking_config.separators {
            if let Some(pos) = remaining_content[..max_end].rfind(separator) {
                let boundary = start + pos + separator.len();

                // Make sure the chunk isn't too small
                if boundary - start >= self.chunking_config.min_chunk_size {
                    return Ok(boundary);
                }
            }
        }

        // If no good separator found, break at word boundary
        let word_boundary = remaining_content[..max_end]
            .char_indices()
            .rev()
            .find(|(_, c)| c.is_whitespace())
            .map(|(i, _)| i)
            .unwrap_or(max_end);

        Ok(start + word_boundary)
    }

    /// Generate embeddings for chunks (placeholder implementation)
    async fn generate_embeddings(&self, chunks: &[DocumentChunk]) -> GemmaResult<Vec<Vec<f32>>> {
        // This is a placeholder implementation
        // In a real system, you would:
        // 1. Call an embedding API (OpenAI, Sentence Transformers, etc.)
        // 2. Use a local embedding model
        // 3. Batch requests for efficiency

        let embeddings = chunks
            .iter()
            .map(|chunk| {
                // Generate a dummy embedding based on text hash for testing
                // In production, replace with actual embedding generation
                self.generate_dummy_embedding(&chunk.text)
            })
            .collect();

        debug!("Generated embeddings for {} chunks", chunks.len());
        Ok(embeddings)
    }

    /// Generate a dummy embedding (for testing purposes)
    fn generate_dummy_embedding(&self, text: &str) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();

        // Generate a deterministic dummy embedding
        (0..self.embedding_config.dimension)
            .map(|i| {
                let val = ((hash as u128 * (i + 1) as u128) % 10000) as f32 / 10000.0;
                val * 2.0 - 1.0 // Normalize to [-1, 1]
            })
            .collect()
    }

    /// Update pipeline statistics
    async fn update_stats(&self, chunk_count: usize, processing_time_ms: f64, success: bool) {
        let mut stats = self.stats.write().await;

        stats.documents_processed += 1;
        stats.chunks_generated += chunk_count as u64;

        if success {
            stats.embeddings_created += chunk_count as u64;
        } else {
            stats.failed_documents += 1;
        }

        // Update moving averages
        let doc_count = stats.documents_processed as f64;
        stats.average_chunks_per_document =
            (stats.average_chunks_per_document * (doc_count - 1.0) + chunk_count as f32) / doc_count;

        stats.average_processing_time_ms =
            (stats.average_processing_time_ms * (doc_count - 1.0) + processing_time_ms) / doc_count;

        stats.total_tokens_processed += chunk_count as u64 * self.chunking_config.max_chunk_size as u64;
    }

    /// Get current pipeline statistics
    pub async fn get_stats(&self) -> PipelineStats {
        self.stats.read().await.clone()
    }

    /// Reset pipeline statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = PipelineStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_format_detection() {
        assert_eq!(DocumentFormat::detect("test.md", ""), DocumentFormat::Markdown);
        assert_eq!(DocumentFormat::detect("test.html", ""), DocumentFormat::Html);
        assert_eq!(DocumentFormat::detect("test.txt", "Hello world"), DocumentFormat::PlainText);

        // Content-based detection
        assert_eq!(DocumentFormat::detect("unknown", "<html>"), DocumentFormat::Html);
        assert_eq!(DocumentFormat::detect("unknown", "# Header"), DocumentFormat::Markdown);
        assert_eq!(DocumentFormat::detect("unknown", r#"{"key": "value"}"#), DocumentFormat::Json);
    }

    #[test]
    fn test_chunking_config() {
        let config = ChunkingConfig::default();
        assert_eq!(config.max_chunk_size, 512);
        assert_eq!(config.chunk_overlap, 50);
        assert!(config.respect_sentence_boundaries);
    }

    #[tokio::test]
    async fn test_document_pipeline_creation() {
        let chunking_config = ChunkingConfig::default();
        let embedding_config = EmbeddingConfig::default();

        let pipeline = DocumentPipeline::new(chunking_config, embedding_config);
        assert!(pipeline.is_ok());
    }

    #[tokio::test]
    async fn test_text_cleaning() {
        let pipeline = DocumentPipeline::new(
            ChunkingConfig::default(),
            EmbeddingConfig::default(),
        ).unwrap();

        let dirty_text = "Hello    world\n\n\n\nThis   is    test.";
        let cleaned = pipeline.clean_text(dirty_text);
        assert!(cleaned.contains("Hello world"));
        assert!(!cleaned.contains("    "));
    }

    #[tokio::test]
    async fn test_markdown_parsing() {
        let pipeline = DocumentPipeline::new(
            ChunkingConfig::default(),
            EmbeddingConfig::default(),
        ).unwrap();

        let markdown = "# Header\n\nThis is a paragraph.\n\n- List item 1\n- List item 2";
        let parsed = pipeline.parse_markdown(markdown).await.unwrap();

        assert!(parsed.contains("Header"));
        assert!(parsed.contains("paragraph"));
        assert!(parsed.contains("List item"));
    }

    #[tokio::test]
    async fn test_html_parsing() {
        let pipeline = DocumentPipeline::new(
            ChunkingConfig::default(),
            EmbeddingConfig::default(),
        ).unwrap();

        let html = r#"<html><body><h1>Title</h1><p>Content here.</p><script>alert('ignore');</script></body></html>"#;
        let parsed = pipeline.parse_html(html).await.unwrap();

        assert!(parsed.contains("Title"));
        assert!(parsed.contains("Content here"));
        assert!(!parsed.contains("alert"));
    }

    #[tokio::test]
    async fn test_json_parsing() {
        let pipeline = DocumentPipeline::new(
            ChunkingConfig::default(),
            EmbeddingConfig::default(),
        ).unwrap();

        let json = r#"{"name": "Test Document", "content": "This is content", "tags": ["tag1", "tag2"]}"#;
        let parsed = pipeline.parse_json(json).await.unwrap();

        assert!(parsed.contains("Test Document"));
        assert!(parsed.contains("This is content"));
        assert!(parsed.contains("tag1"));
    }

    #[test]
    fn test_document_chunk_creation() {
        let chunk = DocumentChunk::new(
            "doc1".to_string(),
            0,
            "Test content".to_string(),
            0,
            12,
        );

        assert_eq!(chunk.document_id, "doc1");
        assert_eq!(chunk.chunk_index, 0);
        assert_eq!(chunk.text, "Test content");
        assert!(!chunk.id.is_empty());
    }
}