//! Document processing benchmarks for the RAG-Redis system
//!
//! These benchmarks measure the performance of:
//! - Document parsing and preprocessing
//! - Text chunking with various strategies
//! - Token counting and text analysis
//! - Metadata extraction and enrichment
//! - Concurrent document processing
//! - Memory efficiency during processing

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

use rag_redis_system::{
    config::{ChunkingConfig, ChunkingMethod, DocumentConfig, PreprocessingConfig},
    Document, DocumentChunk, DocumentPipeline,
};

use serde_json::json;

/// Generate test documents of various sizes and types
fn generate_test_documents() -> Vec<(String, String, serde_json::Value)> {
    vec![
        // Short technical document
        (
            "short_tech".to_string(),
            "Machine Learning Fundamentals\n\nMachine learning is a subset of artificial intelligence that focuses on developing algorithms. Key concepts include supervised learning, unsupervised learning, and reinforcement learning. Popular algorithms include linear regression, decision trees, and neural networks.".to_string(),
            json!({"category": "technology", "length": "short", "language": "en"})
        ),

        // Medium research paper excerpt
        (
            "medium_research".to_string(),
            generate_research_paper_text(1000),
            json!({"category": "research", "length": "medium", "language": "en", "domain": "computer_science"})
        ),

        // Long documentation
        (
            "long_docs".to_string(),
            generate_documentation_text(5000),
            json!({"category": "documentation", "length": "long", "language": "en", "format": "technical"})
        ),

        // Code-heavy document
        (
            "code_heavy".to_string(),
            generate_code_heavy_text(2000),
            json!({"category": "programming", "length": "medium", "language": "mixed", "has_code": true})
        ),

        // Structured data document
        (
            "structured_data".to_string(),
            generate_structured_text(3000),
            json!({"category": "data", "length": "medium", "language": "en", "structure": "high"})
        ),

        // Very long document
        (
            "very_long".to_string(),
            generate_long_article_text(10000),
            json!({"category": "article", "length": "very_long", "language": "en", "complexity": "high"})
        ),
    ]
}

fn generate_research_paper_text(target_words: usize) -> String {
    let base_text = r#"
# Abstract

Recent advances in natural language processing have demonstrated the effectiveness of transformer-based architectures
for various downstream tasks. This paper presents a comprehensive analysis of attention mechanisms and their impact
on model performance across different domains.

## Introduction

Natural language processing (NLP) has experienced remarkable progress in recent years, largely driven by the development
of transformer architectures. The attention mechanism, first introduced in sequence-to-sequence models, has become a
fundamental component of modern NLP systems.

### Background

The transformer architecture, introduced by Vaswani et al. in "Attention Is All You Need," revolutionized the field
by replacing recurrent neural networks with self-attention mechanisms. This approach enables parallel processing of
sequences and captures long-range dependencies more effectively.

## Methodology

Our experimental setup consists of multiple datasets spanning various NLP tasks including:
1. Text classification
2. Named entity recognition
3. Question answering
4. Text summarization

### Model Architecture

We implement a multi-layer transformer with the following specifications:
- 12 attention heads
- 768-dimensional embeddings
- Layer normalization
- Residual connections
- Position encodings

### Training Procedure

Models are trained using the Adam optimizer with a learning rate schedule that includes:
- Warm-up phase: 10,000 steps
- Peak learning rate: 5e-4
- Linear decay schedule
- Weight decay: 0.01

## Results and Discussion

Our experiments demonstrate significant improvements across all evaluated tasks. The attention mechanism shows
particular effectiveness in capturing semantic relationships between distant tokens in the input sequence.

### Performance Metrics

The model achieves state-of-the-art results on several benchmarks:
- GLUE score: 88.9
- SQuAD v2.0: 89.3 F1
- CoNLL-2003 NER: 92.4 F1

## Conclusion

This work provides evidence for the effectiveness of attention mechanisms in modern NLP systems. Future research
directions include exploring sparse attention patterns and improving computational efficiency.
"#;

    let words: Vec<&str> = base_text.split_whitespace().collect();
    let base_word_count = words.len();

    if target_words <= base_word_count {
        words[..target_words].join(" ")
    } else {
        let repetitions = (target_words / base_word_count) + 1;
        let extended_text = (0..repetitions)
            .map(|i| format!("\n\n## Section {}\n\n{}", i + 1, base_text))
            .collect::<Vec<_>>()
            .join("");

        let extended_words: Vec<&str> = extended_text.split_whitespace().collect();
        extended_words[..std::cmp::min(target_words, extended_words.len())].join(" ")
    }
}

fn generate_documentation_text(target_words: usize) -> String {
    let base_text = r#"
# API Documentation

## Overview

This API provides comprehensive access to data processing and analysis capabilities. The system supports multiple
data formats and offers both synchronous and asynchronous processing modes.

### Authentication

All API endpoints require authentication using Bearer tokens. Tokens can be obtained through the authentication
endpoint using valid credentials.

```
POST /auth/token
Content-Type: application/json

{
    "username": "your_username",
    "password": "your_password"
}
```

### Rate Limiting

The API implements rate limiting to ensure fair usage across all clients:
- Free tier: 100 requests per hour
- Professional: 1,000 requests per hour
- Enterprise: 10,000 requests per hour

## Endpoints

### Data Processing

#### POST /process/text

Processes text data using various NLP techniques including tokenization, named entity recognition, and sentiment analysis.

**Parameters:**
- `text` (string, required): Input text to process
- `options` (object, optional): Processing configuration
  - `language` (string): Language code (default: "en")
  - `include_sentiment` (boolean): Include sentiment analysis
  - `include_entities` (boolean): Include named entity recognition

**Response:**
```json
{
    "status": "success",
    "data": {
        "tokens": ["example", "tokens"],
        "entities": [],
        "sentiment": {
            "score": 0.85,
            "label": "positive"
        }
    }
}
```

#### POST /process/batch

Processes multiple documents in a single request. Supports up to 100 documents per batch.

**Parameters:**
- `documents` (array, required): Array of document objects
- `parallel` (boolean, optional): Enable parallel processing

### Data Retrieval

#### GET /data/{id}

Retrieves processed data by ID.

**Response:**
```json
{
    "id": "doc_12345",
    "status": "completed",
    "results": {
        "processing_time": 1.23,
        "word_count": 567,
        "metadata": {}
    }
}
```

#### GET /data/search

Searches through processed documents using vector similarity.

**Parameters:**
- `query` (string, required): Search query
- `limit` (integer, optional): Maximum results (default: 10)
- `threshold` (float, optional): Similarity threshold

## Error Handling

The API uses standard HTTP status codes and provides detailed error messages in JSON format:

```json
{
    "error": {
        "code": "INVALID_REQUEST",
        "message": "Missing required parameter: text",
        "details": {}
    }
}
```

### Common Error Codes

- `INVALID_REQUEST`: Malformed request
- `UNAUTHORIZED`: Invalid or expired token
- `RATE_LIMITED`: Exceeded rate limit
- `INTERNAL_ERROR`: Server error

## SDKs and Libraries

Official SDKs are available for popular programming languages:
- Python: `pip install api-client`
- JavaScript: `npm install api-client`
- Java: Available on Maven Central
- Go: `go get github.com/company/api-client`

## Examples

### Python Example

```python
import api_client

client = api_client.Client(token="your_token")
result = client.process_text("Hello, world!")
print(result.sentiment)
```

### JavaScript Example

```javascript
const client = new ApiClient('your_token');
const result = await client.processText('Hello, world!');
console.log(result.sentiment);
```

## Changelog

### Version 2.1.0
- Added batch processing endpoint
- Improved error messages
- Enhanced rate limiting

### Version 2.0.0
- Breaking: Changed authentication method
- Added vector search capabilities
- Performance improvements
"#;

    let words: Vec<&str> = base_text.split_whitespace().collect();
    let base_word_count = words.len();

    if target_words <= base_word_count {
        words[..target_words].join(" ")
    } else {
        let repetitions = (target_words / base_word_count) + 1;
        let extended_text = (0..repetitions)
            .map(|i| format!("\n\n## Additional Section {}\n\n{}", i + 1, base_text))
            .collect::<Vec<_>>()
            .join("");

        let extended_words: Vec<&str> = extended_text.split_whitespace().collect();
        extended_words[..std::cmp::min(target_words, extended_words.len())].join(" ")
    }
}

fn generate_code_heavy_text(target_words: usize) -> String {
    let base_text = r#"
# Rust Programming Guide

## Memory Management

Rust's ownership system ensures memory safety without garbage collection. Here's a basic example:

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1; // s1 is moved to s2
    // println!("{}", s1); // This would cause a compile error
    println!("{}", s2); // This works fine
}
```

### Borrowing and References

Instead of transferring ownership, you can borrow values:

```rust
fn calculate_length(s: &String) -> usize {
    s.len()
} // s goes out of scope, but nothing happens because we don't have ownership

fn main() {
    let s1 = String::from("hello");
    let len = calculate_length(&s1);
    println!("The length of '{}' is {}.", s1, len);
}
```

### Mutable References

You can also have mutable references:

```rust
fn change(some_string: &mut String) {
    some_string.push_str(", world");
}

fn main() {
    let mut s = String::from("hello");
    change(&mut s);
    println!("{}", s); // Prints "hello, world"
}
```

## Error Handling

Rust uses Result and Option types for error handling:

```rust
use std::fs::File;
use std::io::ErrorKind;

fn main() {
    let f = File::open("hello.txt");

    let f = match f {
        Ok(file) => file,
        Err(error) => match error.kind() {
            ErrorKind::NotFound => match File::create("hello.txt") {
                Ok(fc) => fc,
                Err(e) => panic!("Problem creating the file: {:?}", e),
            },
            other_error => {
                panic!("Problem opening the file: {:?}", other_error)
            }
        },
    };
}
```

### Using the ? Operator

The ? operator makes error propagation more concise:

```rust
use std::fs::File;
use std::io;
use std::io::Read;

fn read_username_from_file() -> Result<String, io::Error> {
    let mut f = File::open("hello.txt")?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s)
}
```

## Generic Types

Generics allow you to write flexible, reusable code:

```rust
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];

    for item in list {
        if item > largest {
            largest = item;
        }
    }

    largest
}

fn main() {
    let number_list = vec![34, 50, 25, 100, 65];
    let result = largest(&number_list);
    println!("The largest number is {}", result);

    let char_list = vec!['y', 'm', 'a', 'q'];
    let result = largest(&char_list);
    println!("The largest char is {}", result);
}
```

### Struct Generics

Structs can also use generics:

```rust
struct Point<T> {
    x: T,
    y: T,
}

impl<T> Point<T> {
    fn new(x: T, y: T) -> Self {
        Point { x, y }
    }
}

impl Point<f32> {
    fn distance_from_origin(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}
```

## Traits

Traits define shared behavior:

```rust
pub trait Summary {
    fn summarize(&self) -> String;
}

pub struct NewsArticle {
    pub headline: String,
    pub location: String,
    pub author: String,
    pub content: String,
}

impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{}, by {} ({})", self.headline, self.author, self.location)
    }
}

pub struct Tweet {
    pub username: String,
    pub content: String,
    pub reply: bool,
    pub retweet: bool,
}

impl Summary for Tweet {
    fn summarize(&self) -> String {
        format!("{}: {}", self.username, self.content)
    }
}
```
"#;

    let words: Vec<&str> = base_text.split_whitespace().collect();
    let base_word_count = words.len();

    if target_words <= base_word_count {
        words[..target_words].join(" ")
    } else {
        let repetitions = (target_words / base_word_count) + 1;
        let extended_text = (0..repetitions)
            .map(|i| format!("\n\n## Advanced Topic {}\n\n{}", i + 1, base_text))
            .collect::<Vec<_>>()
            .join("");

        let extended_words: Vec<&str> = extended_text.split_whitespace().collect();
        extended_words[..std::cmp::min(target_words, extended_words.len())].join(" ")
    }
}

fn generate_structured_text(target_words: usize) -> String {
    let base_text = r#"
# Data Analysis Report

## Executive Summary

| Metric | Q1 2024 | Q2 2024 | Change |
|--------|---------|---------|--------|
| Revenue | $2.5M | $3.1M | +24% |
| Users | 150K | 185K | +23% |
| Retention | 85% | 88% | +3% |

Key findings:
1. Revenue growth exceeded expectations
2. User acquisition remained strong
3. Customer satisfaction improved
4. Market share increased by 15%

## Detailed Analysis

### Revenue Performance

The company achieved significant revenue growth in Q2 2024:

- **Product Sales**: $1.8M (58% of total)
- **Subscription Revenue**: $1.0M (32% of total)
- **Professional Services**: $0.3M (10% of total)

Monthly breakdown:
- April: $950K
- May: $1.05M
- June: $1.1M

### User Metrics

User engagement metrics show positive trends:

```
Active Users by Month:
- April: 165,000 (+10% MoM)
- May: 175,000 (+6% MoM)
- June: 185,000 (+6% MoM)
```

Cohort analysis reveals:
- Week 1 retention: 72%
- Week 4 retention: 45%
- Week 12 retention: 28%

### Geographic Distribution

| Region | Users | Revenue | Growth |
|--------|-------|---------|--------|
| North America | 95K | $1.9M | +18% |
| Europe | 55K | $0.8M | +31% |
| Asia Pacific | 25K | $0.3M | +42% |
| Other | 10K | $0.1M | +15% |

### Product Performance

Top performing features:
1. **Dashboard Analytics** - 89% adoption
2. **API Integration** - 67% adoption
3. **Mobile App** - 78% adoption
4. **Reporting Suite** - 54% adoption

Feature usage correlation with retention:
- High API usage: 95% retention
- Dashboard power users: 92% retention
- Mobile-only users: 78% retention

### Market Analysis

Competitive positioning:
- Market Leader: 35% share
- Competitor A: 28% share
- Our Company: 15% share (+3% QoQ)
- Competitor B: 12% share
- Others: 10% share

Industry trends:
- Overall market growth: 8% QoQ
- Cloud adoption: +45% YoY
- Mobile usage: +67% YoY
- API integrations: +89% YoY

## Technical Metrics

System performance indicators:

```yaml
availability: 99.97%
response_time_p95: 245ms
error_rate: 0.03%
throughput: 15000 rps
```

Infrastructure costs:
- Compute: $45K/month
- Storage: $12K/month
- Network: $8K/month
- Monitoring: $3K/month

## Recommendations

### Short Term (Q3 2024)
1. Increase marketing spend in APAC region
2. Enhance mobile application features
3. Expand API capabilities
4. Improve onboarding experience

### Medium Term (Q4 2024 - Q1 2025)
1. Launch enterprise tier
2. Develop partnerships in Europe
3. Implement advanced analytics
4. Build customer success program

### Long Term (2025+)
1. Explore new market segments
2. Consider acquisition opportunities
3. Expand into adjacent products
4. Develop AI/ML capabilities

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Competitive pressure | High | Medium | Innovation focus |
| Economic downturn | Medium | High | Diversification |
| Technical debt | Medium | Medium | Engineering investment |
| Talent retention | Low | High | Compensation review |

## Conclusion

Q2 2024 results demonstrate strong business momentum with revenue growth of 24% and user growth of 23%.
The company is well-positioned for continued expansion, particularly in international markets.

Key success factors:
- Strong product-market fit
- Effective go-to-market strategy
- Robust technical infrastructure
- Engaged customer base

Next steps include expanding international presence, enhancing product capabilities, and building
strategic partnerships to accelerate growth.
"#;

    let words: Vec<&str> = base_text.split_whitespace().collect();
    let base_word_count = words.len();

    if target_words <= base_word_count {
        words[..target_words].join(" ")
    } else {
        let repetitions = (target_words / base_word_count) + 1;
        let extended_text = (0..repetitions)
            .map(|i| format!("\n\n## Additional Analysis {}\n\n{}", i + 1, base_text))
            .collect::<Vec<_>>()
            .join("");

        let extended_words: Vec<&str> = extended_text.split_whitespace().collect();
        extended_words[..std::cmp::min(target_words, extended_words.len())].join(" ")
    }
}

fn generate_long_article_text(target_words: usize) -> String {
    let sections = vec![
        "Introduction to the topic and its importance in modern society.",
        "Historical background and evolution of key concepts over time.",
        "Current state of research and recent developments in the field.",
        "Methodology and approach used in comprehensive analysis.",
        "Detailed findings and their implications for future work.",
        "Comparative analysis with existing solutions and alternatives.",
        "Case studies demonstrating practical applications and outcomes.",
        "Challenges and limitations identified during research process.",
        "Future directions and potential areas for further investigation.",
        "Conclusions and recommendations for practitioners and researchers.",
    ];

    let paragraph_template = "This section explores various aspects of the topic with detailed analysis and supporting evidence. Research indicates significant trends and patterns that merit further investigation. The methodology employed ensures robust and reliable results that contribute to the broader understanding of the subject matter. Expert opinions and peer-reviewed sources provide additional validation for the presented findings.";

    let mut content = Vec::new();
    let mut word_count = 0;
    let target_per_section = target_words / sections.len();

    for (i, section_title) in sections.iter().enumerate() {
        content.push(format!("## Section {}: {}\n", i + 1, section_title));

        while word_count < (i + 1) * target_per_section && word_count < target_words {
            content.push(paragraph_template.to_string());
            content.push("\n\n".to_string());
            word_count += paragraph_template.split_whitespace().count();
        }

        if word_count >= target_words {
            break;
        }
    }

    let full_text = content.join("\n");
    let words: Vec<&str> = full_text.split_whitespace().collect();
    words[..std::cmp::min(target_words, words.len())].join(" ")
}

/// Create document processing pipeline with specified configuration
fn create_document_pipeline(
    chunking_method: ChunkingMethod,
    chunk_size: usize,
) -> DocumentPipeline {
    let config = DocumentConfig {
        chunking: ChunkingConfig {
            method: chunking_method,
            chunk_size,
            chunk_overlap: chunk_size / 10, // 10% overlap
            min_chunk_size: chunk_size / 4,
            max_chunk_size: chunk_size * 2,
            separator_priority: vec![
                "\n\n".to_string(),
                "\n".to_string(),
                ". ".to_string(),
                "! ".to_string(),
                "? ".to_string(),
                "; ".to_string(),
                ", ".to_string(),
                " ".to_string(),
            ],
        },
        supported_formats: vec!["txt".to_string(), "md".to_string(), "html".to_string()],
        max_file_size: 50 * 1024 * 1024, // 50MB
        preprocessing: PreprocessingConfig {
            clean_whitespace: true,
            normalize_unicode: true,
            remove_html_tags: true,
            extract_metadata: true,
        },
    };

    DocumentPipeline::new(config)
}

/// Benchmark document processing with various configurations
fn bench_document_processing(c: &mut Criterion) {
    let test_documents = generate_test_documents();
    let chunking_methods = [
        ChunkingMethod::TokenBased,
        ChunkingMethod::CharacterBased,
        ChunkingMethod::Semantic,
    ];
    let chunk_sizes = [256, 512, 1024];

    let mut group = c.benchmark_group("document_processing");

    for (doc_type, content, metadata) in &test_documents {
        let word_count = content.split_whitespace().count();
        group.throughput(Throughput::Elements(word_count as u64));

        for &chunking_method in &chunking_methods {
            for &chunk_size in &chunk_sizes {
                let pipeline = create_document_pipeline(chunking_method, chunk_size);

                group.bench_with_input(
                    BenchmarkId::new(format!("{:?}_{}", chunking_method, chunk_size), doc_type),
                    &(pipeline, content.clone(), metadata.clone()),
                    |b, (pipeline, content, metadata)| {
                        b.iter(|| pipeline.process(content, metadata.clone()).unwrap())
                    },
                );
            }
        }
    }

    group.finish();
}

/// Benchmark text chunking strategies
fn bench_text_chunking(c: &mut Criterion) {
    let test_texts = vec![
        ("short", generate_research_paper_text(500)),
        ("medium", generate_documentation_text(2000)),
        ("long", generate_long_article_text(5000)),
    ];

    let chunking_configs = [
        (ChunkingMethod::TokenBased, 256),
        (ChunkingMethod::TokenBased, 512),
        (ChunkingMethod::CharacterBased, 1000),
        (ChunkingMethod::CharacterBased, 2000),
        (ChunkingMethod::Semantic, 512),
    ];

    let mut group = c.benchmark_group("text_chunking");

    for (text_type, text) in &test_texts {
        for &(method, size) in &chunking_configs {
            let pipeline = create_document_pipeline(method, size);
            let document = Document {
                id: "benchmark_doc".to_string(),
                title: format!("Benchmark Document - {}", text_type),
                content: text.clone(),
                metadata: json!({"type": text_type}),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
                tags: vec!["benchmark".to_string()],
                source_url: None,
                language: Some("en".to_string()),
                word_count: text.split_whitespace().count(),
                embedding: None,
            };

            group.bench_with_input(
                BenchmarkId::new(format!("{:?}_{}", method, size), text_type),
                &(pipeline, document),
                |b, (pipeline, doc)| b.iter(|| pipeline.chunk_document(doc).unwrap()),
            );
        }
    }

    group.finish();
}

/// Benchmark concurrent document processing
fn bench_concurrent_processing(c: &mut Criterion) {
    let documents = generate_test_documents();
    let pipeline = create_document_pipeline(ChunkingMethod::TokenBased, 512);
    let pipeline = std::sync::Arc::new(pipeline);

    let mut group = c.benchmark_group("concurrent_processing");

    // Sequential processing baseline
    group.bench_function("sequential", |b| {
        b.iter(|| {
            for (_, content, metadata) in &documents {
                pipeline.process(content, metadata.clone()).unwrap();
            }
        })
    });

    // Concurrent processing
    group.bench_function("concurrent", |b| {
        b.to_async(tokio::runtime::Runtime::new().unwrap())
            .iter(|| async {
                let futures: Vec<_> = documents
                    .iter()
                    .map(|(_, content, metadata)| {
                        let pipeline_clone = pipeline.clone();
                        let content_clone = content.clone();
                        let metadata_clone = metadata.clone();

                        tokio::spawn(async move {
                            pipeline_clone
                                .process(&content_clone, metadata_clone)
                                .unwrap()
                        })
                    })
                    .collect();

                for future in futures {
                    future.await.unwrap();
                }
            })
    });

    group.finish();
}

/// Benchmark preprocessing operations
fn bench_preprocessing(c: &mut Criterion) {
    let test_texts = vec![
        ("clean", "This is clean text with proper formatting."),
        ("messy_whitespace", "This    has  \n\n  lots   of   \t  whitespace  \r\n  issues."),
        ("html_tags", "<html><body><h1>Title</h1><p>Content with <b>bold</b> and <i>italic</i> text.</p></body></html>"),
        ("unicode", "Text with émojis 🚀 and spëcial chäractérs ñoñó 中文 العربية"),
        ("mixed", "<div>Mixed content with 🌟 emojis,    whitespace,   and <b>HTML</b> tags.</div>"),
    ];

    let preprocessing_configs = vec![
        (
            "minimal",
            PreprocessingConfig {
                clean_whitespace: false,
                normalize_unicode: false,
                remove_html_tags: false,
                extract_metadata: false,
            },
        ),
        (
            "standard",
            PreprocessingConfig {
                clean_whitespace: true,
                normalize_unicode: true,
                remove_html_tags: true,
                extract_metadata: true,
            },
        ),
        (
            "aggressive",
            PreprocessingConfig {
                clean_whitespace: true,
                normalize_unicode: true,
                remove_html_tags: true,
                extract_metadata: true,
            },
        ),
    ];

    let mut group = c.benchmark_group("preprocessing");

    for (text_type, text) in &test_texts {
        for (config_type, preprocessing_config) in &preprocessing_configs {
            let config = DocumentConfig {
                preprocessing: preprocessing_config.clone(),
                ..Default::default()
            };
            let pipeline = DocumentPipeline::new(config);

            group.bench_with_input(
                BenchmarkId::new(config_type, text_type),
                &(pipeline, text.to_string()),
                |b, (pipeline, text)| b.iter(|| pipeline.process(text, json!({})).unwrap()),
            );
        }
    }

    group.finish();
}

/// Benchmark memory efficiency during processing
fn bench_memory_efficiency(c: &mut Criterion) {
    let document_sizes = [1000, 5000, 10000, 20000]; // Word counts
    let chunk_sizes = [256, 512, 1024];

    let mut group = c.benchmark_group("memory_efficiency");

    for &doc_size in &document_sizes {
        for &chunk_size in &chunk_sizes {
            let content = generate_long_article_text(doc_size);
            let pipeline = create_document_pipeline(ChunkingMethod::TokenBased, chunk_size);

            group.throughput(Throughput::Elements(doc_size as u64));
            group.bench_with_input(
                BenchmarkId::new("memory_usage", format!("{}w_{}c", doc_size, chunk_size)),
                &(pipeline, content),
                |b, (pipeline, content)| {
                    b.iter_batched(
                        || content.clone(),
                        |text| {
                            let document = pipeline.process(&text, json!({})).unwrap();
                            let chunks = pipeline.chunk_document(&document).unwrap();
                            (document, chunks)
                        },
                        criterion::BatchSize::LargeInput,
                    )
                },
            );
        }
    }

    group.finish();
}

/// Benchmark token counting and text analysis
fn bench_text_analysis(c: &mut Criterion) {
    let test_texts = vec![
        ("simple", "Simple text with basic words and punctuation."),
        ("complex", "Complex text with contractions, abbreviations, URLs like https://example.com, and @mentions."),
        ("multilingual", "Mixed languages: Hello, 你好, مرحبا, Здравствуйте, Bonjour, Hola, こんにちは"),
        ("technical", "Technical content with code snippets: `let x = 42;` and formulas like E=mc²."),
        ("long", &generate_research_paper_text(1000)),
    ];

    let mut group = c.benchmark_group("text_analysis");

    for (text_type, text) in &test_texts {
        // Benchmark word counting
        group.bench_with_input(
            BenchmarkId::new("word_count", text_type),
            text,
            |b, text| b.iter(|| text.split_whitespace().count()),
        );

        // Benchmark character counting
        group.bench_with_input(
            BenchmarkId::new("char_count", text_type),
            text,
            |b, text| b.iter(|| text.chars().count()),
        );

        // Benchmark token-based chunking preparation
        group.bench_with_input(
            BenchmarkId::new("tokenization", text_type),
            text,
            |b, text| {
                b.iter(|| {
                    // Simplified tokenization for benchmarking
                    text.split_whitespace().enumerate().collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}

/// Benchmark chunk overlap calculations
fn bench_chunk_overlap(c: &mut Criterion) {
    let chunk_sizes = [256, 512, 1024];
    let overlap_percentages = [10, 20, 50]; // Percentage of chunk size
    let text = generate_documentation_text(5000);

    let mut group = c.benchmark_group("chunk_overlap");

    for &chunk_size in &chunk_sizes {
        for &overlap_pct in &overlap_percentages {
            let overlap_size = (chunk_size * overlap_pct) / 100;
            let pipeline = create_document_pipeline(ChunkingMethod::TokenBased, chunk_size);

            group.bench_with_input(
                BenchmarkId::new(
                    "overlap_calculation",
                    format!("{}size_{}overlap", chunk_size, overlap_pct),
                ),
                &(pipeline, text.clone()),
                |b, (pipeline, text)| {
                    b.iter(|| {
                        let document = pipeline.process(text, json!({})).unwrap();
                        pipeline.chunk_document(&document).unwrap()
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark document format detection and parsing
fn bench_format_detection(c: &mut Criterion) {
    let format_samples = vec![
        ("plain_text", "This is plain text content without any special formatting."),
        ("markdown", "# Markdown Document\n\n## Section\n\nThis is **bold** and *italic* text.\n\n- List item 1\n- List item 2"),
        ("html", "<html><head><title>Test</title></head><body><h1>HTML Document</h1><p>Paragraph content.</p></body></html>"),
        ("json_like", r#"{"title": "Document", "content": "JSON-like content", "metadata": {"type": "test"}}"#),
        ("code", "fn main() {\n    println!(\"Hello, world!\");\n    let x = 42;\n    // This is a comment\n}"),
    ];

    let mut group = c.benchmark_group("format_detection");

    for (format_type, content) in &format_samples {
        group.bench_with_input(
            BenchmarkId::new("detect_format", format_type),
            content,
            |b, content| {
                b.iter(|| {
                    // Simulate format detection logic
                    if content.contains('<') && content.contains('>') {
                        "html"
                    } else if content.contains('#') && content.contains('\n') {
                        "markdown"
                    } else if content.contains('{') && content.contains('}') {
                        "json"
                    } else if content.contains("fn ") || content.contains("//") {
                        "code"
                    } else {
                        "plain_text"
                    }
                })
            },
        );
    }

    group.finish();
}

/// Comprehensive benchmark suite
criterion_group!(
    name = document_processing_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(15))
        .warm_up_time(Duration::from_secs(3))
        .sample_size(50);
    targets =
        bench_document_processing,
        bench_text_chunking,
        bench_concurrent_processing,
        bench_preprocessing,
        bench_memory_efficiency,
        bench_text_analysis,
        bench_chunk_overlap,
        bench_format_detection
);

criterion_main!(document_processing_benches);

#[cfg(test)]
mod benchmark_tests {
    use super::*;

    #[test]
    fn test_document_generation() {
        let docs = generate_test_documents();
        assert!(!docs.is_empty());

        for (doc_type, content, metadata) in &docs {
            assert!(!doc_type.is_empty());
            assert!(!content.is_empty());
            assert!(metadata.is_object());

            let word_count = content.split_whitespace().count();
            assert!(word_count > 0);

            println!("Generated {}: {} words", doc_type, word_count);
        }
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = create_document_pipeline(ChunkingMethod::TokenBased, 512);

        // Test processing a simple document
        let content = "This is a test document for pipeline validation.";
        let metadata = json!({"test": true});

        let document = pipeline.process(content, metadata).unwrap();
        assert!(!document.id.is_empty());
        assert!(!document.content.is_empty());

        let chunks = pipeline.chunk_document(&document).unwrap();
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_text_generation_functions() {
        let research_text = generate_research_paper_text(500);
        assert!(research_text.split_whitespace().count() <= 500);

        let doc_text = generate_documentation_text(1000);
        assert!(doc_text.split_whitespace().count() <= 1000);

        let code_text = generate_code_heavy_text(800);
        assert!(code_text.contains("rust") || code_text.contains("fn "));

        let structured_text = generate_structured_text(1200);
        assert!(structured_text.contains("|") || structured_text.contains("##"));

        let long_text = generate_long_article_text(2000);
        assert!(long_text.split_whitespace().count() <= 2000);
    }

    #[test]
    fn test_chunking_methods() {
        let content = generate_research_paper_text(1000);
        let methods = [
            ChunkingMethod::TokenBased,
            ChunkingMethod::CharacterBased,
            ChunkingMethod::Semantic,
        ];

        for method in &methods {
            let pipeline = create_document_pipeline(*method, 256);
            let document = pipeline.process(&content, json!({})).unwrap();
            let chunks = pipeline.chunk_document(&document).unwrap();

            assert!(!chunks.is_empty());
            println!("Method {:?} produced {} chunks", method, chunks.len());
        }
    }
}
