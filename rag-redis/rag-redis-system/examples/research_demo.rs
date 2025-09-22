//! Research module demonstration
//!
//! This example shows how to use the research module with:
//! - Web scraping with rate limiting
//! - Content extraction and quality scoring
//! - Domain filtering
//! - API integration
//! - Concurrent request processing

use std::time::Duration;
use tokio;

// Simplified version for demo
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DemoError {
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("Request timeout: {0}")]
    Timeout(String),
    #[error("Content too large: {0}")]
    ContentTooLarge(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

type Result<T> = std::result::Result<T, DemoError>;

impl From<reqwest::Error> for DemoError {
    fn from(err: reqwest::Error) -> Self {
        DemoError::Http(err.to_string())
    }
}

impl From<serde_json::Error> for DemoError {
    fn from(err: serde_json::Error) -> Self {
        DemoError::Serialization(err.to_string())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ResearchConfig {
    max_concurrent_requests: Option<usize>,
    request_timeout_secs: Option<u64>,
    rate_limit_per_minute: Option<u32>,
    user_agent: Option<String>,
    blocked_domains: Vec<String>,
    allowed_domains: Option<Vec<String>>,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: Some(10),
            request_timeout_secs: Some(30),
            rate_limit_per_minute: Some(60),
            user_agent: Some("RAG-Research-Bot/1.0".to_string()),
            blocked_domains: vec!["facebook.com".to_string(), "twitter.com".to_string()],
            allowed_domains: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchResult {
    pub id: String,
    pub url: String,
    pub title: String,
    pub text: String,
    pub score: f64,
    pub metadata: serde_json::Value,
}

async fn demonstrate_research_capabilities() -> Result<()> {
    println!("🔍 Research Module Demonstration");
    println!("================================");

    // 1. Configuration Demo
    let config = ResearchConfig::default();
    println!("\n✅ Created research configuration:");
    println!(
        "   - Max concurrent requests: {:?}",
        config.max_concurrent_requests
    );
    println!(
        "   - Rate limit: {:?} requests/minute",
        config.rate_limit_per_minute
    );
    println!("   - Blocked domains: {:?}", config.blocked_domains);

    // 2. Rate Limiting Demo
    println!("\n⏱️ Rate limiting demonstration:");
    let start = std::time::Instant::now();

    // Simulate rate-limited requests
    for i in 1..=5 {
        tokio::time::sleep(Duration::from_millis(100)).await;
        println!("   Request {} completed after {:?}", i, start.elapsed());
    }

    // 3. Content Quality Scoring Demo
    println!("\n📊 Content quality scoring:");

    let high_quality_content = "This comprehensive article discusses the implications of artificial intelligence in modern software development. It covers multiple aspects including machine learning algorithms, neural networks, and their practical applications in various industries. The content is well-structured with clear sections and provides detailed examples.";

    let low_quality_content = "AI is good.";

    let high_score = calculate_quality_score(high_quality_content, "AI Article");
    let low_score = calculate_quality_score(low_quality_content, "");

    println!("   High-quality content score: {:.2}", high_score);
    println!("   Low-quality content score: {:.2}", low_score);

    // 4. Domain Filtering Demo
    println!("\n🚫 Domain filtering demonstration:");
    let test_domains = vec![
        "facebook.com",
        "example.com",
        "twitter.com",
        "wikipedia.org",
        "github.com",
    ];

    for domain in test_domains {
        let is_allowed = !config
            .blocked_domains
            .iter()
            .any(|blocked| domain.contains(blocked));
        println!(
            "   Domain {}: {}",
            domain,
            if is_allowed {
                "✅ Allowed"
            } else {
                "❌ Blocked"
            }
        );
    }

    // 5. Concurrent Processing Demo
    println!("\n🔄 Concurrent processing simulation:");
    let urls = vec![
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
        "https://example.com/page4",
    ];

    let start = std::time::Instant::now();
    let mut tasks = Vec::new();

    for (i, url) in urls.iter().enumerate() {
        let url = url.to_string();
        let task = tokio::spawn(async move {
            // Simulate processing time
            tokio::time::sleep(Duration::from_millis(200 + i as u64 * 50)).await;
            SearchResult {
                id: format!("result-{}", i),
                url,
                title: format!("Page {}", i + 1),
                text: format!("Content from page {}", i + 1),
                score: 0.8 - (i as f64 * 0.1),
                metadata: serde_json::json!({"processing_time_ms": 200 + i * 50}),
            }
        });
        tasks.push(task);
    }

    let results = futures::future::try_join_all(tasks)
        .await
        .map_err(|e| DemoError::Internal(format!("Task join error: {}", e)))?;

    println!(
        "   Processed {} URLs in {:?}",
        results.len(),
        start.elapsed()
    );
    for result in &results {
        println!("   - {}: score={:.2}", result.title, result.score);
    }

    // 6. Error Handling Demo
    println!("\n❌ Error handling demonstration:");

    // Simulate various error types
    let errors = vec![
        DemoError::Timeout("Request timeout after 30s".to_string()),
        DemoError::ContentTooLarge("Content size 15MB exceeds 10MB limit".to_string()),
        DemoError::Http("HTTP 404 Not Found".to_string()),
    ];

    for error in errors {
        println!("   Handled error: {}", error);
    }

    // 7. Retry Logic Demo
    println!("\n🔄 Retry logic demonstration:");
    let mut attempts = 0;
    let max_retries = 3;

    loop {
        attempts += 1;
        println!("   Attempt {} of {}", attempts, max_retries);

        // Simulate success on 3rd attempt
        if attempts == 3 {
            println!("   ✅ Request succeeded!");
            break;
        } else if attempts < max_retries {
            println!(
                "   ❌ Request failed, retrying in {:?}...",
                Duration::from_millis(100 * attempts as u64)
            );
            tokio::time::sleep(Duration::from_millis(100 * attempts as u64)).await;
        } else {
            println!("   ❌ Max retries exceeded");
            break;
        }
    }

    println!("\n🎉 Research module demonstration complete!");
    Ok(())
}

fn calculate_quality_score(text: &str, title: &str) -> f64 {
    let mut score = 0.0;

    // Content length score (normalized)
    let length_score = (text.len() as f64 / 1000.0).min(1.0);
    score += length_score * 0.3;

    // Readability score (simplified)
    let words = text.split_whitespace().count();
    let sentences = text.matches(&['.', '!', '?'][..]).count().max(1);
    let avg_sentence_length = words as f64 / sentences as f64;
    let readability_score = (20.0 - avg_sentence_length.abs()).max(0.0) / 20.0;
    score += readability_score * 0.3;

    // Structure score
    let has_title = !title.is_empty();
    let has_paragraphs = text.contains('\n') || text.len() > 500;
    let structure_score = match (has_title, has_paragraphs) {
        (true, true) => 1.0,
        (true, false) | (false, true) => 0.7,
        (false, false) => 0.3,
    };
    score += structure_score * 0.2;

    // Freshness score (assume recent)
    let freshness_score = 0.8;
    score += freshness_score * 0.2;

    score.min(1.0)
}

#[tokio::main]
async fn main() -> Result<()> {
    demonstrate_research_capabilities().await
}
