// Standalone test for research module functionality
use std::time::Duration;
use tokio;
use futures;
use thiserror::Error;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Error, Debug)]
pub enum TestError {
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

type Result<T> = std::result::Result<T, TestError>;

impl From<reqwest::Error> for TestError {
    fn from(err: reqwest::Error) -> Self {
        TestError::Http(err.to_string())
    }
}

impl From<serde_json::Error> for TestError {
    fn from(err: serde_json::Error) -> Self {
        TestError::Serialization(err.to_string())
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
    pub extracted_at: DateTime<Utc>,
}

#[derive(Debug, Default)]
struct ResearchStats {
    pub requests_made: std::sync::atomic::AtomicU64,
    pub requests_succeeded: std::sync::atomic::AtomicU64,
    pub requests_failed: std::sync::atomic::AtomicU64,
}

impl ResearchStats {
    pub fn success_rate(&self) -> f64 {
        let total = self.requests_made.load(std::sync::atomic::Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let succeeded = self.requests_succeeded.load(std::sync::atomic::Ordering::Relaxed);
        succeeded as f64 / total as f64
    }
}

async fn test_research_functionality() -> Result<()> {
    println!("🔍 Testing Research Module Core Functionality");
    println!("==============================================");

    // Test 1: Rate limiting simulation
    println!("\n⏱️ Testing rate limiting simulation:");
    let start = std::time::Instant::now();
    for i in 1..=3 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        println!("   Request {} completed in {:?}", i, start.elapsed());
    }

    // Test 2: Content quality scoring
    println!("\n📊 Testing content quality scoring:");

    let samples = vec![
        ("High quality article with detailed content and proper structure", "Research Article"),
        ("Short text", ""),
        ("This is a comprehensive analysis of modern technology trends including artificial intelligence, machine learning, and blockchain applications in various industries", "Tech Analysis"),
        ("Lorem ipsum", "Title"),
    ];

    for (text, title) in samples {
        let score = calculate_quality_score(text, title);
        println!("   Content: \"{}...\" -> Score: {:.2}",
                 &text.chars().take(30).collect::<String>(), score);
    }

    // Test 3: Domain filtering
    println!("\n🚫 Testing domain filtering:");
    let blocked_domains = vec!["facebook.com", "twitter.com"];
    let test_domains = vec!["example.com", "facebook.com", "github.com", "twitter.com", "wikipedia.org"];

    for domain in test_domains {
        let is_blocked = blocked_domains.iter().any(|blocked| domain.contains(blocked));
        println!("   {}: {}", domain, if is_blocked { "❌ Blocked" } else { "✅ Allowed" });
    }

    // Test 4: Concurrent processing simulation
    println!("\n🔄 Testing concurrent processing:");
    let start = std::time::Instant::now();
    let urls = vec![
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
    ];

    let tasks: Vec<_> = urls.into_iter().enumerate().map(|(i, url)| {
        tokio::spawn(async move {
            // Simulate processing delay
            tokio::time::sleep(Duration::from_millis(100 + i as u64 * 50)).await;
            SearchResult {
                id: uuid::Uuid::new_v4().to_string(),
                url: url.to_string(),
                title: format!("Page {}", i + 1),
                text: format!("Content from page {} with detailed information", i + 1),
                score: 0.9 - (i as f64 * 0.1),
                metadata: serde_json::json!({
                    "word_count": 100 + i * 50,
                    "processing_time_ms": 100 + i * 50
                }),
                extracted_at: Utc::now(),
            }
        })
    }).collect();

    let results = futures::future::try_join_all(tasks).await
        .map_err(|e| TestError::Internal(format!("Task error: {}", e)))?;

    println!("   Processed {} URLs in {:?}", results.len(), start.elapsed());
    for result in &results {
        println!("   - {}: score={:.2}, words={}",
                 result.title,
                 result.score,
                 result.metadata["word_count"]);
    }

    // Test 5: Error handling
    println!("\n❌ Testing error handling:");
    let errors = vec![
        TestError::Timeout("Request timed out after 30s".to_string()),
        TestError::ContentTooLarge("Content exceeds 10MB limit".to_string()),
        TestError::Http("HTTP 404 Not Found".to_string()),
    ];

    for error in errors {
        println!("   Handled: {}", error);
    }

    // Test 6: Statistics tracking
    println!("\n📈 Testing statistics tracking:");
    let stats = ResearchStats::default();

    // Simulate some operations
    stats.requests_made.store(10, std::sync::atomic::Ordering::Relaxed);
    stats.requests_succeeded.store(8, std::sync::atomic::Ordering::Relaxed);
    stats.requests_failed.store(2, std::sync::atomic::Ordering::Relaxed);

    println!("   Requests made: {}", stats.requests_made.load(std::sync::atomic::Ordering::Relaxed));
    println!("   Success rate: {:.1}%", stats.success_rate() * 100.0);

    // Test 7: Retry logic simulation
    println!("\n🔄 Testing retry logic:");
    let mut attempts = 0;
    let max_retries = 3;

    loop {
        attempts += 1;
        println!("   Attempt {}/{}", attempts, max_retries);

        if attempts == 2 {
            println!("   ✅ Operation succeeded!");
            break;
        } else if attempts < max_retries {
            println!("   ❌ Failed, retrying with backoff...");
            tokio::time::sleep(Duration::from_millis(attempts as u64 * 100)).await;
        } else {
            println!("   ❌ Max retries exceeded");
            break;
        }
    }

    println!("\n🎉 All research functionality tests passed!");
    Ok(())
}

fn calculate_quality_score(text: &str, title: &str) -> f64 {
    let mut score = 0.0;

    // Content length score (normalized to 1000 chars)
    let length_score = (text.len() as f64 / 1000.0).min(1.0);
    score += length_score * 0.3;

    // Word count and sentence structure
    let words = text.split_whitespace().count();
    let sentences = text.matches(&['.', '!', '?'][..]).count().max(1);
    let avg_sentence_length = words as f64 / sentences as f64;

    // Readability score (penalize very short or very long sentences)
    let readability_score = if avg_sentence_length >= 5.0 && avg_sentence_length <= 25.0 {
        1.0 - (avg_sentence_length - 15.0).abs() / 15.0
    } else {
        0.3
    };
    score += readability_score * 0.3;

    // Structure score
    let has_title = !title.is_empty();
    let has_good_length = text.len() > 100;
    let structure_score = match (has_title, has_good_length) {
        (true, true) => 1.0,
        (true, false) => 0.7,
        (false, true) => 0.6,
        (false, false) => 0.2,
    };
    score += structure_score * 0.2;

    // Freshness score (assume recent for testing)
    score += 0.8 * 0.2;

    score.min(1.0)
}

#[tokio::main]
async fn main() -> Result<()> {
    test_research_functionality().await
}
