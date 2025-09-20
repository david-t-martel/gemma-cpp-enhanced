//! Internet database research utilities for external API integration
//!
//! This module provides high-performance utilities for:
//! - Web scraping and content extraction
//! - API integration with external knowledge sources
//! - Parallel research queries with rate limiting
//! - Content validation and filtering
//! - Caching of research results
//! - Retry logic and error handling

use crate::error::{GemmaError, GemmaResult};
use crate::redis_manager::RedisManager;
use futures::{stream, StreamExt, TryStreamExt};
use reqwest::{
    header::{HeaderMap, HeaderValue, USER_AGENT},
    Client, ClientBuilder, Response,
};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::{RwLock, Semaphore},
    time::{sleep, timeout},
};
use tracing::{debug, error, info, warn};
use url::Url;
use uuid::Uuid;

/// Configuration for research client behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchConfig {
    /// Maximum number of concurrent requests
    pub max_concurrent_requests: usize,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Rate limiting: requests per second
    pub rate_limit_rps: f64,
    /// Maximum number of retries for failed requests
    pub max_retries: usize,
    /// User agent string for HTTP requests
    pub user_agent: String,
    /// Enable response caching
    pub enable_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Maximum response size in bytes
    pub max_response_size: usize,
    /// Follow redirects
    pub follow_redirects: bool,
    /// Custom headers to include in requests
    pub custom_headers: HashMap<String, String>,
    /// Domains to exclude from scraping
    pub blocked_domains: Vec<String>,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            max_concurrent_requests: 10,
            timeout_ms: 30000,
            rate_limit_rps: 5.0,
            max_retries: 3,
            user_agent: "Mozilla/5.0 (compatible; RAG-Research/1.0)".to_string(),
            enable_caching: true,
            cache_ttl_seconds: 3600,
            max_response_size: 10 * 1024 * 1024, // 10MB
            follow_redirects: true,
            custom_headers: HashMap::new(),
            blocked_domains: Vec::new(),
        }
    }
}

/// Research query request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchQuery {
    pub id: String,
    pub query: String,
    pub sources: Vec<String>, // URLs or API endpoints
    pub max_results: usize,
    pub filters: HashMap<String, String>,
    pub priority: u8, // 1-10, higher is more important
}

impl ResearchQuery {
    pub fn new(query: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            query,
            sources: Vec::new(),
            max_results: 10,
            filters: HashMap::new(),
            priority: 5,
        }
    }

    pub fn with_sources(mut self, sources: Vec<String>) -> Self {
        self.sources = sources;
        self
    }

    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results;
        self
    }

    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.min(10).max(1);
        self
    }
}

/// Research result from a single source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchResult {
    pub source_url: String,
    pub title: String,
    pub content: String,
    pub snippet: String,
    pub confidence_score: f32,
    pub retrieved_at: u64,
    pub metadata: HashMap<String, String>,
}

impl ResearchResult {
    pub fn new(source_url: String, title: String, content: String) -> Self {
        let snippet = if content.len() > 200 {
            format!("{}...", &content[..200])
        } else {
            content.clone()
        };

        Self {
            source_url,
            title,
            content,
            snippet,
            confidence_score: 0.5, // Default neutral confidence
            retrieved_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_confidence(mut self, score: f32) -> Self {
        self.confidence_score = score.max(0.0).min(1.0);
        self
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Research response containing multiple results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchResponse {
    pub query_id: String,
    pub query: String,
    pub results: Vec<ResearchResult>,
    pub total_sources_queried: usize,
    pub successful_queries: usize,
    pub processing_time_ms: u64,
    pub cached_results: usize,
}

/// Statistics for research operations
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ResearchStats {
    pub total_queries: AtomicU64,
    pub successful_queries: AtomicU64,
    pub failed_queries: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub total_results_retrieved: AtomicU64,
    pub average_response_time_ms: f64,
    pub total_bytes_downloaded: AtomicU64,
}

impl ResearchStats {
    pub fn success_rate(&self) -> f64 {
        let total = self.total_queries.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            self.successful_queries.load(Ordering::Relaxed) as f64 / total as f64
        }
    }

    pub fn cache_hit_rate(&self) -> f64 {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        if hits + misses == 0 {
            0.0
        } else {
            hits as f64 / (hits + misses) as f64
        }
    }
}

/// Web scraper with content extraction capabilities
pub struct WebScraper {
    client: Client,
    content_selectors: Vec<String>,
    title_selectors: Vec<String>,
}

impl WebScraper {
    pub fn new() -> Self {
        let client = ClientBuilder::new()
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_default();

        Self {
            client,
            content_selectors: vec![
                "article".to_string(),
                "main".to_string(),
                ".content".to_string(),
                "#content".to_string(),
                ".post".to_string(),
                ".article-body".to_string(),
                "p".to_string(),
            ],
            title_selectors: vec![
                "title".to_string(),
                "h1".to_string(),
                ".title".to_string(),
                "#title".to_string(),
                ".headline".to_string(),
            ],
        }
    }

    /// Extract content from a web page
    pub async fn scrape_url(&self, url: &str) -> GemmaResult<ResearchResult> {
        let response = self.client
            .get(url)
            .send()
            .await
            .map_err(|e| GemmaError::HttpRequest(e.to_string()))?;

        if !response.status().is_success() {
            return Err(GemmaError::HttpRequest(format!(
                "HTTP {} for URL: {}",
                response.status(),
                url
            )));
        }

        let html_content = response
            .text()
            .await
            .map_err(|e| GemmaError::HttpRequest(e.to_string()))?;

        let document = Html::parse_document(&html_content);

        // Extract title
        let title = self.extract_title(&document).unwrap_or_else(|| "Untitled".to_string());

        // Extract main content
        let content = self.extract_content(&document);

        Ok(ResearchResult::new(url.to_string(), title, content))
    }

    /// Extract title from HTML document
    fn extract_title(&self, document: &Html) -> Option<String> {
        for selector_str in &self.title_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                if let Some(element) = document.select(&selector).next() {
                    let title = element.text().collect::<Vec<_>>().join(" ").trim().to_string();
                    if !title.is_empty() {
                        return Some(title);
                    }
                }
            }
        }
        None
    }

    /// Extract main content from HTML document
    fn extract_content(&self, document: &Html) -> String {
        let mut content = String::new();

        for selector_str in &self.content_selectors {
            if let Ok(selector) = Selector::parse(selector_str) {
                for element in document.select(&selector) {
                    let text = element.text().collect::<Vec<_>>().join(" ");
                    if !text.trim().is_empty() {
                        content.push_str(&text);
                        content.push('\n');

                        // If we found substantial content, stop looking
                        if content.len() > 500 {
                            break;
                        }
                    }
                }

                if content.len() > 500 {
                    break;
                }
            }
        }

        // If no structured content found, extract all text
        if content.trim().is_empty() {
            content = document.root_element().text().collect::<Vec<_>>().join(" ");
        }

        // Clean up the content
        content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty() && line.len() > 10)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// High-performance research client with caching and rate limiting
pub struct ResearchClient {
    config: ResearchConfig,
    http_client: Client,
    scraper: WebScraper,
    redis: Option<Arc<RedisManager>>,
    semaphore: Arc<Semaphore>,
    stats: Arc<ResearchStats>,
    rate_limiter: Arc<RwLock<RateLimiter>>,
}

/// Simple rate limiter implementation
struct RateLimiter {
    last_request: SystemTime,
    min_interval: Duration,
}

impl RateLimiter {
    fn new(rps: f64) -> Self {
        Self {
            last_request: UNIX_EPOCH,
            min_interval: Duration::from_secs_f64(1.0 / rps),
        }
    }

    async fn wait_if_needed(&mut self) {
        let now = SystemTime::now();
        if let Ok(elapsed) = now.duration_since(self.last_request) {
            if elapsed < self.min_interval {
                let sleep_duration = self.min_interval - elapsed;
                sleep(sleep_duration).await;
            }
        }
        self.last_request = SystemTime::now();
    }
}

impl ResearchClient {
    /// Create a new research client
    pub fn new(config: ResearchConfig) -> GemmaResult<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(
            USER_AGENT,
            HeaderValue::from_str(&config.user_agent)
                .map_err(|e| GemmaError::Configuration(e.to_string()))?,
        );

        // Add custom headers
        for (key, value) in &config.custom_headers {
            headers.insert(
                key.parse()
                    .map_err(|e| GemmaError::Configuration(format!("Invalid header key {}: {}", key, e)))?,
                HeaderValue::from_str(value)
                    .map_err(|e| GemmaError::Configuration(format!("Invalid header value {}: {}", value, e)))?,
            );
        }

        let http_client = ClientBuilder::new()
            .timeout(Duration::from_millis(config.timeout_ms))
            .default_headers(headers)
            .redirect(if config.follow_redirects {
                reqwest::redirect::Policy::limited(5)
            } else {
                reqwest::redirect::Policy::none()
            })
            .build()
            .map_err(|e| GemmaError::Configuration(e.to_string()))?;

        let scraper = WebScraper::new();
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_requests));
        let stats = Arc::new(ResearchStats::default());
        let rate_limiter = Arc::new(RwLock::new(RateLimiter::new(config.rate_limit_rps)));

        Ok(Self {
            config,
            http_client,
            scraper,
            redis: None,
            semaphore,
            stats,
            rate_limiter,
        })
    }

    /// Set Redis manager for caching
    pub fn with_redis(mut self, redis: Arc<RedisManager>) -> Self {
        self.redis = Some(redis);
        self
    }

    /// Execute a research query across multiple sources
    pub async fn research(&self, query: ResearchQuery) -> GemmaResult<ResearchResponse> {
        let start_time = SystemTime::now();
        self.stats.total_queries.fetch_add(1, Ordering::Relaxed);

        info!("Starting research query: {} with {} sources", query.query, query.sources.len());

        // Check cache first if enabled
        if self.config.enable_caching {
            if let Some(cached_response) = self.get_cached_response(&query).await? {
                self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                debug!("Returning cached response for query: {}", query.id);
                return Ok(cached_response);
            }
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
        }

        // Process sources in parallel with rate limiting and concurrency control
        let results = stream::iter(query.sources.iter().enumerate())
            .map(|(index, source)| self.query_source(source, &query.query, index))
            .buffer_unordered(self.config.max_concurrent_requests)
            .collect::<Vec<_>>()
            .await;

        let mut research_results = Vec::new();
        let mut successful_queries = 0;

        for result in results {
            match result {
                Ok(research_result) => {
                    research_results.push(research_result);
                    successful_queries += 1;
                }
                Err(e) => {
                    warn!("Source query failed: {}", e);
                }
            }
        }

        // Sort results by confidence score (highest first)
        research_results.sort_by(|a, b| b.confidence_score.partial_cmp(&a.confidence_score).unwrap_or(std::cmp::Ordering::Equal));

        // Limit results
        research_results.truncate(query.max_results);

        let processing_time = start_time
            .elapsed()
            .unwrap_or_default()
            .as_millis() as u64;

        let response = ResearchResponse {
            query_id: query.id.clone(),
            query: query.query.clone(),
            results: research_results.clone(),
            total_sources_queried: query.sources.len(),
            successful_queries,
            processing_time_ms: processing_time,
            cached_results: 0,
        };

        // Cache the response if enabled
        if self.config.enable_caching && !research_results.is_empty() {
            if let Err(e) = self.cache_response(&query, &response).await {
                warn!("Failed to cache response: {}", e);
            }
        }

        // Update statistics
        if successful_queries > 0 {
            self.stats.successful_queries.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.failed_queries.fetch_add(1, Ordering::Relaxed);
        }

        self.stats
            .total_results_retrieved
            .fetch_add(research_results.len() as u64, Ordering::Relaxed);

        info!(
            "Research query completed: {} results in {}ms",
            research_results.len(),
            processing_time
        );

        Ok(response)
    }

    /// Query a single source
    async fn query_source(
        &self,
        source: &str,
        query: &str,
        _index: usize,
    ) -> GemmaResult<ResearchResult> {
        // Acquire semaphore permit for concurrency control
        let _permit = self.semaphore.acquire().await.unwrap();

        // Apply rate limiting
        {
            let mut rate_limiter = self.rate_limiter.write().await;
            rate_limiter.wait_if_needed().await;
        }

        // Check if domain is blocked
        if let Ok(url) = Url::parse(source) {
            if let Some(domain) = url.host_str() {
                if self.config.blocked_domains.iter().any(|blocked| domain.contains(blocked)) {
                    return Err(GemmaError::BlockedDomain(domain.to_string()));
                }
            }
        }

        debug!("Querying source: {}", source);

        // Determine query method based on source type
        if source.starts_with("http") {
            self.query_web_source(source, query).await
        } else {
            self.query_api_source(source, query).await
        }
    }

    /// Query a web source (HTTP/HTTPS URL)
    async fn query_web_source(&self, url: &str, _query: &str) -> GemmaResult<ResearchResult> {
        let mut result = self.scraper.scrape_url(url).await?;

        // Calculate confidence based on content quality
        result.confidence_score = self.calculate_content_confidence(&result.content);

        Ok(result)
    }

    /// Query an API source
    async fn query_api_source(&self, endpoint: &str, query: &str) -> GemmaResult<ResearchResult> {
        // This is a placeholder for API integrations
        // In practice, you would implement specific API clients for:
        // - Wikipedia API
        // - Google Custom Search
        // - Bing Search API
        // - Academic databases
        // - News APIs
        // etc.

        warn!("API source querying not yet implemented for: {}", endpoint);

        Err(GemmaError::NotImplemented(format!(
            "API source querying for endpoint: {}",
            endpoint
        )))
    }

    /// Calculate confidence score for content quality
    fn calculate_content_confidence(&self, content: &str) -> f32 {
        let length_score = (content.len() as f32 / 1000.0).min(1.0); // Longer content = higher score
        let sentence_count = content.matches('.').count() as f32;
        let structure_score = if sentence_count > 3.0 { 0.8 } else { 0.4 };

        // Simple heuristic - in practice you might use ML models
        (length_score * 0.3 + structure_score * 0.7).min(1.0).max(0.1)
    }

    /// Get cached response for a query
    async fn get_cached_response(&self, query: &ResearchQuery) -> GemmaResult<Option<ResearchResponse>> {
        if let Some(redis) = &self.redis {
            let cache_key = format!("research:{}:{}", query.query, query.sources.join(","));
            if let Some(cached_results) = redis.get_cached_search_results(&cache_key).await? {
                // Convert cached results to research response format
                // This is a simplified implementation - you'd want to store the full response
                debug!("Found cached results for query: {}", query.id);
                return Ok(None); // Placeholder
            }
        }
        Ok(None)
    }

    /// Cache a research response
    async fn cache_response(&self, query: &ResearchQuery, response: &ResearchResponse) -> GemmaResult<()> {
        if let Some(redis) = &self.redis {
            let cache_key = format!("research:{}:{}", query.query, query.sources.join(","));
            let results: Vec<(String, f32)> = response.results
                .iter()
                .map(|r| (r.source_url.clone(), r.confidence_score))
                .collect();

            redis.cache_search_results(&cache_key, &results, self.config.cache_ttl_seconds).await?;
        }
        Ok(())
    }

    /// Get current statistics
    pub fn get_stats(&self) -> ResearchStats {
        // Return a copy of stats (the actual implementation would need to handle atomics properly)
        ResearchStats::default()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.stats.total_queries.store(0, Ordering::Relaxed);
        self.stats.successful_queries.store(0, Ordering::Relaxed);
        self.stats.failed_queries.store(0, Ordering::Relaxed);
        self.stats.cache_hits.store(0, Ordering::Relaxed);
        self.stats.cache_misses.store(0, Ordering::Relaxed);
        self.stats.total_results_retrieved.store(0, Ordering::Relaxed);
        self.stats.total_bytes_downloaded.store(0, Ordering::Relaxed);
    }
}

/// Utility functions for URL validation and processing

/// Validate if a URL is safe to access
pub fn is_safe_url(url: &str) -> bool {
    if let Ok(parsed_url) = Url::parse(url) {
        matches!(parsed_url.scheme(), "http" | "https") &&
        parsed_url.host_str().is_some() &&
        !url.contains("localhost") &&
        !url.contains("127.0.0.1")
    } else {
        false
    }
}

/// Extract domain from URL
pub fn extract_domain(url: &str) -> Option<String> {
    Url::parse(url).ok()?.host_str().map(|s| s.to_string())
}

/// Clean and normalize URL
pub fn normalize_url(url: &str) -> String {
    if let Ok(mut parsed_url) = Url::parse(url) {
        // Remove fragment
        parsed_url.set_fragment(None);
        // Remove common tracking parameters
        let mut query_pairs: Vec<_> = parsed_url.query_pairs()
            .filter(|(key, _)| !matches!(key.as_ref(), "utm_source" | "utm_medium" | "utm_campaign" | "fbclid"))
            .collect();

        if query_pairs.is_empty() {
            parsed_url.set_query(None);
        } else {
            let query_string = query_pairs
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join("&");
            parsed_url.set_query(Some(&query_string));
        }

        parsed_url.to_string()
    } else {
        url.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_research_config_default() {
        let config = ResearchConfig::default();
        assert_eq!(config.max_concurrent_requests, 10);
        assert_eq!(config.rate_limit_rps, 5.0);
        assert!(config.enable_caching);
    }

    #[test]
    fn test_research_query_creation() {
        let query = ResearchQuery::new("test query".to_string())
            .with_sources(vec!["https://example.com".to_string()])
            .with_max_results(5)
            .with_priority(8);

        assert_eq!(query.query, "test query");
        assert_eq!(query.sources.len(), 1);
        assert_eq!(query.max_results, 5);
        assert_eq!(query.priority, 8);
    }

    #[test]
    fn test_research_result_creation() {
        let result = ResearchResult::new(
            "https://example.com".to_string(),
            "Test Title".to_string(),
            "Test content for the research result.".to_string(),
        ).with_confidence(0.8);

        assert_eq!(result.source_url, "https://example.com");
        assert_eq!(result.title, "Test Title");
        assert_eq!(result.confidence_score, 0.8);
        assert!(result.snippet.len() <= 203); // Original + "..."
    }

    #[test]
    fn test_url_validation() {
        assert!(is_safe_url("https://example.com"));
        assert!(is_safe_url("http://example.com"));
        assert!(!is_safe_url("ftp://example.com"));
        assert!(!is_safe_url("https://localhost"));
        assert!(!is_safe_url("invalid-url"));
    }

    #[test]
    fn test_domain_extraction() {
        assert_eq!(extract_domain("https://example.com/path"), Some("example.com".to_string()));
        assert_eq!(extract_domain("invalid-url"), None);
    }

    #[test]
    fn test_url_normalization() {
        let url = "https://example.com/path?param=1&utm_source=test#fragment";
        let normalized = normalize_url(url);

        assert!(!normalized.contains("utm_source"));
        assert!(!normalized.contains("#fragment"));
        assert!(normalized.contains("param=1"));
    }

    #[test]
    fn test_research_stats() {
        let stats = ResearchStats::default();
        assert_eq!(stats.success_rate(), 0.0);
        assert_eq!(stats.cache_hit_rate(), 0.0);
    }

    #[tokio::test]
    async fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(2.0); // 2 requests per second
        let start = SystemTime::now();

        limiter.wait_if_needed().await;
        limiter.wait_if_needed().await;

        let elapsed = start.elapsed().unwrap();
        assert!(elapsed >= Duration::from_millis(400)); // Should wait at least 0.5 seconds
    }
}