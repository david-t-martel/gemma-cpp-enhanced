#[cfg(feature = "metrics")]
use prometheus::{
    register_counter, register_gauge, register_histogram, register_histogram_vec, Counter, Gauge,
    Histogram, HistogramVec,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub counters: HashMap<String, u64>,
    pub gauges: HashMap<String, f64>,
    pub histograms: HashMap<String, HistogramData>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramData {
    pub count: u64,
    pub sum: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

pub struct MetricsCollector {
    #[cfg(feature = "metrics")]
    pub documents_processed: Counter,
    #[cfg(feature = "metrics")]
    pub vectors_indexed: Counter,
    #[cfg(feature = "metrics")]
    pub searches_performed: Counter,
    #[cfg(feature = "metrics")]
    pub redis_operations: Counter,
    #[cfg(feature = "metrics")]
    pub errors_total: Counter,

    #[cfg(feature = "metrics")]
    pub active_connections: Gauge,
    #[cfg(feature = "metrics")]
    pub memory_usage_bytes: Gauge,
    #[cfg(feature = "metrics")]
    pub vector_count: Gauge,
    #[cfg(feature = "metrics")]
    pub cache_hit_rate: Gauge,

    #[cfg(feature = "metrics")]
    pub operation_duration: HistogramVec,
    #[cfg(feature = "metrics")]
    pub embedding_latency: Histogram,
    #[cfg(feature = "metrics")]
    pub search_latency: Histogram,
    #[cfg(feature = "metrics")]
    pub document_processing_time: Histogram,

    custom_metrics: Arc<RwLock<HashMap<String, f64>>>,
}

impl MetricsCollector {
    #[cfg(feature = "metrics")]
    pub fn new(_config: &crate::config::MetricsConfig) -> crate::error::Result<Self> {
        let documents_processed = register_counter!(
            "rag_documents_processed_total",
            "Total number of documents processed"
        )?;

        let vectors_indexed = register_counter!(
            "rag_vectors_indexed_total",
            "Total number of vectors indexed"
        )?;

        let searches_performed = register_counter!(
            "rag_searches_performed_total",
            "Total number of searches performed"
        )?;

        let redis_operations = register_counter!(
            "rag_redis_operations_total",
            "Total number of Redis operations"
        )?;

        let errors_total = register_counter!("rag_errors_total", "Total number of errors")?;

        let active_connections =
            register_gauge!("rag_active_connections", "Number of active connections")?;

        let memory_usage_bytes =
            register_gauge!("rag_memory_usage_bytes", "Memory usage in bytes")?;

        let vector_count = register_gauge!("rag_vector_count", "Number of vectors in the index")?;

        let cache_hit_rate = register_gauge!("rag_cache_hit_rate", "Cache hit rate percentage")?;

        let operation_duration = register_histogram_vec!(
            "rag_operation_duration_seconds",
            "Operation duration in seconds",
            &["operation"]
        )?;

        let embedding_latency = register_histogram!(
            "rag_embedding_latency_seconds",
            "Embedding generation latency in seconds"
        )?;

        let search_latency = register_histogram!(
            "rag_search_latency_seconds",
            "Search operation latency in seconds"
        )?;

        let document_processing_time = register_histogram!(
            "rag_document_processing_seconds",
            "Document processing time in seconds"
        )?;

        Ok(Self {
            documents_processed,
            vectors_indexed,
            searches_performed,
            redis_operations,
            errors_total,
            active_connections,
            memory_usage_bytes,
            vector_count,
            cache_hit_rate,
            operation_duration,
            embedding_latency,
            search_latency,
            document_processing_time,
            custom_metrics: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    #[cfg(not(feature = "metrics"))]
    pub fn new(_config: &()) -> crate::error::Result<Self> {
        Ok(Self {
            custom_metrics: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn record_document_processed(&self) {
        #[cfg(feature = "metrics")]
        self.documents_processed.inc();

        let mut custom = self.custom_metrics.write().await;
        *custom
            .entry("documents_processed".to_string())
            .or_insert(0.0) += 1.0;
    }

    pub async fn record_vectors_indexed(&self, count: u64) {
        #[cfg(feature = "metrics")]
        self.vectors_indexed.inc_by(count as f64);

        let mut custom = self.custom_metrics.write().await;
        *custom.entry("vectors_indexed".to_string()).or_insert(0.0) += count as f64;
    }

    pub async fn record_search(&self) {
        #[cfg(feature = "metrics")]
        self.searches_performed.inc();

        let mut custom = self.custom_metrics.write().await;
        *custom
            .entry("searches_performed".to_string())
            .or_insert(0.0) += 1.0;
    }

    pub async fn record_redis_operation(&self) {
        #[cfg(feature = "metrics")]
        self.redis_operations.inc();

        let mut custom = self.custom_metrics.write().await;
        *custom.entry("redis_operations".to_string()).or_insert(0.0) += 1.0;
    }

    pub async fn record_error(&self) {
        #[cfg(feature = "metrics")]
        self.errors_total.inc();

        let mut custom = self.custom_metrics.write().await;
        *custom.entry("errors_total".to_string()).or_insert(0.0) += 1.0;
    }

    pub async fn set_active_connections(&self, count: f64) {
        #[cfg(feature = "metrics")]
        self.active_connections.set(count);

        let mut custom = self.custom_metrics.write().await;
        custom.insert("active_connections".to_string(), count);
    }

    pub async fn set_memory_usage(&self, bytes: f64) {
        #[cfg(feature = "metrics")]
        self.memory_usage_bytes.set(bytes);

        let mut custom = self.custom_metrics.write().await;
        custom.insert("memory_usage_bytes".to_string(), bytes);
    }

    pub async fn set_vector_count(&self, count: f64) {
        #[cfg(feature = "metrics")]
        self.vector_count.set(count);

        let mut custom = self.custom_metrics.write().await;
        custom.insert("vector_count".to_string(), count);
    }

    pub async fn set_cache_hit_rate(&self, rate: f64) {
        #[cfg(feature = "metrics")]
        self.cache_hit_rate.set(rate);

        let mut custom = self.custom_metrics.write().await;
        custom.insert("cache_hit_rate".to_string(), rate);
    }

    pub async fn record_operation_duration(&self, operation: &str, duration: Duration) {
        #[cfg(feature = "metrics")]
        self.operation_duration
            .with_label_values(&[operation])
            .observe(duration.as_secs_f64());

        let mut custom = self.custom_metrics.write().await;
        let key = format!("operation_duration_{}", operation);
        custom.insert(key, duration.as_secs_f64());
    }

    pub async fn record_embedding_latency(&self, duration: Duration) {
        #[cfg(feature = "metrics")]
        self.embedding_latency.observe(duration.as_secs_f64());

        let mut custom = self.custom_metrics.write().await;
        custom.insert("embedding_latency".to_string(), duration.as_secs_f64());
    }

    pub async fn record_search_latency(&self, duration: Duration) {
        #[cfg(feature = "metrics")]
        self.search_latency.observe(duration.as_secs_f64());

        let mut custom = self.custom_metrics.write().await;
        custom.insert("search_latency".to_string(), duration.as_secs_f64());
    }

    pub async fn record_document_processing_time(&self, duration: Duration) {
        #[cfg(feature = "metrics")]
        self.document_processing_time
            .observe(duration.as_secs_f64());

        let mut custom = self.custom_metrics.write().await;
        custom.insert(
            "document_processing_time".to_string(),
            duration.as_secs_f64(),
        );
    }

    pub async fn get_snapshot(&self) -> MetricsSnapshot {
        let custom = self.custom_metrics.read().await;

        let mut counters = HashMap::new();
        let mut gauges = HashMap::new();

        for (key, value) in custom.iter() {
            if key.contains("_total") || key.contains("_processed") || key.contains("_performed") {
                counters.insert(key.clone(), *value as u64);
            } else {
                gauges.insert(key.clone(), *value);
            }
        }

        MetricsSnapshot {
            counters,
            gauges,
            histograms: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    pub async fn reset(&self) {
        let mut custom = self.custom_metrics.write().await;
        custom.clear();
    }

    pub async fn export_prometheus(&self) -> String {
        #[cfg(feature = "metrics")]
        {
            use prometheus::Encoder;
            let encoder = prometheus::TextEncoder::new();
            let metric_families = prometheus::gather();
            let mut buffer = Vec::new();
            encoder.encode(&metric_families, &mut buffer).unwrap();
            String::from_utf8(buffer).unwrap()
        }
        #[cfg(not(feature = "metrics"))]
        {
            "# Metrics feature not enabled\n".to_string()
        }
    }
}

pub struct Timer {
    start: Instant,
    metric_name: String,
    collector: Option<Arc<MetricsCollector>>,
}

impl Timer {
    pub fn new(metric_name: String, collector: Option<Arc<MetricsCollector>>) -> Self {
        Self {
            start: Instant::now(),
            metric_name,
            collector,
        }
    }

    pub async fn stop(self) {
        let duration = self.start.elapsed();
        if let Some(collector) = self.collector {
            collector
                .record_operation_duration(&self.metric_name, duration)
                .await;
        }
    }
}

#[derive(Default)]
pub struct NoOpMetrics;

impl NoOpMetrics {
    pub async fn record_document_processed(&self) {}
    pub async fn record_vectors_indexed(&self, _count: u64) {}
    pub async fn record_search(&self) {}
    pub async fn record_redis_operation(&self) {}
    pub async fn record_error(&self) {}
    pub async fn set_active_connections(&self, _count: f64) {}
    pub async fn set_memory_usage(&self, _bytes: f64) {}
    pub async fn set_vector_count(&self, _count: f64) {}
    pub async fn set_cache_hit_rate(&self, _rate: f64) {}
    pub async fn record_operation_duration(&self, _operation: &str, _duration: Duration) {}
    pub async fn record_embedding_latency(&self, _duration: Duration) {}
    pub async fn record_search_latency(&self, _duration: Duration) {}
    pub async fn record_document_processing_time(&self, _duration: Duration) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_collection() {
        let config = crate::config::MetricsConfig::default();
        let collector = MetricsCollector::new(&config).unwrap();

        collector.record_document_processed().await;
        collector.record_vectors_indexed(10).await;
        collector.record_search().await;
        collector.set_cache_hit_rate(0.85).await;

        let snapshot = collector.get_snapshot().await;
        assert!(snapshot.counters.contains_key("documents_processed"));
        assert!(snapshot.gauges.contains_key("cache_hit_rate"));
    }

    #[tokio::test]
    async fn test_timer() {
        let config = crate::config::MetricsConfig::default();
        let collector = Arc::new(MetricsCollector::new(&config).unwrap());

        let timer = Timer::new("test_operation".to_string(), Some(collector.clone()));
        tokio::time::sleep(Duration::from_millis(10)).await;
        timer.stop().await;

        let snapshot = collector.get_snapshot().await;
        assert!(snapshot
            .gauges
            .contains_key("operation_duration_test_operation"));
    }
}
