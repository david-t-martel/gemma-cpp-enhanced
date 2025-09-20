use crate::{
    config::{DistanceMetric, IndexType, VectorStoreConfig},
    VectorStore,
};
use std::time::Instant;

fn generate_test_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    let mut vectors = Vec::with_capacity(count);
    let mut rng_state = 42u64;

    for _ in 0..count {
        let mut vector = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let random_val = ((rng_state >> 16) & 0xFFFF) as f32 / 65536.0 - 0.5;
            vector.push(random_val);
        }

        // Normalize vector
        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            vector.iter_mut().for_each(|x| *x /= magnitude);
        }

        vectors.push(vector);
    }

    vectors
}

#[derive(Default)]
pub struct PerformanceReport {
    pub vector_addition_time: std::time::Duration,
    pub search_times: Vec<(usize, std::time::Duration)>, // (limit, avg_time)
    pub memory_usage: usize,
    pub vectors_per_sec: f64,
    pub queries_per_sec: Vec<(usize, f64)>, // (limit, qps)
}

pub fn run_baseline_performance_test(
    dataset_size: usize,
    dimension: usize,
    metric: DistanceMetric,
) -> crate::Result<PerformanceReport> {
    let mut report = PerformanceReport::default();

    // Create vector store
    let config = VectorStoreConfig {
        dimension,
        distance_metric: metric,
        index_type: IndexType::Flat,
        hnsw_m: 16,
        hnsw_ef_construction: 50,
        hnsw_ef_search: 20,
        max_vectors: Some(dataset_size + 100),
    };

    let store = VectorStore::new(config)?;
    let vectors = generate_test_vectors(dataset_size + 1, dimension);

    // Benchmark vector addition
    let start = Instant::now();
    for (i, vector) in vectors[..dataset_size].iter().enumerate() {
        let id = format!("vec_{}", i);
        let metadata = serde_json::json!({
            "doc_id": format!("doc_{}", i / 10),
            "chunk_id": i,
            "category": match i % 4 {
                0 => "tech",
                1 => "science",
                2 => "business",
                _ => "general"
            }
        });
        store.add_vector(&id, vector, metadata)?;
    }
    report.vector_addition_time = start.elapsed();
    report.vectors_per_sec = dataset_size as f64 / report.vector_addition_time.as_secs_f64();

    // Benchmark search with different limits
    let query_vector = &vectors[dataset_size];
    let search_limits = [1, 5, 10, 20];

    for &limit in &search_limits {
        let mut total_time = std::time::Duration::new(0, 0);
        let num_queries = 20;

        for _ in 0..num_queries {
            let start = Instant::now();
            let _results = store.search(query_vector, limit, None)?;
            total_time += start.elapsed();
        }

        let avg_time = total_time / num_queries;
        let qps = 1.0 / avg_time.as_secs_f64();

        report.search_times.push((limit, avg_time));
        report.queries_per_sec.push((limit, qps));
    }

    // Get memory usage
    let stats = store.get_stats();
    report.memory_usage = stats.memory_usage;

    Ok(report)
}

pub fn print_performance_report(
    report: &PerformanceReport,
    dataset_size: usize,
    dimension: usize,
    metric: DistanceMetric,
) {
    println!(
        "=== Performance Report: {} vectors ({}D) - {:?} ===",
        dataset_size, dimension, metric
    );
    println!(
        "Vector Addition: {:?} ({:.2} vectors/sec)",
        report.vector_addition_time, report.vectors_per_sec
    );

    for &(limit, ref avg_time) in &report.search_times {
        let qps = report
            .queries_per_sec
            .iter()
            .find(|(l, _)| *l == limit)
            .map(|(_, qps)| *qps)
            .unwrap_or(0.0);

        println!(
            "Search (limit={}): {:?} ({:.2} queries/sec)",
            limit, avg_time, qps
        );
    }

    println!(
        "Memory usage: {} bytes ({:.2} MB)",
        report.memory_usage,
        report.memory_usage as f64 / (1024.0 * 1024.0)
    );
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_performance_small() {
        let report = run_baseline_performance_test(100, 128, DistanceMetric::Cosine).unwrap();
        print_performance_report(&report, 100, 128, DistanceMetric::Cosine);

        // Basic sanity checks
        assert!(report.vectors_per_sec > 0.0);
        assert!(!report.search_times.is_empty());
        assert!(report.memory_usage > 0);
    }

    #[test]
    fn test_baseline_performance_medium() {
        let report = run_baseline_performance_test(500, 384, DistanceMetric::Cosine).unwrap();
        print_performance_report(&report, 500, 384, DistanceMetric::Cosine);

        assert!(report.vectors_per_sec > 0.0);
        assert!(!report.search_times.is_empty());
    }

    #[test]
    fn test_distance_metric_comparison() {
        let metrics = [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
        ];

        for metric in &metrics {
            let report = run_baseline_performance_test(200, 256, *metric).unwrap();
            print_performance_report(&report, 200, 256, *metric);
        }
    }
}
