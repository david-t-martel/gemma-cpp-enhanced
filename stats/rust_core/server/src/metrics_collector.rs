//! Metrics collection and Prometheus export

pub struct MetricsCollector;

impl MetricsCollector {
    pub fn new() -> Self {
        Self
    }

    pub fn export_prometheus(&self) -> String {
        "# HELP requests_total Total requests\n# TYPE requests_total counter\nrequests_total 0\n".to_string()
    }
}
