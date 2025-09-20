//! Memory usage dashboard and reporting system
//!
//! This module provides real-time memory monitoring, visualization, and reporting
//! for the RAG-Redis system.

use crate::memory_profiler::{MemoryProfiler, MemoryReport, MemorySnapshot};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub update_interval: Duration,
    pub history_size: usize,
    pub alert_thresholds: AlertThresholds,
    pub export_format: ExportFormat,
    pub enable_alerts: bool,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_secs(5),
            history_size: 1000,
            alert_thresholds: AlertThresholds::default(),
            export_format: ExportFormat::Json,
            enable_alerts: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub memory_usage_percent: f64,
    pub allocation_rate: usize, // allocations per second
    pub fragmentation_ratio: f64,
    pub leak_growth_rate: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            memory_usage_percent: 80.0,
            allocation_rate: 10000,
            fragmentation_ratio: 2.0,
            leak_growth_rate: 0.1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Csv,
    Html,
    Markdown,
}

/// Real-time memory metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub timestamp: DateTime<Utc>,
    pub current_usage_bytes: usize,
    pub peak_usage_bytes: usize,
    pub allocation_rate: f64,
    pub deallocation_rate: f64,
    pub fragmentation_ratio: f64,
    pub cache_hit_rate: f64,
    pub component_metrics: Vec<ComponentMetric>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetric {
    pub name: String,
    pub memory_bytes: usize,
    pub object_count: usize,
    pub growth_rate: f64,
}

/// Memory alert event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAlert {
    pub timestamp: DateTime<Utc>,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub current_value: f64,
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighMemoryUsage,
    MemoryLeak,
    HighFragmentation,
    AllocationSpike,
    ComponentGrowth(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Memory usage dashboard
pub struct MemoryDashboard {
    profiler: Arc<MemoryProfiler>,
    config: DashboardConfig,
    metrics_history: Arc<RwLock<Vec<MemoryMetrics>>>,
    alerts: Arc<RwLock<Vec<MemoryAlert>>>,
    alert_tx: mpsc::UnboundedSender<MemoryAlert>,
    alert_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<MemoryAlert>>>>,
    last_snapshot: Arc<RwLock<Option<MemorySnapshot>>>,
}

impl MemoryDashboard {
    pub fn new(profiler: Arc<MemoryProfiler>, config: DashboardConfig) -> Arc<Self> {
        let (alert_tx, alert_rx) = mpsc::unbounded_channel();

        Arc::new(Self {
            profiler,
            config,
            metrics_history: Arc::new(RwLock::new(Vec::new())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            alert_tx,
            alert_rx: Arc::new(RwLock::new(Some(alert_rx))),
            last_snapshot: Arc::new(RwLock::new(None)),
        })
    }

    /// Start the dashboard monitoring
    pub fn start(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        let dashboard = Arc::clone(&self);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(dashboard.config.update_interval);

            loop {
                interval.tick().await;
                dashboard.update_metrics().await;
                dashboard.check_alerts().await;
            }
        })
    }

    /// Take the alert receiver for external alert handling
    pub fn take_alert_receiver(&self) -> Option<mpsc::UnboundedReceiver<MemoryAlert>> {
        self.alert_rx.write().take()
    }

    /// Update metrics from profiler
    async fn update_metrics(&self) {
        let snapshot = MemorySnapshot::capture();
        let report = self.profiler.generate_report();

        // Calculate rates
        let (alloc_rate, dealloc_rate) = if let Some(last) = self.last_snapshot.read().as_ref() {
            let time_delta = snapshot
                .timestamp
                .signed_duration_since(last.timestamp)
                .num_seconds() as f64;

            if time_delta > 0.0 {
                let alloc_delta = snapshot
                    .allocation_count
                    .saturating_sub(last.allocation_count);
                let dealloc_delta = snapshot
                    .deallocation_count
                    .saturating_sub(last.deallocation_count);

                (
                    alloc_delta as f64 / time_delta,
                    dealloc_delta as f64 / time_delta,
                )
            } else {
                (0.0, 0.0)
            }
        } else {
            (0.0, 0.0)
        };

        // Build component metrics
        let component_metrics: Vec<ComponentMetric> = report
            .component_breakdown
            .into_iter()
            .map(|(name, usage)| ComponentMetric {
                name,
                memory_bytes: usage.allocated_bytes,
                object_count: usage.object_count,
                growth_rate: 0.0, // TODO: Calculate from history
            })
            .collect();

        let metrics = MemoryMetrics {
            timestamp: Utc::now(),
            current_usage_bytes: snapshot.current_usage,
            peak_usage_bytes: snapshot.peak_usage,
            allocation_rate: alloc_rate,
            deallocation_rate: dealloc_rate,
            fragmentation_ratio: snapshot.fragmentation_ratio,
            cache_hit_rate: 0.0, // TODO: Get from cache stats
            component_metrics,
        };

        // Update history
        {
            let mut history = self.metrics_history.write();
            history.push(metrics);

            // Trim history to configured size
            if history.len() > self.config.history_size {
                let excess = history.len() - self.config.history_size;
                history.drain(0..excess);
            }
        }

        // Update last snapshot
        *self.last_snapshot.write() = Some(snapshot);
    }

    /// Check for alert conditions
    async fn check_alerts(&self) {
        if !self.config.enable_alerts {
            return;
        }

        let history = self.metrics_history.read();
        if let Some(latest) = history.last() {
            // Check memory usage
            if let Ok(mem_info) = sys_info::mem_info() {
                let total_memory = mem_info.total as f64 * 1024.0;
                let usage_percent = (latest.current_usage_bytes as f64 / total_memory) * 100.0;

                if usage_percent > self.config.alert_thresholds.memory_usage_percent {
                    self.send_alert(MemoryAlert {
                        timestamp: Utc::now(),
                        alert_type: AlertType::HighMemoryUsage,
                        severity: if usage_percent > 90.0 {
                            AlertSeverity::Critical
                        } else {
                            AlertSeverity::Warning
                        },
                        message: format!("Memory usage at {:.1}% of system memory", usage_percent),
                        current_value: usage_percent,
                        threshold: self.config.alert_thresholds.memory_usage_percent,
                    });
                }
            }

            // Check allocation rate
            if latest.allocation_rate > self.config.alert_thresholds.allocation_rate as f64 {
                self.send_alert(MemoryAlert {
                    timestamp: Utc::now(),
                    alert_type: AlertType::AllocationSpike,
                    severity: AlertSeverity::Warning,
                    message: format!(
                        "High allocation rate: {:.0} allocs/sec",
                        latest.allocation_rate
                    ),
                    current_value: latest.allocation_rate,
                    threshold: self.config.alert_thresholds.allocation_rate as f64,
                });
            }

            // Check fragmentation
            if latest.fragmentation_ratio > self.config.alert_thresholds.fragmentation_ratio {
                self.send_alert(MemoryAlert {
                    timestamp: Utc::now(),
                    alert_type: AlertType::HighFragmentation,
                    severity: AlertSeverity::Warning,
                    message: format!(
                        "High memory fragmentation: {:.2}",
                        latest.fragmentation_ratio
                    ),
                    current_value: latest.fragmentation_ratio,
                    threshold: self.config.alert_thresholds.fragmentation_ratio,
                });
            }

            // Check for memory leaks
            if history.len() >= 10 {
                let recent = &history[history.len() - 10..];
                let growth_rate = Self::calculate_growth_rate(recent);

                if growth_rate > self.config.alert_thresholds.leak_growth_rate {
                    self.send_alert(MemoryAlert {
                        timestamp: Utc::now(),
                        alert_type: AlertType::MemoryLeak,
                        severity: AlertSeverity::Critical,
                        message: format!(
                            "Potential memory leak detected: {:.1}% growth",
                            growth_rate * 100.0
                        ),
                        current_value: growth_rate,
                        threshold: self.config.alert_thresholds.leak_growth_rate,
                    });
                }
            }
        }
    }

    fn calculate_growth_rate(metrics: &[MemoryMetrics]) -> f64 {
        if metrics.len() < 2 {
            return 0.0;
        }

        let first = metrics.first().unwrap();
        let last = metrics.last().unwrap();

        if first.current_usage_bytes == 0 {
            return 0.0;
        }

        (last.current_usage_bytes as f64 - first.current_usage_bytes as f64)
            / first.current_usage_bytes as f64
    }

    fn send_alert(&self, alert: MemoryAlert) {
        self.alerts.write().push(alert.clone());
        let _ = self.alert_tx.send(alert);
    }

    /// Get current metrics
    pub fn get_current_metrics(&self) -> Option<MemoryMetrics> {
        self.metrics_history.read().last().cloned()
    }

    /// Get metrics history
    pub fn get_metrics_history(&self, limit: Option<usize>) -> Vec<MemoryMetrics> {
        let history = self.metrics_history.read();
        if let Some(limit) = limit {
            history.iter().rev().take(limit).rev().cloned().collect()
        } else {
            history.clone()
        }
    }

    /// Get active alerts
    pub fn get_alerts(&self, since: Option<DateTime<Utc>>) -> Vec<MemoryAlert> {
        let alerts = self.alerts.read();
        if let Some(since) = since {
            alerts
                .iter()
                .filter(|a| a.timestamp > since)
                .cloned()
                .collect()
        } else {
            alerts.clone()
        }
    }

    /// Export dashboard data
    pub fn export(&self, format: ExportFormat) -> String {
        let metrics = self.get_metrics_history(None);
        let alerts = self.get_alerts(None);
        let report = self.profiler.generate_report();

        match format {
            ExportFormat::Json => self.export_json(metrics, alerts, report),
            ExportFormat::Csv => self.export_csv(metrics),
            ExportFormat::Html => self.export_html(metrics, alerts, report),
            ExportFormat::Markdown => self.export_markdown(metrics, alerts, report),
        }
    }

    fn export_json(
        &self,
        metrics: Vec<MemoryMetrics>,
        alerts: Vec<MemoryAlert>,
        report: MemoryReport,
    ) -> String {
        serde_json::json!({
            "metrics": metrics,
            "alerts": alerts,
            "report": report,
            "exported_at": Utc::now(),
        })
        .to_string()
    }

    fn export_csv(&self, metrics: Vec<MemoryMetrics>) -> String {
        let mut csv = String::from(
            "timestamp,usage_bytes,peak_bytes,alloc_rate,dealloc_rate,fragmentation\n",
        );

        for metric in metrics {
            csv.push_str(&format!(
                "{},{},{},{:.2},{:.2},{:.2}\n",
                metric.timestamp,
                metric.current_usage_bytes,
                metric.peak_usage_bytes,
                metric.allocation_rate,
                metric.deallocation_rate,
                metric.fragmentation_ratio,
            ));
        }

        csv
    }

    fn export_html(
        &self,
        _metrics: Vec<MemoryMetrics>,
        alerts: Vec<MemoryAlert>,
        report: MemoryReport,
    ) -> String {
        format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>Memory Dashboard Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .alert {{ background: #fff3cd; padding: 10px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107; }}
        .critical {{ border-left-color: #dc3545; background: #f8d7da; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Memory Dashboard Report</h1>

    <h2>Current Status</h2>
    <div class="metric">
        <strong>Current Usage:</strong> {} MB<br>
        <strong>Peak Usage:</strong> {} MB<br>
        <strong>Fragmentation:</strong> {:.2}
    </div>

    <h2>Recent Alerts</h2>
    {}

    <h2>Component Breakdown</h2>
    <table>
        <tr>
            <th>Component</th>
            <th>Memory (MB)</th>
            <th>Objects</th>
        </tr>
        {}
    </table>

    <p><em>Generated at {}</em></p>
</body>
</html>
        "#,
            report.current_usage / 1_048_576,
            report.peak_usage / 1_048_576,
            report.fragmentation_ratio,
            self.format_alerts_html(&alerts),
            self.format_components_html(&report),
            Utc::now()
        )
    }

    fn format_alerts_html(&self, alerts: &[MemoryAlert]) -> String {
        if alerts.is_empty() {
            return String::from("<p>No recent alerts</p>");
        }

        alerts
            .iter()
            .take(10)
            .map(|alert| {
                let class = match alert.severity {
                    AlertSeverity::Critical => "alert critical",
                    _ => "alert",
                };
                format!(
                    r#"<div class="{}"><strong>{}:</strong> {}</div>"#,
                    class,
                    alert.timestamp.format("%Y-%m-%d %H:%M:%S"),
                    alert.message
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn format_components_html(&self, report: &MemoryReport) -> String {
        report
            .component_breakdown
            .iter()
            .map(|(name, usage)| {
                format!(
                    "<tr><td>{}</td><td>{}</td><td>{}</td></tr>",
                    name,
                    usage.allocated_bytes / 1_048_576,
                    usage.object_count
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn export_markdown(
        &self,
        metrics: Vec<MemoryMetrics>,
        alerts: Vec<MemoryAlert>,
        report: MemoryReport,
    ) -> String {
        format!(
            r#"# Memory Dashboard Report

## Current Status
- **Current Usage:** {} MB
- **Peak Usage:** {} MB
- **Average Usage:** {} MB
- **Fragmentation Ratio:** {:.2}
- **Total Allocations:** {}
- **Total Deallocations:** {}

## Recent Alerts
{}

## Component Breakdown
| Component | Memory (MB) | Objects | Avg Size (KB) |
|-----------|-------------|---------|---------------|
{}

## Memory Trends
{}

*Generated at {}*
"#,
            report.current_usage / 1_048_576,
            report.peak_usage / 1_048_576,
            report.average_usage / 1_048_576,
            report.fragmentation_ratio,
            report.allocation_count,
            report.deallocation_count,
            self.format_alerts_markdown(&alerts),
            self.format_components_markdown(&report),
            self.format_trends_markdown(&metrics),
            Utc::now()
        )
    }

    fn format_alerts_markdown(&self, alerts: &[MemoryAlert]) -> String {
        if alerts.is_empty() {
            return String::from("No recent alerts");
        }

        alerts
            .iter()
            .take(10)
            .map(|alert| {
                let severity_icon = match alert.severity {
                    AlertSeverity::Critical => "🔴",
                    AlertSeverity::Warning => "🟡",
                    AlertSeverity::Info => "🔵",
                };
                format!(
                    "- {} **{}:** {}",
                    severity_icon,
                    alert.timestamp.format("%Y-%m-%d %H:%M:%S"),
                    alert.message
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn format_components_markdown(&self, report: &MemoryReport) -> String {
        report
            .component_breakdown
            .iter()
            .map(|(name, usage)| {
                format!(
                    "| {} | {} | {} | {} |",
                    name,
                    usage.allocated_bytes / 1_048_576,
                    usage.object_count,
                    usage.avg_object_size / 1024
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn format_trends_markdown(&self, metrics: &[MemoryMetrics]) -> String {
        if metrics.len() < 2 {
            return String::from("Insufficient data for trends");
        }

        let first = metrics.first().unwrap();
        let last = metrics.last().unwrap();

        let memory_change = (last.current_usage_bytes as i64 - first.current_usage_bytes as i64)
            as f64
            / first.current_usage_bytes as f64
            * 100.0;

        let trend = if memory_change > 5.0 {
            "📈 Increasing"
        } else if memory_change < -5.0 {
            "📉 Decreasing"
        } else {
            "➡️ Stable"
        };

        format!(
            "Memory usage trend: {} ({:+.1}% over {} samples)",
            trend,
            memory_change,
            metrics.len()
        )
    }

    /// Clear dashboard data
    pub fn clear(&self) {
        self.metrics_history.write().clear();
        self.alerts.write().clear();
        *self.last_snapshot.write() = None;
    }
}

/// Terminal UI renderer for console output
pub struct TerminalRenderer {
    dashboard: Arc<MemoryDashboard>,
}

impl TerminalRenderer {
    pub fn new(dashboard: Arc<MemoryDashboard>) -> Self {
        Self { dashboard }
    }

    /// Render dashboard to terminal
    pub fn render(&self) -> String {
        let mut output = String::new();

        // Header
        output.push_str(&format!("{:=^80}\n", " Memory Dashboard "));

        // Current metrics
        if let Some(metrics) = self.dashboard.get_current_metrics() {
            output.push_str(&format!(
                "\n📊 Current Status @ {}\n",
                metrics.timestamp.format("%H:%M:%S")
            ));
            output.push_str(&format!(
                "  Memory: {} MB / {} MB peak\n",
                metrics.current_usage_bytes / 1_048_576,
                metrics.peak_usage_bytes / 1_048_576
            ));
            output.push_str(&format!(
                "  Rates: {:.0} allocs/s, {:.0} deallocs/s\n",
                metrics.allocation_rate, metrics.deallocation_rate
            ));
            output.push_str(&format!(
                "  Fragmentation: {:.2}x\n",
                metrics.fragmentation_ratio
            ));

            // Component breakdown
            if !metrics.component_metrics.is_empty() {
                output.push_str("\n📦 Components:\n");
                for comp in &metrics.component_metrics {
                    output.push_str(&format!(
                        "  {}: {} MB ({} objects)\n",
                        comp.name,
                        comp.memory_bytes / 1_048_576,
                        comp.object_count
                    ));
                }
            }
        }

        // Recent alerts
        let alerts = self
            .dashboard
            .get_alerts(Some(Utc::now() - chrono::Duration::minutes(5)));
        if !alerts.is_empty() {
            output.push_str("\n⚠️ Recent Alerts:\n");
            for alert in alerts.iter().take(5) {
                let icon = match alert.severity {
                    AlertSeverity::Critical => "🔴",
                    AlertSeverity::Warning => "🟡",
                    AlertSeverity::Info => "🔵",
                };
                output.push_str(&format!(
                    "  {} {}: {}\n",
                    icon,
                    alert.timestamp.format("%H:%M:%S"),
                    alert.message
                ));
            }
        }

        // Memory usage bar chart
        output.push_str("\n📈 Memory Usage (last 20 samples):\n");
        output.push_str(&self.render_bar_chart());

        output.push_str(&format!("\n{:=^80}\n", ""));

        output
    }

    fn render_bar_chart(&self) -> String {
        let history = self.dashboard.get_metrics_history(Some(20));
        if history.is_empty() {
            return String::from("  No data available\n");
        }

        let max_usage = history
            .iter()
            .map(|m| m.current_usage_bytes)
            .max()
            .unwrap_or(1);

        let mut chart = String::new();
        for metrics in &history {
            let bar_width = (metrics.current_usage_bytes as f64 / max_usage as f64 * 40.0) as usize;
            let bar = "█".repeat(bar_width);
            chart.push_str(&format!(
                "  {:>6} MB |{:<40}|\n",
                metrics.current_usage_bytes / 1_048_576,
                bar
            ));
        }

        chart
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_creation() {
        let profiler = Arc::new(MemoryProfiler::new());
        let dashboard = MemoryDashboard::new(profiler, DashboardConfig::default());

        assert!(dashboard.get_current_metrics().is_none());
        assert!(dashboard.get_alerts(None).is_empty());
    }

    #[test]
    fn test_alert_thresholds() {
        let thresholds = AlertThresholds::default();
        assert_eq!(thresholds.memory_usage_percent, 80.0);
        assert_eq!(thresholds.allocation_rate, 10000);
    }

    #[test]
    fn test_export_formats() {
        let profiler = Arc::new(MemoryProfiler::new());
        let dashboard = MemoryDashboard::new(profiler, DashboardConfig::default());

        let json = dashboard.export(ExportFormat::Json);
        assert!(json.contains("metrics"));

        let csv = dashboard.export(ExportFormat::Csv);
        assert!(csv.contains("timestamp"));
    }
}
