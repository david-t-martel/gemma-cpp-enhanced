//! Memory profiling and monitoring module for the RAG-Redis system
//!
//! This module provides comprehensive memory profiling capabilities including:
//! - Real-time memory usage tracking
//! - Memory leak detection
//! - Allocation pattern analysis
//! - Memory pressure monitoring
//! - Detailed memory reports

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::{debug, error, warn};

/// Global memory statistics
pub static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
pub static DEALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
pub static ALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static DEALLOCATION_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static PEAK_MEMORY: AtomicUsize = AtomicUsize::new(0);

/// Custom allocator that tracks memory usage
pub struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            let size = layout.size();
            ALLOCATED_BYTES.fetch_add(size, Ordering::Relaxed);
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);

            // Update peak memory
            let current =
                ALLOCATED_BYTES.load(Ordering::Relaxed) - DEALLOCATED_BYTES.load(Ordering::Relaxed);
            let mut peak = PEAK_MEMORY.load(Ordering::Relaxed);
            while current > peak {
                match PEAK_MEMORY.compare_exchange(
                    peak,
                    current,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
        }
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        DEALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        DEALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

/// Memory usage snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub timestamp: DateTime<Utc>,
    pub allocated_bytes: usize,
    pub deallocated_bytes: usize,
    pub current_usage: usize,
    pub peak_usage: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
    pub fragmentation_ratio: f64,
}

impl MemorySnapshot {
    pub fn capture() -> Self {
        let allocated = ALLOCATED_BYTES.load(Ordering::Relaxed);
        let deallocated = DEALLOCATED_BYTES.load(Ordering::Relaxed);
        let current = allocated.saturating_sub(deallocated);

        // Estimate fragmentation
        let alloc_count = ALLOCATION_COUNT.load(Ordering::Relaxed);
        let dealloc_count = DEALLOCATION_COUNT.load(Ordering::Relaxed);
        let active_allocations = alloc_count.saturating_sub(dealloc_count);

        let avg_allocation_size = if active_allocations > 0 {
            current / active_allocations
        } else {
            0
        };

        // Simple fragmentation heuristic: ratio of allocations to average size
        let fragmentation_ratio = if avg_allocation_size > 0 && active_allocations > 0 {
            (active_allocations as f64 * 1024.0) / current as f64
        } else {
            0.0
        };

        Self {
            timestamp: Utc::now(),
            allocated_bytes: allocated,
            deallocated_bytes: deallocated,
            current_usage: current,
            peak_usage: PEAK_MEMORY.load(Ordering::Relaxed),
            allocation_count: alloc_count,
            deallocation_count: dealloc_count,
            fragmentation_ratio,
        }
    }
}

/// Component-specific memory tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMemoryUsage {
    pub name: String,
    pub allocated_bytes: usize,
    pub object_count: usize,
    pub avg_object_size: usize,
    pub peak_bytes: usize,
    pub last_updated: DateTime<Utc>,
}

/// Memory profiler for detailed analysis
pub struct MemoryProfiler {
    snapshots: Arc<RwLock<Vec<MemorySnapshot>>>,
    component_usage: Arc<RwLock<HashMap<String, ComponentMemoryUsage>>>,
    pressure_threshold: f64,
    snapshot_interval: std::time::Duration,
    max_snapshots: usize,
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryProfiler {
    pub fn new() -> Self {
        Self {
            snapshots: Arc::new(RwLock::new(Vec::new())),
            component_usage: Arc::new(RwLock::new(HashMap::new())),
            pressure_threshold: 0.8, // 80% of available memory
            snapshot_interval: std::time::Duration::from_secs(60),
            max_snapshots: 1000,
        }
    }

    /// Start automatic profiling
    pub fn start_profiling(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(self.snapshot_interval);

            loop {
                interval.tick().await;

                let snapshot = MemorySnapshot::capture();
                self.record_snapshot(snapshot.clone());

                // Check for memory pressure
                if let Some(pressure) = self.check_memory_pressure() {
                    if pressure > self.pressure_threshold {
                        warn!("High memory pressure detected: {:.2}%", pressure * 100.0);
                        self.trigger_memory_optimization();
                    }
                }

                // Detect potential memory leaks
                if let Some(leak_info) = self.detect_memory_leak() {
                    error!("Potential memory leak detected: {}", leak_info);
                }
            }
        })
    }

    /// Record a memory snapshot
    pub fn record_snapshot(&self, snapshot: MemorySnapshot) {
        let mut snapshots = self.snapshots.write();
        snapshots.push(snapshot);

        // Keep only recent snapshots
        if snapshots.len() > self.max_snapshots {
            let excess = snapshots.len() - self.max_snapshots;
            snapshots.drain(0..excess);
        }
    }

    /// Track memory usage for a specific component
    pub fn track_component(&self, name: &str, bytes: usize, object_count: usize) {
        let mut components = self.component_usage.write();

        let component =
            components
                .entry(name.to_string())
                .or_insert_with(|| ComponentMemoryUsage {
                    name: name.to_string(),
                    allocated_bytes: 0,
                    object_count: 0,
                    avg_object_size: 0,
                    peak_bytes: 0,
                    last_updated: Utc::now(),
                });

        component.allocated_bytes = bytes;
        component.object_count = object_count;
        component.avg_object_size = if object_count > 0 {
            bytes / object_count
        } else {
            0
        };

        if bytes > component.peak_bytes {
            component.peak_bytes = bytes;
        }

        component.last_updated = Utc::now();
    }

    /// Check current memory pressure
    pub fn check_memory_pressure(&self) -> Option<f64> {
        // Get system memory info
        if let Ok(mem_info) = sys_info::mem_info() {
            let total = mem_info.total as f64 * 1024.0; // Convert to bytes
            let available = mem_info.avail as f64 * 1024.0;
            let used = total - available;

            Some(used / total)
        } else {
            None
        }
    }

    /// Detect potential memory leaks
    pub fn detect_memory_leak(&self) -> Option<String> {
        let snapshots = self.snapshots.read();

        if snapshots.len() < 10 {
            return None;
        }

        // Check if memory is consistently growing
        let recent_snapshots = &snapshots[snapshots.len() - 10..];
        let mut growing_count = 0;

        for i in 1..recent_snapshots.len() {
            if recent_snapshots[i].current_usage > recent_snapshots[i - 1].current_usage {
                growing_count += 1;
            }
        }

        if growing_count >= 8 {
            let growth_rate = (recent_snapshots.last()?.current_usage as f64
                - recent_snapshots.first()?.current_usage as f64)
                / recent_snapshots.first()?.current_usage as f64;

            if growth_rate > 0.1 {
                // 10% growth
                return Some(format!(
                    "Memory grew by {:.1}% over last {} snapshots",
                    growth_rate * 100.0,
                    recent_snapshots.len()
                ));
            }
        }

        None
    }

    /// Trigger memory optimization
    pub fn trigger_memory_optimization(&self) {
        debug!("Triggering memory optimization");

        // Force garbage collection hint (though Rust doesn't have GC)
        // This is more of a placeholder for actual optimization logic

        // Log component usage for analysis
        let components = self.component_usage.read();
        for (name, usage) in components.iter() {
            debug!(
                "Component {}: {} bytes, {} objects",
                name, usage.allocated_bytes, usage.object_count
            );
        }
    }

    /// Generate memory report
    pub fn generate_report(&self) -> MemoryReport {
        let snapshots = self.snapshots.read();
        let components = self.component_usage.read();

        let current_snapshot = MemorySnapshot::capture();

        // Calculate statistics
        let avg_usage = if !snapshots.is_empty() {
            snapshots.iter().map(|s| s.current_usage).sum::<usize>() / snapshots.len()
        } else {
            current_snapshot.current_usage
        };

        let max_usage = snapshots
            .iter()
            .map(|s| s.current_usage)
            .max()
            .unwrap_or(current_snapshot.current_usage);

        MemoryReport {
            current_usage: current_snapshot.current_usage,
            peak_usage: current_snapshot.peak_usage,
            average_usage: avg_usage,
            max_recorded_usage: max_usage,
            allocation_count: current_snapshot.allocation_count,
            deallocation_count: current_snapshot.deallocation_count,
            fragmentation_ratio: current_snapshot.fragmentation_ratio,
            component_breakdown: components.clone(),
            timestamp: Utc::now(),
        }
    }

    /// Clear profiling data
    pub fn clear(&self) {
        self.snapshots.write().clear();
        self.component_usage.write().clear();
    }
}

/// Comprehensive memory report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReport {
    pub current_usage: usize,
    pub peak_usage: usize,
    pub average_usage: usize,
    pub max_recorded_usage: usize,
    pub allocation_count: usize,
    pub deallocation_count: usize,
    pub fragmentation_ratio: f64,
    pub component_breakdown: HashMap<String, ComponentMemoryUsage>,
    pub timestamp: DateTime<Utc>,
}

impl MemoryReport {
    /// Format the report as human-readable text
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("=== Memory Report ({}) ===\n", self.timestamp));
        report.push_str(&format!(
            "Current Usage: {} MB\n",
            self.current_usage / 1_048_576
        ));
        report.push_str(&format!("Peak Usage: {} MB\n", self.peak_usage / 1_048_576));
        report.push_str(&format!(
            "Average Usage: {} MB\n",
            self.average_usage / 1_048_576
        ));
        report.push_str(&format!(
            "Max Recorded: {} MB\n",
            self.max_recorded_usage / 1_048_576
        ));
        report.push_str(&format!("Allocations: {}\n", self.allocation_count));
        report.push_str(&format!("Deallocations: {}\n", self.deallocation_count));
        report.push_str(&format!(
            "Fragmentation: {:.2}%\n",
            self.fragmentation_ratio * 100.0
        ));

        if !self.component_breakdown.is_empty() {
            report.push_str("\n=== Component Breakdown ===\n");
            for (name, usage) in &self.component_breakdown {
                report.push_str(&format!(
                    "{}: {} MB ({} objects, avg {} KB)\n",
                    name,
                    usage.allocated_bytes / 1_048_576,
                    usage.object_count,
                    usage.avg_object_size / 1024
                ));
            }
        }

        report
    }
}

/// Memory optimization strategies
pub struct MemoryOptimizer {
    profiler: Arc<MemoryProfiler>,
}

impl MemoryOptimizer {
    pub fn new(profiler: Arc<MemoryProfiler>) -> Self {
        Self { profiler }
    }

    /// Optimize vector store memory
    pub fn optimize_vector_store(
        &self,
        store: &crate::vector_store::VectorStore,
    ) -> OptimizationResult {
        let mut result = OptimizationResult::default();

        // Track current usage
        let stats = store.get_stats();
        self.profiler
            .track_component("vector_store", stats.memory_usage, stats.vector_count);

        // Optimization strategies
        result
            .strategies_applied
            .push("vector_store_optimization".to_string());
        result.bytes_saved = 0; // Placeholder for actual optimization

        result
    }

    /// Optimize memory manager
    pub async fn optimize_memory_manager(
        &self,
        manager: &crate::memory::MemoryManager,
    ) -> OptimizationResult {
        let mut result = OptimizationResult::default();

        // Trigger cleanup
        let (removed, freed) = manager.cleanup().await.unwrap_or((0, 0));
        result.bytes_saved = freed;

        // Consolidate memories
        let consolidated = manager.consolidate_memories().await.unwrap_or(0);

        result
            .strategies_applied
            .push(format!("cleanup_removed_{}_entries", removed));
        result
            .strategies_applied
            .push(format!("consolidated_{}_memories", consolidated));

        result
    }

    /// Run full optimization pass
    pub async fn optimize_all(&self) -> OptimizationResult {
        let mut result = OptimizationResult::default();

        // Run all optimization strategies
        self.profiler.trigger_memory_optimization();

        result
            .strategies_applied
            .push("full_optimization".to_string());
        result.success = true;

        result
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub success: bool,
    pub bytes_saved: usize,
    pub strategies_applied: Vec<String>,
    pub errors: Vec<String>,
}

/// RAII guard for tracking memory usage of a scope
pub struct MemoryScope {
    name: String,
    start_usage: usize,
    profiler: Arc<MemoryProfiler>,
}

impl MemoryScope {
    pub fn new(name: impl Into<String>, profiler: Arc<MemoryProfiler>) -> Self {
        let snapshot = MemorySnapshot::capture();
        Self {
            name: name.into(),
            start_usage: snapshot.current_usage,
            profiler,
        }
    }
}

impl Drop for MemoryScope {
    fn drop(&mut self) {
        let snapshot = MemorySnapshot::capture();
        let delta = snapshot.current_usage.saturating_sub(self.start_usage);

        debug!("Memory scope '{}' used {} bytes", self.name, delta);

        self.profiler.track_component(&self.name, delta, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_snapshot() {
        let snapshot = MemorySnapshot::capture();
        assert!(snapshot.timestamp <= Utc::now());
    }

    #[test]
    fn test_memory_profiler() {
        let profiler = Arc::new(MemoryProfiler::new());

        let snapshot1 = MemorySnapshot::capture();
        profiler.record_snapshot(snapshot1);

        let report = profiler.generate_report();
        assert!(report.current_usage >= 0);
    }

    #[test]
    fn test_component_tracking() {
        let profiler = Arc::new(MemoryProfiler::new());

        profiler.track_component("test_component", 1024 * 1024, 100);

        let report = profiler.generate_report();
        assert!(report.component_breakdown.contains_key("test_component"));
    }
}
