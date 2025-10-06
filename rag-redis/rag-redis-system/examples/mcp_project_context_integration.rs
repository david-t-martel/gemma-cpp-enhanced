//! MCP Project Context Integration Example
//!
//! This example demonstrates how to integrate the project context system
//! with an MCP server to provide project state management capabilities.

use rag_redis_system::{Config, RagSystem, project_context::SaveOptions};
use serde_json::json;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 MCP Project Context Integration Demo");
    println!("=====================================\n");

    // Initialize the RAG system
    let config = Config::default();
    let rag_system = RagSystem::new(config).await?;

    // Warm up all components
    rag_system.warm_up().await?;

    println!("✅ RAG system initialized and warmed up");

    // Demo MCP operations as they would be called from an MCP client
    let project_id = "mcp-demo-project";
    let current_dir = env::current_dir()?;

    // Demo 1: Save project context via MCP
    println!("\n📦 Demo 1: MCP save_project_context");
    println!("------------------------------------");

    let save_options = json!({
        "include_files": true,
        "include_memories": true,
        "include_vectors": false, // Skip for demo speed
        "include_config": true,
        "compress": true,
        "deduplicate": true,
        "description": "MCP integration demo snapshot",
        "snapshot_type": "Full"
    });

    let snapshot_id = rag_system
        .handle_save_project_context(
            project_id.to_string(),
            Some(current_dir.to_string_lossy().to_string()),
            Some(save_options),
        )
        .await?;

    println!("✅ Project context saved via MCP");
    println!("   Snapshot ID: {}", snapshot_id);

    // Demo 2: Load project context via MCP
    println!("\n📥 Demo 2: MCP load_project_context");
    println!("------------------------------------");

    let loaded_context = rag_system
        .handle_load_project_context(project_id.to_string(), Some(snapshot_id.clone()))
        .await?;

    println!("✅ Project context loaded via MCP");
    if let Some(obj) = loaded_context.as_object() {
        println!("   📊 Files: {}", obj.get("project_files")
            .and_then(|f| f.get("file_count"))
            .and_then(|c| c.as_u64())
            .unwrap_or(0));
        println!("   🔖 Version: {}", obj.get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown"));
        println!("   📏 Storage size: {} bytes", obj.get("snapshot_metadata")
            .and_then(|m| m.get("storage_size"))
            .and_then(|s| s.as_u64())
            .unwrap_or(0));
    }

    // Demo 3: Quick session save via MCP
    println!("\n⚡ Demo 3: MCP quick_save_session");
    println!("----------------------------------");

    let session_id = rag_system
        .handle_quick_save_session(
            project_id.to_string(),
            "MCP session: implementing project context features".to_string(),
        )
        .await?;

    println!("✅ Quick session saved via MCP");
    println!("   Session ID: {}", session_id);

    // Demo 4: List project snapshots via MCP
    println!("\n📋 Demo 4: MCP list_project_snapshots");
    println!("--------------------------------------");

    let snapshots = rag_system
        .handle_list_project_snapshots(project_id.to_string(), Some(10))
        .await?;

    println!("✅ Listed {} snapshots via MCP", snapshots.len());
    for (i, snapshot) in snapshots.iter().enumerate() {
        if let Some(obj) = snapshot.as_object() {
            println!("   {}. {} - {} ({})",
                i + 1,
                obj.get("version").and_then(|v| v.as_str()).unwrap_or("unknown"),
                obj.get("description").and_then(|d| d.as_str()).unwrap_or("no description"),
                obj.get("snapshot_type").and_then(|t| t.as_str()).unwrap_or("unknown")
            );
        }
    }

    // Demo 5: Generate a simple diff via MCP (if we have multiple snapshots)
    if snapshots.len() >= 2 {
        println!("\n🔍 Demo 5: MCP diff_contexts");
        println!("-----------------------------");

        let snapshot1 = &snapshots[1]; // Older snapshot
        let snapshot2 = &snapshots[0]; // Newer snapshot

        let version1 = snapshot1.get("version").and_then(|v| v.as_str()).unwrap_or("1.0.0");
        let version2 = snapshot2.get("version").and_then(|v| v.as_str()).unwrap_or("1.0.1");

        let diff = rag_system
            .handle_diff_contexts(
                project_id.to_string(),
                version1.to_string(),
                version2.to_string(),
            )
            .await?;

        println!("✅ Context diff generated via MCP");
        if let Some(summary) = diff.get("summary") {
            println!("   📊 Total changes: {}", summary.get("total_changes")
                .and_then(|c| c.as_u64())
                .unwrap_or(0));
            println!("   📁 Files affected: {}", summary.get("files_affected")
                .and_then(|f| f.as_u64())
                .unwrap_or(0));
        }
    }

    // Demo 6: Get project statistics via MCP
    println!("\n📈 Demo 6: MCP get_project_statistics");
    println!("--------------------------------------");

    let stats = rag_system
        .handle_get_project_statistics(project_id.to_string())
        .await?;

    println!("✅ Project statistics retrieved via MCP");
    if let Some(obj) = stats.as_object() {
        println!("   📊 Total snapshots: {}", obj.get("total_snapshots")
            .and_then(|s| s.as_u64())
            .unwrap_or(0));
        println!("   💾 Total storage: {} bytes", obj.get("total_storage_size")
            .and_then(|s| s.as_u64())
            .unwrap_or(0));
    }

    // Demo 7: Cleanup old snapshots via MCP
    println!("\n🧹 Demo 7: MCP cleanup_old_snapshots");
    println!("-------------------------------------");

    let cleaned_count = rag_system
        .handle_cleanup_old_snapshots(project_id.to_string())
        .await?;

    println!("✅ Cleaned up {} old snapshots via MCP", cleaned_count);

    println!("\n🎉 MCP Project Context Integration Demo Completed!");
    println!("===================================================");
    println!("The following MCP tools are now available:");
    println!("• save_project_context     - Save complete project state");
    println!("• load_project_context     - Load project state by ID");
    println!("• quick_save_session       - Quick session save");
    println!("• quick_load_session       - Quick session load");
    println!("• list_project_snapshots   - List all project snapshots");
    println!("• diff_contexts            - Compare two project contexts");
    println!("• get_project_statistics   - Get project analytics");
    println!("• cleanup_old_snapshots    - Clean up old snapshots");

    println!("\n🔧 Integration Points:");
    println!("• Redis backend for persistent storage");
    println!("• Memory system integration for context");
    println!("• Vector store integration for embeddings");
    println!("• Compression and deduplication for efficiency");
    println!("• Versioning system for tracking changes");
    println!("• Session management for LLM continuity");

    Ok(())
}
"