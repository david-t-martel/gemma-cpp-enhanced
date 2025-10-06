//! Project Context Storage and Retrieval System Demo
//!
//! This example demonstrates the comprehensive project context management capabilities
//! of the RAG-Redis system, including saving/loading project state, versioning,
//! session management, and context diffing.

use rag_redis_system::{
    Config, RagSystem,
    project_context::{ProjectContextManager, SaveOptions, SnapshotType, SessionType},
};
use std::path::PathBuf;
use tokio::fs;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🚀 RAG-Redis Project Context System Demo");
    println!("=========================================\n");

    // Initialize the RAG system
    let config = Config::default();
    let rag_system = RagSystem::new(config).await?;

    // Project setup
    let project_id = "demo-project";
    let demo_project_root = create_demo_project().await?;

    println!("📁 Created demo project at: {}", demo_project_root.display());

    // Demo 1: Full Project Context Save
    println!("\n📦 Demo 1: Saving Full Project Context");
    println!("--------------------------------------");

    let save_options = SaveOptions {
        include_files: true,
        include_memories: true,
        include_vectors: true,
        include_config: true,
        compress: true,
        deduplicate: true,
        description: Some("Initial project snapshot with full context".to_string()),
        snapshot_type: SnapshotType::Full,
        ..Default::default()
    };

    let snapshot_id = rag_system
        .handle_save_project_context(
            project_id.to_string(),
            Some(demo_project_root.to_string_lossy().to_string()),
            Some(serde_json::to_value(save_options)?),
        )
        .await?;

    println!("✅ Project context saved with ID: {}", snapshot_id);

    // Demo 2: Load Project Context
    println!("\n📥 Demo 2: Loading Project Context");
    println!("----------------------------------");

    let loaded_context = rag_system
        .handle_load_project_context(project_id.to_string(), Some(snapshot_id.clone()))
        .await?;

    println!("✅ Project context loaded successfully");
    if let Some(obj) = loaded_context.as_object() {
        println!("   📊 Files: {}", obj.get("project_files")
            .and_then(|f| f.get("file_count"))
            .and_then(|c| c.as_u64())
            .unwrap_or(0));
        println!("   📏 Total size: {} bytes", obj.get("project_files")
            .and_then(|f| f.get("total_size"))
            .and_then(|s| s.as_u64())
            .unwrap_or(0));
        println!("   🏷️  Version: {}", obj.get("version")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown"));
    }

    // Demo 3: Quick Session Save/Load
    println!("\n⚡ Demo 3: Quick Session Management");
    println!("-----------------------------------");

    // Set current directory for quick save
    std::env::set_current_dir(&demo_project_root)?;

    let session_id = rag_system
        .handle_quick_save_session(
            project_id.to_string(),
            "Working on feature X - debugging authentication".to_string(),
        )
        .await?;

    println!("✅ Quick session saved with ID: {}", session_id);

    let session_context = rag_system
        .handle_quick_load_session(project_id.to_string(), session_id)
        .await?;

    println!("✅ Quick session loaded successfully");
    if let Some(obj) = session_context.as_object() {
        println!("   📝 Description: {}", obj.get("description")
            .and_then(|d| d.as_str())
            .unwrap_or("none"));
        println!("   🔖 Type: {:?}", obj.get("snapshot_metadata")
            .and_then(|m| m.get("snapshot_type"))
            .and_then(|t| t.as_str())
            .unwrap_or("unknown"));
    }

    // Demo 4: Project Modifications and Versioning
    println!("\n🔄 Demo 4: Project Modifications and Versioning");
    println!("------------------------------------------------");

    // Modify some files
    modify_demo_project(&demo_project_root).await?;
    println!("📝 Modified project files");

    // Save another snapshot
    let modified_snapshot_id = rag_system
        .handle_save_project_context(
            project_id.to_string(),
            Some(demo_project_root.to_string_lossy().to_string()),
            Some(serde_json::to_value(SaveOptions {
                description: Some("After implementing authentication feature".to_string()),
                snapshot_type: SnapshotType::Milestone,
                ..Default::default()
            })?),
        )
        .await?;

    println!("✅ Modified project context saved with ID: {}", modified_snapshot_id);

    // Demo 5: Context Diffing
    println!("\n🔍 Demo 5: Context Diffing");
    println!("--------------------------");

    // Load both snapshots to get their versions
    let original_snapshot = rag_system
        .handle_load_project_context(project_id.to_string(), Some(snapshot_id))
        .await?;

    let modified_snapshot = rag_system
        .handle_load_project_context(project_id.to_string(), Some(modified_snapshot_id))
        .await?;

    let original_version = original_snapshot.get("version")
        .and_then(|v| v.as_str())
        .unwrap_or("1.0.0");

    let modified_version = modified_snapshot.get("version")
        .and_then(|v| v.as_str())
        .unwrap_or("1.0.1");

    let diff = rag_system
        .handle_diff_contexts(
            project_id.to_string(),
            original_version.to_string(),
            modified_version.to_string(),
        )
        .await?;

    println!("✅ Context diff generated");
    if let Some(summary) = diff.get("summary") {
        println!("   📊 Total changes: {}", summary.get("total_changes")
            .and_then(|c| c.as_u64())
            .unwrap_or(0));
        println!("   📁 Files affected: {}", summary.get("files_affected")
            .and_then(|f| f.as_u64())
            .unwrap_or(0));
        println!("   🧠 Memories affected: {}", summary.get("memories_affected")
            .and_then(|m| m.as_u64())
            .unwrap_or(0));
        println!("   ⚠️  Impact level: {}", summary.get("overall_impact")
            .and_then(|i| i.as_str())
            .unwrap_or("unknown"));
    }

    // Demo 6: List Project Snapshots
    println!("\n📋 Demo 6: Listing Project Snapshots");
    println!("-------------------------------------");

    let snapshots = rag_system
        .handle_list_project_snapshots(project_id.to_string(), Some(10))
        .await?;

    println!("✅ Found {} snapshots:", snapshots.len());
    for (i, snapshot) in snapshots.iter().enumerate() {
        if let Some(obj) = snapshot.as_object() {
            println!("   {}. {} - {} ({})",
                i + 1,
                obj.get("version").and_then(|v| v.as_str()).unwrap_or("unknown"),
                obj.get("description").and_then(|d| d.as_str()).unwrap_or("no description"),
                obj.get("snapshot_type").and_then(|t| t.as_str()).unwrap_or("unknown")
            );
            println!("      📁 {} files, 📏 {} bytes, 🗜️ {:.2}% compression",
                obj.get("file_count").and_then(|f| f.as_u64()).unwrap_or(0),
                obj.get("total_size").and_then(|s| s.as_u64()).unwrap_or(0),
                (1.0 - obj.get("compression_ratio").and_then(|r| r.as_f64()).unwrap_or(1.0)) * 100.0
            );
        }
    }

    // Demo 7: Project Statistics
    println!("\n📈 Demo 7: Project Statistics");
    println!("------------------------------");

    let stats = rag_system
        .handle_get_project_statistics(project_id.to_string())
        .await?;

    println!("✅ Project statistics:");
    if let Some(obj) = stats.as_object() {
        println!("   📊 Total snapshots: {}", obj.get("total_snapshots")
            .and_then(|s| s.as_u64())
            .unwrap_or(0));
        println!("   💾 Total storage: {} bytes", obj.get("total_storage_size")
            .and_then(|s| s.as_u64())
            .unwrap_or(0));
        println!("   📅 First snapshot: {}", obj.get("oldest_snapshot")
            .and_then(|d| d.as_str())
            .unwrap_or("unknown"));
        println!("   📅 Latest snapshot: {}", obj.get("newest_snapshot")
            .and_then(|d| d.as_str())
            .unwrap_or("unknown"));
    }

    // Demo 8: Cleanup Old Snapshots
    println!("\n🧹 Demo 8: Cleanup Old Snapshots");
    println!("----------------------------------");

    let cleaned_count = rag_system
        .handle_cleanup_old_snapshots(project_id.to_string())
        .await?;

    println!("✅ Cleaned up {} old snapshots", cleaned_count);

    println!("\n🎉 Demo completed successfully!");
    println!("===============================");
    println!("The project context system provides:");
    println!("• 📦 Complete project state capture");
    println!("• 🔄 Versioning and snapshot management");
    println!("• ⚡ Quick session save/load");
    println!("• 🔍 Context diffing and change tracking");
    println!("• 📊 Project analytics and statistics");
    println!("• 🗜️ Compression and deduplication");
    println!("• 🧹 Automatic cleanup and maintenance");

    Ok(())
}

/// Create a demo project with various file types
async fn create_demo_project() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let temp_dir = std::env::temp_dir().join("rag_redis_demo");
    fs::create_dir_all(&temp_dir).await?;

    // Main Rust files
    fs::write(temp_dir.join("main.rs"), r#"
use std::io;

fn main() {
    println!("Welcome to RAG-Redis Demo Project!");

    let mut input = String::new();
    println!("Enter your name: ");
    io::stdin().read_line(&mut input).expect("Failed to read input");

    println!("Hello, {}!", input.trim());
}
"#).await?;

    fs::write(temp_dir.join("lib.rs"), r#"
//! Demo library for RAG-Redis project context system

pub mod auth;
pub mod utils;

/// Add two numbers together
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Multiply two numbers
pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
    }

    #[test]
    fn test_multiply() {
        assert_eq!(multiply(4, 5), 20);
    }
}
"#).await?;

    // Create subdirectories
    fs::create_dir_all(temp_dir.join("src")).await?;
    fs::create_dir_all(temp_dir.join("tests")).await?;
    fs::create_dir_all(temp_dir.join("docs")).await?;

    // Auth module
    fs::write(temp_dir.join("src").join("auth.rs"), r#"
//! Authentication module

use std::collections::HashMap;

pub struct User {
    pub id: u32,
    pub username: String,
    pub email: String,
}

pub struct AuthManager {
    users: HashMap<String, User>,
}

impl AuthManager {
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
        }
    }

    pub fn register_user(&mut self, username: String, email: String) -> Result<u32, String> {
        if self.users.contains_key(&username) {
            return Err("User already exists".to_string());
        }

        let id = self.users.len() as u32 + 1;
        let user = User { id, username: username.clone(), email };
        self.users.insert(username, user);
        Ok(id)
    }

    pub fn authenticate(&self, username: &str) -> Option<&User> {
        self.users.get(username)
    }
}
"#).await?;

    // Utils module
    fs::write(temp_dir.join("src").join("utils.rs"), r#"
//! Utility functions

use std::time::{SystemTime, UNIX_EPOCH};

/// Get current timestamp
pub fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs()
}

/// Format file size in human readable format
pub fn format_file_size(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.1} {}", size, UNITS[unit_index])
}
"#).await?;

    // Configuration files
    fs::write(temp_dir.join("Cargo.toml"), r#"
[package]
name = "rag-redis-demo"
version = "0.1.0"
edition = "2021"
authors = ["RAG-Redis Demo <demo@rag-redis.com>"]
description = "Demo project for RAG-Redis context system"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
uuid = "1.0"
chrono = { version = "0.4", features = ["serde"] }

[dev-dependencies]
tempfile = "3.0"
"#).await?;

    fs::write(temp_dir.join("README.md"), r#"
# RAG-Redis Demo Project

This is a demonstration project for the RAG-Redis project context storage and retrieval system.

## Features

- User authentication system
- Utility functions
- Comprehensive test coverage
- Configuration management

## Building

```bash
cargo build --release
```

## Running

```bash
cargo run
```

## Testing

```bash
cargo test
```

## Project Structure

- `main.rs` - Main application entry point
- `lib.rs` - Library with core functionality
- `src/auth.rs` - Authentication module
- `src/utils.rs` - Utility functions
- `tests/` - Integration tests
- `docs/` - Documentation

## Context System Demo

This project demonstrates the following RAG-Redis context features:

1. **Complete State Capture** - All files, configurations, and metadata
2. **Versioning** - Semantic versioning with snapshot management
3. **Session Management** - Quick save/load for LLM sessions
4. **Change Tracking** - Git-like diffing between contexts
5. **Compression** - Automatic compression and deduplication
6. **Analytics** - Project statistics and trends
7. **Cleanup** - Automatic maintenance and cleanup
"#).await?;

    // Test files
    fs::write(temp_dir.join("tests").join("integration_test.rs"), r#"
use rag_redis_demo::*;

#[test]
fn test_integration() {
    assert_eq!(add(1, 2), 3);
    assert_eq!(multiply(3, 4), 12);
}
"#).await?;

    // Documentation
    fs::write(temp_dir.join("docs").join("API.md"), r#"
# API Documentation

## Functions

### `add(a: i32, b: i32) -> i32`

Adds two integers together.

### `multiply(a: i32, b: i32) -> i32`

Multiplies two integers together.

## Modules

### `auth`

Provides user authentication functionality.

### `utils`

Contains utility functions for common operations.
"#).await?;

    // Environment file
    fs::write(temp_dir.join(".env"), r#"
DATABASE_URL=postgresql://localhost:5432/rag_redis_demo
REDIS_URL=redis://127.0.0.1:6379
LOG_LEVEL=info
PORT=8080
"#).await?;

    // Git ignore
    fs::write(temp_dir.join(".gitignore"), r#"
target/
*.tmp
*.log
.env.local
"#).await?;

    Ok(temp_dir)
}

/// Modify the demo project to simulate development changes
async fn modify_demo_project(project_root: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    // Modify main.rs to add more functionality
    fs::write(project_root.join("main.rs"), r#"
use std::io;
use rag_redis_demo::auth::AuthManager;

fn main() {
    println!("Welcome to RAG-Redis Demo Project v2.0!");

    let mut auth_manager = AuthManager::new();

    // Demo user registration
    match auth_manager.register_user("demo_user".to_string(), "demo@example.com".to_string()) {
        Ok(user_id) => println!("User registered with ID: {}", user_id),
        Err(e) => println!("Registration failed: {}", e),
    }

    let mut input = String::new();
    println!("Enter your username: ");
    io::stdin().read_line(&mut input).expect("Failed to read input");

    let username = input.trim();
    if let Some(user) = auth_manager.authenticate(username) {
        println!("Welcome back, {}! (ID: {})", user.username, user.id);
    } else {
        println!("User not found: {}", username);
    }
}
"#).await?;

    // Add a new feature file
    fs::write(project_root.join("src").join("analytics.rs"), r#"
//! Analytics module for tracking user behavior

use std::collections::HashMap;
use crate::utils::current_timestamp;

pub struct Analytics {
    events: Vec<Event>,
    metrics: HashMap<String, u64>,
}

pub struct Event {
    pub name: String,
    pub timestamp: u64,
    pub user_id: Option<u32>,
    pub metadata: HashMap<String, String>,
}

impl Analytics {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            metrics: HashMap::new(),
        }
    }

    pub fn track_event(&mut self, name: String, user_id: Option<u32>) {
        let event = Event {
            name: name.clone(),
            timestamp: current_timestamp(),
            user_id,
            metadata: HashMap::new(),
        };

        self.events.push(event);

        // Update metrics
        let counter = self.metrics.entry(name).or_insert(0);
        *counter += 1;
    }

    pub fn get_metric(&self, name: &str) -> u64 {
        self.metrics.get(name).copied().unwrap_or(0)
    }
}
"#).await?;

    // Update lib.rs to include the new module
    fs::write(project_root.join("lib.rs"), r#"
//! Demo library for RAG-Redis project context system

pub mod auth;
pub mod utils;
pub mod analytics;

/// Add two numbers together
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Multiply two numbers
pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}

/// New function: Calculate factorial
pub fn factorial(n: u32) -> u64 {
    match n {
        0 | 1 => 1,
        _ => n as u64 * factorial(n - 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
    }

    #[test]
    fn test_multiply() {
        assert_eq!(multiply(4, 5), 20);
    }

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1);
        assert_eq!(factorial(1), 1);
        assert_eq!(factorial(5), 120);
    }
}
"#).await?;

    // Update Cargo.toml version
    fs::write(project_root.join("Cargo.toml"), r#"
[package]
name = "rag-redis-demo"
version = "0.2.0"
edition = "2021"
authors = ["RAG-Redis Demo <demo@rag-redis.com>"]
description = "Demo project for RAG-Redis context system - Updated with analytics"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
uuid = "1.0"
chrono = { version = "0.4", features = ["serde"] }

[dev-dependencies]
tempfile = "3.0"

[features]
default = []
analytics = []
"#).await?;

    Ok(())
}
"