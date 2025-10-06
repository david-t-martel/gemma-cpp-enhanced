//! Tests for project context storage and retrieval system

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::redis_backend::RedisManager;
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::fs;

    async fn setup_test_environment() -> (Arc<RedisManager>, TempDir) {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let config = Config::default();
        let redis_manager = Arc::new(RedisManager::new(&config.redis).await.expect("Failed to create Redis manager"));
        (redis_manager, temp_dir)
    }

    async fn create_test_project(temp_dir: &TempDir) -> std::path::PathBuf {
        let project_root = temp_dir.path().join("test_project");
        fs::create_dir_all(&project_root).await.expect("Failed to create project directory");

        // Create some test files
        fs::write(project_root.join("main.rs"), "fn main() { println!(\"Hello, world!\"); }")
            .await.expect("Failed to write main.rs");

        fs::write(project_root.join("Cargo.toml"), r#"
[package]
name = "test_project"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = "1.0"
"#).await.expect("Failed to write Cargo.toml");

        fs::create_dir_all(project_root.join("src")).await.expect("Failed to create src directory");
        fs::write(project_root.join("src").join("lib.rs"), "pub fn add(a: i32, b: i32) -> i32 { a + b }")
            .await.expect("Failed to write lib.rs");

        fs::write(project_root.join("README.md"), "# Test Project\n\nThis is a test project.")
            .await.expect("Failed to write README.md");

        project_root
    }

    #[tokio::test]
    async fn test_save_and_load_project_context() {
        let (redis_manager, temp_dir) = setup_test_environment().await;
        let project_root = create_test_project(&temp_dir).await;

        let context_manager = ProjectContextManager::new(redis_manager.clone())
            .await
            .expect("Failed to create context manager");

        let project_id = "test_project";
        let save_options = SaveOptions::default();

        // Save project context
        let snapshot_id = context_manager
            .save_project_context(project_id, &project_root, save_options)
            .await
            .expect("Failed to save project context");

        assert!(!snapshot_id.is_empty());

        // Load project context
        let loaded_snapshot = context_manager
            .load_project_context(project_id, Some(&snapshot_id))
            .await
            .expect("Failed to load project context");

        assert_eq!(loaded_snapshot.project_id, project_id);
        assert_eq!(loaded_snapshot.id, snapshot_id);
        assert!(loaded_snapshot.project_files.files.len() > 0);
        assert!(loaded_snapshot.project_files.files.contains_key(&std::path::PathBuf::from("main.rs")));
        assert!(loaded_snapshot.project_files.files.contains_key(&std::path::PathBuf::from("Cargo.toml")));
    }

    #[tokio::test]
    async fn test_quick_save_and_load_session() {
        let (redis_manager, temp_dir) = setup_test_environment().await;
        let project_root = create_test_project(&temp_dir).await;

        let context_manager = ProjectContextManager::new(redis_manager.clone())
            .await
            .expect("Failed to create context manager");

        let project_id = "test_project";
        let description = "Working on main function";

        // Set current directory to project root for quick save
        std::env::set_current_dir(&project_root).expect("Failed to set current directory");

        // Quick save session
        let session_id = context_manager
            .quick_save_session(project_id, description)
            .await
            .expect("Failed to quick save session");

        assert!(!session_id.is_empty());

        // Quick load session
        let loaded_snapshot = context_manager
            .quick_load_session(project_id, &session_id)
            .await
            .expect("Failed to quick load session");

        assert_eq!(loaded_snapshot.project_id, project_id);
        assert_eq!(loaded_snapshot.description, Some(description.to_string()));
        assert!(matches!(loaded_snapshot.snapshot_metadata.snapshot_type, SnapshotType::Session));
    }

    #[tokio::test]
    async fn test_list_project_snapshots() {
        let (redis_manager, temp_dir) = setup_test_environment().await;
        let project_root = create_test_project(&temp_dir).await;

        let context_manager = ProjectContextManager::new(redis_manager.clone())
            .await
            .expect("Failed to create context manager");

        let project_id = "test_project";

        // Create multiple snapshots
        let snapshot1 = context_manager
            .save_project_context(project_id, &project_root, SaveOptions::default())
            .await
            .expect("Failed to save first snapshot");

        let snapshot2 = context_manager
            .save_project_context(project_id, &project_root, SaveOptions {
                description: Some("Second snapshot".to_string()),
                ..Default::default()
            })
            .await
            .expect("Failed to save second snapshot");

        // List snapshots
        let snapshots = context_manager
            .list_project_snapshots(project_id, None)
            .await
            .expect("Failed to list snapshots");

        assert_eq!(snapshots.len(), 2);
        assert!(snapshots.iter().any(|s| s.id == snapshot1));
        assert!(snapshots.iter().any(|s| s.id == snapshot2));

        // Test with limit
        let limited_snapshots = context_manager
            .list_project_snapshots(project_id, Some(1))
            .await
            .expect("Failed to list limited snapshots");

        assert_eq!(limited_snapshots.len(), 1);
    }

    #[tokio::test]
    async fn test_diff_contexts() {
        let (redis_manager, temp_dir) = setup_test_environment().await;
        let project_root = create_test_project(&temp_dir).await;

        let context_manager = ProjectContextManager::new(redis_manager.clone())
            .await
            .expect("Failed to create context manager");

        let project_id = "test_project";

        // Save first snapshot
        let snapshot1_id = context_manager
            .save_project_context(project_id, &project_root, SaveOptions::default())
            .await
            .expect("Failed to save first snapshot");

        // Modify a file
        fs::write(project_root.join("main.rs"), "fn main() { println!(\"Hello, Rust!\"); }")
            .await
            .expect("Failed to modify main.rs");

        // Add a new file
        fs::write(project_root.join("lib.rs"), "pub fn multiply(a: i32, b: i32) -> i32 { a * b }")
            .await
            .expect("Failed to add lib.rs");

        // Save second snapshot
        let snapshot2_id = context_manager
            .save_project_context(project_id, &project_root, SaveOptions::default())
            .await
            .expect("Failed to save second snapshot");

        // Load snapshots to get versions
        let snapshot1 = context_manager
            .load_project_context(project_id, Some(&snapshot1_id))
            .await
            .expect("Failed to load first snapshot");

        let snapshot2 = context_manager
            .load_project_context(project_id, Some(&snapshot2_id))
            .await
            .expect("Failed to load second snapshot");

        // Generate diff
        let diff = context_manager
            .diff_contexts(project_id, &snapshot1.version, &snapshot2.version)
            .await
            .expect("Failed to generate diff");

        assert!(diff.file_changes.added.len() > 0 || diff.file_changes.modified.len() > 0);
        assert!(diff.summary.total_changes > 0);
    }

    #[tokio::test]
    async fn test_project_statistics() {
        let (redis_manager, temp_dir) = setup_test_environment().await;
        let project_root = create_test_project(&temp_dir).await;

        let context_manager = ProjectContextManager::new(redis_manager.clone())
            .await
            .expect("Failed to create context manager");

        let project_id = "test_project";

        // Create several snapshots
        for i in 0..3 {
            context_manager
                .save_project_context(project_id, &project_root, SaveOptions {
                    description: Some(format!("Snapshot {}", i + 1)),
                    ..Default::default()
                })
                .await
                .expect("Failed to save snapshot");
        }

        // Get statistics
        let stats = context_manager
            .get_project_statistics(project_id)
            .await
            .expect("Failed to get project statistics");

        assert_eq!(stats.project_id, project_id);
        assert_eq!(stats.total_snapshots, 3);
        assert!(stats.total_storage_size > 0);
        assert!(stats.oldest_snapshot.is_some());
        assert!(stats.newest_snapshot.is_some());
    }

    #[tokio::test]
    async fn test_compression_and_deduplication() {
        let (redis_manager, temp_dir) = setup_test_environment().await;
        let project_root = create_test_project(&temp_dir).await;

        let context_manager = ProjectContextManager::new(redis_manager.clone())
            .await
            .expect("Failed to create context manager");

        // Create a large file that should be compressed
        let large_content = "a".repeat(10000);
        fs::write(project_root.join("large_file.txt"), &large_content)
            .await
            .expect("Failed to write large file");

        let project_id = "test_project";
        let save_options = SaveOptions {
            compress: true,
            deduplicate: true,
            ..Default::default()
        };

        // Save project context with compression and deduplication
        let snapshot_id = context_manager
            .save_project_context(project_id, &project_root, save_options)
            .await
            .expect("Failed to save project context");

        // Load and verify compression was applied
        let loaded_snapshot = context_manager
            .load_project_context(project_id, Some(&snapshot_id))
            .await
            .expect("Failed to load project context");

        assert!(loaded_snapshot.snapshot_metadata.compression_ratio < 1.0);
        assert!(loaded_snapshot.project_files.files.contains_key(&std::path::PathBuf::from("large_file.txt")));
    }

    #[tokio::test]
    async fn test_file_filtering() {
        let (redis_manager, temp_dir) = setup_test_environment().await;
        let project_root = create_test_project(&temp_dir).await;

        // Create files that should be excluded
        fs::write(project_root.join("temp.tmp"), "temporary file").await.expect("Failed to write temp file");
        fs::write(project_root.join("debug.log"), "log content").await.expect("Failed to write log file");

        let context_manager = ProjectContextManager::new(redis_manager.clone())
            .await
            .expect("Failed to create context manager");

        let project_id = "test_project";
        let save_options = SaveOptions {
            exclude_patterns: Some(vec!["*.tmp".to_string(), "*.log".to_string()]),
            ..Default::default()
        };

        // Save project context with exclusion patterns
        let snapshot_id = context_manager
            .save_project_context(project_id, &project_root, save_options)
            .await
            .expect("Failed to save project context");

        // Load and verify excluded files are not present
        let loaded_snapshot = context_manager
            .load_project_context(project_id, Some(&snapshot_id))
            .await
            .expect("Failed to load project context");

        assert!(!loaded_snapshot.project_files.files.contains_key(&std::path::PathBuf::from("temp.tmp")));
        assert!(!loaded_snapshot.project_files.files.contains_key(&std::path::PathBuf::from("debug.log")));
        assert!(loaded_snapshot.project_files.files.contains_key(&std::path::PathBuf::from("main.rs")));
    }

    #[tokio::test]
    async fn test_cleanup_old_snapshots() {
        let (redis_manager, temp_dir) = setup_test_environment().await;
        let project_root = create_test_project(&temp_dir).await;

        let mut context_manager = ProjectContextManager::new(redis_manager.clone())
            .await
            .expect("Failed to create context manager");

        // Set a very short max age for testing
        context_manager.max_snapshot_age = chrono::Duration::seconds(1);

        let project_id = "test_project";

        // Create a snapshot
        let snapshot_id = context_manager
            .save_project_context(project_id, &project_root, SaveOptions::default())
            .await
            .expect("Failed to save project context");

        // Wait for snapshot to be old enough
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Create another snapshot (milestone type should not be cleaned up)
        context_manager
            .save_project_context(project_id, &project_root, SaveOptions {
                snapshot_type: SnapshotType::Milestone,
                ..Default::default()
            })
            .await
            .expect("Failed to save milestone snapshot");

        // Run cleanup
        let cleaned_count = context_manager
            .cleanup_old_snapshots(project_id)
            .await
            .expect("Failed to cleanup old snapshots");

        // Should have cleaned up the old non-milestone snapshot
        assert_eq!(cleaned_count, 1);

        // Verify the snapshot was actually deleted
        let load_result = context_manager.load_project_context(project_id, Some(&snapshot_id)).await;
        assert!(load_result.is_err());
    }

    #[tokio::test]
    async fn test_integrity_validation() {
        let (redis_manager, temp_dir) = setup_test_environment().await;
        let project_root = create_test_project(&temp_dir).await;

        let context_manager = ProjectContextManager::new(redis_manager.clone())
            .await
            .expect("Failed to create context manager");

        let project_id = "test_project";

        // Save project context
        let snapshot_id = context_manager
            .save_project_context(project_id, &project_root, SaveOptions::default())
            .await
            .expect("Failed to save project context");

        // Load project context - should pass integrity check
        let loaded_snapshot = context_manager
            .load_project_context(project_id, Some(&snapshot_id))
            .await
            .expect("Failed to load project context");

        assert_eq!(loaded_snapshot.id, snapshot_id);
        assert!(!loaded_snapshot.snapshot_metadata.validation_hash.is_empty());
    }
}