//! Cross-compilation build script for all targets

use clap::Parser;
use anyhow::{Context, Result};
use std::process::Command;
use std::path::PathBuf;
use tokio::process::Command as AsyncCommand;

#[derive(Parser)]
#[command(name = "build-all-targets")]
#[command(about = "Build Gemma Rust components for all supported targets")]
struct Args {
    /// Targets to build (comma-separated)
    #[arg(long, default_value = "x86_64-unknown-linux-gnu,aarch64-unknown-linux-gnu,x86_64-pc-windows-msvc,x86_64-apple-darwin,aarch64-apple-darwin,wasm32-unknown-unknown")]
    targets: String,

    /// Build profile
    #[arg(long, default_value = "release")]
    profile: String,

    /// Output directory
    #[arg(long, default_value = "target/cross-builds")]
    output_dir: String,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Clean before building
    #[arg(long)]
    clean: bool,
}

/// Supported build targets with their configurations
#[derive(Debug, Clone)]
struct BuildTarget {
    name: String,
    triple: String,
    features: Vec<String>,
    env_vars: Vec<(String, String)>,
    requires_cross: bool,
}

impl BuildTarget {
    fn all_targets() -> Vec<Self> {
        vec![
            BuildTarget {
                name: "Linux x86_64".to_string(),
                triple: "x86_64-unknown-linux-gnu".to_string(),
                features: vec!["simd".to_string(), "parallel".to_string()],
                env_vars: vec![],
                requires_cross: cfg!(not(all(target_os = "linux", target_arch = "x86_64"))),
            },
            BuildTarget {
                name: "Linux ARM64".to_string(),
                triple: "aarch64-unknown-linux-gnu".to_string(),
                features: vec!["simd".to_string(), "parallel".to_string()],
                env_vars: vec![],
                requires_cross: cfg!(not(all(target_os = "linux", target_arch = "aarch64"))),
            },
            BuildTarget {
                name: "Windows x86_64".to_string(),
                triple: "x86_64-pc-windows-msvc".to_string(),
                features: vec!["simd".to_string(), "parallel".to_string()],
                env_vars: vec![],
                requires_cross: cfg!(not(all(target_os = "windows", target_arch = "x86_64"))),
            },
            BuildTarget {
                name: "macOS x86_64".to_string(),
                triple: "x86_64-apple-darwin".to_string(),
                features: vec!["simd".to_string(), "parallel".to_string(), "accelerate".to_string()],
                env_vars: vec![],
                requires_cross: cfg!(not(all(target_os = "macos", target_arch = "x86_64"))),
            },
            BuildTarget {
                name: "macOS ARM64".to_string(),
                triple: "aarch64-apple-darwin".to_string(),
                features: vec!["simd".to_string(), "parallel".to_string(), "accelerate".to_string()],
                env_vars: vec![],
                requires_cross: cfg!(not(all(target_os = "macos", target_arch = "aarch64"))),
            },
            BuildTarget {
                name: "WebAssembly".to_string(),
                triple: "wasm32-unknown-unknown".to_string(),
                features: vec!["simd".to_string(), "wasm-optimizations".to_string()],
                env_vars: vec![],
                requires_cross: cfg!(not(target_arch = "wasm32")),
            },
        ]
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("🚀 Building Gemma Rust components for all targets");
    println!("Profile: {}", args.profile);
    println!("Output directory: {}", args.output_dir);

    let target_names: Vec<&str> = args.targets.split(',').collect();
    let all_targets = BuildTarget::all_targets();
    let selected_targets: Vec<&BuildTarget> = all_targets
        .iter()
        .filter(|target| target_names.contains(&target.triple.as_str()))
        .collect();

    if selected_targets.is_empty() {
        anyhow::bail!("No valid targets found in: {}", args.targets);
    }

    println!("Selected targets: {}", selected_targets.len());
    for target in &selected_targets {
        println!("  - {} ({})", target.name, target.triple);
    }

    // Clean if requested
    if args.clean {
        println!("\n🧹 Cleaning workspace...");
        run_command(Command::new("cargo").arg("clean")).await?;
    }

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)
        .context("Failed to create output directory")?;

    // Build each target
    let mut successful_builds = 0;
    let mut failed_builds = Vec::new();

    for target in selected_targets {
        println!("\n📦 Building {} ({})", target.name, target.triple);

        match build_target(target, &args).await {
            Ok(_) => {
                successful_builds += 1;
                println!("✅ Successfully built {}", target.name);
            }
            Err(e) => {
                failed_builds.push(target.name.clone());
                eprintln!("❌ Failed to build {}: {}", target.name, e);
            }
        }
    }

    // Summary
    println!("\n📊 Build Summary");
    println!("Successful builds: {}", successful_builds);
    println!("Failed builds: {}", failed_builds.len());

    if !failed_builds.is_empty() {
        println!("Failed targets:");
        for target in failed_builds {
            println!("  - {}", target);
        }
        anyhow::bail!("Some builds failed");
    }

    println!("🎉 All builds completed successfully!");
    Ok(())
}

async fn build_target(target: &BuildTarget, args: &Args) -> Result<()> {
    // Install target if needed
    install_target(&target.triple).await?;

    // Build inference crate
    build_crate("inference", target, args).await?;

    // Build server crate
    build_crate("server", target, args).await?;

    // Build WASM crate for WASM target
    if target.triple.starts_with("wasm32") {
        build_wasm_crate(target, args).await?;
    }

    Ok(())
}

async fn install_target(target: &str) -> Result<()> {
    let output = AsyncCommand::new("rustup")
        .args(&["target", "add", target])
        .output()
        .await
        .context("Failed to run rustup")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Failed to install target {}: {}", target, stderr);
    }

    Ok(())
}

async fn build_crate(crate_name: &str, target: &BuildTarget, args: &Args) -> Result<()> {
    let mut cmd = AsyncCommand::new("cargo");

    cmd.arg("build")
       .arg("--package")
       .arg(format!("gemma-{}", crate_name))
       .arg("--target")
       .arg(&target.triple);

    if args.profile == "release" {
        cmd.arg("--release");
    }

    if !target.features.is_empty() {
        cmd.arg("--features");
        cmd.arg(target.features.join(","));
    }

    if args.verbose {
        cmd.arg("--verbose");
    }

    // Set environment variables
    for (key, value) in &target.env_vars {
        cmd.env(key, value);
    }

    let output = cmd.output().await
        .context(format!("Failed to build {} for {}", crate_name, target.name))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Build failed for {} on {}: {}", crate_name, target.name, stderr);
    }

    Ok(())
}

async fn build_wasm_crate(target: &BuildTarget, args: &Args) -> Result<()> {
    // Use wasm-pack for WASM builds
    let mut cmd = AsyncCommand::new("wasm-pack");

    cmd.arg("build")
       .arg("wasm")
       .arg("--target")
       .arg("web");

    if args.profile == "release" {
        cmd.arg("--release");
    }

    if args.verbose {
        cmd.arg("--verbose");
    }

    let output = cmd.output().await
        .context("Failed to build WASM crate")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("WASM build failed: {}", stderr);
    }

    Ok(())
}

async fn run_command(mut cmd: Command) -> Result<()> {
    let output = cmd.output()
        .context("Failed to run command")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Command failed: {}", stderr);
    }

    Ok(())
}
