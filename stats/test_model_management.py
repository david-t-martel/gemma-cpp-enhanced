#!/usr/bin/env python3
"""Test script for model management system.

Tests ModelManager, ProfileManager, and HardwareDetector functionality.
"""

from pathlib import Path

from rich.console import Console

from src.gemma_cli.config.models import HardwareDetector, ModelManager, ProfileManager

console = Console()


def test_hardware_detector():
    """Test HardwareDetector functionality."""
    console.print("\n[bold cyan]Testing HardwareDetector[/bold cyan]")

    detector = HardwareDetector()

    # Test CPU detection
    cpu_info = detector.detect_cpu()
    console.print(f"[green]OK[/green] CPU detection: {cpu_info['physical_cores']} cores")

    # Test memory detection
    mem_info = detector.detect_memory()
    console.print(
        f"[green]OK[/green] Memory detection: {mem_info['total_gb']:.1f} GB total"
    )

    # Test GPU detection
    has_gpu, gpu_info = detector.detect_gpu()
    console.print(
        f"[green]OK[/green] GPU detection: {'Found' if has_gpu else 'Not found'}"
    )

    # Test comprehensive hardware info
    hw_info = detector.get_hardware_info()
    console.print(f"[green]OK[/green] Hardware info: {hw_info.os_system}")

    # Test display
    detector.display_hardware_info(hw_info)

    return hw_info


def test_model_manager(config_path: Path, hw_info):
    """Test ModelManager functionality."""
    console.print("\n[bold cyan]Testing ModelManager[/bold cyan]")

    mgr = ModelManager(config_path)

    # Test list models
    models = mgr.list_models()
    console.print(f"[green]OK[/green] List models: {len(models)} found")

    # Test get model
    if models:
        model = mgr.get_model(models[0].name)
        console.print(f"[green]OK[/green] Get model: {model.name if model else 'None'}")

        # Test validate model
        is_valid, errors = mgr.validate_model(models[0])
        console.print(
            f"[green]OK[/green] Validate model: "
            f"{'Valid' if is_valid else f'Invalid ({len(errors)} errors)'}"
        )

        # Test get model info
        info = mgr.get_model_info(models[0])
        console.print(f"[green]OK[/green] Get model info: {len(info)} fields")

    # Test default model
    default = mgr.get_default_model()
    console.print(
        f"[green]OK[/green] Default model: {default.name if default else 'None'}"
    )

    # Test detect models
    detected = mgr.detect_models()
    console.print(f"[green]OK[/green] Detect models: {len(detected)} found")

    # Test display
    mgr.display_models_table()

    return mgr


def test_profile_manager(config_path: Path):
    """Test ProfileManager functionality."""
    console.print("\n[bold cyan]Testing ProfileManager[/bold cyan]")

    mgr = ProfileManager(config_path)

    # Test list profiles
    profiles = mgr.list_profiles()
    console.print(f"[green]OK[/green] List profiles: {len(profiles)} found")

    # Test get profile
    if profiles:
        profile = mgr.get_profile(profiles[0].name)
        console.print(
            f"[green]OK[/green] Get profile: {profile.name if profile else 'None'}"
        )

    # Test display
    mgr.display_profiles_table()

    return mgr


def test_recommendations(detector, model_mgr, profile_mgr):
    """Test recommendation functionality."""
    console.print("\n[bold cyan]Testing Recommendations[/bold cyan]")

    hw_info = detector.get_hardware_info()

    # Test model recommendation
    recommended_model = detector.recommend_model(model_mgr, hw_info)
    console.print(
        f"[green]OK[/green] Recommended model: "
        f"{recommended_model.name if recommended_model else 'None'}"
    )

    # Test settings recommendation
    settings = detector.recommend_settings(hw_info)
    console.print(f"[green]OK[/green] Recommended settings: {len(settings)} parameters")

    # Test profile recommendation
    recommended_profile = profile_mgr.recommend_profile(hw_info)
    console.print(
        f"[green]OK[/green] Recommended profile: "
        f"{recommended_profile.name if recommended_profile else 'None'}"
    )


def main():
    """Run all tests."""
    console.print("[bold]Model Management System Test Suite[/bold]")

    # Use enhanced config
    config_path = Path("config/config_enhanced.toml")

    if not config_path.exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        return

    try:
        # Test components
        hw_info = test_hardware_detector()
        model_mgr = test_model_manager(config_path, hw_info)
        profile_mgr = test_profile_manager(config_path)
        test_recommendations(
            HardwareDetector(), model_mgr, profile_mgr
        )

        console.print("\n[bold green]All tests passed![/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]Test failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    main()
