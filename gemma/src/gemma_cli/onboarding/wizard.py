"""First-run onboarding wizard for gemma-cli."""

import asyncio
from pathlib import Path
from typing import Any

import toml
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.validation import ValidationError, Validator
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from .checks import (
    check_model_files,
    check_redis_connection,
    check_system_requirements,
    display_health_check_results,
)
from .templates import get_template, list_templates


class PathValidator(Validator):
    """Validator for file/directory paths."""

    def __init__(self, must_exist: bool = True, must_be_file: bool = False) -> None:
        self.must_exist = must_exist
        self.must_be_file = must_be_file

    def validate(self, document: Any) -> None:
        text = document.text
        if not text:
            return

        path = Path(text)

        if self.must_exist and not path.exists():
            raise ValidationError(
                message=f"Path does not exist: {text}",
                cursor_position=len(text),
            )

        if self.must_be_file and path.exists() and not path.is_file():
            raise ValidationError(
                message=f"Path is not a file: {text}",
                cursor_position=len(text),
            )


class OnboardingWizard:
    """Interactive setup wizard for first-time users."""

    def __init__(
        self,
        console: Console | None = None,
        config_path: Path | None = None,
    ) -> None:
        """
        Initialize the onboarding wizard.

        Args:
            console: Rich Console instance for output (injected dependency)
            config_path: Path where config will be saved (default: ~/.gemma_cli/config.toml)
        """
        self.console = console or Console()  # Fallback for backward compatibility
        self.session = PromptSession()
        self.config: dict[str, Any] = {}
        self.config_path = (
            config_path or Path.home() / ".gemma_cli" / "config.toml"
        )

    async def run(self) -> dict[str, Any]:
        """
        Run the onboarding wizard.

        Returns:
            Generated configuration dictionary
        """
        # Display welcome screen
        self._display_welcome()

        # Check for existing configuration
        if self.config_path.exists():
            self.console.print(
                f"\n[yellow]Found existing configuration at {self.config_path}[/yellow]"
            )
            if not Confirm.ask("Do you want to reconfigure?", default=False):
                self.console.print("[green]Keeping existing configuration.[/green]")
                return {}

        # Step 1: Run system checks
        self.console.print("\n[bold cyan]Step 1/6: System Health Check[/bold cyan]")
        await self._run_health_checks()

        # Step 2: Model selection
        self.console.print("\n[bold cyan]Step 2/6: Model Selection[/bold cyan]")
        model_config = await self._step_model_selection()

        # Step 3: Redis configuration
        self.console.print("\n[bold cyan]Step 3/6: Redis Configuration[/bold cyan]")
        redis_config = await self._step_redis_config()

        # Step 4: Performance profile
        self.console.print("\n[bold cyan]Step 4/6: Performance Profile[/bold cyan]")
        profile_config = await self._step_performance_profile()

        # Step 5: UI preferences
        self.console.print("\n[bold cyan]Step 5/6: UI Preferences[/bold cyan]")
        ui_config = await self._step_ui_preferences()

        # Step 6: Optional features
        self.console.print("\n[bold cyan]Step 6/6: Optional Features[/bold cyan]")
        features_config = await self._step_optional_features()

        # Merge all configurations
        self.config = self._merge_configurations(
            model_config, redis_config, profile_config, ui_config, features_config
        )

        # Test configuration
        self.console.print("\n[bold cyan]Testing Configuration...[/bold cyan]")
        test_passed = await self._test_configuration(self.config)

        if test_passed:
            # Save configuration
            self._save_configuration()

            # Display success message
            self._display_success()

            # Ask if user wants tutorial
            if Confirm.ask("\nWould you like to run the interactive tutorial?"):
                from .tutorial import InteractiveTutorial

                tutorial = InteractiveTutorial()
                await tutorial.run()
        else:
            self.console.print(
                "\n[yellow]Configuration test failed. Please review settings.[/yellow]"
            )

        return self.config

    def _display_welcome(self) -> None:
        """Display welcome screen."""
        welcome_text = """
        [bold cyan]Welcome to Gemma CLI![/bold cyan]

        This wizard will help you configure gemma-cli for first use.
        We'll guide you through:

        • System health checks
        • Model selection
        • Redis/RAG configuration
        • Performance tuning
        • UI customization
        • Optional features

        The process takes about 5 minutes.
        You can reconfigure anytime by running: gemma-cli init --force
        """

        self.console.print(Panel(welcome_text, border_style="cyan", padding=(1, 2)))

    async def _run_health_checks(self) -> None:
        """Run system health checks."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Running system checks...", total=None)
            checks = await check_system_requirements()
            progress.update(task, completed=True)

        all_passed = display_health_check_results(checks)

        if not all_passed:
            self.console.print(
                "\n[yellow]Some checks failed. You can continue, but some features may not work.[/yellow]"
            )
            if not Confirm.ask("Continue anyway?", default=True):
                raise SystemExit(0)

    async def _step_model_selection(self) -> dict[str, Any]:
        """Guide user through model selection."""
        self.console.print("\n[bold]Model Configuration[/bold]")
        self.console.print(
            "Select a Gemma model for inference. Models should be in .sbs format.\n"
        )
        # TODO: [Model Management] Integrate model download functionality here.
        # Offer to download recommended models if none are found or if user prefers.
        # This is crucial for a standalone local LLM TUI.

        # Show common model locations
        common_paths = [
            Path("C:/codedev/llm/.models"),
            Path.home() / ".cache" / "gemma",
            Path.home() / "models",
        ]

        self.console.print("[dim]Common model locations:[/dim]")
        for path in common_paths:
            if path.exists():
                sbs_files = list(path.glob("**/*.sbs"))
                if sbs_files:
                    self.console.print(f"  [green]✓[/green] {path} ({len(sbs_files)} models)")
                else:
                    self.console.print(f"  [dim]• {path} (empty)[/dim]")
            else:
                self.console.print(f"  [dim]• {path} (not found)[/dim]")

        # Detect available models
        available_models = self._detect_available_models()

        if not available_models:
            self.console.print("\n[yellow]No local models found.[/yellow]")
            if Confirm.ask("\nDownload the recommended default model (gemma-2b-it-sfp)?"):
                # TODO: [Model Management] Call the download logic here.
                # For now, we'll just print a message.
                self.console.print("\n[green]Model download functionality to be implemented.[/green]")
                # After download, we would need to re-detect models or directly use the downloaded path.
                # For now, we'll proceed to manual entry.

        elif available_models:
            self.console.print(f"\n[bold]Found {len(available_models)} model(s):[/bold]")
            table = Table(show_header=True)
            table.add_column("ID", style="cyan", width=4)
            table.add_column("Model", style="white")
            table.add_column("Size", style="yellow", justify="right")
            table.add_column("Tokenizer", style="green")

            for i, model_info in enumerate(available_models, 1):
                size_mb = model_info["size_mb"]
                has_tokenizer = "✓" if model_info["tokenizer"] else "✗"
                table.add_row(
                    str(i),
                    str(model_info["path"]),
                    f"{size_mb:.0f} MB",
                    has_tokenizer,
                )

            self.console.print(table)

            # Let user select from available models
            if Confirm.ask("\nUse one of these models?", default=True):
                choice = IntPrompt.ask(
                    "Select model number",
                    default=1,
                    show_default=True,
                )
                if 1 <= choice <= len(available_models):
                    selected = available_models[choice - 1]
                    return {
                        "gemma": {
                            "default_model": str(selected["path"]),
                            "default_tokenizer": str(selected["tokenizer"])
                            if selected["tokenizer"]
                            else "",
                            # TODO: [Executable Discovery] The _find_gemma_executable call here should be more robust,
                            # potentially pointing to a bundled uvx wrapper or a known installation path.
                            "executable": self._find_gemma_executable(),
                        }
                    }

        # Manual path entry
        self.console.print("\n[bold]Enter model path manually:[/bold]")
        while True:
            model_path = Prompt.ask("Model path (.sbs file or directory)")

            success, message = await check_model_files(model_path)
            if success:
                self.console.print(f"[green]✓ {message}[/green]")

                # Try to find tokenizer
                path = Path(model_path)
                tokenizer_path = ""

                if path.is_dir():
                    spm_files = list(path.glob("*.spm"))
                    if spm_files:
                        tokenizer_path = str(spm_files[0])
                elif path.is_file():
                    # Look for tokenizer in same directory
                    spm_files = list(path.parent.glob("*.spm"))
                    if spm_files:
                        tokenizer_path = str(spm_files[0])

                if not tokenizer_path:
                    if Confirm.ask("Specify tokenizer path?", default=False):
                        tokenizer_path = Prompt.ask("Tokenizer path (.spm file)")

                return {
                    "gemma": {
                        "default_model": str(model_path),
                        "default_tokenizer": tokenizer_path,
                        "executable": self._find_gemma_executable(),
                    }
                }
            else:
                self.console.print(f"[red]✗ {message}[/red]")
                if not Confirm.ask("Try again?", default=True):
                    raise SystemExit(0)

    def _detect_available_models(self) -> list[dict[str, Any]]:
        """Detect available Gemma models."""
        models = []
        search_paths = [
            Path("C:/codedev/llm/.models"),
            Path.home() / ".cache" / "gemma",
            Path.home() / "models",
        ]

        for search_path in search_paths:
            if not search_path.exists():
                continue

            for sbs_file in search_path.glob("**/*.sbs"):
                # Find corresponding tokenizer
                tokenizer = None
                spm_files = list(sbs_file.parent.glob("*.spm"))
                if spm_files:
                    tokenizer = spm_files[0]

                size_mb = sbs_file.stat().st_size / (1024**2)

                models.append(
                    {
                        "path": sbs_file,
                        "tokenizer": tokenizer,
                        "size_mb": size_mb,
                    }
                )

        return models

    def _find_gemma_executable(self) -> str:
        """Find gemma.exe executable."""
        possible_paths = [
            Path("C:/codedev/llm/gemma/build/Release/gemma.exe"),
            Path("C:/codedev/llm/gemma/build-avx2-sycl/bin/RELEASE/gemma.exe"),
            Path("./build/Release/gemma.exe"),
            Path("./gemma.exe"),
        ]

        for path in possible_paths:
            if path.exists():
                return str(path.resolve())

        return "gemma.exe"

    async def _step_redis_config(self) -> dict[str, Any]:
        """Configure Redis connection or embedded store."""
        self.console.print("\n[bold]Memory Storage Configuration[/bold]")
        self.console.print(
            "The 5-tier memory system and RAG capabilities can use either:\n"
            "  • [green]Embedded storage[/green] (default) - File-based, no setup required\n"
            "  • [cyan]Redis[/cyan] (optional) - For distributed access or large datasets\n"
        )

        # Default to embedded store (standalone mode)
        self.console.print("[dim]Checking for Redis (optional)...[/dim]")
        success, message = await check_redis_connection("localhost", 6379)

        if success:
            self.console.print(f"[green]✓ Redis detected: {message}[/green]")
            if Confirm.ask("Use Redis for memory storage?", default=False):
                return {
                    "redis": {
                        "host": "localhost",
                        "port": 6379,
                        "db": 0,
                        "enable_fallback": False,  # Use Redis
                    }
                }
        else:
            self.console.print(f"[dim]✗ {message}[/dim]")

        # Manual Redis configuration (for advanced users)
        if Confirm.ask("\nConfigure Redis manually? (advanced)", default=False):
            host = Prompt.ask("Redis host", default="localhost")
            port = IntPrompt.ask("Redis port", default=6379)
            db = IntPrompt.ask("Redis database", default=0)

            # Test connection
            self.console.print(f"\nTesting connection to {host}:{port}...")
            success, message = await check_redis_connection(host, port)

            if success:
                self.console.print(f"[green]✓ {message}[/green]")
                return {
                    "redis": {
                        "host": host,
                        "port": port,
                        "db": db,
                        "pool_size": 10,
                        "connection_timeout": 5,
                        "enable_fallback": False,  # Use Redis
                    }
                }
            else:
                self.console.print(f"[red]✗ {message}[/red]")
                self.console.print(
                    "[yellow]Falling back to embedded storage.[/yellow]"
                )

        # Default to embedded store
        self.console.print("[green]Using embedded file-based storage (recommended for local use)[/green]")
        return {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "enable_fallback": True,  # Use embedded store
            }
        }

    async def _step_performance_profile(self) -> dict[str, Any]:
        """Select performance profile."""
        self.console.print("\n[bold]Performance Profile[/bold]")
        self.console.print("Choose a configuration template:\n")

        templates = list_templates()
        table = Table(show_header=True)
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Profile", style="white", width=20)
        table.add_column("Description", style="dim")

        for i, (key, name, desc) in enumerate(templates, 1):
            table.add_row(str(i), name, desc)

        self.console.print(table)

        choice = IntPrompt.ask(
            "\nSelect profile",
            default=1,
            show_default=True,
        )

        if 1 <= choice <= len(templates):
            template_key = templates[choice - 1][0]
            return get_template(template_key)["config"]

        # Default to minimal
        return get_template("minimal")["config"]

    async def _step_ui_preferences(self) -> dict[str, Any]:
        """Configure UI preferences."""
        self.console.print("\n[bold]UI Preferences[/bold]")

        show_stats = Confirm.ask("Show memory statistics?", default=True)
        show_perf = Confirm.ask("Show performance metrics?", default=True)
        show_status = Confirm.ask("Show status bar?", default=True)

        theme_completer = WordCompleter(
            ["default", "monokai", "dracula", "solarized"], ignore_case=True
        )
        theme = await self.session.prompt_async(
            "Color theme (default/monokai/dracula/solarized): ",
            completer=theme_completer,
            default="default",
        )

        return {
            "ui": {
                "theme": theme,
                "show_memory_stats": show_stats,
                "show_performance": show_perf,
                "show_status_bar": show_status,
                "progress_style": "rich",
                "color_scheme": "auto",
            }
        }

    async def _step_optional_features(self) -> dict[str, Any]:
        """Configure optional features."""
        self.console.print("\n[bold]Optional Features[/bold]")

        enable_mcp = Confirm.ask(
            "Enable MCP (Model Context Protocol) integration?", default=True
        )
        enable_rag = Confirm.ask("Enable RAG context in responses?", default=True)
        enable_monitoring = Confirm.ask(
            "Enable performance monitoring?", default=True
        )

        return {
            "mcp": {"enabled": enable_mcp},
            "system": {"enable_rag_context": enable_rag},
            "monitoring": {"enabled": enable_monitoring},
        }

    def _merge_configurations(self, *configs: dict[str, Any]) -> dict[str, Any]:
        """Merge multiple configuration dictionaries."""
        merged: dict[str, Any] = {}

        for config in configs:
            for key, value in config.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key].update(value)
                else:
                    merged[key] = value

        return merged

    async def _test_configuration(self, config: dict[str, Any]) -> bool:
        """
        Test generated configuration.

        Args:
            config: Configuration to test

        Returns:
            True if tests passed
        """
        all_passed = True

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            # Test model loading
            task1 = progress.add_task("Testing model configuration...", total=None)
            if "gemma" in config:
                model_path = config["gemma"].get("default_model", "")
                if model_path:
                    success, msg = await check_model_files(model_path)
                    if success:
                        self.console.print(f"  [green]✓ Model: {msg}[/green]")
                    else:
                        self.console.print(f"  [red]✗ Model: {msg}[/red]")
                        all_passed = False
            progress.update(task1, completed=True)

            # Test memory storage configuration
            task2 = progress.add_task("Testing memory storage...", total=None)
            if "redis" in config:
                redis_cfg = config["redis"]
                if redis_cfg.get("enable_fallback", True):
                    self.console.print(f"  [green]✓ Memory: Using embedded storage (standalone mode)[/green]")
                else:
                    success, msg = await check_redis_connection(
                        redis_cfg.get("host", "localhost"), redis_cfg.get("port", 6379)
                    )
                    if success:
                        self.console.print(f"  [green]✓ Memory: Redis connected - {msg}[/green]")
                    else:
                        self.console.print(f"  [yellow]⚠ Memory: Redis unavailable, will use embedded storage[/yellow]")
            progress.update(task2, completed=True)

        return all_passed

    def _save_configuration(self) -> None:
        """Save configuration to file."""
        # Create directory if needed
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as TOML
        with open(self.config_path, "w", encoding="utf-8") as f:
            toml.dump(self.config, f)

        self.console.print(
            f"\n[green]✓ Configuration saved to {self.config_path}[/green]"
        )

    def _display_success(self) -> None:
        """Display success message."""
        success_text = f"""
        [bold green]Setup Complete![/bold green]

        Configuration saved to: [cyan]{self.config_path}[/cyan]

        [bold]Next Steps:[/bold]
        1. Start gemma-cli: [cyan]gemma-cli chat[/cyan]
        2. View help: [cyan]gemma-cli --help[/cyan]
        3. Check status: [cyan]gemma-cli health[/cyan]

        [dim]To reconfigure: gemma-cli init --force[/dim]
        """

        self.console.print(Panel(success_text, border_style="green", padding=(1, 2)))
