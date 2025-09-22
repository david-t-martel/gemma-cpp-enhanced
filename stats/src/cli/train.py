"""Training and fine-tuning commands for Gemma models.

This module provides comprehensive training capabilities including
data preparation, fine-tuning, evaluation, and model management.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeRemainingColumn
from rich.table import Table
from rich.tree import Tree

from .utils import format_size
from .utils import get_console
from .utils import handle_exceptions

# Create training subcommand app
train_app = typer.Typer(
    name="train", help="🎓 Training and fine-tuning commands", rich_markup_mode="rich"
)

console = get_console()


@train_app.command("prepare")
@handle_exceptions(console)
def prepare_data(
    input_path: Annotated[Path, typer.Argument(help="Input data file or directory")],
    output_path: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory for processed data")
    ],
    format_type: Annotated[
        str, typer.Option("--format", "-f", help="Data format: jsonl, csv, parquet, text")
    ] = "jsonl",
    split_ratio: Annotated[
        str, typer.Option("--split", help="Train/val/test split ratio (e.g., '0.8,0.1,0.1')")
    ] = "0.8,0.2,0.0",
    max_sequence_length: Annotated[
        int, typer.Option("--max-length", help="Maximum sequence length for tokenization")
    ] = 2048,
    tokenizer_name: Annotated[
        str | None, typer.Option("--tokenizer", help="Tokenizer to use for preprocessing")
    ] = None,
    chunk_size: Annotated[int, typer.Option("--chunk-size", help="Processing chunk size")] = 1000,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose output")] = False,
) -> None:
    """Prepare training data from various formats."""
    asyncio.run(
        _prepare_training_data(
            input_path=input_path,
            output_path=output_path,
            format_type=format_type,
            split_ratio=split_ratio,
            max_sequence_length=max_sequence_length,
            tokenizer_name=tokenizer_name,
            chunk_size=chunk_size,
            verbose=verbose,
        )
    )


async def _prepare_training_data(
    input_path: Path,
    output_path: Path,
    format_type: str,
    split_ratio: str,
    max_sequence_length: int,
    tokenizer_name: str | None,
    chunk_size: int,
    verbose: bool,
) -> None:
    """Prepare training data implementation."""
    try:
        from ..training.data_processor import DataProcessor
        from ..training.tokenization import TokenizationPipeline

        console.print(f"📊 Preparing training data from: {input_path}", style="blue")

        # Validate inputs
        if not input_path.exists():
            console.print(f"❌ Input path does not exist: {input_path}", style="red")
            raise typer.Exit(1)

        # Parse split ratio
        try:
            split_parts = [float(x.strip()) for x in split_ratio.split(",")]
            if len(split_parts) not in [2, 3]:
                raise ValueError("Split ratio must have 2 or 3 parts")
            if abs(sum(split_parts) - 1.0) > 0.01:
                raise ValueError("Split ratios must sum to 1.0")
        except ValueError as e:
            console.print(f"❌ Invalid split ratio: {e}", style="red")
            raise typer.Exit(1)

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize data processor
        processor = DataProcessor(
            output_dir=output_path,
            max_sequence_length=max_sequence_length,
            chunk_size=chunk_size,
            verbose=verbose,
        )

        # Initialize tokenization pipeline
        tokenizer = None
        if tokenizer_name:
            tokenizer = TokenizationPipeline(tokenizer_name)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # Process data
            if input_path.is_file():
                task = progress.add_task("Processing file...", total=100)
                result = await processor.process_file(
                    input_path, format_type, tokenizer, progress, task
                )
            else:
                task = progress.add_task("Processing directory...", total=100)
                result = await processor.process_directory(
                    input_path, format_type, tokenizer, progress, task
                )

            progress.update(task, completed=50, description="Splitting data...")

            # Split data
            split_result = await processor.split_data(split_parts, progress, task)

            progress.update(task, completed=100, description="Data preparation complete")

        # Show results
        _show_data_preparation_results(result, split_result, output_path, console)

    except Exception as e:
        console.print(f"❌ Data preparation failed: {e}", style="red")
        if verbose:
            import traceback

            console.print(traceback.format_exc(), style="dim red")
        raise typer.Exit(1)


def _show_data_preparation_results(
    process_result: dict[str, Any],
    split_result: dict[str, Any],
    output_path: Path,
    console: Console,
) -> None:
    """Display data preparation results."""
    # Processing results table
    process_table = Table(title="Data Processing Results")
    process_table.add_column("Metric", style="cyan")
    process_table.add_column("Value", style="white")

    process_table.add_row("Total Records", str(process_result.get("total_records", 0)))
    process_table.add_row("Valid Records", str(process_result.get("valid_records", 0)))
    process_table.add_row("Skipped Records", str(process_result.get("skipped_records", 0)))
    process_table.add_row("Average Length", f"{process_result.get('avg_length', 0):.1f} tokens")
    process_table.add_row("Total Size", format_size(process_result.get("total_size", 0)))

    console.print(process_table)

    # Split results table
    if split_result:
        console.print()
        split_table = Table(title="Data Split Results")
        split_table.add_column("Split", style="cyan")
        split_table.add_column("Records", style="white", justify="right")
        split_table.add_column("Percentage", style="blue", justify="right")
        split_table.add_column("File", style="dim")

        for split_name, split_info in split_result.items():
            split_table.add_row(
                split_name.title(),
                str(split_info["count"]),
                f"{split_info['percentage']:.1f}%",
                split_info["file"],
            )

        console.print(split_table)

    # Output files
    console.print()
    console.print("📁 Output files:", style="bold green")
    output_tree = Tree(f"[bold]{output_path}[/bold]")

    for file_path in output_path.iterdir():
        if file_path.is_file():
            size = format_size(file_path.stat().st_size)
            output_tree.add(f"[blue]{file_path.name}[/blue] [dim]({size})[/dim]")

    console.print(output_tree)


@train_app.command("finetune")
@handle_exceptions(console)
def finetune(
    data_path: Annotated[Path, typer.Argument(help="Path to training data directory")],
    model_name: Annotated[
        str | None, typer.Option("--model", "-m", help="Base model to fine-tune")
    ] = None,
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="Output directory for fine-tuned model")
    ] = Path("./fine_tuned_model"),
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Training configuration file")
    ] = None,
    epochs: Annotated[int, typer.Option("--epochs", help="Number of training epochs")] = 3,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Training batch size")] = 4,
    learning_rate: Annotated[float, typer.Option("--lr", help="Learning rate")] = 2e-5,
    warmup_steps: Annotated[int, typer.Option("--warmup", help="Number of warmup steps")] = 100,
    save_steps: Annotated[
        int, typer.Option("--save-steps", help="Save checkpoint every N steps")
    ] = 500,
    eval_steps: Annotated[int, typer.Option("--eval-steps", help="Evaluate every N steps")] = 100,
    max_grad_norm: Annotated[
        float, typer.Option("--max-grad-norm", help="Maximum gradient norm for clipping")
    ] = 1.0,
    use_lora: Annotated[
        bool, typer.Option("--lora/--no-lora", help="Use LoRA for parameter-efficient fine-tuning")
    ] = True,
    lora_rank: Annotated[int, typer.Option("--lora-rank", help="LoRA rank")] = 16,
    lora_alpha: Annotated[int, typer.Option("--lora-alpha", help="LoRA alpha")] = 32,
    resume: Annotated[
        str | None, typer.Option("--resume", help="Resume training from checkpoint")
    ] = None,
    push_to_hub: Annotated[
        bool, typer.Option("--push-to-hub", help="Push model to Hugging Face Hub")
    ] = False,
    hub_name: Annotated[
        str | None, typer.Option("--hub-name", help="Model name on Hugging Face Hub")
    ] = None,
) -> None:
    """Fine-tune a Gemma model on your data."""
    asyncio.run(
        _run_finetuning(
            data_path=data_path,
            model_name=model_name,
            output_dir=output_dir,
            config_file=config_file,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            max_grad_norm=max_grad_norm,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            resume=resume,
            push_to_hub=push_to_hub,
            hub_name=hub_name,
        )
    )


async def _run_finetuning(
    data_path: Path,
    model_name: str | None,
    output_dir: Path,
    config_file: Path | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_steps: int,
    save_steps: int,
    eval_steps: int,
    max_grad_norm: float,
    use_lora: bool,
    lora_rank: int,
    lora_alpha: int,
    resume: str | None,
    push_to_hub: bool,
    hub_name: str | None,
) -> None:
    """Run fine-tuning implementation."""
    try:
        from ..shared.config.settings import Settings
        from ..training.config import TrainingConfig
        from ..training.trainer import GemmaTrainer

        console.print("🎓 Starting fine-tuning process", style="blue bold")

        # Validate data path
        if not data_path.exists():
            console.print(f"❌ Data path does not exist: {data_path}", style="red")
            raise typer.Exit(1)

        # Load or create training configuration
        if config_file and config_file.exists():
            console.print(f"📖 Loading config from: {config_file}")
            with open(config_file) as f:
                if config_file.suffix.lower() == ".yaml" or config_file.suffix.lower() == ".yml":
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)

            training_config = TrainingConfig(**config_dict)
        else:
            # Create config from command line arguments
            training_config = TrainingConfig(
                model_name=model_name or "google/gemma-2b",
                data_path=str(data_path),
                output_dir=str(output_dir),
                num_epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                save_steps=save_steps,
                eval_steps=eval_steps,
                max_grad_norm=max_grad_norm,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                push_to_hub=push_to_hub,
                hub_model_name=hub_name,
            )

        # Show training configuration
        _show_training_config(training_config, console)

        # Initialize trainer
        settings = Settings()
        trainer = GemmaTrainer(training_config, settings)

        # Initialize training
        console.print("\n🔧 Initializing training...", style="blue")
        await trainer.initialize()

        if resume:
            console.print(f"📂 Resuming from checkpoint: {resume}")
            await trainer.load_checkpoint(resume)

        # Start training
        console.print("\n🚀 Starting training...", style="green bold")

        training_metrics = await trainer.train()

        # Show final results
        _show_training_results(training_metrics, output_dir, console)

        # Push to hub if requested
        if push_to_hub:
            if not hub_name:
                hub_name = f"gemma-finetuned-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            console.print(f"\n📤 Pushing model to Hub as: {hub_name}")
            await trainer.push_to_hub(hub_name)
            console.print(f"✅ Model pushed to: https://huggingface.co/{hub_name}", style="green")

    except Exception as e:
        console.print(f"❌ Fine-tuning failed: {e}", style="red")
        raise typer.Exit(1)


def _show_training_config(config: Any, console: Console) -> None:
    """Display training configuration."""
    config_table = Table(title="Training Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="white")

    # Add configuration rows
    config_table.add_row("Base Model", str(config.model_name))
    config_table.add_row("Data Path", str(config.data_path))
    config_table.add_row("Output Directory", str(config.output_dir))
    config_table.add_row("Epochs", str(config.num_epochs))
    config_table.add_row("Batch Size", str(config.batch_size))
    config_table.add_row("Learning Rate", str(config.learning_rate))
    config_table.add_row("Warmup Steps", str(config.warmup_steps))

    if config.use_lora:
        config_table.add_row("LoRA Enabled", "Yes")
        config_table.add_row("LoRA Rank", str(config.lora_rank))
        config_table.add_row("LoRA Alpha", str(config.lora_alpha))
    else:
        config_table.add_row("LoRA Enabled", "No")

    console.print(config_table)


def _show_training_results(metrics: dict[str, Any], output_dir: Path, console: Console) -> None:
    """Display training results."""
    results_table = Table(title="Training Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="white")

    # Training metrics
    if "final_loss" in metrics:
        results_table.add_row("Final Loss", f"{metrics['final_loss']:.4f}")
    if "best_eval_loss" in metrics:
        results_table.add_row("Best Eval Loss", f"{metrics['best_eval_loss']:.4f}")
    if "total_steps" in metrics:
        results_table.add_row("Total Steps", str(metrics["total_steps"]))
    if "training_time" in metrics:
        results_table.add_row("Training Time", f"{metrics['training_time']:.1f}s")

    console.print(results_table)

    # Model outputs
    console.print(f"\n📁 Model saved to: [bold green]{output_dir}[/bold green]")

    if output_dir.exists():
        console.print("\n📄 Output files:")
        output_tree = Tree(f"[bold]{output_dir}[/bold]")

        for file_path in output_dir.iterdir():
            if file_path.is_file():
                size = format_size(file_path.stat().st_size)
                output_tree.add(f"[blue]{file_path.name}[/blue] [dim]({size})[/dim]")
            elif file_path.is_dir():
                output_tree.add(f"[cyan]{file_path.name}/[/cyan]")

        console.print(output_tree)


@train_app.command("evaluate")
@handle_exceptions(console)
def evaluate(
    model_path: Annotated[Path, typer.Argument(help="Path to model to evaluate")],
    data_path: Annotated[Path, typer.Argument(help="Path to evaluation data")],
    output_file: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output file for evaluation results")
    ] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Evaluation batch size")] = 8,
    max_samples: Annotated[
        int | None, typer.Option("--max-samples", help="Maximum number of samples to evaluate")
    ] = None,
    metrics: Annotated[
        list[str] | None, typer.Option("--metric", help="Metrics to compute (can be repeated)")
    ] = None,
) -> None:
    """Evaluate a fine-tuned model."""
    if metrics is None:
        metrics = ["perplexity", "bleu"]
    asyncio.run(
        _run_evaluation(
            model_path=model_path,
            data_path=data_path,
            output_file=output_file,
            batch_size=batch_size,
            max_samples=max_samples,
            metrics=metrics,
        )
    )


async def _run_evaluation(
    model_path: Path,
    data_path: Path,
    output_file: Path | None,
    batch_size: int,
    max_samples: int | None,
    metrics: list[str],
) -> None:
    """Run model evaluation."""
    try:
        from ..training.evaluator import ModelEvaluator

        console.print("📊 Starting model evaluation", style="blue bold")

        # Validate paths
        if not model_path.exists():
            console.print(f"❌ Model path does not exist: {model_path}", style="red")
            raise typer.Exit(1)

        if not data_path.exists():
            console.print(f"❌ Data path does not exist: {data_path}", style="red")
            raise typer.Exit(1)

        # Initialize evaluator
        evaluator = ModelEvaluator(
            model_path=str(model_path),
            batch_size=batch_size,
            max_samples=max_samples,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Loading model...", total=100)
            await evaluator.load_model()
            progress.update(task, completed=25)

            progress.update(task, description="Loading data...")
            await evaluator.load_data(str(data_path))
            progress.update(task, completed=50)

            progress.update(task, description="Running evaluation...")
            results = await evaluator.evaluate(metrics, progress, task)
            progress.update(task, completed=100, description="Evaluation complete")

        # Display results
        _show_evaluation_results(results, console)

        # Save results if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"\n💾 Results saved to: {output_file}", style="green")

    except Exception as e:
        console.print(f"❌ Evaluation failed: {e}", style="red")
        raise typer.Exit(1)


def _show_evaluation_results(results: dict[str, Any], console: Console) -> None:
    """Display evaluation results."""
    results_table = Table(title="Evaluation Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="white")
    results_table.add_column("Description", style="dim")

    metric_descriptions = {
        "perplexity": "Lower is better - measure of prediction uncertainty",
        "bleu": "Higher is better - translation/generation quality",
        "rouge-1": "Higher is better - recall of unigrams",
        "rouge-2": "Higher is better - recall of bigrams",
        "rouge-l": "Higher is better - longest common subsequence",
        "accuracy": "Higher is better - exact match accuracy",
        "f1": "Higher is better - harmonic mean of precision/recall",
    }

    for metric_name, value in results.get("metrics", {}).items():
        formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)

        description = metric_descriptions.get(metric_name.lower(), "")
        results_table.add_row(metric_name.upper(), formatted_value, description)

    console.print(results_table)

    # Show sample predictions if available
    if results.get("samples"):
        console.print("\n🔍 Sample Predictions:")
        for i, sample in enumerate(results["samples"][:3], 1):
            sample_panel = Panel(
                f"[bold]Input:[/bold] {sample.get('input', '')[:100]}...\n"
                f"[bold green]Prediction:[/bold green] {sample.get('prediction', '')[:100]}...\n"
                f"[bold blue]Target:[/bold blue] {sample.get('target', '')[:100]}...",
                title=f"Sample {i}",
                border_style="blue",
            )
            console.print(sample_panel)


@train_app.command("list")
@handle_exceptions(console)
def list_checkpoints(
    model_dir: Annotated[
        Path | None, typer.Argument(help="Model directory to list checkpoints from")
    ] = None,
    all_models: Annotated[
        bool, typer.Option("--all", "-a", help="List all available models")
    ] = False,
) -> None:
    """List available models and checkpoints."""
    asyncio.run(_list_available_models(model_dir, all_models))


async def _list_available_models(model_dir: Path | None, all_models: bool) -> None:
    """List available models and checkpoints."""
    try:
        from .utils import find_model_checkpoints
        from .utils import get_local_models

        if model_dir:
            # List specific model checkpoints
            if not model_dir.exists():
                console.print(f"❌ Model directory does not exist: {model_dir}", style="red")
                return

            checkpoints = find_model_checkpoints(model_dir)

            if not checkpoints:
                console.print(f"📭 No checkpoints found in {model_dir}", style="yellow")
                return

            table = Table(title=f"Checkpoints in {model_dir}")
            table.add_column("Checkpoint", style="cyan")
            table.add_column("Step", style="blue", justify="right")
            table.add_column("Size", style="white")
            table.add_column("Modified", style="dim")

            for checkpoint in checkpoints:
                table.add_row(
                    checkpoint["name"],
                    str(checkpoint.get("step", "Unknown")),
                    format_size(checkpoint.get("size", 0)),
                    checkpoint.get("modified", "Unknown"),
                )

            console.print(table)

        else:
            # List all available models
            models = get_local_models()

            if not models:
                console.print("📭 No local models found", style="yellow")
                console.print("Use 'gemma-cli models --download <model-name>' to download a model")
                return

            table = Table(title="Available Models")
            table.add_column("Model", style="cyan")
            table.add_column("Type", style="blue")
            table.add_column("Size", style="white")
            table.add_column("Checkpoints", style="green", justify="right")
            table.add_column("Path", style="dim")

            for model in models:
                table.add_row(
                    model["name"],
                    model.get("type", "Unknown"),
                    format_size(model.get("size", 0)),
                    str(model.get("checkpoint_count", 0)),
                    (
                        str(model["path"])[:50] + "..."
                        if len(str(model["path"])) > 50
                        else str(model["path"])
                    ),
                )

            console.print(table)

    except Exception as e:
        console.print(f"❌ Error listing models: {e}", style="red")


@train_app.command("config")
@handle_exceptions(console)
def generate_config(
    output_file: Annotated[
        Path, typer.Option("--output", "-o", help="Output configuration file")
    ] = Path("training_config.yaml"),
    template: Annotated[
        str, typer.Option("--template", help="Configuration template: basic, advanced, lora")
    ] = "basic",
) -> None:
    """Generate a training configuration file."""
    try:
        from ..training.config import generate_config_template

        console.print(f"📝 Generating {template} configuration template", style="blue")

        config_dict = generate_config_template(template)

        # Save as YAML
        if output_file.suffix.lower() in [".yaml", ".yml"]:
            with open(output_file, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            with open(output_file, "w") as f:
                json.dump(config_dict, f, indent=2)

        console.print(f"✅ Configuration saved to: {output_file}", style="green")

        # Show the configuration
        console.print("\n📋 Generated configuration:")
        if output_file.suffix.lower() in [".yaml", ".yml"]:
            console.print(yaml.dump(config_dict, default_flow_style=False, indent=2))
        else:
            console.print(json.dumps(config_dict, indent=2))

    except Exception as e:
        console.print(f"❌ Error generating configuration: {e}", style="red")
        raise typer.Exit(1)
