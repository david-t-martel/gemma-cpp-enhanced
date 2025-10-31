# Integration Guide: Using Rich UI in Gemma CLI

This guide shows how to integrate the Rich UI system into the main `cli.py` file.

## Step 1: Import UI Components

```python
# At top of cli.py
from src.gemma_cli.ui import (
    get_console,
    format_user_message,
    format_assistant_message,
    format_system_message,
    format_error_message,
    MemoryDashboard,
    StatusBar,
    StartupBanner,
    LoadingSpinner,
    ConversationView,
)
```

## Step 2: Replace Basic Print Statements

### Before (Basic Print)
```python
print(f"User: {user_input}")
print(f"Assistant: {response}")
```

### After (Rich UI)
```python
console = get_console()
console.print(format_user_message(user_input, timestamp=datetime.now()))
console.print(format_assistant_message(response, timestamp=datetime.now(), render_markdown=True))
```

## Step 3: Add Startup Banner

```python
def main():
    console = get_console()

    # Show startup banner
    banner = StartupBanner(app_name="Gemma CLI", version="1.0.0")
    console.print(banner.render())
    console.print()

    # Continue with CLI initialization
    ...
```

## Step 4: Add Status Bar

```python
class GemmaCLI:
    def __init__(self):
        self.console = get_console()
        self.status = StatusBar(
            model_name="gemma-2b-it",
            memory_usage=0,
            total_memory=2000,
            tokens_used=0,
            max_tokens=8192,
        )

    def show_status(self):
        """Show current status."""
        self.console.print(self.status.render())

    def update_status(self, **kwargs):
        """Update status values."""
        self.status.update(**kwargs)
```

## Step 5: Add Memory Dashboard

```python
def show_memory_dashboard(memory_system):
    """Display memory usage dashboard."""
    console = get_console()

    # Gather stats from memory system
    memory_stats = {
        "working": {
            "used": len(memory_system.working_memory),
            "capacity": memory_system.WORKING_MEMORY_SIZE,
            "items": len(memory_system.working_memory),
        },
        "short": {
            "used": len(memory_system.short_term_memory),
            "capacity": memory_system.SHORT_TERM_SIZE,
            "items": len(memory_system.short_term_memory),
        },
        # ... other tiers
    }

    dashboard = MemoryDashboard(memory_stats)
    console.print(dashboard.render())
```

## Step 6: Add Loading Spinners

```python
def load_model(model_path):
    """Load model with loading spinner."""
    with LoadingSpinner("Loading model...") as spinner:
        # Load model weights
        model = load_weights(model_path)
        spinner.update("Initializing tokenizer...")
        tokenizer = load_tokenizer(model_path)
        spinner.update("Model loaded successfully")
        return model, tokenizer
```

## Step 7: Error Handling

```python
try:
    # Some operation
    result = risky_operation()
except Exception as e:
    console = get_console()
    console.print(format_error_message(
        error=str(e),
        suggestion="Try checking your input and try again",
        traceback=traceback.format_exc() if args.debug else None,
    ))
```

## Step 8: Interactive Command Palette

```python
def show_help():
    """Show available commands."""
    console = get_console()

    commands = {
        "help": "Show this help message",
        "clear": "Clear conversation history",
        "save": "Save current conversation",
        "load": "Load saved conversation",
        "models": "List available models",
        "switch": "Switch to different model",
        "memory": "Show memory dashboard",
        "stats": "Show statistics",
        "exit": "Exit the CLI",
    }

    from src.gemma_cli.ui import CommandPalette
    palette = CommandPalette(commands)
    console.print(palette.render())
```

## Step 9: Conversation History

```python
class ConversationManager:
    def __init__(self):
        self.view = ConversationView()

    def add_message(self, role, content):
        """Add message to history."""
        self.view.add_message(role, content, timestamp=datetime.now())

    def show_history(self, max_messages=20):
        """Display conversation history."""
        console = get_console()
        console.print(self.view.render(max_messages=max_messages))
```

## Complete Integration Example

```python
# cli.py

from datetime import datetime
from src.gemma_cli.ui import (
    get_console,
    format_user_message,
    format_assistant_message,
    format_system_message,
    format_error_message,
    MemoryDashboard,
    StatusBar,
    StartupBanner,
    LoadingSpinner,
    CommandPalette,
)


class GemmaCLI:
    def __init__(self, model_path: str):
        self.console = get_console()
        self.status = StatusBar()
        self.conversation = []

    def start(self):
        """Start the CLI."""
        # Show banner
        banner = StartupBanner(app_name="Gemma CLI", version="1.0.0")
        self.console.print(banner.render())
        self.console.print()

        # Load model with spinner
        with LoadingSpinner("Loading model..."):
            self.model = self.load_model()

        # Show success message
        self.console.print(format_system_message(
            "Model loaded successfully",
            level="success",
        ))
        self.console.print()

        # Main loop
        self.main_loop()

    def main_loop(self):
        """Main CLI loop."""
        while True:
            # Show status
            self.console.print(self.status.render())
            self.console.print()

            # Get user input
            try:
                user_input = self.console.input("[bold cyan]You:[/bold cyan] ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                self.handle_command(user_input)
                continue

            # Show user message
            self.console.print(format_user_message(
                user_input,
                timestamp=datetime.now(),
            ))

            # Get response
            try:
                with LoadingSpinner("Thinking..."):
                    response = self.get_response(user_input)

                # Show assistant message
                self.console.print(format_assistant_message(
                    response,
                    timestamp=datetime.now(),
                    render_markdown=True,
                ))

                # Update status
                self.status.update(
                    tokens_used=self.get_token_count(),
                    memory_usage=self.get_memory_usage(),
                )

            except Exception as e:
                self.console.print(format_error_message(
                    error=str(e),
                    suggestion="Check your model configuration and try again",
                ))

    def handle_command(self, command: str):
        """Handle CLI commands."""
        cmd = command[1:].lower()

        if cmd == "help":
            self.show_help()
        elif cmd == "clear":
            self.console.clear()
            self.conversation.clear()
            self.console.print(format_system_message(
                "Conversation cleared",
                level="success",
            ))
        elif cmd == "memory":
            self.show_memory_dashboard()
        elif cmd == "exit":
            raise KeyboardInterrupt
        else:
            self.console.print(format_error_message(
                error=f"Unknown command: {command}",
                suggestion="Type /help to see available commands",
            ))

    def show_help(self):
        """Show command palette."""
        commands = {
            "help": "Show this help message",
            "clear": "Clear conversation history",
            "memory": "Show memory dashboard",
            "exit": "Exit the CLI",
        }
        palette = CommandPalette(commands)
        self.console.print(palette.render())

    def show_memory_dashboard(self):
        """Show memory usage dashboard."""
        memory_stats = self.get_memory_stats()
        dashboard = MemoryDashboard(memory_stats)
        self.console.print(dashboard.render())

    def load_model(self):
        """Load model (placeholder)."""
        import time
        time.sleep(1)  # Simulate loading
        self.status.update(model_name="gemma-2b-it")
        return None

    def get_response(self, user_input: str) -> str:
        """Get model response (placeholder)."""
        return f"Echo: {user_input}"

    def get_token_count(self) -> int:
        """Get current token count."""
        return len(self.conversation) * 10  # Placeholder

    def get_memory_usage(self) -> int:
        """Get memory usage."""
        return 500  # Placeholder

    def get_memory_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "working": {"used": 5, "capacity": 10, "items": 5},
            "short": {"used": 30, "capacity": 100, "items": 30},
            "long": {"used": 1000, "capacity": 10000, "items": 1000},
            "episodic": {"used": 50, "capacity": 1000, "items": 50},
            "semantic": {"used": 2000, "capacity": 50000, "items": 2000},
        }


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/gemma-2b-it"

    cli = GemmaCLI(model_path)
    cli.start()


if __name__ == "__main__":
    main()
```

## Testing the Integration

```bash
# Run the integrated CLI
cd /c/codedev/llm/stats
uv run python -m src.gemma_cli.cli

# Or if using the above standalone example
uv run python cli_example.py
```

## Best Practices

1. **Use Console Singleton**: Always get console via `get_console()` instead of creating new instances
2. **Format Messages**: Always use formatters for consistent styling
3. **Loading Indicators**: Use `LoadingSpinner` context manager for long operations
4. **Error Handling**: Always wrap risky operations in try/except with `format_error_message()`
5. **Status Updates**: Update StatusBar after each interaction
6. **Timestamps**: Include timestamps for better conversation tracking

## Performance Tips

1. **Lazy Rendering**: Only render widgets when needed
2. **Batch Updates**: Update status bar once per turn, not per token
3. **Efficient Logging**: Use Rich logging handlers for structured logs
4. **Memory**: Widgets have minimal overhead (~2MB total)

## Troubleshooting

### Unicode Errors on Windows
Set environment variable before running:
```bash
export PYTHONIOENCODING=utf-8
```

### Colors Not Showing
Ensure terminal supports 24-bit color (most modern terminals do)

### Slow Rendering
- Check console width (shouldn't exceed 200 characters)
- Reduce markdown complexity in messages
- Use `show_lines=False` in tables for faster rendering
