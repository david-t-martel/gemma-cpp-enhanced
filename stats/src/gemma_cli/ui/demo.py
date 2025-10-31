"""Demo script to showcase Rich UI components.

Run this script to see examples of all UI components in action.
"""

from datetime import datetime
from . import (
    get_console,
    print_header,
    print_separator,
    format_user_message,
    format_assistant_message,
    format_system_message,
    format_error_message,
    format_success_message,
    MemoryDashboard,
    StatusBar,
    StartupBanner,
    CommandPalette,
    ModelSelector,
    create_table,
    create_syntax_highlighted_code,
    create_markdown_panel,
)


def demo_banner():
    """Demo startup banner."""
    console = get_console()
    banner = StartupBanner(app_name="Gemma CLI", version="1.0.0")
    console.print(banner.render())
    console.print()


def demo_messages():
    """Demo message formatting."""
    console = get_console()
    print_header("Message Formatting Demo")

    # User message
    console.print(format_user_message(
        "What is the capital of France?",
        timestamp=datetime.now(),
    ))
    console.print()

    # Assistant message
    console.print(format_assistant_message(
        "The capital of France is **Paris**. It is located in the north-central part of the country.",
        timestamp=datetime.now(),
        render_markdown=True,
    ))
    console.print()

    # System message
    console.print(format_system_message(
        "Model loaded successfully",
        level="success",
    ))
    console.print()


def demo_errors():
    """Demo error formatting."""
    console = get_console()
    print_header("Error Formatting Demo")

    console.print(format_error_message(
        error="Failed to connect to server",
        suggestion="Check your network connection and try again",
    ))
    console.print()

    console.print(format_success_message(
        message="Operation completed successfully",
        details="All files processed without errors",
    ))
    console.print()


def demo_memory_dashboard():
    """Demo memory dashboard."""
    console = get_console()
    print_header("Memory Dashboard Demo")

    memory_stats = {
        "working": {"used": 7, "capacity": 10, "items": 7},
        "short": {"used": 45, "capacity": 100, "items": 45},
        "long": {"used": 3500, "capacity": 10000, "items": 3500},
        "episodic": {"used": 150, "capacity": 1000, "items": 150},
        "semantic": {"used": 8000, "capacity": 50000, "items": 8000},
    }

    dashboard = MemoryDashboard(memory_stats)
    console.print(dashboard.render())
    console.print()


def demo_status_bar():
    """Demo status bar."""
    console = get_console()
    print_header("Status Bar Demo")

    status = StatusBar(
        model_name="gemma-2b-it",
        memory_usage=1200,
        total_memory=2000,
        tokens_used=512,
        max_tokens=8192,
    )

    console.print(status.render())
    console.print()


def demo_command_palette():
    """Demo command palette."""
    console = get_console()
    print_header("Command Palette Demo")

    commands = {
        "help": "Show this help message",
        "clear": "Clear conversation history",
        "save": "Save current conversation",
        "load": "Load saved conversation",
        "models": "List available models",
        "switch": "Switch to different model",
        "stats": "Show statistics",
        "exit": "Exit the CLI",
    }

    palette = CommandPalette(commands)
    console.print(palette.render())
    console.print()


def demo_model_selector():
    """Demo model selector."""
    console = get_console()
    print_header("Model Selector Demo")

    models = [
        {
            "name": "gemma-2b-it",
            "size": "2.5 GB",
            "description": "Fast 2B parameter model for quick responses",
        },
        {
            "name": "gemma-7b-it",
            "size": "8.5 GB",
            "description": "High-quality 7B parameter model",
        },
        {
            "name": "codegemma-2b",
            "size": "2.5 GB",
            "description": "Specialized for code generation",
        },
    ]

    selector = ModelSelector(models)
    console.print(selector.render(highlight_selected=True))
    console.print()


def demo_code_syntax():
    """Demo code syntax highlighting."""
    console = get_console()
    print_header("Syntax Highlighting Demo")

    code = '''def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Example usage
result = fibonacci(10)
print(f"Fibonacci(10) = {result}")
'''

    syntax = create_syntax_highlighted_code(
        code,
        language="python",
        theme="monokai",
        line_numbers=True,
    )

    console.print(syntax)
    console.print()


def demo_markdown():
    """Demo markdown rendering."""
    console = get_console()
    print_header("Markdown Rendering Demo")

    markdown_text = """
# Gemma CLI Features

## Memory System
The CLI includes a **5-tier memory system**:
- Working Memory (short-term)
- Short-term Memory
- Long-term Memory
- Episodic Memory
- Semantic Memory

## Commands
Use `/help` to see available commands.

## Code Example
```python
from gemma_cli import GemmaAgent
agent = GemmaAgent()
response = agent.chat("Hello!")
```
"""

    panel = create_markdown_panel(markdown_text, title="Documentation")
    console.print(panel)
    console.print()


def demo_table():
    """Demo table creation."""
    console = get_console()
    print_header("Table Demo")

    headers = ["Model", "Size", "Speed", "Quality"]
    rows = [
        ["gemma-2b-it", "2.5 GB", "Fast", "Good"],
        ["gemma-7b-it", "8.5 GB", "Medium", "Excellent"],
        ["codegemma-2b", "2.5 GB", "Fast", "Good (Code)"],
    ]

    table = create_table(
        headers=headers,
        rows=rows,
        title="Available Models",
        show_lines=True,
    )

    console.print(table)
    console.print()


def main():
    """Run all demos."""
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'

    console = get_console()

    # Clear screen for clean demo
    console.clear()

    # Run all demos
    demo_banner()
    print_separator()

    demo_messages()
    print_separator()

    demo_errors()
    print_separator()

    demo_memory_dashboard()
    print_separator()

    demo_status_bar()
    print_separator()

    demo_command_palette()
    print_separator()

    demo_model_selector()
    print_separator()

    demo_code_syntax()
    print_separator()

    demo_markdown()
    print_separator()

    demo_table()

    # Final message
    console.print()
    console.print("[success]Demo completed successfully![/success]", justify="center")
    console.print()


if __name__ == "__main__":
    main()
