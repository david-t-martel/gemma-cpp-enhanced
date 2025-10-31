# Gemma CLI Rich UI System

A production-ready Rich terminal UI system for the Gemma CLI, providing comprehensive components for building beautiful, interactive command-line interfaces.

## Architecture

```
ui/
├── __init__.py          # Public API exports
├── theme.py             # Color schemes and style definitions
├── console.py           # Global console configuration
├── components.py        # Reusable UI components
├── formatters.py        # Message formatting utilities
├── widgets.py           # Complex interactive widgets
├── demo.py              # Demo script showcasing all features
└── README.md            # This file
```

## Features

### Theme System (`theme.py`)
- **Dark and Light themes** with semantic colors
- **Memory tier colors** (working, short, long, episodic, semantic)
- **Status colors** (success, error, warning, info)
- **Role colors** (user, assistant, system)
- Easy theme switching with `get_theme()` and `set_theme()`

### Console Management (`console.py`)
- **Singleton console instance** with consistent configuration
- **Context managers** for scoped styling and output capture
- **24-bit true color** support
- **Modern Windows terminal** compatibility
- Helper functions: `print_header()`, `print_separator()`, `clear_screen()`

### Reusable Components (`components.py`)
- **Panels**: Bordered content containers
- **Tables**: Tabular data with styling
- **Progress bars**: Task progress visualization
- **Tree views**: Hierarchical data display
- **Syntax highlighting**: Code blocks with color
- **Markdown rendering**: Rich formatted text
- **Key-value displays**: Configuration/metadata views

### Message Formatters (`formatters.py`)
- **User messages**: Styled user input
- **Assistant messages**: AI responses with Markdown
- **System messages**: Status/info messages
- **Error messages**: Errors with suggestions
- **Success messages**: Operation confirmations
- **Memory entries**: Memory tier formatting
- **Conversation history**: Chat history tables
- **Model info**: Model metadata display
- **Statistics**: Metric tables

### Complex Widgets (`widgets.py`)
- **MemoryDashboard**: 5-tier memory usage visualization
- **StatusBar**: Bottom status bar (model/memory/tokens)
- **StartupBanner**: Animated splash screen
- **CommandPalette**: Interactive command list
- **ModelSelector**: Interactive model picker
- **LoadingSpinner**: Context manager for long operations
- **ConversationView**: Scrollable chat history

## Usage Examples

### Basic Setup

```python
from gemma_cli.ui import get_console, set_theme

# Get console instance
console = get_console()

# Change theme (optional)
set_theme("dark")  # or "light"

# Print with Rich markup
console.print("[bold cyan]Hello, World![/bold cyan]")
```

### Message Formatting

```python
from gemma_cli.ui import (
    format_user_message,
    format_assistant_message,
    format_error_message,
)

# User message
console.print(format_user_message("What is the capital of France?"))

# Assistant response with Markdown
console.print(format_assistant_message(
    "The capital is **Paris**.",
    render_markdown=True,
))

# Error with suggestion
console.print(format_error_message(
    error="Connection failed",
    suggestion="Check your network settings",
))
```

### Memory Dashboard

```python
from gemma_cli.ui import MemoryDashboard

memory_stats = {
    "working": {"used": 7, "capacity": 10, "items": 7},
    "short": {"used": 45, "capacity": 100, "items": 45},
    "long": {"used": 3500, "capacity": 10000, "items": 3500},
    "episodic": {"used": 150, "capacity": 1000, "items": 150},
    "semantic": {"used": 8000, "capacity": 50000, "items": 8000},
}

dashboard = MemoryDashboard(memory_stats)
console.print(dashboard.render())
```

### Status Bar

```python
from gemma_cli.ui import StatusBar

status = StatusBar(
    model_name="gemma-2b-it",
    memory_usage=1200,
    total_memory=2000,
    tokens_used=512,
    max_tokens=8192,
)

console.print(status.render())

# Update dynamically
status.update(tokens_used=600)
console.print(status.render())
```

### Loading Spinner

```python
from gemma_cli.ui import LoadingSpinner
import time

with LoadingSpinner("Loading model...") as spinner:
    time.sleep(2)  # Long operation
    spinner.update("Initializing...")
    time.sleep(1)
```

### Syntax Highlighting

```python
from gemma_cli.ui import create_syntax_highlighted_code

code = '''def hello():
    print("Hello, World!")
'''

syntax = create_syntax_highlighted_code(
    code,
    language="python",
    theme="monokai",
    line_numbers=True,
)

console.print(syntax)
```

### Interactive Model Selector

```python
from gemma_cli.ui import ModelSelector

models = [
    {"name": "gemma-2b-it", "size": "2.5 GB", "description": "Fast model"},
    {"name": "gemma-7b-it", "size": "8.5 GB", "description": "High quality"},
]

selector = ModelSelector(models)
console.print(selector.render())

# Navigate
selector.select_next()
selected = selector.get_selected_model()
```

### Context Managers

```python
from gemma_cli.ui import scoped_style, status_spinner, capture_output

# Scoped styling
with scoped_style("bold red"):
    console.print("This is bold and red")

# Status spinner
with status_spinner("Processing..."):
    # Long operation
    process_data()

# Capture output
with capture_output() as (console, get_output):
    console.print("Test output")
    output = get_output()
    print(output)  # Plain text output
```

## Running the Demo

See all components in action:

```bash
cd /c/codedev/llm/stats
uv run python -m src.gemma_cli.ui.demo
```

## Color Scheme

### Dark Theme (Default)
- **Primary**: Cyan (#00D9FF)
- **Secondary**: Magenta (#FF00FF)
- **Accent**: Gold (#FFD700)
- **Success**: Green (#00FF00)
- **Error**: Red (#FF0000)
- **Warning**: Orange (#FFA500)
- **Info**: Cyan (#00D9FF)

### Memory Tier Colors
- **Working**: Red (#FF6B6B)
- **Short**: Orange (#FFA500)
- **Long**: Gold (#FFD700)
- **Episodic**: Cyan (#00D9FF)
- **Semantic**: Green (#00FF00)

## Best Practices

### 1. Use Console Singleton
```python
# ✅ Good
from gemma_cli.ui import get_console
console = get_console()
console.print("Hello")

# ❌ Avoid
from rich.console import Console
console = Console()  # Creates new instance
```

### 2. Use Formatters for Consistency
```python
# ✅ Good
console.print(format_user_message("Hello"))

# ❌ Avoid
console.print("[blue]User:[/blue] Hello")
```

### 3. Use Context Managers for Long Operations
```python
# ✅ Good
with LoadingSpinner("Processing..."):
    process_data()

# ❌ Avoid
console.print("Processing...")
process_data()
```

### 4. Use Semantic Styles
```python
# ✅ Good
console.print("[success]Operation complete[/success]")

# ❌ Avoid
console.print("[green bold]Operation complete[/green bold]")
```

### 5. Handle Rendering Errors
```python
# ✅ Good
try:
    console.print(complex_widget.render())
except Exception as e:
    console.print(f"[error]Rendering failed: {e}[/error]")
```

## Integration with Gemma CLI

The UI system is designed to integrate seamlessly with the main CLI:

```python
from gemma_cli.ui import (
    get_console,
    StartupBanner,
    MemoryDashboard,
    StatusBar,
    format_user_message,
    format_assistant_message,
)

class GemmaCLI:
    def __init__(self):
        self.console = get_console()
        self.status = StatusBar()

    def start(self):
        # Show banner
        banner = StartupBanner()
        self.console.print(banner.render())

        # Main loop
        while True:
            # Show status
            self.console.print(self.status.render())

            # Get user input
            user_input = self.console.input("[user]You:[/user] ")

            # Show formatted message
            self.console.print(format_user_message(user_input))

            # Get response (from model)
            response = self.get_response(user_input)

            # Show formatted response
            self.console.print(format_assistant_message(response))
```

## Performance Considerations

- **Lazy rendering**: Widgets only render when needed
- **Efficient updates**: StatusBar and widgets support incremental updates
- **No blocking**: All rendering is synchronous and fast
- **Memory efficient**: Reuses console instance

## Testing

The UI system includes comprehensive type hints and can be tested:

```python
from gemma_cli.ui import capture_output, format_user_message

def test_user_message():
    with capture_output() as (console, get_output):
        console.print(format_user_message("Test"))
        output = get_output()
        assert "Test" in output
```

## Dependencies

- **rich**: Core Rich library (all UI components)
- **typing**: Type hints for Python 3.8+

Install via:
```bash
uv pip install rich
```

## License

MIT License - See project root for details.
