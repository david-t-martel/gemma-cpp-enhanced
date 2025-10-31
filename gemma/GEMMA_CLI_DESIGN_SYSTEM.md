# Gemma CLI Terminal UI Design System

> **Version:** 1.0.0
> **Last Updated:** 2025-10-13
> **Framework:** Python Rich Library

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Color System](#color-system)
3. [Typography](#typography)
4. [Component Library](#component-library)
5. [Layout System](#layout-system)
6. [Interactive Patterns](#interactive-patterns)
7. [Animation & Feedback](#animation--feedback)
8. [Accessibility](#accessibility)
9. [User Flows](#user-flows)
10. [Implementation Guide](#implementation-guide)

---

## Design Philosophy

### Core Principles

1. **Developer-First**: Terminal UIs should enhance productivity, not distract
2. **Information Density**: Balance between rich information and visual clarity
3. **Performance Feedback**: Always show what's happening (loading, processing, streaming)
4. **Graceful Degradation**: Work well in various terminal environments
5. **Semantic Color Usage**: Colors convey meaning, not just decoration

### Inspiration Sources

- **Claude CLI**: Clean conversation flow, excellent streaming responses
- **GitHub CLI (gh)**: Clear command structure, helpful tables
- **Vercel CLI**: Beautiful status indicators, smooth animations
- **Rich Library Examples**: Proper use of panels, syntax highlighting, progress bars

---

## Color System

### Primary Color Palette

```python
# Rich Color Names and Hex Equivalents

PRIMARY = {
    "brand_primary": "bright_cyan",      # #00D7FF - Main brand color
    "brand_secondary": "cyan",           # #00AFAF - Secondary actions
    "brand_accent": "deep_sky_blue3",    # #0087D7 - Highlights
}

SEMANTIC = {
    "success": "bright_green",           # #00FF00 - Success states
    "warning": "yellow",                 # #FFFF00 - Warnings
    "error": "bright_red",               # #FF0000 - Errors
    "info": "bright_blue",               # #0000FF - Information
    "muted": "grey50",                   # #7F7F7F - Deemphasized text
}

MESSAGE_TYPES = {
    "user_message": "bright_magenta",    # #FF00FF - User input
    "assistant_message": "bright_cyan",  # #00D7FF - AI responses
    "system_message": "yellow",          # #FFFF00 - System notifications
    "memory_recall": "bright_blue",      # #0000FF - RAG memory
    "code_block": "green",               # #00FF00 - Code highlighting
}

STATUS = {
    "active": "bright_green",            # #00FF00 - Active/running
    "idle": "grey50",                    # #7F7F7F - Idle state
    "processing": "bright_yellow",       # #FFFF00 - Processing
    "error": "bright_red",               # #FF0000 - Error state
}

MEMORY_TIERS = {
    "working": "bright_magenta",         # #FF00FF - Immediate context
    "short_term": "bright_cyan",         # #00D7FF - Recent memories
    "long_term": "bright_blue",          # #0000FF - Consolidated
    "episodic": "bright_green",          # #00FF00 - Event sequences
    "semantic": "yellow",                # #FFFF00 - Concepts/relations
}
```

### Theme Support

#### Dark Theme (Default)
```python
DARK_THEME = {
    "background": "grey0",               # #000000
    "foreground": "grey100",             # #FFFFFF
    "panel_border": "bright_cyan",       # #00D7FF
    "panel_title": "bright_white",       # #FFFFFF
    "dim_text": "grey50",                # #7F7F7F
}
```

#### Light Theme
```python
LIGHT_THEME = {
    "background": "grey100",             # #FFFFFF
    "foreground": "grey0",               # #000000
    "panel_border": "blue",              # #0000AF
    "panel_title": "grey0",              # #000000
    "dim_text": "grey50",                # #7F7F7F
}
```

### Color Usage Guidelines

| Element | Color | When to Use |
|---------|-------|-------------|
| **Prompts** | `bright_magenta` | User input indicators |
| **Responses** | `bright_cyan` | AI-generated text |
| **Commands** | `bright_blue` | Command names, keywords |
| **Success** | `bright_green` | Completed operations, checkmarks |
| **Warnings** | `yellow` | Non-critical alerts |
| **Errors** | `bright_red` | Critical errors, failures |
| **Code** | `green` | Code blocks, syntax elements |
| **Numbers** | `cyan` | Metrics, counts, percentages |
| **Borders** | `bright_cyan` | Panel borders, dividers |

---

## Typography

### Text Styles

```python
from rich.text import Text
from rich.style import Style

# Heading Styles
HEADING_1 = Style(color="bright_white", bold=True, underline=True)
HEADING_2 = Style(color="bright_cyan", bold=True)
HEADING_3 = Style(color="cyan", bold=True)

# Body Text
BODY_TEXT = Style(color="grey100")
DIM_TEXT = Style(color="grey50", dim=True)
BOLD_TEXT = Style(color="bright_white", bold=True)
ITALIC_TEXT = Style(color="grey100", italic=True)

# Special Text
CODE_INLINE = Style(color="green", bgcolor="grey15")
CODE_BLOCK = Style(color="bright_green")
QUOTE = Style(color="grey70", italic=True)
LINK = Style(color="bright_blue", underline=True)

# Message Styles
USER_MESSAGE = Style(color="bright_magenta", bold=True)
ASSISTANT_MESSAGE = Style(color="bright_cyan")
SYSTEM_MESSAGE = Style(color="yellow", italic=True)
ERROR_MESSAGE = Style(color="bright_red", bold=True)
```

### Typography Scale

```
H1: 18pt equivalent (double-width with box drawing)
H2: 16pt equivalent (bold with underline)
H3: 14pt equivalent (bold, no underline)
Body: 12pt equivalent (normal weight)
Small: 10pt equivalent (dim)
Tiny: 8pt equivalent (very dim, metadata)
```

### Message Formatting Patterns

#### User Message
```python
"""
╭─ You ────────────────────────────────────────────────
│ What is the capital of France?
╰───────────────────────────────────────────────────────
"""
```

#### Assistant Response
```python
"""
╭─ Gemma ──────────────────────────────────────────────
│ The capital of France is Paris. It's known for its
│ iconic landmarks like the Eiffel Tower, Louvre Museum,
│ and Notre-Dame Cathedral.
│
│ Memory: Retrieved 2 relevant facts
│ Tokens: 45 | Time: 0.8s
╰───────────────────────────────────────────────────────
"""
```

#### System Message
```python
"""
⚙ System: Model loaded successfully (gemma-2b-it)
"""
```

#### Error Message
```python
"""
✗ Error: Failed to load model weights
  └─ File not found: /path/to/model.sbs
  └─ Try: gemma-cli --model-path <path>
"""
```

---

## Component Library

### 1. Startup Banner

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║      ██████╗ ███████╗███╗   ███╗███╗   ███╗ █████╗              ║
║     ██╔════╝ ██╔════╝████╗ ████║████╗ ████║██╔══██╗             ║
║     ██║  ███╗█████╗  ██╔████╔██║██╔████╔██║███████║             ║
║     ██║   ██║██╔══╝  ██║╚██╔╝██║██║╚██╔╝██║██╔══██║             ║
║     ╚██████╔╝███████╗██║ ╚═╝ ██║██║ ╚═╝ ██║██║  ██║             ║
║      ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝             ║
║                                                                  ║
║              High-Performance LLM Inference CLI                  ║
║                      Version 2.0.0                               ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Status                                                          ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║   ✓ Model: gemma-2b-it (SFP format)                             ║
║   ✓ Memory: Redis connected (5 tiers active)                    ║
║   ✓ MCP: 3 servers available                                    ║
║   ✓ RAG: Document store ready (1,234 chunks)                    ║
║                                                                  ║
║  Quick Start                                                     ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║   /help       Show all commands                                 ║
║   /memory     View memory dashboard                             ║
║   /mcp        List available tools                              ║
║   /settings   Configure preferences                             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

Type your message or /help for commands...
```

**Implementation:**
```python
from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.text import Text

def render_startup_banner(console: Console):
    banner_text = Text()
    banner_text.append("GEMMA\n", style="bold bright_cyan")
    banner_text.append("High-Performance LLM Inference CLI\n", style="grey70")
    banner_text.append("Version 2.0.0", style="dim")

    status_text = Text()
    status_text.append("✓ ", style="bright_green")
    status_text.append("Model: gemma-2b-it\n", style="grey100")
    # ... add more status lines

    panel = Panel(
        Align.center(banner_text),
        border_style="bright_cyan",
        padding=(1, 2)
    )
    console.print(panel)
```

---

### 2. Status Bar (Persistent Bottom)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Model: gemma-2b │ Memory: 45/100 MB │ Tokens: 1.2K/8K │ Latency: 45ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**States:**
```
# Active Inference
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 🔄 Processing... │ Memory: 45/100 MB │ Tokens: 1.2K/8K │ 2.3s elapsed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Error State
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ✗ Error │ Memory: disconnected │ /help for assistance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Implementation:**
```python
from rich.live import Live
from rich.table import Table

def create_status_bar(
    model: str,
    memory_used: int,
    memory_total: int,
    tokens_used: int,
    tokens_max: int,
    latency_ms: int
) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left")

    table.add_row(
        f"Model: [bright_cyan]{model}[/] │ "
        f"Memory: [cyan]{memory_used}/{memory_total} MB[/] │ "
        f"Tokens: [cyan]{tokens_used}/{tokens_max}[/] │ "
        f"Latency: [green]{latency_ms}ms[/]"
    )
    return table
```

---

### 3. Conversation View

```
╭─ Conversation ─────────────────────────────────────────────────────╮
│                                                                    │
│ ╭─ You ──────────────────────────────────────────── 10:45:32 AM ─╮ │
│ │ Explain the difference between AVX2 and SSE4 in SIMD           │ │
│ │ optimizations for LLM inference.                               │ │
│ ╰────────────────────────────────────────────────────────────────╯ │
│                                                                    │
│ ╭─ Gemma ────────────────────────────────────────── 10:45:35 AM ─╮ │
│ │ AVX2 and SSE4 are both SIMD instruction sets for x86 CPUs:    │ │
│ │                                                                 │ │
│ │ SSE4 (Streaming SIMD Extensions 4):                            │ │
│ │  • 128-bit registers (4 x float32)                             │ │
│ │  • Released 2006-2008                                          │ │
│ │  • Good baseline compatibility                                 │ │
│ │                                                                 │ │
│ │ AVX2 (Advanced Vector Extensions 2):                           │ │
│ │  • 256-bit registers (8 x float32)                             │ │
│ │  • Released 2013                                               │ │
│ │  • ~2x throughput for vectorized ops                           │ │
│ │  • Required for modern inference engines                       │ │
│ │                                                                 │ │
│ │ For gemma.cpp, AVX2 provides ~2-4x speedup over SSE4.         │ │
│ │                                                                 │ │
│ │ ┌─ Memory Recall ─────────────────────────────────────────┐    │ │
│ │ │ Retrieved 2 facts from Long-Term Memory:                │    │ │
│ │ │  • Highway SIMD library supports both instruction sets  │    │ │
│ │ │  • gemma.cpp auto-enables AVX2 on x86_64 builds        │    │ │
│ │ └──────────────────────────────────────────────────────────┘    │ │
│ │                                                                 │ │
│ │ 📊 Tokens: 156 | ⏱ Time: 2.8s | 🧠 Memory: Long-Term          │ │
│ ╰────────────────────────────────────────────────────────────────╯ │
│                                                                    │
│ ╭─ You ──────────────────────────────────────────── 10:46:10 AM ─╮ │
│ │ How do I check if AVX2 is enabled in my build?                │ │
│ ╰────────────────────────────────────────────────────────────────╯ │
│                                                                    │
│ ⏳ Gemma is thinking...                                           │
│                                                                    │
╰────────────────────────────────────────────────────────────────────╯
```

**Implementation:**
```python
from rich.panel import Panel
from rich.console import Group
from datetime import datetime

def render_message(
    role: str,  # "You" or "Gemma" or "System"
    content: str,
    timestamp: datetime,
    metadata: dict = None
) -> Panel:
    title = f"{role} {timestamp.strftime('%I:%M:%S %p')}"

    style_map = {
        "You": "bright_magenta",
        "Gemma": "bright_cyan",
        "System": "yellow"
    }

    panel = Panel(
        content,
        title=title,
        title_align="left",
        border_style=style_map.get(role, "grey50"),
        padding=(1, 2)
    )

    return panel
```

---

### 4. Memory Dashboard

```
╭─ Memory System Dashboard ──────────────────────────────────────────╮
│                                                                    │
│ Overall Status                                                     │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  Total Entries: 1,234 | Redis: Connected | Last Sync: 2s ago     │
│                                                                    │
│ Memory Tiers                                                       │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                    │
│ Working Memory        [████████░░] 8/10 entries (80%)             │
│  └─ Current conversation context, immediate recall                │
│                                                                    │
│ Short-Term Memory     [██████████] 100/100 entries (100%)         │
│  └─ Recent interactions, ~24hr retention                          │
│                                                                    │
│ Long-Term Memory      [███░░░░░░░] 342/10K entries (3%)           │
│  └─ Consolidated facts, persistent storage                        │
│                                                                    │
│ Episodic Memory       [█░░░░░░░░░] 56/500 events (11%)            │
│  └─ Event sequences with timestamps                               │
│                                                                    │
│ Semantic Memory       [██░░░░░░░░] 128/1K concepts (13%)          │
│  └─ Graph-based concept relationships                             │
│                                                                    │
│ Recent Activity                                                    │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  10:45:35  Recalled 2 facts from Long-Term                        │
│  10:44:12  Added 1 concept to Semantic                            │
│  10:42:50  Consolidated 5 entries to Long-Term                    │
│                                                                    │
│ Actions                                                            │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  /memory clear <tier>     Clear specific memory tier              │
│  /memory export           Export to JSON file                     │
│  /memory stats            Detailed statistics                     │
│  /memory search <query>   Search across all tiers                 │
│                                                                    │
╰────────────────────────────────────────────────────────────────────╯
```

**Implementation:**
```python
from rich.progress import Progress, BarColumn, TextColumn

def render_memory_tier(
    name: str,
    current: int,
    maximum: int,
    color: str,
    description: str
) -> Group:
    percentage = (current / maximum) * 100

    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(complete_style=color, finished_style=color),
        TextColumn("{task.percentage:>3.0f}%"),
    )
    task = progress.add_task(name, total=maximum, completed=current)

    description_text = Text(f"  └─ {description}", style="grey50")

    return Group(progress, description_text)
```

---

### 5. Command Palette

```
╭─ Available Commands ───────────────────────────────────────────────╮
│                                                                    │
│ Conversation                                                       │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  /clear          Clear conversation history                        │
│  /save <name>    Save conversation to file                         │
│  /load <name>    Load previous conversation                        │
│  /export         Export to markdown/JSON                           │
│                                                                    │
│ Memory & RAG                                                       │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  /memory         Show memory dashboard                             │
│  /recall <query> Search memory for query                           │
│  /ingest <path>  Ingest documents into RAG                         │
│  /forget <tier>  Clear specific memory tier                        │
│                                                                    │
│ MCP Tools                                                          │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  /mcp            List available MCP tools                          │
│  /mcp use <tool> Use specific MCP tool                             │
│  /mcp status     Check MCP server status                           │
│                                                                    │
│ Configuration                                                      │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  /settings       Open settings menu                                │
│  /model <name>   Switch model                                      │
│  /theme <name>   Change color theme (dark/light)                   │
│  /logs           View system logs                                  │
│                                                                    │
│ System                                                             │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  /help           Show this help                                    │
│  /version        Show version information                          │
│  /quit           Exit gemma-cli                                    │
│                                                                    │
│ Tip: Tab for autocomplete, Ctrl+C to cancel, Ctrl+D to exit       │
│                                                                    │
╰────────────────────────────────────────────────────────────────────╯
```

---

### 6. Progress Indicators

#### Loading Spinner
```python
from rich.spinner import Spinner

# Different contexts
SPINNERS = {
    "inference": Spinner("dots", text="Generating response..."),
    "memory_recall": Spinner("arc", text="Searching memory..."),
    "document_processing": Spinner("bouncingBar", text="Processing documents..."),
    "model_loading": Spinner("simpleDotsScrolling", text="Loading model..."),
}
```

**Visual Examples:**
```
⠋ Generating response...
◜ Searching memory...
[=   ] Processing documents... 25%
⣾ Loading model...
```

#### Progress Bar (Long Operations)
```
╭─ Ingesting Documents ──────────────────────────────────────────────╮
│                                                                    │
│ Processing: technical_docs.pdf                                     │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│                                                                    │
│ Progress: 456/1000 chunks ━━━━━━━━━━░░░░░░░░░░░░░░░ 45%          │
│ Speed: 23 chunks/s | ETA: 24s                                      │
│                                                                    │
│ Stats:                                                             │
│  ✓ Parsed: 456 chunks                                             │
│  ✓ Embedded: 456 vectors                                          │
│  ✓ Indexed: 456 entries                                           │
│                                                                    │
╰────────────────────────────────────────────────────────────────────╯
```

**Implementation:**
```python
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

progress = Progress(
    SpinnerColumn(),
    TextColumn("[bold bright_cyan]{task.description}"),
    BarColumn(complete_style="bright_green", finished_style="bright_green"),
    TaskProgressColumn(),
    TimeRemainingColumn(),
    console=console,
)
```

---

### 7. Tables (MCP Tools, Configuration)

```
╭─ Available MCP Tools ──────────────────────────────────────────────╮
│                                                                    │
│ ┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┓ │
│ ┃ Tool           ┃ Description         ┃ Status  ┃ Last Used  ┃ │
│ ┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━┩ │
│ │ filesystem     │ Read/write files    │ ✓ Ready │ 2m ago     │ │
│ │ web_search     │ Search the web      │ ✓ Ready │ 5m ago     │ │
│ │ code_analysis  │ Analyze codebases   │ ✗ Error │ Never      │ │
│ │ database       │ Query databases     │ ⏸ Idle  │ 1h ago     │ │
│ └────────────────┴─────────────────────┴─────────┴────────────┘ │
│                                                                    │
│ Use /mcp use <tool> to invoke a tool                              │
│                                                                    │
╰────────────────────────────────────────────────────────────────────╯
```

**Implementation:**
```python
from rich.table import Table

def create_mcp_tools_table() -> Table:
    table = Table(
        title="Available MCP Tools",
        border_style="bright_cyan",
        header_style="bold bright_white",
        show_lines=True,
    )

    table.add_column("Tool", style="bright_cyan", no_wrap=True)
    table.add_column("Description", style="grey100")
    table.add_column("Status", justify="center")
    table.add_column("Last Used", justify="right", style="grey50")

    # Status icons
    status_map = {
        "ready": "[bright_green]✓ Ready[/]",
        "error": "[bright_red]✗ Error[/]",
        "idle": "[yellow]⏸ Idle[/]",
    }

    table.add_row(
        "filesystem",
        "Read/write files",
        status_map["ready"],
        "2m ago"
    )

    return table
```

---

### 8. Error States

```
╭─ Error ────────────────────────────────────────────────────────────╮
│                                                                    │
│ ✗ Model Loading Failed                                            │
│                                                                    │
│ Details:                                                           │
│  • File not found: C:\models\gemma-2b-it.sbs                      │
│  • Error code: ENOENT (2)                                         │
│                                                                    │
│ Suggestions:                                                       │
│  1. Verify model path: /settings or --model-path flag             │
│  2. Download models: python -m src.gcp.gemma_download --auto      │
│  3. Check file permissions                                        │
│                                                                    │
│ Documentation: https://github.com/example/gemma-cli#models        │
│                                                                    │
╰────────────────────────────────────────────────────────────────────╯
```

**Severity Levels:**
```python
ERROR_STYLES = {
    "critical": ("bright_red", "✗"),
    "error": ("red", "✗"),
    "warning": ("yellow", "⚠"),
    "info": ("bright_blue", "ℹ"),
}
```

---

### 9. Success States

```
╭─ Success ──────────────────────────────────────────────────────────╮
│                                                                    │
│ ✓ Documents Ingested Successfully                                 │
│                                                                    │
│ Summary:                                                           │
│  • Files processed: 5                                             │
│  • Chunks created: 1,234                                          │
│  • Vectors embedded: 1,234                                        │
│  • Time elapsed: 45.2s                                            │
│                                                                    │
│ Next Steps:                                                        │
│  • Use /recall to search ingested content                         │
│  • View statistics with /memory stats                             │
│                                                                    │
╰────────────────────────────────────────────────────────────────────╯
```

---

## Layout System

### Grid System

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Header/Banner (100%)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                                                                     │
│                      Main Content Area (100%)                       │
│                        (Scrollable region)                          │
│                                                                     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                   Status Bar (100%, fixed bottom)                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Spacing Scale

```python
SPACING = {
    "xs": 1,   # 1 line
    "sm": 2,   # 2 lines
    "md": 3,   # 3 lines (default)
    "lg": 4,   # 4 lines
    "xl": 6,   # 6 lines
}

PADDING = {
    "xs": (0, 1),  # (vertical, horizontal)
    "sm": (1, 2),
    "md": (1, 2),  # Default
    "lg": (2, 3),
    "xl": (2, 4),
}
```

### Border Styles

```python
BORDER_STYLES = {
    "default": "bright_cyan",        # Standard panels
    "success": "bright_green",       # Success messages
    "error": "bright_red",           # Error messages
    "warning": "yellow",             # Warnings
    "info": "bright_blue",           # Information
    "muted": "grey50",               # Deemphasized content
}

BORDER_TYPES = {
    "rounded": "rounded",   # Friendly, modern (default)
    "square": "square",     # Technical, precise
    "double": "double",     # Emphasis, headers
    "heavy": "heavy",       # Strong separation
    "minimal": "ascii",     # Maximum compatibility
}
```

---

## Interactive Patterns

### 1. Input Prompts

#### Basic Prompt
```python
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML

session = PromptSession()

message = session.prompt(
    HTML('<ansimagenta><b>You:</b></ansimagenta> '),
    multiline=False,
)
```

**Visual:**
```
You: █ (cursor)
```

#### Multiline Prompt
```
You: (press Alt+Enter for multiline, Enter to send)
┃ First line of message
┃ Second line of message
┃ Third line of message█
```

### 2. Autocomplete

```
You: /mem█

Suggestions:
╭─────────────────────────────────────╮
│ /memory        Show memory dashboard │
│ /memory clear  Clear memory tier     │
│ /memory stats  Memory statistics     │
│ /memory export Export to file        │
╰─────────────────────────────────────╯
```

**Implementation:**
```python
from prompt_toolkit.completion import WordCompleter

commands = WordCompleter(
    ['/memory', '/memory clear', '/memory stats', '/help', '/quit'],
    meta_dict={
        '/memory': 'Show memory dashboard',
        '/help': 'Show help',
    },
    ignore_case=True,
)

session.prompt('You: ', completer=commands)
```

### 3. Confirmation Dialogs

```
╭─ Confirm Action ───────────────────────────────────────────────────╮
│                                                                    │
│ Are you sure you want to clear Long-Term Memory?                  │
│                                                                    │
│ This will permanently delete 342 entries.                         │
│ This action cannot be undone.                                     │
│                                                                    │
│ [Y]es  [N]o  (default: No)                                        │
│                                                                    │
╰────────────────────────────────────────────────────────────────────╯
```

**Implementation:**
```python
from rich.prompt import Confirm

confirmed = Confirm.ask(
    "Are you sure you want to clear Long-Term Memory?",
    default=False
)
```

### 4. Selection Menu

```
╭─ Select Model ─────────────────────────────────────────────────────╮
│                                                                    │
│ Available Models:                                                  │
│                                                                    │
│   ● gemma-2b-it (2.5GB, fastest)          [CURRENT]               │
│   ○ gemma-4b-it (4.8GB, better quality)                           │
│   ○ gemma-7b-it (8.5GB, highest quality)                          │
│   ○ codegemma-2b (2.5GB, code specialized)                        │
│                                                                    │
│ Use ↑↓ to navigate, Enter to select, Esc to cancel                │
│                                                                    │
╰────────────────────────────────────────────────────────────────────╯
```

**Implementation:**
```python
from rich.prompt import Prompt

choices = [
    "gemma-2b-it (2.5GB, fastest)",
    "gemma-4b-it (4.8GB, better quality)",
    "gemma-7b-it (8.5GB, highest quality)",
]

selected = Prompt.ask(
    "Select model",
    choices=["1", "2", "3"],
    default="1"
)
```

---

## Animation & Feedback

### Timing Guidelines

```python
ANIMATION_TIMINGS = {
    "instant": 0,           # No delay
    "fast": 0.1,            # Quick feedback
    "normal": 0.3,          # Standard transitions
    "slow": 0.5,            # Emphasized actions
    "very_slow": 1.0,       # Loading operations
}

REFRESH_RATES = {
    "status_bar": 0.5,      # 2 Hz
    "spinner": 0.1,         # 10 Hz
    "progress_bar": 0.2,    # 5 Hz
    "streaming_text": 0.05, # 20 Hz (smooth streaming)
}
```

### Streaming Text Animation

```python
from rich.live import Live
from time import sleep

def stream_response(text: str, console: Console):
    """Stream text character by character"""
    with Live(console=console, refresh_per_second=20) as live:
        buffer = ""
        for char in text:
            buffer += char
            live.update(Panel(buffer, border_style="bright_cyan"))
            sleep(0.05)  # 50ms per character
```

**Visual Effect:**
```
╭─ Gemma ────────────────────────────────────────────────────────────╮
│ The capital of Fr█                                                 │
╰────────────────────────────────────────────────────────────────────╯

# ... animates to ...

╭─ Gemma ────────────────────────────────────────────────────────────╮
│ The capital of France is Paris.█                                   │
╰────────────────────────────────────────────────────────────────────╯
```

### Loading States

```python
# Indeterminate (unknown duration)
⠋ Loading model...
⠙ Loading model...
⠹ Loading model...
⠸ Loading model...

# Determinate (known progress)
Loading model... ━━━━━━━━━━░░░░░░░░░░ 50% (2.5/5.0 GB)
```

### Success Animation

```python
# Before
⏳ Processing documents...

# After (with green checkmark)
✓ Documents processed successfully
```

---

## Accessibility

### High Contrast Mode

```python
HIGH_CONTRAST_THEME = {
    "foreground": "bright_white",
    "background": "black",
    "primary": "bright_yellow",
    "success": "bright_green",
    "error": "bright_red",
    "borders": "bright_white",
}
```

### Screen Reader Support

- Use semantic markup (titles, descriptions)
- Avoid ASCII art for critical information
- Provide text alternatives for progress indicators
- Announce state changes clearly

```python
# Good: Semantic announcement
console.print("[bold]Status:[/] Model loaded successfully")

# Avoid: Visual-only indicators
console.print("🎉 ✨ 🚀")  # Screen readers can't interpret
```

### Keyboard Navigation

| Key | Action |
|-----|--------|
| `Tab` | Autocomplete suggestion |
| `Ctrl+C` | Cancel current operation |
| `Ctrl+D` | Exit application |
| `Ctrl+L` | Clear screen |
| `↑` | Previous command history |
| `↓` | Next command history |
| `Alt+Enter` | Multiline input mode |
| `Esc` | Cancel input/dialog |

---

## User Flows

### 1. First-Time Setup

```
┌─ Step 1: Welcome ─────────────────────────────────────────────────┐
│ Welcome to Gemma CLI!                                             │
│ Let's set up your environment.                                    │
└───────────────────────────────────────────────────────────────────┘
                            ↓
┌─ Step 2: Model Selection ─────────────────────────────────────────┐
│ Select a model to download:                                       │
│   ○ gemma-2b-it (2.5GB, fastest)       [Recommended]             │
│   ○ gemma-4b-it (4.8GB, better quality)                          │
└───────────────────────────────────────────────────────────────────┘
                            ↓
┌─ Step 3: Download Progress ───────────────────────────────────────┐
│ Downloading gemma-2b-it...                                        │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━░░░░░░░░ 75% (1.9/2.5 GB)          │
│ Speed: 12 MB/s | ETA: 45s                                         │
└───────────────────────────────────────────────────────────────────┘
                            ↓
┌─ Step 4: Memory Setup ────────────────────────────────────────────┐
│ Enable RAG memory system?                                         │
│   ● Yes (requires Redis)                                          │
│   ○ No (conversation-only mode)                                   │
└───────────────────────────────────────────────────────────────────┘
                            ↓
┌─ Step 5: Ready ───────────────────────────────────────────────────┐
│ ✓ Setup complete!                                                 │
│ Type /help for available commands or start chatting.             │
└───────────────────────────────────────────────────────────────────┘
```

### 2. Typical Conversation Flow

```
User types message
        ↓
Display user message panel
        ↓
Show "Gemma is thinking..." spinner
        ↓
    [If RAG enabled]
        ↓
Show "Searching memory..." spinner
        ↓
Display memory recall results (if any)
        ↓
Stream response character by character
        ↓
Display metadata (tokens, time, memory tier)
        ↓
Update status bar with new stats
        ↓
Return to input prompt
```

### 3. Memory Management Flow

```
User types /memory
        ↓
Display memory dashboard
        ↓
User types /recall "query"
        ↓
Show search spinner
        ↓
Display search results table
        ↓
User types /ingest file.pdf
        ↓
Show progress bar (chunking → embedding → indexing)
        ↓
Display success message with stats
        ↓
Update memory dashboard
```

### 4. Error Recovery Flow

```
Error occurs
        ↓
Cancel current operation
        ↓
Display error panel with:
    • Clear error message
    • Technical details
    • Suggestions for resolution
    • Documentation links
        ↓
Return to safe state (input prompt)
        ↓
Log error to system logs
```

---

## Implementation Guide

### Project Structure

```python
gemma_cli/
├── ui/
│   ├── __init__.py
│   ├── theme.py              # Color schemes and styles
│   ├── components.py         # Reusable UI components
│   ├── layouts.py            # Layout managers
│   ├── animations.py         # Animation utilities
│   └── prompts.py            # Interactive prompts
├── cli.py                    # Main CLI entry point
└── config.py                 # User preferences
```

### Core Components Module

```python
# ui/components.py
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from typing import Optional

class UIComponents:
    """Reusable UI components following design system"""

    def __init__(self, console: Console, theme: dict):
        self.console = console
        self.theme = theme

    def message_panel(
        self,
        role: str,
        content: str,
        timestamp: str,
        metadata: Optional[dict] = None
    ) -> Panel:
        """Create a formatted message panel"""
        # Implementation
        pass

    def status_bar(
        self,
        model: str,
        memory: tuple,
        tokens: tuple,
        latency: int
    ) -> str:
        """Create status bar text"""
        # Implementation
        pass

    def memory_dashboard(self, memory_stats: dict) -> Panel:
        """Create memory system dashboard"""
        # Implementation
        pass

    def error_panel(
        self,
        title: str,
        details: str,
        suggestions: list[str]
    ) -> Panel:
        """Create error display panel"""
        # Implementation
        pass
```

### Theme Configuration

```python
# ui/theme.py
from rich.theme import Theme

def create_theme(mode: str = "dark") -> Theme:
    """Create Rich theme based on design system"""

    if mode == "dark":
        return Theme({
            "primary": "bright_cyan",
            "secondary": "cyan",
            "success": "bright_green",
            "warning": "yellow",
            "error": "bright_red",
            "user": "bright_magenta",
            "assistant": "bright_cyan",
            "system": "yellow",
            "code": "green",
            "dim": "grey50",
        })
    else:  # light mode
        return Theme({
            "primary": "blue",
            "secondary": "blue",
            "success": "green",
            "warning": "yellow",
            "error": "red",
            "user": "magenta",
            "assistant": "blue",
            "system": "yellow",
            "code": "green",
            "dim": "grey50",
        })
```

### Animation Manager

```python
# ui/animations.py
from rich.console import Console
from rich.live import Live
from time import sleep
from typing import Iterator

class AnimationManager:
    """Manage animations and streaming updates"""

    def __init__(self, console: Console):
        self.console = console

    def stream_text(
        self,
        text_generator: Iterator[str],
        panel_kwargs: dict
    ):
        """Stream text with live updates"""
        with Live(console=self.console, refresh_per_second=20) as live:
            buffer = ""
            for chunk in text_generator:
                buffer += chunk
                panel = Panel(buffer, **panel_kwargs)
                live.update(panel)
                sleep(0.05)

    def progress_bar(
        self,
        task_description: str,
        total: int
    ):
        """Create progress bar for long operations"""
        # Implementation
        pass
```

### Usage Example

```python
# cli.py
from rich.console import Console
from ui.components import UIComponents
from ui.theme import create_theme
from ui.animations import AnimationManager

def main():
    # Initialize
    console = Console(theme=create_theme("dark"))
    ui = UIComponents(console, theme)
    animator = AnimationManager(console)

    # Show startup banner
    console.print(ui.startup_banner())

    # Main conversation loop
    while True:
        user_input = Prompt.ask("[bright_magenta]You[/]")

        # Display user message
        console.print(ui.message_panel(
            role="You",
            content=user_input,
            timestamp=datetime.now()
        ))

        # Stream AI response
        response_generator = model.generate(user_input)
        animator.stream_text(
            response_generator,
            panel_kwargs={
                "title": "Gemma",
                "border_style": "bright_cyan"
            }
        )

        # Update status bar
        console.print(ui.status_bar(
            model="gemma-2b-it",
            memory=(45, 100),
            tokens=(1200, 8000),
            latency=45
        ))
```

---

## Style Guide Reference

### Quick Decision Matrix

| Scenario | Component | Style |
|----------|-----------|-------|
| User types message | Panel | `bright_magenta` border |
| AI responds | Panel | `bright_cyan` border |
| System notification | Text | `yellow` italic |
| Success operation | Panel | `bright_green` border |
| Error occurred | Panel | `bright_red` border |
| Loading/processing | Spinner | Context-specific icon |
| Long operation | Progress Bar | `bright_cyan` complete style |
| Data display | Table | `bright_cyan` borders |
| Grouped info | Panel | `bright_cyan` border |
| Memory recall | Nested Panel | `bright_blue` border |
| Code snippet | Syntax | `green` base color |
| Metadata | Text | `grey50` dim |

### Component Selection

**Use Panels when:**
- Grouping related content
- Displaying messages
- Showing errors/success states
- Creating dashboards

**Use Tables when:**
- Displaying structured data
- Comparing multiple items
- Listing tools/commands
- Showing statistics

**Use Spinners when:**
- Operation duration unknown
- Quick operations (<5s)
- Background processing

**Use Progress Bars when:**
- Operation duration known
- Long operations (>5s)
- File uploads/downloads
- Batch processing

---

## Conclusion

This design system provides a comprehensive foundation for building a modern, professional terminal UI for gemma-cli. Key principles:

1. **Consistency**: Reuse components and colors systematically
2. **Clarity**: Information hierarchy through typography and spacing
3. **Feedback**: Always show system state and progress
4. **Performance**: Smooth animations, responsive interactions
5. **Accessibility**: High contrast, keyboard navigation, semantic markup

**Next Steps:**
1. Implement core components module (`ui/components.py`)
2. Create theme configuration (`ui/theme.py`)
3. Build animation manager (`ui/animations.py`)
4. Integrate with existing CLI logic
5. Add user preferences for theme/layout customization
6. Conduct usability testing with target users

**Resources:**
- [Rich Documentation](https://rich.readthedocs.io/)
- [Prompt Toolkit Documentation](https://python-prompt-toolkit.readthedocs.io/)
- [Terminal Color Reference](https://rich.readthedocs.io/en/stable/appendix/colors.html)
