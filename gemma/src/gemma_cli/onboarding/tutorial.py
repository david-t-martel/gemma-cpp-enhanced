"""Interactive tutorial for new users."""

import asyncio
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

console = Console()


class InteractiveTutorial:
    """Step-by-step tutorial with examples."""

    def __init__(self) -> None:
        """Initialize the tutorial."""
        self.completed_lessons: list[str] = []

    async def run(self) -> None:
        """Run interactive tutorial."""
        self._display_tutorial_intro()

        lessons = [
            ("basic_chat", "Basic Chat"),
            ("memory_system", "Memory System & RAG"),
            ("mcp_tools", "MCP Tools"),
            ("advanced_features", "Advanced Features"),
        ]

        for lesson_id, lesson_name in lessons:
            if not Confirm.ask(f"\nStart lesson: {lesson_name}?", default=True):
                continue

            console.print(f"\n[bold cyan]Lesson: {lesson_name}[/bold cyan]\n")

            if lesson_id == "basic_chat":
                await self._lesson_basic_chat()
            elif lesson_id == "memory_system":
                await self._lesson_memory_system()
            elif lesson_id == "mcp_tools":
                await self._lesson_mcp_tools()
            elif lesson_id == "advanced_features":
                await self._lesson_advanced_features()

            self.completed_lessons.append(lesson_id)

        self._display_tutorial_complete()

    def _display_tutorial_intro(self) -> None:
        """Display tutorial introduction."""
        intro_text = """
        [bold cyan]Gemma CLI Interactive Tutorial[/bold cyan]

        This tutorial will teach you:
        • Basic chat interactions
        • Using the 5-tier memory system
        • Working with MCP tools
        • Advanced features and commands

        Each lesson includes examples you can try.
        Feel free to skip lessons you're familiar with.
        """

        console.print(Panel(intro_text, border_style="cyan", padding=(1, 2)))

    async def _lesson_basic_chat(self) -> None:
        """Teach basic chat interaction."""
        markdown_content = """
# Basic Chat

## Starting a Conversation

Simply type your message and press Enter:

```
You: Hello! What can you help me with?
```

The assistant will respond with helpful information.

## Chat Commands

Use slash commands for special operations:

- `/help` - Show all available commands
- `/clear` - Clear conversation history
- `/status` - Show session information
- `/settings` - View current configuration
- `/quit` - Exit the application

## Context Awareness

Gemma CLI maintains conversation context automatically:

```
You: What's the capital of France?
Assistant: The capital of France is Paris.

You: What's its population?
Assistant: Paris has approximately 2.2 million residents...
```

The assistant remembers "its" refers to Paris from previous context.

## Interrupting Generation

Press `Ctrl+C` during generation to stop the response.
The partial response will be discarded.
"""

        console.print(Markdown(markdown_content))

        if Confirm.ask("\nTry a basic chat command?", default=False):
            example_cmd = Prompt.ask(
                "Enter a command to try",
                default="/help",
            )
            console.print(
                f"\n[dim]In a real session, you would run: {example_cmd}[/dim]\n"
            )

        self._wait_for_next()

    async def _lesson_memory_system(self) -> None:
        """Teach memory/RAG features."""
        markdown_content = """
# Memory System & RAG

## 5-Tier Memory Architecture

Gemma CLI uses a hierarchical memory system:

1. **Working Memory** (15 min) - Active conversation context
2. **Short-term** (1 hour) - Recent interactions
3. **Long-term** (30 days) - Important information
4. **Episodic** (7 days) - Event sequences
5. **Semantic** (permanent) - Knowledge base

## Storing Memories

Store information in a specific tier:

```
/store "Project deadline is Dec 15" long_term 0.9
```

Arguments:
- Text to store (quoted)
- Memory tier (optional, default: short_term)
- Importance 0-1 (optional, default: 0.5)

## Recalling Memories

Search memories by semantic similarity:

```
/recall "project deadlines" long_term 5
```

This retrieves the 5 most relevant memories about project deadlines.

## Searching Memories

Search by exact content match:

```
/search "December" long_term 0.7
```

Finds memories containing "December" with importance >= 0.7.

## Ingesting Documents

Load entire documents into memory:

```
/ingest C:\\docs\\project_notes.txt long_term
```

The document is automatically chunked and stored with embeddings.

## Memory Statistics

View memory usage across tiers:

```
/memory_stats
```

Shows count and capacity for each tier.

## Cleanup

Remove expired memories:

```
/cleanup
```
"""

        console.print(Markdown(markdown_content))

        if Confirm.ask("\nView example memory commands?", default=True):
            examples = [
                ("/store", 'Store information: /store "Important note" long_term 0.8'),
                ("/recall", "Recall similar: /recall \"project info\" long_term 3"),
                ("/memory_stats", "View statistics: /memory_stats"),
            ]

            console.print("\n[bold]Example Commands:[/bold]\n")
            for cmd, desc in examples:
                console.print(f"  [cyan]{cmd:15}[/cyan] {desc}")

        self._wait_for_next()

    async def _lesson_mcp_tools(self) -> None:
        """Teach MCP integration."""
        markdown_content = """
# MCP Tools

## What is MCP?

Model Context Protocol (MCP) allows Gemma to interact with external tools:

- File system operations
- Web searches
- API integrations
- Database queries
- Custom tools

## Enabling MCP

MCP is enabled in your configuration. No additional setup needed!

## Available MCP Servers

Common MCP servers include:

1. **Filesystem** - Read/write files, search directories
2. **Web Search** - Search the internet
3. **GitHub** - Repository operations
4. **Database** - Query databases
5. **Custom** - Your own tools

## How It Works

When you ask Gemma to perform a task that requires external data:

```
You: Read the README.md file and summarize it
```

Gemma will:
1. Recognize it needs file system access
2. Call the MCP filesystem server
3. Read the file
4. Summarize the contents

All automatically!

## MCP Configuration

Configure MCP servers in: `config/mcp_servers.toml`

Example configuration:

```toml
[servers.filesystem]
enabled = true
allowed_paths = ["/home/user/projects"]

[servers.web_search]
enabled = true
api_key = "your-api-key"
```

## Checking MCP Status

```
gemma-cli health
```

Shows status of all MCP servers.
"""

        console.print(Markdown(markdown_content))

        self._wait_for_next()

    async def _lesson_advanced_features(self) -> None:
        """Teach advanced features."""
        markdown_content = """
# Advanced Features

## Saving Conversations

Save your chat history:

```
/save my_conversation.json
```

Files are saved to: `~/.gemma_conversations/`

## Loading Conversations

Resume a previous conversation:

```
/load my_conversation.json
```

## Listing Conversations

```
/list
```

Shows all saved conversations with timestamps.

## Session Management

View session information:

```
/status
```

Shows:
- Message count
- Session duration
- Memory usage
- Active context

## Configuration Tuning

Advanced settings in `~/.gemma_cli/config.toml`:

```toml
[conversation]
max_context_length = 8192
max_history_messages = 50
auto_save = true

[system]
enable_rag_context = true
max_rag_context_tokens = 2000
```

## Performance Monitoring

Enable detailed monitoring:

```toml
[monitoring]
enabled = true
track_latency = true
track_memory = true
track_token_usage = true
```

## Model Selection

Switch models by updating configuration:

```toml
[gemma]
default_model = "C:/path/to/model.sbs"
default_tokenizer = "C:/path/to/tokenizer.spm"
```

Then restart gemma-cli.

## Reconfiguration

Re-run setup wizard anytime:

```
gemma-cli init --force
```

## Getting Help

- Documentation: Check README files
- Health check: `gemma-cli health`
- Configuration: `gemma-cli config --show`
"""

        console.print(Markdown(markdown_content))

        self._wait_for_next()

    def _display_tutorial_complete(self) -> None:
        """Display tutorial completion message."""
        completed_count = len(self.completed_lessons)
        total_count = 4

        completion_text = f"""
        [bold green]Tutorial Complete![/bold green]

        You completed {completed_count} of {total_count} lessons.

        [bold]Ready to Start:[/bold]
        • Run: [cyan]gemma-cli chat[/cyan]
        • Help: [cyan]gemma-cli --help[/cyan]
        • Health: [cyan]gemma-cli health[/cyan]

        [bold]Resources:[/bold]
        • Documentation: README.md
        • Configuration: ~/.gemma_cli/config.toml
        • Conversations: ~/.gemma_conversations/

        [dim]Restart tutorial: gemma-cli tutorial[/dim]
        """

        console.print(Panel(completion_text, border_style="green", padding=(1, 2)))

    def _wait_for_next(self) -> None:
        """Wait for user to continue."""
        if not Confirm.ask("\n[dim]Continue to next section?[/dim]", default=True):
            console.print("[yellow]Skipping remaining tutorial.[/yellow]")
            raise SystemExit(0)


async def run_quick_start() -> None:
    """Run a quick-start guide (shorter than full tutorial)."""
    console.print("[bold cyan]Gemma CLI Quick Start[/bold cyan]\n")

    quick_start = """
# Quick Start Guide

## Basic Usage

1. **Start chatting:**
   ```
   gemma-cli chat
   ```

2. **Type your message:**
   ```
   You: Hello! How do I use this?
   ```

3. **View commands:**
   ```
   /help
   ```

## Essential Commands

- `/help` - Show all commands
- `/clear` - Clear conversation
- `/save` - Save conversation
- `/quit` - Exit

## Memory Commands (if Redis enabled)

- `/store "text" tier importance` - Store memory
- `/recall "query"` - Search memories
- `/memory_stats` - View statistics

## Configuration

- Config file: `~/.gemma_cli/config.toml`
- Reconfigure: `gemma-cli init --force`
- Health check: `gemma-cli health`

That's it! You're ready to use Gemma CLI.
"""

    console.print(Markdown(quick_start))
    console.print("\n[dim]For detailed tutorial: gemma-cli tutorial[/dim]\n")
