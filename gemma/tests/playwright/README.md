# Terminal UI Test Automation

Comprehensive test suite for gemma-cli terminal interface with snapshot generation and visual validation.

## Overview

This test suite provides automated testing for the gemma-cli terminal-based user interface, capturing visual snapshots and validating rendering, colors, and interactive behavior.

**Key Features:**
- ✅ Terminal session recording with asciinema
- ✅ Visual snapshot generation (SVG, PNG, HTML)
- ✅ Rich console output validation
- ✅ Color-coded UI component testing
- ✅ Interactive command flow testing
- ✅ Performance benchmarking
- ✅ Side-by-side snapshot comparison

## Architecture

### Terminal Testing Approach

Unlike Playwright (designed for web browsers), this framework uses:
- **pyte** - Terminal emulator for rendering capture
- **Rich** - Console output validation and SVG export
- **asciinema** - Terminal session recording
- **pytest-asyncio** - Async test execution
- **Pillow** - Image comparison and manipulation

### Test Organization

```
tests/playwright/
├── conftest.py                 # Pytest fixtures and configuration
├── config.py                   # Test settings and constants
├── README.md                   # This file
│
├── utils/                      # Testing utilities
│   ├── terminal_recorder.py   # Recording and snapshot generation
│   └── cli_runner.py           # Async CLI execution
│
├── test_startup.py             # Startup banner and initialization
├── test_chat_ui.py             # Chat interface components
├── test_memory_dashboard.py   # Memory visualization
├── test_command_palette.py    # Help and command system
└── test_integration.py         # End-to-end workflows
```

## Setup

### Prerequisites

```bash
# Install Python dependencies
pip install -r playwright_requirements.txt

# Install asciinema (optional, for terminal recordings)
# Windows: scoop install asciinema
# Linux: apt install asciinema
# macOS: brew install asciinema

# Install cairo for SVG to PNG conversion (optional)
# Windows: pip install cairosvg
# Linux: apt install python3-cairosvg
# macOS: brew install cairo && pip install cairosvg
```

### Dependencies

Core packages (from `playwright_requirements.txt`):
```
pytest>=7.4.0
pytest-asyncio>=0.21.0
pyte>=0.8.1
rich>=13.0.0
pillow>=10.0.0
asciinema>=2.3.0  # Optional
cairosvg>=2.7.0   # Optional
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest tests/playwright/ -v

# Run specific test file
pytest tests/playwright/test_chat_ui.py -v

# Run specific test
pytest tests/playwright/test_startup.py::test_startup_banner_displays -v

# Run with markers
pytest tests/playwright/ -m ui        # UI tests only
pytest tests/playwright/ -m slow      # Slow tests only
pytest tests/playwright/ -m integration  # Integration tests
```

### Test Filtering

```bash
# Skip slow tests
pytest tests/playwright/ -m "not slow"

# Run only snapshot tests
pytest tests/playwright/ -m snapshot

# Run memory-related tests
pytest tests/playwright/ -k memory

# Run chat UI tests
pytest tests/playwright/ -k chat
```

### Verbose Output

```bash
# Extra verbose with captured output
pytest tests/playwright/ -vvs

# Show test durations
pytest tests/playwright/ --durations=10

# Show stdout/stderr
pytest tests/playwright/ --capture=no
```

## Snapshot Management

### Viewing Snapshots

Snapshots are saved to `tests/playwright/screenshots/` organized by test name:

```
screenshots/
├── test_startup_banner_displays/
│   └── startup_banner.svg
├── test_user_message_format/
│   └── user_message_format.svg
└── test_memory_tier_bars/
    └── memory_tier_bars.svg
```

**View snapshots:**
- SVG: Open in web browser or image viewer
- PNG: Any image viewer
- HTML: Open in web browser for interactive viewing

### Snapshot Formats

```python
# Generate SVG (default)
snapshot_path = await snapshot_recorder.take_snapshot(
    output, "test_name", format="svg", theme="monokai"
)

# Generate PNG
snapshot_path = await snapshot_recorder.take_snapshot(
    output, "test_name", format="png"
)

# Generate HTML
snapshot_path = await snapshot_recorder.take_snapshot(
    output, "test_name", format="html"
)
```

### Snapshot Comparison

```python
# Generate side-by-side comparison
comparison = await snapshot_recorder.generate_comparison(
    snapshot1_path,
    snapshot2_path,
    "comparison_name"
)

# Generate diff highlighting
diff = await snapshot_recorder.generate_comparison(
    snapshot1_path,
    snapshot2_path,
    "diff_name",
    diff_only=True  # Only show differences
)
```

## Test Coverage

### Startup Tests (`test_startup.py`)

- ✅ Startup banner rendering
- ✅ System health checks display
- ✅ Model loading spinner animation
- ✅ Startup error handling
- ✅ Welcome message content
- ✅ Color output validation
- ✅ Startup performance

### Chat UI Tests (`test_chat_ui.py`)

- ✅ User message formatting (cyan panels)
- ✅ Assistant response formatting (green panels)
- ✅ Streaming text animation
- ✅ Error message display (red panels)
- ✅ Markdown code blocks with syntax highlighting
- ✅ Multiline input handling
- ✅ Conversation history display
- ✅ Token count and timing metadata
- ✅ Long response scrolling

### Memory Dashboard Tests (`test_memory_dashboard.py`)

- ✅ 5-tier memory progress bars
- ✅ Live dashboard updates
- ✅ Memory table rendering
- ✅ Capacity indicators
- ✅ Search results display
- ✅ Consolidation visualization
- ✅ Export preview
- ✅ Color-coded tiers
- ✅ Empty state display

### Command Palette Tests (`test_command_palette.py`)

- ✅ Help command display
- ✅ Chat help commands
- ✅ Command examples
- ✅ Subcommand help
- ✅ Command descriptions
- ✅ Option flags
- ✅ Version display
- ✅ Table formatting
- ✅ Invalid command suggestions
- ✅ Command categories

### Integration Tests (`test_integration.py`)

- ✅ Complete conversation flow
- ✅ Onboarding wizard
- ✅ Error recovery workflow
- ✅ Memory workflow (store/search/recall/export)
- ✅ Streaming with memory context
- ✅ Multi-session workflow
- ✅ Export/import conversation

## Configuration

### Test Settings (`config.py`)

```python
TEST_CONFIG = {
    "default_timeout": 30.0,
    "terminal_width": 120,
    "terminal_height": 40,
    "snapshot_format": "svg",
    "snapshot_theme": "monokai",
}
```

### Environment Variables

```bash
# Enable test mode
export GEMMA_TEST_MODE=1

# Set log level
export GEMMA_LOG_LEVEL=DEBUG

# Use test Redis database
export GEMMA_REDIS_DB=15

# Force colors
export FORCE_COLOR=1
```

## Writing New Tests

### Basic Test Template

```python
@pytest.mark.asyncio
@pytest.mark.ui
async def test_my_feature(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test my feature description."""
    # Run CLI command
    result = await cli_runner.run(
        ["python", "gemma-cli.py", "command"],
        timeout=10.0,
    )

    # Assertions
    assert result.returncode == 0
    assert "expected" in result.stdout.lower()

    # Capture snapshot
    snapshot_path = await snapshot_recorder.take_snapshot(
        result.terminal_display,
        "test_my_feature",
        format="svg",
    )

    assert snapshot_path.exists()
```

### Interactive Test Template

```python
@pytest.mark.asyncio
@pytest.mark.ui
async def test_interactive_feature(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test interactive command flow."""
    result = await cli_runner.run_interactive(
        ["python", "gemma-cli.py", "chat"],
        interactions=[
            (1.0, "First input"),      # Wait 1s, send input
            (2.0, "Second input"),     # Wait 2s, send input
            (1.0, "/exit"),            # Wait 1s, exit
        ],
        timeout=15.0,
    )

    # Validate and snapshot
    assert "expected" in result.stdout
    await snapshot_recorder.take_snapshot(
        result.terminal_display, "interactive_test"
    )
```

### Live Capture Template

```python
@pytest.mark.asyncio
@pytest.mark.slow
async def test_live_animation(
    cli_runner: AsyncCLIRunner,
    snapshot_recorder,
):
    """Test animated UI component."""
    # Start process
    process = await asyncio.create_subprocess_exec(
        "python", "gemma-cli.py", "command",
        stdout=asyncio.subprocess.PIPE,
    )

    # Capture frames
    snapshots = await snapshot_recorder.capture_live_process(
        process,
        "animation_test",
        duration=5.0,
    )

    # Cleanup
    process.terminate()
    await process.wait()

    # Verify
    assert len(snapshots) >= 3
```

## Troubleshooting

### Common Issues

**1. Tests timing out:**
```bash
# Increase timeout
pytest tests/playwright/ --timeout=60
```

**2. Snapshots not generated:**
```bash
# Check output directory permissions
ls -la tests/playwright/screenshots/

# Ensure Rich installed
pip install rich>=13.0.0
```

**3. Color codes not detected:**
```bash
# Force color output
export FORCE_COLOR=1
export NO_COLOR=0
pytest tests/playwright/
```

**4. Redis connection errors:**
```bash
# Start Redis
redis-server

# Use test database
export GEMMA_REDIS_DB=15
```

**5. Model loading errors:**
```bash
# Use mock model
export GEMMA_TEST_MODE=1

# Or specify valid model path
export GEMMA_MODEL_PATH=/path/to/model.sbs
```

### Debug Mode

```bash
# Enable debug logging
export GEMMA_LOG_LEVEL=DEBUG

# Run with verbose output
pytest tests/playwright/ -vvs --log-cli-level=DEBUG

# Save debug logs
pytest tests/playwright/ --log-file=test_debug.log
```

## Performance Benchmarks

Expected performance targets:

| Operation | Target | Slow |
|-----------|--------|------|
| Startup | < 5s | > 10s |
| Help display | < 2s | > 5s |
| Memory stats | < 10s | > 20s |
| Chat response | < 30s | > 60s |

Run benchmarks:
```bash
pytest tests/playwright/ --benchmark-only
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Terminal UI Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r playwright_requirements.txt

      - name: Run tests
        run: |
          pytest tests/playwright/ -v

      - name: Upload snapshots
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-snapshots
          path: tests/playwright/screenshots/
```

## Best Practices

1. **Always capture snapshots** for visual validation
2. **Use descriptive test names** for easy snapshot identification
3. **Test both success and error cases**
4. **Validate colors** to ensure proper UI rendering
5. **Keep tests isolated** using separate Redis databases
6. **Use mock models** for faster test execution
7. **Clean up resources** in finally blocks or fixtures
8. **Document expected behavior** in test docstrings

## Support

For issues or questions:
1. Check this README for common solutions
2. Review test logs in `tests/playwright/screenshots/`
3. Run with `-vvs` for detailed output
4. Check gemma-cli logs for errors

## License

Same license as gemma.cpp project.
