# Quick Start Guide - Terminal UI Testing

Get started with gemma-cli terminal UI testing in 5 minutes.

## 1. Install Dependencies

### Windows
```powershell
cd tests\playwright
.\setup.ps1
```

### Linux/macOS
```bash
cd tests/playwright
chmod +x setup.sh
./setup.sh
```

### Manual Installation
```bash
pip install -r playwright_requirements.txt
```

## 2. Run Your First Test

```bash
# Run all tests
pytest tests/playwright/ -v

# Or use the test runner
python run_tests.py
```

## 3. View Snapshots

Snapshots are saved in `screenshots/` directory:
```bash
# Open in browser (SVG files)
open screenshots/test_startup_banner_displays/startup_banner.svg

# Windows
start screenshots\test_startup_banner_displays\startup_banner.svg
```

## 4. Run Specific Tests

```bash
# Test startup banner
pytest tests/playwright/test_startup.py::test_startup_banner_displays -v

# Test chat UI
pytest tests/playwright/test_chat_ui.py -v

# Test memory dashboard
pytest tests/playwright/test_memory_dashboard.py -v
```

## 5. Common Commands

```bash
# Run only UI tests
python run_tests.py -m ui

# Run without slow tests
python run_tests.py -m "not slow"

# Run with verbose output
python run_tests.py -v

# Clean old snapshots and run
python run_tests.py --clean
```

## What Gets Tested?

✅ **Startup**
- Banner rendering
- Health checks
- Model loading animation

✅ **Chat Interface**
- User messages (cyan panels)
- Assistant responses (green panels)
- Streaming animation
- Error handling (red panels)

✅ **Memory Dashboard**
- 5-tier progress bars
- Capacity indicators
- Search results
- Export/import

✅ **Commands**
- Help system
- Command palette
- Error suggestions

✅ **Integration**
- Complete workflows
- Multi-session handling
- Error recovery

## Snapshot Formats

```bash
# SVG (default - best for viewing)
python run_tests.py --snapshot-format=svg

# PNG (best for comparison)
python run_tests.py --snapshot-format=png

# HTML (interactive)
python run_tests.py --snapshot-format=html
```

## Troubleshooting

### Tests timeout
```bash
# Increase timeout
pytest tests/playwright/ --timeout=120
```

### Redis errors
```bash
# Start Redis
redis-server

# Or use mock mode
export GEMMA_TEST_MODE=1
```

### No snapshots generated
```bash
# Check Rich installed
pip install rich>=13.0.0

# Verify directory permissions
ls -la screenshots/
```

## Next Steps

1. Read [README.md](README.md) for detailed documentation
2. Write custom tests using templates in README
3. Integrate with CI/CD pipeline
4. Compare snapshots for visual regression testing

## Help

Run with verbose output to see what's happening:
```bash
pytest tests/playwright/ -vvs --log-cli-level=DEBUG
```

Check test configuration in `config.py` for customization options.
