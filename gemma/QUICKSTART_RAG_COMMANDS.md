# Quick Start Guide: RAG Commands

Get started with Gemma CLI RAG commands in 5 minutes.

## Prerequisites

1. **Redis Server** (required)
2. **Python 3.11+** (required)
3. **Dependencies** (install below)

## Installation

### 1. Start Redis

```bash
# Install Redis (if not already installed)
# Windows: Download from https://redis.io/download
# Linux/macOS: brew install redis

# Start Redis on port 6380
redis-server --port 6380
```

### 2. Install Dependencies

```bash
cd C:\codedev\llm\gemma

# Install required packages
pip install click rich redis numpy pydantic aiofiles

# Install optional packages (recommended)
pip install sentence-transformers tiktoken
```

### 3. Verify Installation

```bash
# Test Redis connection
redis-cli -p 6380 ping
# Expected output: PONG

# Check Python imports
python -c "import click, rich, redis; print('✓ Dependencies OK')"
```

## Basic Usage

### Store Your First Memory

```bash
python -c "
from gemma_cli.commands.rag_commands import store_command
from click.testing import CliRunner

runner = CliRunner()
result = runner.invoke(store_command, [
    'Python is a dynamically typed language',
    '--tier', 'semantic',
    '--importance', '0.8'
])
print(result.output)
"
```

**Expected Output**:
```
✓ Memory stored successfully
ID: entry-abc123...
Tier: semantic
```

### Recall Memories

```bash
python -c "
from gemma_cli.commands.rag_commands import recall_command
from click.testing import CliRunner

runner = CliRunner()
result = runner.invoke(recall_command, ['Python programming'])
print(result.output)
"
```

### View Memory Dashboard

```bash
python -c "
from gemma_cli.commands.rag_commands import memory_commands
from click.testing import CliRunner

runner = CliRunner()
result = runner.invoke(memory_commands, ['dashboard'])
print(result.output)
"
```

## Interactive Demo

Run the comprehensive demo script:

```bash
python examples/demo_rag_commands.py
```

This will demonstrate:
- ✅ Storing memories in different tiers
- ✅ Semantic recall
- ✅ Keyword search
- ✅ Document ingestion
- ✅ Cleanup operations

## Common Commands

### 1. Store Important Information

```bash
# Permanent semantic memory
gemma /store "Key concept or fact" --tier=semantic --importance=0.9

# Long-term memory
gemma /store "Project notes" --tier=long_term --importance=0.7

# Short-term reminder
gemma /store "Quick note" --tier=short_term
```

### 2. Recall Similar Information

```bash
# Basic recall
gemma /recall "your query"

# Limit results
gemma /recall "your query" --limit=10

# Search specific tier
gemma /recall "your query" --tier=semantic
```

### 3. Search by Keyword

```bash
# Basic search
gemma /search "keyword"

# Filter by importance
gemma /search "critical" --min-importance=0.8

# Search specific tier
gemma /search "API" --tier=long_term
```

### 4. Ingest Documents

```bash
# Basic ingestion
gemma /ingest document.txt

# Custom chunk size
gemma /ingest large_doc.md --chunk-size=1000

# Store in semantic tier
gemma /ingest reference.txt --tier=semantic
```

### 5. Maintenance

```bash
# View statistics
gemma /memory dashboard

# Preview cleanup
gemma /cleanup --dry-run

# Execute cleanup
gemma /cleanup
```

## Configuration

Create `config/config.toml`:

```toml
[redis]
host = "localhost"
port = 6380
db = 0
pool_size = 10

[memory]
working_ttl = 900          # 15 minutes
short_term_ttl = 3600      # 1 hour
long_term_ttl = 2592000    # 30 days

[embedding]
provider = "local"
model = "all-MiniLM-L6-v2"
dimension = 384
```

## Troubleshooting

### Redis Connection Failed

```bash
# Check Redis is running
redis-cli -p 6380 ping

# If not running, start it
redis-server --port 6380
```

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade click rich redis numpy pydantic

# Check Python path
python -c "import sys; print(sys.path)"
```

### Embedding Model Download

First run may take 1-2 minutes to download the embedding model:

```bash
# Pre-download the model
python -c "from sentence_transformers import SentenceTransformer; \
  SentenceTransformer('all-MiniLM-L6-v2')"
```

## Testing

Run the test suite:

```bash
# Install pytest
pip install pytest pytest-asyncio

# Run tests
pytest tests/unit/test_rag_commands.py -v

# With coverage
pytest tests/unit/test_rag_commands.py --cov=gemma_cli.commands
```

## Next Steps

1. **Read Full Documentation**: `src/gemma_cli/commands/README.md`
2. **Explore Examples**: `examples/demo_rag_commands.py`
3. **Customize Config**: Edit `config/config.toml`
4. **Build Integrations**: Use commands in your workflows

## Help

For command-specific help:

```bash
gemma /memory --help
gemma /recall --help
gemma /store --help
```

## Quick Reference

| Command | Purpose | Example |
|---------|---------|---------|
| `/store` | Save memory | `gemma /store "text" --tier=semantic` |
| `/recall` | Semantic search | `gemma /recall "query" --limit=5` |
| `/search` | Keyword search | `gemma /search "word" --min-importance=0.7` |
| `/ingest` | Import document | `gemma /ingest file.txt --chunk-size=500` |
| `/cleanup` | Remove expired | `gemma /cleanup` |
| `/memory dashboard` | View stats | `gemma /memory dashboard` |

## Support

- **Documentation**: `src/gemma_cli/commands/README.md`
- **Examples**: `examples/demo_rag_commands.py`
- **Tests**: `tests/unit/test_rag_commands.py`
- **Issues**: Report bugs in project issue tracker

---

**Ready to start?** Run `python examples/demo_rag_commands.py` now!
