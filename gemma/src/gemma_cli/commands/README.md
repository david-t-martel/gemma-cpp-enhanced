# Gemma CLI Commands

Comprehensive command handlers for Phase 2 MCP and RAG features.

## Overview

This module provides CLI commands for:

- **RAG Memory System**: 5-tier memory architecture with Redis backend
- **MCP Integration**: Model Context Protocol server management
- **Document Ingestion**: Intelligent document chunking and storage
- **Memory Operations**: Store, recall, search, consolidate, cleanup

## Architecture

### Command Groups

```
gemma_cli.commands
├── rag_commands.py          # Main command implementations
│   ├── memory_commands      # Memory/RAG command group
│   └── mcp_commands         # MCP command group
└── __init__.py             # Module exports
```

### Dependencies

- **Click**: Command-line interface framework
- **Rich**: Terminal formatting and UI components
- **PythonRAGBackend**: Redis-backed memory system
- **Settings**: Configuration management

## Memory Commands

### `/memory dashboard`

Show comprehensive memory system statistics with real-time monitoring.

**Usage:**
```bash
gemma /memory dashboard
gemma /memory dashboard --refresh 5  # Auto-refresh every 5 seconds
```

**Output:**
- Memory tier counts (working, short_term, long_term, episodic, semantic)
- Total entries across all tiers
- Redis memory usage
- Capacity utilization

**Example:**
```
┌─ Memory System Dashboard ─┐
│ Tier         Count  Max    │
│ working         5    15    │
│ short_term     20   100    │
│ long_term     100 10000    │
│ episodic       50  5000    │
│ semantic      200 50000    │
│                            │
│ TOTAL         375         │
└────────────────────────────┘

Redis Memory Usage: 10.24 MB
```

### `/memory recall <query>`

Semantic similarity search across memory tiers using embeddings.

**Usage:**
```bash
gemma /recall "machine learning concepts"
gemma /recall "error handling" --tier=long_term --limit=10
gemma /recall "yesterday's discussion" --tier=episodic
```

**Options:**
- `--tier, -t`: Specific tier to search (working|short_term|long_term|episodic|semantic)
- `--limit, -l`: Maximum results (default: 5)

**How it works:**
1. Generates embedding for query text
2. Computes cosine similarity with stored embeddings
3. Ranks results by similarity × importance
4. Returns top N results

**Example Output:**
```
Found 3 relevant memories:

Result 1
ID             entry-abc123...
Type           long_term
Importance     0.85
Similarity     0.923
Access Count   5
Created        2025-01-10 14:30:00
Content        Machine learning uses statistical models to learn patterns...
Tags           ml, concepts
```

### `/memory store <text>`

Store content in specified memory tier with importance weighting.

**Usage:**
```bash
gemma /store "Python uses duck typing" --tier=semantic --importance=0.8
gemma /store "Meeting at 3pm tomorrow" --tier=episodic --tags=meeting --tags=reminder
gemma /store "Debugging findings" --tier=long_term --importance=0.7
```

**Options:**
- `--tier, -t`: Memory tier (default: short_term)
- `--importance, -i`: Score 0.0-1.0 (default: 0.5)
- `--tags, -g`: Tags (repeatable)

**Importance Guidelines:**
- **0.0-0.3**: Low importance (quick pruning)
- **0.4-0.6**: Normal importance
- **0.7-0.9**: High importance (retained longer)
- **1.0**: Critical importance (highest priority)

**Example:**
```bash
$ gemma /store "Redis connection pooling improves performance" \
    --tier=long_term \
    --importance=0.8 \
    --tags=redis --tags=performance

✓ Memory stored successfully
ID: entry-xyz789...
Tier: long_term
```

### `/memory search <query>`

Keyword-based search with importance filtering (exact substring matching).

**Usage:**
```bash
gemma /search "error" --min-importance=0.5
gemma /search "API" --tier=long_term
gemma /search "critical" --min-importance=0.8 --tier=semantic
```

**Options:**
- `--tier, -t`: Specific tier to search
- `--min-importance, -m`: Minimum importance threshold (default: 0.0)

**Difference from `/recall`:**
- `/recall`: Semantic similarity (uses embeddings)
- `/search`: Exact substring matching (faster for known keywords)

### `/memory ingest <file_path>`

Ingest documents by intelligent chunking and embedding.

**Usage:**
```bash
gemma /ingest docs/readme.md
gemma /ingest notes.txt --tier=semantic --chunk-size=1000
gemma /ingest data.json --tier=long_term
```

**Options:**
- `--tier, -t`: Storage tier (default: long_term)
- `--chunk-size, -c`: Chunk size in tokens/chars (default: 500)

**Supported Formats:**
- `.txt`: Plain text
- `.md`: Markdown
- `.html`: HTML documents
- `.json`: JSON data

**Chunking Strategy:**
- **With tiktoken**: Token-based chunking (intelligent)
- **Without tiktoken**: Sentence-based chunking (fallback)

**Example:**
```bash
$ gemma /ingest technical_docs.md --tier=semantic --chunk-size=800

Ingesting document: technical_docs.md
Chunk size: 800 | Tier: semantic

✓ Successfully ingested 23 chunks
```

### `/memory cleanup`

Remove expired memory entries across all tiers.

**Usage:**
```bash
gemma /cleanup
gemma /cleanup --dry-run  # Preview without deleting
```

**Options:**
- `--dry-run, -d`: Show what would be deleted without deleting

**Expiration Rules:**
- **Working**: 15 minutes
- **Short-term**: 1 hour
- **Long-term**: 30 days
- **Episodic**: 7 days
- **Semantic**: Never expires

**Example:**
```bash
$ gemma /cleanup

✓ Cleaned up 42 expired entries
```

### `/memory consolidate`

Run memory consolidation across tiers (Phase 2B feature).

**Usage:**
```bash
gemma /consolidate
gemma /consolidate --force  # Force even if threshold not met
```

**Consolidation Process:**
1. Analyze access patterns and importance
2. Promote frequently accessed memories to higher tiers
3. Demote rarely accessed memories to lower tiers
4. Merge duplicate or similar entries

**Note**: Full implementation coming in Phase 2B.

## MCP Commands

MCP (Model Context Protocol) commands for server management and tool execution.

### `/mcp status`

Show MCP server status and connection health.

**Usage:**
```bash
gemma /mcp status
```

**Example Output:**
```
┌─ MCP Server Status ─────┐
│ Server              Status          Tools │
│ filesystem          connected          12 │
│ redis-cache         connected           8 │
│ sequential-thinking not connected       0 │
└──────────────────────────────────────────┘
```

### `/mcp list [tools|resources|servers]`

List MCP items.

**Usage:**
```bash
gemma /mcp list servers     # List all MCP servers
gemma /mcp list tools       # List available tools
gemma /mcp list resources   # List available resources
```

### `/mcp call <server> <tool> <args...>`

Execute an MCP tool.

**Usage:**
```bash
gemma /mcp call filesystem read_file /path/to/file.txt
gemma /mcp call redis-cache get key123
gemma /mcp call sequential-thinking analyze "complex problem"
```

**Example:**
```bash
$ gemma /mcp call filesystem list_directory /home/user/projects

Files:
- project1/
- project2/
- notes.txt
- README.md
```

### `/mcp connect <server>`

Connect to an MCP server.

**Usage:**
```bash
gemma /mcp connect filesystem
gemma /mcp connect redis-cache
```

### `/mcp disconnect <server>`

Disconnect from an MCP server.

**Usage:**
```bash
gemma /mcp disconnect filesystem
```

### `/mcp health [server]`

Perform health check on MCP servers.

**Usage:**
```bash
gemma /mcp health              # Check all servers
gemma /mcp health filesystem   # Check specific server
```

**Example Output:**
```
Server: filesystem
Status: ✓ Healthy
Latency: 12ms
Uptime: 5h 23m

Server: redis-cache
Status: ✗ Unhealthy
Error: Connection timeout
```

## Configuration

Commands load settings from `config/config.toml`:

```toml
[redis]
host = "localhost"
port = 6380
db = 0
pool_size = 10
connection_timeout = 5
command_timeout = 10

[memory]
working_ttl = 900          # 15 minutes
short_term_ttl = 3600      # 1 hour
long_term_ttl = 2592000    # 30 days
episodic_ttl = 604800      # 7 days
semantic_ttl = 0           # Permanent

working_capacity = 15
short_term_capacity = 100
long_term_capacity = 10000
episodic_capacity = 5000
semantic_capacity = 50000

[embedding]
provider = "local"
model = "all-MiniLM-L6-v2"
dimension = 384

[document]
chunk_size = 512
chunk_overlap = 50
supported_formats = ["txt", "md", "html", "json", "pdf"]
```

## Error Handling

Commands provide user-friendly error messages with suggestions:

### Redis Connection Failure
```
[red]Failed to initialize RAG backend[/red]
[yellow]Make sure Redis is running on localhost:6380[/yellow]

Suggestions:
  1. Start Redis: redis-server --port 6380
  2. Check Redis status: redis-cli -p 6380 ping
  3. Verify config: cat config/config.toml
```

### Invalid Arguments
```
[red]Importance must be between 0.0 and 1.0[/red]

Example: gemma /store "text" --importance 0.8
```

### File Not Found
```
[red]File not found: /path/to/doc.txt[/red]

Check:
  - File path is correct
  - File has read permissions
  - File format is supported: txt, md, html, json, pdf
```

## Output Formatting

All commands use **Rich** for beautiful terminal output:

### Features
- **Tables**: Structured data with aligned columns
- **Panels**: Highlighted sections with borders
- **Progress**: Spinners for async operations
- **Colors**: Semantic color coding (green=success, red=error, yellow=warning, cyan=info)
- **Syntax**: Code highlighting for JSON, TOML, etc.

### Example
```python
# Rich table output
┌─────────────────────────────────────────┐
│ Memory System Dashboard                 │
├──────────────┬──────────┬───────────────┤
│ Tier         │ Count    │ Max Size      │
├──────────────┼──────────┼───────────────┤
│ working      │     5    │    15         │
│ short_term   │    20    │   100         │
│ long_term    │   100    │ 10000         │
└──────────────┴──────────┴───────────────┘
```

## Testing

Comprehensive test suite in `tests/unit/test_rag_commands.py`:

```bash
# Run all command tests
pytest tests/unit/test_rag_commands.py -v

# Run specific test class
pytest tests/unit/test_rag_commands.py::TestRecallCommand -v

# Run with coverage
pytest tests/unit/test_rag_commands.py --cov=gemma_cli.commands --cov-report=term-missing
```

**Test Coverage:**
- Command parsing and validation
- Async operation handling
- Error scenarios and recovery
- Output formatting
- Integration workflows

## Development

### Adding New Commands

1. **Define command function:**
```python
@memory_commands.command("new_cmd")
@click.argument("required_arg")
@click.option("--optional", "-o", default="value")
def new_command(required_arg: str, optional: str):
    """Command description for help text.

    Examples:
        gemma /memory new_cmd arg1
        gemma /memory new_cmd arg1 --optional custom
    """
    async def _impl():
        backend = await get_rag_backend()
        # Implementation
        console.print("[green]Success![/green]")

    asyncio.run(_impl())
```

2. **Add tests:**
```python
class TestNewCommand:
    def test_new_command_basic(self, cli_runner, mock_rag_backend):
        result = cli_runner.invoke(new_command, ["arg1"])
        assert result.exit_code == 0
```

3. **Update documentation:**
- Add to this README
- Add docstring with examples
- Update main CLI help

### Code Style

- **Type hints**: All functions must have type annotations
- **Docstrings**: Google-style with examples
- **Error handling**: User-friendly messages with suggestions
- **Async**: Use `asyncio.run()` for async operations
- **Rich output**: Consistent formatting patterns

### Performance

- **Connection pooling**: Reuse Redis connections
- **Batch operations**: Use pipelines for multiple operations
- **Lazy loading**: Initialize backends on first use
- **Progress indicators**: Show feedback for long operations

## Troubleshooting

### Common Issues

**1. Redis connection refused**
```bash
# Start Redis
redis-server --port 6380

# Test connection
redis-cli -p 6380 ping
```

**2. Embedding model download**
```bash
# Pre-download model (first run only)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**3. Config not found**
```bash
# Create default config
mkdir -p config
cp config/config.example.toml config/config.toml
```

## Future Enhancements

### Phase 2B
- [ ] Full MCP server integration
- [ ] Memory consolidation algorithm
- [ ] Graph-based semantic relationships
- [ ] Advanced search with filters
- [ ] Export/import memory snapshots
- [ ] Memory analytics and insights

### Phase 3
- [ ] Multi-user memory isolation
- [ ] Distributed memory across nodes
- [ ] Real-time memory replication
- [ ] Memory versioning and rollback
- [ ] Advanced compression strategies

## References

- [Click Documentation](https://click.palletsprojects.com/)
- [Rich Documentation](https://rich.readthedocs.io/)
- [Redis Documentation](https://redis.io/docs/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [SentenceTransformers](https://www.sbert.net/)

---

**Status**: Phase 2A Complete | Phase 2B In Progress
**Last Updated**: 2025-01-13
