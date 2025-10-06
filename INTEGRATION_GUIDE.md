# LLM Development Ecosystem - Integration Guide

## System Overview

This guide covers the integration between three core components:
- **Gemma C++ Engine**: High-performance inference with gemma.exe
- **RAG-Redis System**: Multi-tier memory and document management
- **Python Agent Framework**: Orchestration and tool integration

### Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                           User Interface                            │
│                    (CLI / HTTP Server / MCP Client)                │
└──────────────────────┬─────────────────────────┬──────────────────┘
                       │                         │
                       ▼                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                     gemma-cli.py Wrapper                          │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ • Session Management    • Context Window (8K-16K tokens) │    │
│  │ • Command Processing    • Memory Consolidation           │    │
│  │ • RAG Integration       • Conversation History           │    │
│  └──────────────────────────────────────────────────────────┘    │
└────────┬──────────────────────┬──────────────────┬───────────────┘
         │                      │                  │
         ▼                      ▼                  ▼
┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
│  gemma.exe      │  │  Redis Server    │  │  Embedding Model    │
│  C++ Inference  │  │  Memory Store    │  │  Sentence-BERT      │
├─────────────────┤  ├──────────────────┤  ├─────────────────────┤
│ • SFP Models    │  │ • 5 Memory Tiers │  │ • Document Vectors  │
│ • 2B/4B/7B      │  │ • Vector Search  │  │ • Similarity Search │
│ • Highway SIMD  │  │ • TTL Management │  │ • Semantic Matching │
└─────────────────┘  └──────────────────┘  └─────────────────────┘
```

### Component Relationships

1. **Data Flow**:
   - User Input → gemma-cli.py → RAG Retrieval → Context Enhancement → gemma.exe → Response
   - Document Ingestion → Embedding Generation → Redis Storage → Vector Index

2. **Memory Tiers** (Redis-backed):
   - **Working Memory**: 10 items, immediate context (TTL: 1 hour)
   - **Short-term**: 100 items, recent interactions (TTL: 24 hours)
   - **Long-term**: 10,000 items, consolidated facts (TTL: 30 days)
   - **Episodic**: Event sequences with timestamps (TTL: 90 days)
   - **Semantic**: Graph-based concept relationships (No TTL)

3. **Session Management**:
   - Conversation state in JSON files
   - Context window management (8K-16K tokens)
   - Automatic memory consolidation
   - Session save/load capabilities

## Setup Instructions

### 1. Building gemma.exe (Windows Native)

```bash
# Prerequisites
# - Visual Studio 2022 (v143 toolset)
# - CMake 3.20+
# - Git

# Clone and prepare
cd /c/codedev/llm/gemma
git submodule update --init --recursive

# Configure build (Visual Studio 2022)
cmake -B build -G "Visual Studio 17 2022" -T v143 \
  -DCMAKE_BUILD_TYPE=Release \
  -DWEIGHT_TYPE=sfp \
  -DBUILD_SINGLE_FILE_INFERENCE=ON

# Build (use all cores)
cmake --build build --config Release -j %NUMBER_OF_PROCESSORS%

# Verify build
./build/bin/RELEASE/gemma.exe --help

# Expected output files:
# - build/bin/RELEASE/gemma.exe
# - build/bin/RELEASE/single_benchmark.exe
# - build/bin/RELEASE/migrate_weights.exe
```

### 2. Installing Python Dependencies

```bash
# Navigate to stats directory
cd /c/codedev/llm/stats

# Create virtual environment with uv (recommended)
uv venv
uv sync --all-groups

# Install core dependencies
uv pip install redis numpy sentence-transformers tiktoken colorama

# Install development dependencies
uv pip install pytest pytest-cov ruff mypy

# Verify installation
uv run python -c "import redis, numpy, sentence_transformers; print('Dependencies OK')"
```

### 3. Starting Redis Server

```bash
# Windows (WSL or native)
redis-server --port 6379 --maxmemory 2gb --maxmemory-policy allkeys-lru

# Alternative: Docker
docker run -d -p 6379:6379 --name redis-rag \
  -v redis-data:/data \
  redis:7-alpine redis-server --appendonly yes

# Verify connection
redis-cli ping
# Should return: PONG
```

### 4. Configuring RAG System

```bash
# Build Rust components (optional, for performance)
cd /c/codedev/llm/stats/rag-redis-system
cargo build --release --features full

# Set environment variables
export REDIS_URL="redis://localhost:6379"
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export GEMMA_MODEL_PATH="/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs"
export GEMMA_TOKENIZER_PATH="/c/codedev/llm/.models/tokenizer.spm"
```

## Using gemma-cli.py

### Basic Chat Usage

```bash
# Simple chat interface
python /c/codedev/llm/gemma/gemma-cli.py \
  --model /c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs \
  --temperature 0.7 \
  --max-tokens 2048

# With increased context window
python gemma-cli.py \
  --model /c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs \
  --max-context 16384 \
  --max-tokens 4096 \
  --temperature 0.8
```

### RAG Commands with Examples

```bash
# Enable RAG system
python gemma-cli.py \
  --model /c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs \
  --enable-rag \
  --redis-host localhost \
  --redis-port 6379

# During chat, use these commands:
# /rag help                 - Show RAG commands
# /rag ingest <file>        - Ingest a document
# /rag search <query>       - Search memories
# /rag stats                - Show memory statistics
# /rag consolidate          - Force memory consolidation
# /rag clear <tier>         - Clear specific memory tier
```

#### Example RAG Session

```python
# Interactive session example
User> /rag ingest /path/to/technical_docs.txt
System> ✅ Ingested document: doc_7834_15234 (342 chunks)

User> What are the main components of our system?
System> Based on the ingested documentation, the main components are:
[RAG Context: Retrieved 3 relevant chunks with similarity > 0.75]
1. Gemma C++ inference engine for model execution
2. Redis-based multi-tier memory system
3. Python orchestration layer with tool integration
...

User> /rag stats
System> Memory Statistics:
- Working: 8/10 entries
- Short-term: 45/100 entries
- Long-term: 234/10000 entries
- Episodic: 12 sequences
- Semantic: 56 concepts
```

### Memory Tier Management

```python
#!/usr/bin/env python3
"""Example: Managing memory tiers programmatically"""

import asyncio
import sys
sys.path.append('/c/codedev/llm/gemma')

async def manage_memory():
    # Import the gemma-cli module
    import importlib.util
    spec = importlib.util.spec_from_file_location("gemma_cli", "gemma-cli.py")
    gemma_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemma_cli)

    # Initialize RAG manager
    rag = gemma_cli.RAGRedisManager()
    await rag.initialize()

    # Store memories in different tiers
    memories = [
        ("Current task: integrate RAG system", gemma_cli.MemoryTier.WORKING, 0.9),
        ("User prefers Python examples", gemma_cli.MemoryTier.SHORT_TERM, 0.7),
        ("System architecture uses Redis", gemma_cli.MemoryTier.LONG_TERM, 0.8),
        ("First conversation on Sept 24", gemma_cli.MemoryTier.EPISODIC, 0.6),
        ("RAG means Retrieval Augmented Generation", gemma_cli.MemoryTier.SEMANTIC, 0.9),
    ]

    for content, tier, importance in memories:
        entry_id = await rag.store_memory(content, tier, importance)
        print(f"Stored in {tier}: {entry_id}")

    # Recall relevant memories
    query = "What is RAG?"
    results = await rag.recall_memories(query, limit=3)

    for memory in results:
        print(f"[{memory.memory_type}] {memory.content}")
        print(f"  Relevance: {memory.similarity_score:.3f}")

    # Consolidate memories (promote important ones)
    await rag.consolidate_memories()

    # Get statistics
    stats = await rag.get_memory_stats()
    print(f"Total memories: {stats['total']}")

if __name__ == "__main__":
    asyncio.run(manage_memory())
```

### Document Ingestion Walkthrough

```python
#!/usr/bin/env python3
"""Complete document ingestion example"""

import asyncio
from pathlib import Path

async def ingest_documents():
    # Setup
    import sys
    sys.path.append('/c/codedev/llm/gemma')
    import importlib.util
    spec = importlib.util.spec_from_file_location("gemma_cli", "gemma-cli.py")
    gemma_cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemma_cli)

    rag = gemma_cli.RAGRedisManager()
    await rag.initialize()

    # 1. Ingest a single document
    doc_path = Path("/path/to/document.txt")
    if doc_path.exists():
        content = doc_path.read_text()
        doc_id = await rag.ingest_document(
            content,
            metadata={
                "source": str(doc_path),
                "type": "technical_doc",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        print(f"Ingested: {doc_id}")

    # 2. Batch ingestion with chunking
    docs_dir = Path("/path/to/docs")
    for doc_file in docs_dir.glob("*.txt"):
        content = doc_file.read_text()

        # Chunk large documents
        chunks = gemma_cli.chunk_text(content, chunk_size=512, overlap=50)

        for i, chunk in enumerate(chunks):
            chunk_id = await rag.ingest_document(
                chunk,
                metadata={
                    "source": str(doc_file),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            print(f"  Chunk {i+1}/{len(chunks)}: {chunk_id}")

    # 3. Query the ingested documents
    query = "How does the system handle memory management?"
    results = await rag.search_documents(query, limit=5)

    for doc in results:
        print(f"Document: {doc['id']}")
        print(f"Relevance: {doc['score']:.3f}")
        print(f"Content: {doc['content'][:200]}...")
        print()

if __name__ == "__main__":
    asyncio.run(ingest_documents())
```

## Session Management

### How Sessions Work

Sessions maintain conversation state across interactions:

1. **Session Files**: Stored as JSON in `sessions/` directory
2. **Context Management**: Tracks token usage and manages overflow
3. **Memory Integration**: Links to Redis memory tiers
4. **Persistence**: Auto-save every N turns or on exit

### Saving/Loading Conversations

```bash
# Start with session management
python gemma-cli.py \
  --model /c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs \
  --session my_project_session \
  --auto-save

# Commands during chat:
# /session save [name]     - Save current session
# /session load <name>     - Load a session
# /session list           - List available sessions
# /session clear          - Clear current session
# /session export <path>  - Export to JSON file
```

### Context Window Handling

```python
#!/usr/bin/env python3
"""Context window management example"""

class ContextManager:
    def __init__(self, max_tokens=8192, model="gemma-2b"):
        self.max_tokens = max_tokens
        self.model = model
        self.context = []
        self.token_count = 0

    def add_turn(self, user_input, assistant_response):
        """Add a conversation turn, managing overflow."""
        turn = {
            "user": user_input,
            "assistant": assistant_response,
            "tokens": self.count_tokens(user_input + assistant_response)
        }

        self.context.append(turn)
        self.token_count += turn["tokens"]

        # Handle overflow
        while self.token_count > self.max_tokens * 0.9:  # Keep 10% buffer
            if len(self.context) > 1:
                removed = self.context.pop(0)
                self.token_count -= removed["tokens"]
                print(f"Context overflow: removed turn with {removed['tokens']} tokens")
            else:
                break

    def get_context_string(self):
        """Get formatted context for model input."""
        messages = []
        for turn in self.context:
            messages.append(f"User: {turn['user']}")
            messages.append(f"Assistant: {turn['assistant']}")
        return "\n".join(messages)

    def count_tokens(self, text):
        """Estimate token count (rough approximation)."""
        # More accurate with tiktoken if available
        try:
            from tiktoken import get_encoding
            enc = get_encoding("cl100k_base")
            return len(enc.encode(text))
        except:
            # Fallback: rough estimate
            return len(text) // 4

# Usage
ctx_mgr = ContextManager(max_tokens=8192)
ctx_mgr.add_turn("What is RAG?", "RAG stands for Retrieval Augmented Generation...")
ctx_mgr.add_turn("How does it work?", "It works by first retrieving relevant context...")
print(f"Current context uses {ctx_mgr.token_count} tokens")
```

## Code Examples

### Python: Using the CLI Programmatically

```python
#!/usr/bin/env python3
"""Programmatic usage of gemma-cli.py"""

import subprocess
import json
import asyncio
from pathlib import Path

class GemmaInterface:
    def __init__(self, model_path, tokenizer_path=None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.process = None

    async def start(self):
        """Start the Gemma process."""
        cmd = [
            "python", "/c/codedev/llm/gemma/gemma-cli.py",
            "--model", self.model_path,
            "--max-tokens", "2048",
            "--temperature", "0.7",
            "--enable-rag",
            "--json-mode"  # Enable JSON output for parsing
        ]

        if self.tokenizer_path:
            cmd.extend(["--tokenizer", self.tokenizer_path])

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

    async def query(self, prompt, use_rag=True):
        """Send a query and get response."""
        if not self.process:
            await self.start()

        # Prepare query
        query_data = {
            "prompt": prompt,
            "use_rag": use_rag,
            "max_tokens": 1024
        }

        # Send query
        query_json = json.dumps(query_data) + "\n"
        self.process.stdin.write(query_json.encode())
        await self.process.stdin.drain()

        # Read response
        response_line = await self.process.stdout.readline()
        response = json.loads(response_line.decode())

        return response

    async def close(self):
        """Close the process."""
        if self.process:
            self.process.terminate()
            await self.process.wait()

# Usage example
async def main():
    gemma = GemmaInterface(
        model_path="/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs",
        tokenizer_path="/c/codedev/llm/.models/tokenizer.spm"
    )

    # Query with RAG
    response = await gemma.query(
        "What are the best practices for memory management?",
        use_rag=True
    )

    print(f"Response: {response['text']}")
    print(f"RAG Context Used: {response.get('rag_context', 'None')}")
    print(f"Tokens Used: {response.get('tokens_used', 0)}")

    await gemma.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### C++: Integrating Session Framework

```cpp
// session_framework.hpp
#pragma once

#include <string>
#include <vector>
#include <json/json.h>
#include "gemma.h"

namespace gemma {

class SessionFramework {
public:
    struct Turn {
        std::string user_input;
        std::string assistant_response;
        int token_count;
        double timestamp;
    };

    struct Session {
        std::string id;
        std::vector<Turn> history;
        int total_tokens;
        Json::Value metadata;
    };

private:
    Session current_session_;
    Gemma* model_;
    int max_context_tokens_;

public:
    SessionFramework(Gemma* model, int max_context = 8192)
        : model_(model), max_context_tokens_(max_context) {
        current_session_.id = generate_session_id();
        current_session_.total_tokens = 0;
    }

    // Process input with context management
    std::string process_with_context(const std::string& input) {
        // Build context from history
        std::string context = build_context();

        // Add current input
        std::string full_prompt = context + "\nUser: " + input + "\nAssistant: ";

        // Generate response
        RuntimeConfig config;
        config.max_generated_tokens = 2048;
        config.temperature = 0.7f;

        std::string response;
        std::vector<int> prompt_tokens;

        model_->Tokenizer()->Encode(full_prompt, &prompt_tokens);

        auto stream_callback = [&response](int token, float) {
            std::string piece;
            model_->Tokenizer()->Decode({token}, &piece);
            response += piece;
            return true;  // Continue generation
        };

        model_->Generate(config, prompt_tokens, /*start_pos=*/0, stream_callback);

        // Store turn
        Turn turn;
        turn.user_input = input;
        turn.assistant_response = response;
        turn.token_count = prompt_tokens.size() + count_tokens(response);
        turn.timestamp = get_timestamp();

        current_session_.history.push_back(turn);
        current_session_.total_tokens += turn.token_count;

        // Manage context window overflow
        manage_context_window();

        return response;
    }

    // Save session to file
    void save_session(const std::string& filepath) {
        Json::Value root;
        root["id"] = current_session_.id;
        root["total_tokens"] = current_session_.total_tokens;
        root["metadata"] = current_session_.metadata;

        Json::Value history(Json::arrayValue);
        for (const auto& turn : current_session_.history) {
            Json::Value turn_json;
            turn_json["user"] = turn.user_input;
            turn_json["assistant"] = turn.assistant_response;
            turn_json["tokens"] = turn.token_count;
            turn_json["timestamp"] = turn.timestamp;
            history.append(turn_json);
        }
        root["history"] = history;

        std::ofstream file(filepath);
        Json::StreamWriterBuilder builder;
        std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
        writer->write(root, &file);
    }

    // Load session from file
    void load_session(const std::string& filepath) {
        std::ifstream file(filepath);
        Json::Value root;
        file >> root;

        current_session_.id = root["id"].asString();
        current_session_.total_tokens = root["total_tokens"].asInt();
        current_session_.metadata = root["metadata"];

        current_session_.history.clear();
        for (const auto& turn_json : root["history"]) {
            Turn turn;
            turn.user_input = turn_json["user"].asString();
            turn.assistant_response = turn_json["assistant"].asString();
            turn.token_count = turn_json["tokens"].asInt();
            turn.timestamp = turn_json["timestamp"].asDouble();
            current_session_.history.push_back(turn);
        }
    }

private:
    std::string build_context() {
        std::string context;
        int token_budget = max_context_tokens_ * 0.7;  // Keep 30% for new input/output
        int used_tokens = 0;

        // Build context from most recent turns
        for (auto it = current_session_.history.rbegin();
             it != current_session_.history.rend(); ++it) {
            if (used_tokens + it->token_count > token_budget) break;

            std::string turn = "User: " + it->user_input + "\n";
            turn += "Assistant: " + it->assistant_response + "\n";

            context = turn + context;  // Prepend (building backwards)
            used_tokens += it->token_count;
        }

        return context;
    }

    void manage_context_window() {
        // Remove old turns if total exceeds limit
        while (current_session_.total_tokens > max_context_tokens_ * 0.9
               && current_session_.history.size() > 1) {
            auto& oldest = current_session_.history.front();
            current_session_.total_tokens -= oldest.token_count;
            current_session_.history.erase(current_session_.history.begin());
        }
    }

    int count_tokens(const std::string& text) {
        std::vector<int> tokens;
        model_->Tokenizer()->Encode(text, &tokens);
        return tokens.size();
    }

    std::string generate_session_id() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "session_" << time_t;
        return ss.str();
    }

    double get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration<double>(duration).count();
    }
};

} // namespace gemma
```

### Redis: Direct Memory Operations

```python
#!/usr/bin/env python3
"""Direct Redis memory operations for the RAG system"""

import redis
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

class DirectRedisMemory:
    """Direct interface to Redis memory tiers."""

    TIER_CONFIG = {
        'working': {'capacity': 10, 'ttl': 3600},          # 1 hour
        'short_term': {'capacity': 100, 'ttl': 86400},     # 24 hours
        'long_term': {'capacity': 10000, 'ttl': 2592000},  # 30 days
        'episodic': {'capacity': 1000, 'ttl': 7776000},    # 90 days
        'semantic': {'capacity': 5000, 'ttl': None},       # No expiry
    }

    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(
            host=host, port=port, db=db,
            decode_responses=True
        )
        self.pipeline = self.redis_client.pipeline()

    def store_memory(self, tier: str, content: str,
                    embedding: np.ndarray = None,
                    metadata: Dict[str, Any] = None) -> str:
        """Store a memory entry in specified tier."""

        if tier not in self.TIER_CONFIG:
            raise ValueError(f"Invalid tier: {tier}")

        # Generate ID
        memory_id = f"{tier}:{datetime.utcnow().timestamp()}:{hash(content)}"

        # Prepare entry
        entry = {
            'id': memory_id,
            'tier': tier,
            'content': content,
            'created_at': datetime.utcnow().isoformat(),
            'metadata': metadata or {},
        }

        # Store main entry
        key = f"memory:{memory_id}"
        self.redis_client.hset(key, mapping={
            k: json.dumps(v) if isinstance(v, (dict, list)) else v
            for k, v in entry.items()
        })

        # Set TTL if configured
        ttl = self.TIER_CONFIG[tier]['ttl']
        if ttl:
            self.redis_client.expire(key, ttl)

        # Store embedding for vector search
        if embedding is not None:
            embedding_key = f"embedding:{memory_id}"
            # Store as binary for efficiency
            self.redis_client.set(
                embedding_key,
                embedding.astype(np.float32).tobytes(),
                ex=ttl
            )

        # Add to tier index
        self.redis_client.zadd(
            f"tier:{tier}",
            {memory_id: datetime.utcnow().timestamp()}
        )

        # Enforce capacity limit
        self._enforce_capacity(tier)

        return memory_id

    def recall_memories(self, query_embedding: np.ndarray,
                       tiers: List[str] = None,
                       limit: int = 10) -> List[Dict[str, Any]]:
        """Recall memories using vector similarity."""

        if tiers is None:
            tiers = list(self.TIER_CONFIG.keys())

        results = []

        for tier in tiers:
            # Get memory IDs from tier
            memory_ids = self.redis_client.zrevrange(
                f"tier:{tier}", 0, -1
            )

            for memory_id in memory_ids:
                # Get embedding
                embedding_key = f"embedding:{memory_id}"
                embedding_bytes = self.redis_client.get(embedding_key)

                if embedding_bytes:
                    # Decode embedding
                    stored_embedding = np.frombuffer(
                        embedding_bytes, dtype=np.float32
                    )

                    # Calculate similarity
                    similarity = np.dot(query_embedding, stored_embedding) / (
                        np.linalg.norm(query_embedding) *
                        np.linalg.norm(stored_embedding)
                    )

                    # Get memory content
                    memory_data = self.redis_client.hgetall(f"memory:{memory_id}")

                    if memory_data:
                        memory_data['similarity'] = float(similarity)
                        memory_data['tier'] = tier
                        results.append(memory_data)

        # Sort by similarity and return top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    def consolidate_memories(self, source_tier: str = 'working',
                           target_tier: str = 'short_term',
                           importance_threshold: float = 0.7):
        """Promote important memories to higher tier."""

        # Get memories from source tier
        memory_ids = self.redis_client.zrevrange(
            f"tier:{source_tier}", 0, -1
        )

        promoted_count = 0

        for memory_id in memory_ids:
            memory_data = self.redis_client.hgetall(f"memory:{memory_id}")

            if not memory_data:
                continue

            # Calculate importance (simplified)
            # In production, use more sophisticated scoring
            content = memory_data.get('content', '')
            importance = len(content) / 1000.0  # Simple length-based score

            if importance >= importance_threshold:
                # Promote to target tier
                new_id = memory_id.replace(f"{source_tier}:", f"{target_tier}:")

                # Copy memory with new ID
                new_key = f"memory:{new_id}"
                self.redis_client.hset(new_key, mapping=memory_data)

                # Update TTL
                target_ttl = self.TIER_CONFIG[target_tier]['ttl']
                if target_ttl:
                    self.redis_client.expire(new_key, target_ttl)

                # Copy embedding if exists
                old_embedding = self.redis_client.get(f"embedding:{memory_id}")
                if old_embedding:
                    self.redis_client.set(
                        f"embedding:{new_id}",
                        old_embedding,
                        ex=target_ttl
                    )

                # Add to target tier index
                self.redis_client.zadd(
                    f"tier:{target_tier}",
                    {new_id: datetime.utcnow().timestamp()}
                )

                # Remove from source tier
                self.redis_client.zrem(f"tier:{source_tier}", memory_id)
                self.redis_client.delete(f"memory:{memory_id}")
                self.redis_client.delete(f"embedding:{memory_id}")

                promoted_count += 1

        # Enforce capacity on target tier
        self._enforce_capacity(target_tier)

        return promoted_count

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        stats = {'total': 0, 'by_tier': {}}

        for tier in self.TIER_CONFIG:
            count = self.redis_client.zcard(f"tier:{tier}")
            stats['by_tier'][tier] = {
                'count': count,
                'capacity': self.TIER_CONFIG[tier]['capacity'],
                'utilization': count / self.TIER_CONFIG[tier]['capacity']
            }
            stats['total'] += count

        # Get Redis memory info
        info = self.redis_client.info('memory')
        stats['redis_memory'] = {
            'used_mb': info['used_memory'] / (1024 * 1024),
            'peak_mb': info['used_memory_peak'] / (1024 * 1024),
        }

        return stats

    def _enforce_capacity(self, tier: str):
        """Remove oldest memories if tier exceeds capacity."""
        capacity = self.TIER_CONFIG[tier]['capacity']
        current_size = self.redis_client.zcard(f"tier:{tier}")

        if current_size > capacity:
            # Get oldest memories to remove
            to_remove = self.redis_client.zrange(
                f"tier:{tier}", 0, current_size - capacity - 1
            )

            for memory_id in to_remove:
                # Delete memory and embedding
                self.redis_client.delete(f"memory:{memory_id}")
                self.redis_client.delete(f"embedding:{memory_id}")

            # Remove from index
            if to_remove:
                self.redis_client.zrem(f"tier:{tier}", *to_remove)

# Usage example
if __name__ == "__main__":
    # Initialize
    memory = DirectRedisMemory()

    # Generate fake embedding (in production, use sentence-transformers)
    fake_embedding = np.random.randn(384).astype(np.float32)

    # Store memories
    mem_id = memory.store_memory(
        tier='working',
        content="The RAG system uses 5-tier memory architecture",
        embedding=fake_embedding,
        metadata={'source': 'documentation', 'importance': 0.8}
    )
    print(f"Stored: {mem_id}")

    # Recall memories
    query_embedding = np.random.randn(384).astype(np.float32)
    memories = memory.recall_memories(query_embedding, limit=5)

    for mem in memories:
        print(f"[{mem['tier']}] Similarity: {mem['similarity']:.3f}")
        print(f"  Content: {mem['content'][:100]}")

    # Get statistics
    stats = memory.get_statistics()
    print(f"\nMemory Statistics:")
    print(f"Total memories: {stats['total']}")
    for tier, info in stats['by_tier'].items():
        print(f"  {tier}: {info['count']}/{info['capacity']} "
              f"({info['utilization']:.1%} full)")
```

## Performance Tuning

### Model Selection (2B vs 4B vs 7B)

```bash
# Benchmark different models
cd /c/codedev/llm/gemma

# 2B Model - Fastest, good for chat
./build/bin/RELEASE/single_benchmark \
  --model /c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs \
  --tokenizer /c/codedev/llm/.models/tokenizer.spm \
  --max_tokens 1024

# 4B Model - Balanced (if available)
./build/bin/RELEASE/single_benchmark \
  --model /c/codedev/llm/.models/gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/4b-it.sbs \
  --tokenizer /c/codedev/llm/.models/tokenizer.spm \
  --max_tokens 1024

# 7B Model - Best quality, slower
./build/bin/RELEASE/single_benchmark \
  --model /c/codedev/llm/.models/gemma-7b-it.sbs \
  --tokenizer /c/codedev/llm/.models/tokenizer.spm \
  --max_tokens 1024

# Performance expectations:
# 2B: ~30-50 tokens/sec (CPU), 100+ tokens/sec (GPU)
# 4B: ~15-25 tokens/sec (CPU), 60+ tokens/sec (GPU)
# 7B: ~8-15 tokens/sec (CPU), 30+ tokens/sec (GPU)
```

### Memory Tier Optimization

```python
# Optimal tier configuration for different use cases

# High-throughput chatbot
CHATBOT_CONFIG = {
    'working': {'capacity': 20, 'ttl': 1800},       # 30 min
    'short_term': {'capacity': 200, 'ttl': 43200},  # 12 hours
    'long_term': {'capacity': 5000, 'ttl': 604800}, # 7 days
}

# Research assistant
RESEARCH_CONFIG = {
    'working': {'capacity': 50, 'ttl': 7200},        # 2 hours
    'short_term': {'capacity': 500, 'ttl': 172800},  # 2 days
    'long_term': {'capacity': 20000, 'ttl': None},   # No expiry
    'semantic': {'capacity': 10000, 'ttl': None},    # No expiry
}

# Personal assistant
PERSONAL_CONFIG = {
    'working': {'capacity': 30, 'ttl': 3600},         # 1 hour
    'episodic': {'capacity': 2000, 'ttl': 31536000},  # 1 year
    'semantic': {'capacity': 5000, 'ttl': None},      # No expiry
}
```

### Redis Configuration

```bash
# Optimized redis.conf for RAG system

# Memory settings
maxmemory 4gb
maxmemory-policy allkeys-lru

# Persistence (for important data)
save 900 1      # Save after 900 sec if at least 1 key changed
save 300 10     # Save after 300 sec if at least 10 keys changed
save 60 10000   # Save after 60 sec if at least 10000 keys changed

# AOF for durability
appendonly yes
appendfsync everysec

# Optimizations
rdbcompression yes
rdbchecksum yes

# Connection settings
tcp-backlog 511
tcp-keepalive 300
timeout 0

# Slow log for debugging
slowlog-log-slower-than 10000
slowlog-max-len 128
```

### Context Length vs Quality Tradeoffs

```python
# Configuration profiles for different scenarios

PROFILES = {
    'speed': {
        'max_context': 4096,
        'max_tokens': 512,
        'temperature': 0.7,
        'model': 'gemma-2b',
        'rag_limit': 3,
        'description': 'Fast responses, shorter context'
    },
    'balanced': {
        'max_context': 8192,
        'max_tokens': 1024,
        'temperature': 0.7,
        'model': 'gemma-2b',
        'rag_limit': 5,
        'description': 'Good balance of speed and quality'
    },
    'quality': {
        'max_context': 16384,
        'max_tokens': 2048,
        'temperature': 0.8,
        'model': 'gemma-4b',
        'rag_limit': 10,
        'description': 'Best quality, longer processing'
    },
    'research': {
        'max_context': 32768,
        'max_tokens': 4096,
        'temperature': 0.6,
        'model': 'gemma-7b',
        'rag_limit': 20,
        'description': 'Deep analysis, maximum context'
    }
}

def select_profile(task_type='balanced'):
    """Select optimal configuration profile."""
    profile = PROFILES.get(task_type, PROFILES['balanced'])
    print(f"Using profile: {task_type}")
    print(f"  {profile['description']}")
    print(f"  Context: {profile['max_context']} tokens")
    print(f"  Model: {profile['model']}")
    return profile
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Gemma.exe Build Failures

```bash
# Issue: Missing dependencies
# Solution: Install Visual Studio 2022 with C++ workload

# Issue: CMake configuration fails
# Solution: Clear cache and reconfigure
rm -rf build/CMakeCache.txt build/CMakeFiles
cmake -B build -G "Visual Studio 17 2022" -T v143

# Issue: Link errors with Highway
# Solution: Ensure submodules are updated
git submodule update --init --recursive
```

#### 2. Redis Connection Issues

```bash
# Check Redis is running
redis-cli ping

# If connection refused:
# 1. Start Redis server
redis-server --daemonize yes

# 2. Check firewall settings (Windows)
netsh advfirewall firewall add rule name="Redis" dir=in action=allow protocol=TCP localport=6379

# 3. Verify Redis config
redis-cli CONFIG GET bind
redis-cli CONFIG GET protected-mode
```

#### 3. Memory/Performance Issues

```python
# Monitor memory usage
import psutil
import os

def check_resources():
    process = psutil.Process(os.getpid())

    # Memory
    mem_info = process.memory_info()
    print(f"Memory RSS: {mem_info.rss / 1024 / 1024:.2f} MB")
    print(f"Memory VMS: {mem_info.vms / 1024 / 1024:.2f} MB")

    # CPU
    cpu_percent = process.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_percent}%")

    # Open files/handles
    open_files = len(process.open_files())
    print(f"Open Files: {open_files}")

    return mem_info.rss > 4 * 1024 * 1024 * 1024  # Alert if > 4GB

# Memory leak detection
import tracemalloc
tracemalloc.start()

# ... run your code ...

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

### Debug Mode Usage

```bash
# Enable debug output in gemma-cli.py
python gemma-cli.py \
  --model /path/to/model.sbs \
  --debug \
  --log-level DEBUG

# Debug environment variables
export GEMMA_DEBUG=1
export REDIS_DEBUG=1
export TOKENIZER_DEBUG=1

# Debug with specific components
python -m pdb gemma-cli.py --model /path/to/model.sbs  # Python debugger
strace python gemma-cli.py --model /path/to/model.sbs  # System calls (Linux)
```

### Log Interpretation

```python
# Log parser for troubleshooting
import re
from datetime import datetime

class LogAnalyzer:
    def __init__(self, log_file):
        self.log_file = log_file
        self.patterns = {
            'error': re.compile(r'ERROR.*?: (.*)'),
            'warning': re.compile(r'WARNING.*?: (.*)'),
            'token_count': re.compile(r'Tokens: (\d+)'),
            'latency': re.compile(r'Latency: ([\d.]+)ms'),
            'memory': re.compile(r'Memory: ([\d.]+)MB'),
        }

    def analyze(self):
        issues = {'errors': [], 'warnings': [], 'performance': []}

        with open(self.log_file, 'r') as f:
            for line in f:
                # Check for errors
                if match := self.patterns['error'].search(line):
                    issues['errors'].append({
                        'message': match.group(1),
                        'line': line.strip()
                    })

                # Check for warnings
                if match := self.patterns['warning'].search(line):
                    issues['warnings'].append({
                        'message': match.group(1),
                        'line': line.strip()
                    })

                # Check performance issues
                if match := self.patterns['latency'].search(line):
                    latency = float(match.group(1))
                    if latency > 1000:  # > 1 second
                        issues['performance'].append({
                            'type': 'high_latency',
                            'value': latency,
                            'line': line.strip()
                        })

        return issues

    def generate_report(self):
        issues = self.analyze()

        print("=== Log Analysis Report ===")
        print(f"Errors: {len(issues['errors'])}")
        for err in issues['errors'][:5]:
            print(f"  - {err['message']}")

        print(f"\nWarnings: {len(issues['warnings'])}")
        for warn in issues['warnings'][:5]:
            print(f"  - {warn['message']}")

        print(f"\nPerformance Issues: {len(issues['performance'])}")
        for perf in issues['performance'][:5]:
            print(f"  - {perf['type']}: {perf['value']}")

# Usage
analyzer = LogAnalyzer('gemma-cli.log')
analyzer.generate_report()
```

## Quick Reference

### Essential Commands

```bash
# Start complete system
redis-server --daemonize yes
python gemma-cli.py --model model.sbs --enable-rag

# Basic operations
/help                    # Show commands
/rag help               # RAG commands
/session save           # Save session
/stats                  # System statistics
/clear                  # Clear screen
/exit                   # Quit

# RAG operations
/rag ingest <file>      # Ingest document
/rag search <query>     # Search memories
/rag consolidate        # Consolidate memories
/rag stats              # Memory statistics
```

### Configuration Files

- `gemma-config.json` - Model and generation settings
- `redis.conf` - Redis server configuration
- `rag-config.yaml` - RAG system settings
- `sessions/*.json` - Saved conversation sessions

### Environment Variables

```bash
# Required
export GEMMA_MODEL_PATH="/path/to/model.sbs"
export REDIS_URL="redis://localhost:6379"

# Optional
export GEMMA_TOKENIZER_PATH="/path/to/tokenizer.spm"
export GEMMA_MAX_CONTEXT="8192"
export GEMMA_TEMPERATURE="0.7"
export RAG_EMBEDDING_MODEL="all-MiniLM-L6-v2"
export RAG_CHUNK_SIZE="512"
export RAG_CHUNK_OVERLAP="50"
export SESSION_AUTO_SAVE="true"
export DEBUG_MODE="false"
```

### Performance Benchmarks

| Configuration | Model | Context | RAG Docs | Tokens/sec | Memory |
|--------------|--------|---------|----------|------------|---------|
| Speed | 2B | 4K | 100 | 30-50 | 2GB |
| Balanced | 2B | 8K | 500 | 25-40 | 3GB |
| Quality | 4B | 16K | 1000 | 15-25 | 5GB |
| Research | 7B | 32K | 5000 | 8-15 | 12GB |

---

**Last Updated**: December 2024
**Version**: 1.0.0
**Status**: Production Ready

For additional support, refer to:
- [Gemma C++ Documentation](./gemma/README.md)
- [RAG-Redis System Guide](./stats/rag-redis-system/README.md)
- [Python Agent Framework](./stats/README.md)