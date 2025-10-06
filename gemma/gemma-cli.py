#!/usr/bin/env python3
"""
Gemma CLI Wrapper with RAG-Redis Integration
Enhanced Windows chat interface for native gemma.exe with:
- Direct Redis-based RAG system
- 5-tier memory architecture (Working, Short-term, Long-term, Episodic, Semantic)
- Document ingestion and retrieval
- Advanced conversation memory
- Production-ready error handling
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import signal
import hashlib
import uuid
import tempfile

try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    # Fallback for no colorama
    COLORS_AVAILABLE = False
    class MockFore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class MockStyle:
        BRIGHT = DIM = RESET_ALL = ""
    Fore = MockFore()
    Style = MockStyle()

# Redis and RAG dependencies
try:
    import redis
    import redis.asyncio as aioredis
    import numpy as np
    from sentence_transformers import SentenceTransformer
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print(f"{Fore.YELLOW}Warning: Redis dependencies not available. Install with: pip install redis numpy sentence-transformers")

try:
    from tiktoken import get_encoding
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


# RAG Memory System Classes

class MemoryTier:
    """Represents a memory tier with TTL and capacity settings."""
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"

class MemoryEntry:
    """Memory entry with content, metadata, and importance scoring."""

    def __init__(self, content: str, memory_type: str, importance: float = 0.5):
        self.id = str(uuid.uuid4())
        self.content = content
        self.memory_type = memory_type
        self.importance = importance
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.access_count = 0
        self.tags = set()
        self.metadata = {}
        self.embedding = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            'id': self.id,
            'content': self.content,
            'memory_type': self.memory_type,
            'importance': self.importance,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'tags': list(self.tags),
            'metadata': self.metadata,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary loaded from Redis."""
        entry = cls(data['content'], data['memory_type'], data['importance'])
        entry.id = data['id']
        entry.created_at = datetime.fromisoformat(data['created_at'])
        entry.last_accessed = datetime.fromisoformat(data['last_accessed'])
        entry.access_count = data['access_count']
        entry.tags = set(data.get('tags', []))
        entry.metadata = data.get('metadata', {})
        if data.get('embedding'):
            entry.embedding = np.array(data['embedding'])
        return entry

class RAGRedisManager:
    """Redis-based RAG system with 5-tier memory architecture."""

    # Memory tier configurations (TTL in seconds)
    TIER_CONFIG = {
        MemoryTier.WORKING: {'ttl': 900, 'max_size': 15},       # 15 min
        MemoryTier.SHORT_TERM: {'ttl': 3600, 'max_size': 100},  # 1 hour
        MemoryTier.LONG_TERM: {'ttl': 2592000, 'max_size': 10000}, # 30 days
        MemoryTier.EPISODIC: {'ttl': 604800, 'max_size': 5000}, # 7 days
        MemoryTier.SEMANTIC: {'ttl': None, 'max_size': 50000},  # Permanent
    }

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_client = None
        self.async_redis_client = None
        self.embedding_model = None
        self.encoder = None

    async def initialize(self):
        """Initialize Redis connections and embedding model."""
        try:
            # Test Redis connection
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=False
            )
            self.redis_client.ping()

            self.async_redis_client = aioredis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=False
            )

            # Load embedding model for semantic search
            if REDIS_AVAILABLE:
                print(f"{Fore.CYAN}Loading embedding model...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Initialize tokenizer if available
            if TIKTOKEN_AVAILABLE:
                self.encoder = get_encoding("cl100k_base")

            print(f"{Fore.GREEN}RAG-Redis system initialized successfully")
            return True

        except Exception as e:
            print(f"{Fore.RED}Failed to initialize RAG-Redis system: {e}")
            return False

    def get_redis_key(self, memory_type: str, entry_id: str = None) -> str:
        """Generate Redis key for memory tier and entry."""
        if entry_id:
            return f"gemma:mem:{memory_type}:{entry_id}"
        return f"gemma:mem:{memory_type}:*"

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text."""
        if self.embedding_model:
            return self.embedding_model.encode([text])[0]
        else:
            # Fallback: simple hash-based embedding
            hash_val = hashlib.md5(text.encode()).hexdigest()
            return np.array([ord(c) for c in hash_val[:384]], dtype=np.float32) / 256.0

    async def store_memory(self, content: str, memory_type: str, importance: float = 0.5, tags: List[str] = None) -> str:
        """Store content in specified memory tier."""
        try:
            entry = MemoryEntry(content, memory_type, importance)
            if tags:
                entry.tags.update(tags)

            # Generate embedding
            entry.embedding = self.get_embedding(content)

            # Store in Redis
            key = self.get_redis_key(memory_type, entry.id)
            data = json.dumps(entry.to_dict())

            config = self.TIER_CONFIG.get(memory_type, self.TIER_CONFIG[MemoryTier.SHORT_TERM])
            if config['ttl']:
                await self.async_redis_client.setex(key, config['ttl'], data)
            else:
                await self.async_redis_client.set(key, data)

            # Enforce tier size limits
            await self._enforce_tier_limits(memory_type)

            print(f"{Fore.GREEN}Stored memory in {memory_type} tier: {entry.id[:8]}...")
            return entry.id

        except Exception as e:
            print(f"{Fore.RED}Error storing memory: {e}")
            return None

    async def recall_memories(self, query: str, memory_type: str = None, limit: int = 5) -> List[MemoryEntry]:
        """Retrieve memories based on semantic similarity to query."""
        try:
            query_embedding = self.get_embedding(query)
            results = []

            memory_types = [memory_type] if memory_type else list(self.TIER_CONFIG.keys())

            for tier in memory_types:
                pattern = self.get_redis_key(tier)
                keys = await self.async_redis_client.keys(pattern)

                for key in keys:
                    data = await self.async_redis_client.get(key)
                    if data:
                        entry_dict = json.loads(data.decode())
                        entry = MemoryEntry.from_dict(entry_dict)

                        # Calculate similarity score
                        if entry.embedding is not None:
                            similarity = np.dot(query_embedding, entry.embedding) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(entry.embedding)
                            )
                            entry.similarity_score = similarity
                            results.append(entry)

            # Sort by similarity and importance
            results.sort(key=lambda x: (x.similarity_score * x.importance), reverse=True)
            return results[:limit]

        except Exception as e:
            print(f"{Fore.RED}Error recalling memories: {e}")
            return []

    async def search_memories(self, query: str, memory_type: str = None, min_importance: float = 0.0) -> List[MemoryEntry]:
        """Search memories by content and importance."""
        try:
            results = []
            memory_types = [memory_type] if memory_type else list(self.TIER_CONFIG.keys())

            for tier in memory_types:
                pattern = self.get_redis_key(tier)
                keys = await self.async_redis_client.keys(pattern)

                for key in keys:
                    data = await self.async_redis_client.get(key)
                    if data:
                        entry_dict = json.loads(data.decode())
                        entry = MemoryEntry.from_dict(entry_dict)

                        # Filter by importance and content match
                        if (entry.importance >= min_importance and
                            query.lower() in entry.content.lower()):
                            results.append(entry)

            # Sort by importance and recency
            results.sort(key=lambda x: (x.importance, x.last_accessed), reverse=True)
            return results

        except Exception as e:
            print(f"{Fore.RED}Error searching memories: {e}")
            return []

    async def ingest_document(self, file_path: str, memory_type: str = MemoryTier.LONG_TERM, chunk_size: int = 500) -> int:
        """Ingest a document into the memory system by chunking."""
        try:
            path = Path(file_path)
            if not path.exists():
                print(f"{Fore.RED}File not found: {file_path}")
                return 0

            # Read file content
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Chunk the document
            chunks = self._chunk_text(content, chunk_size)
            stored_count = 0

            for i, chunk in enumerate(chunks):
                importance = 0.7 if memory_type == MemoryTier.LONG_TERM else 0.5
                tags = [f"document:{path.name}", f"chunk:{i}"]

                entry_id = await self.store_memory(chunk, memory_type, importance, tags)
                if entry_id:
                    stored_count += 1

            print(f"{Fore.GREEN}Ingested document: {stored_count} chunks from {path.name}")
            return stored_count

        except Exception as e:
            print(f"{Fore.RED}Error ingesting document: {e}")
            return 0

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into semantic chunks."""
        if self.encoder:
            # Use tiktoken for intelligent chunking
            tokens = self.encoder.encode(text)
            chunks = []

            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = self.encoder.decode(chunk_tokens)
                chunks.append(chunk_text)

            return chunks
        else:
            # Simple sentence-based chunking
            sentences = re.split(r'[.!?]+', text)
            chunks = []
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk) + len(sentence) < chunk_size * 4:  # Rough word estimate
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "

            if current_chunk:
                chunks.append(current_chunk.strip())

            return chunks

    async def _enforce_tier_limits(self, memory_type: str):
        """Enforce size limits for memory tiers."""
        try:
            config = self.TIER_CONFIG.get(memory_type)
            if not config or not config.get('max_size'):
                return

            pattern = self.get_redis_key(memory_type)
            keys = await self.async_redis_client.keys(pattern)

            if len(keys) > config['max_size']:
                # Remove oldest entries
                entries_to_check = []

                for key in keys:
                    data = await self.async_redis_client.get(key)
                    if data:
                        entry_dict = json.loads(data.decode())
                        entries_to_check.append((key, entry_dict['last_accessed']))

                # Sort by last accessed time and remove oldest
                entries_to_check.sort(key=lambda x: x[1])
                excess_count = len(keys) - config['max_size']

                for key, _ in entries_to_check[:excess_count]:
                    await self.async_redis_client.delete(key)

        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Error enforcing tier limits: {e}")

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage across tiers."""
        stats = {}
        total_entries = 0

        try:
            for tier in self.TIER_CONFIG.keys():
                pattern = self.get_redis_key(tier)
                keys = await self.async_redis_client.keys(pattern)
                count = len(keys)
                stats[tier] = count
                total_entries += count

            stats['total'] = total_entries
            stats['redis_memory'] = await self.async_redis_client.memory_usage("*") or 0

        except Exception as e:
            print(f"{Fore.RED}Error getting memory stats: {e}")

        return stats

    async def cleanup_expired(self) -> int:
        """Clean up expired memory entries."""
        cleaned = 0
        try:
            for tier, config in self.TIER_CONFIG.items():
                if config['ttl']:  # Skip permanent memories
                    pattern = self.get_redis_key(tier)
                    keys = await self.async_redis_client.keys(pattern)

                    for key in keys:
                        ttl = await self.async_redis_client.ttl(key)
                        if ttl == -2:  # Key doesn't exist or expired
                            await self.async_redis_client.delete(key)
                            cleaned += 1

            if cleaned > 0:
                print(f"{Fore.YELLOW}Cleaned up {cleaned} expired memory entries")

        except Exception as e:
            print(f"{Fore.RED}Error during cleanup: {e}")

        return cleaned

class ConversationManager:
    """Manages conversation history and context for the chat session."""

    def __init__(self, max_context_length: int = 8192):
        self.messages: List[Dict[str, str]] = []
        self.max_context_length = max_context_length
        self.session_start = datetime.now()

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_context()

    def _trim_context(self) -> None:
        """Trim conversation to stay within context length."""
        total_length = sum(len(msg["content"]) for msg in self.messages)
        while total_length > self.max_context_length and len(self.messages) > 1:
            # Remove oldest messages but keep system message if present
            if self.messages[0].get("role") == "system":
                if len(self.messages) > 2:
                    self.messages.pop(1)
                else:
                    break
            else:
                self.messages.pop(0)
            total_length = sum(len(msg["content"]) for msg in self.messages)

    def get_context_prompt(self) -> str:
        """Generate a context-aware prompt for gemma."""
        if not self.messages:
            return ""

        # Build conversation context
        context_parts = []
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                context_parts.append(f"Assistant: {content}")
            elif role == "system":
                context_parts.append(f"System: {content}")

        return "\n".join(context_parts)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        self.session_start = datetime.now()

    def save_to_file(self, filepath: Path) -> bool:
        """Save conversation to JSON file."""
        try:
            data = {
                "session_start": self.session_start.isoformat(),
                "messages": self.messages,
                "saved_at": datetime.now().isoformat()
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"{Fore.RED}Error saving conversation: {e}")
            return False

    def load_from_file(self, filepath: Path) -> bool:
        """Load conversation from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.messages = data.get("messages", [])
            session_start_str = data.get("session_start")
            if session_start_str:
                self.session_start = datetime.fromisoformat(session_start_str)
            return True
        except Exception as e:
            print(f"{Fore.RED}Error loading conversation: {e}")
            return False


class GemmaInterface:
    """Interface for communicating with the native Windows gemma.exe."""

    def __init__(self,
                 model_path: str,
                 tokenizer_path: Optional[str] = None,
                 gemma_executable: str = r"C:\codedev\llm\gemma\build-avx2-sycl\bin\RELEASE\gemma.exe",
                 max_tokens: int = 2048,
                 temperature: float = 0.7):
        self.model_path = os.path.normpath(model_path)
        self.tokenizer_path = os.path.normpath(tokenizer_path) if tokenizer_path else None
        self.gemma_executable = os.path.normpath(gemma_executable)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.process: Optional[subprocess.Popen] = None

        # Verify executable exists
        if not os.path.exists(self.gemma_executable):
            raise FileNotFoundError(f"Gemma executable not found: {self.gemma_executable}")

    def _build_command(self, prompt: str) -> List[str]:
        """Build the native Windows command to execute the gemma executable."""
        cmd = [self.gemma_executable, "--weights", self.model_path]

        if self.tokenizer_path:
            cmd.extend(["--tokenizer", self.tokenizer_path])

        # Add generation parameters
        cmd.extend([
            "--max_generated_tokens", str(self.max_tokens),
            "--temperature", str(self.temperature),
            "--prompt", prompt
        ])

        return cmd

    async def generate_response(self, prompt: str, stream_callback=None) -> str:
        """Generate response from gemma with optional streaming callback."""
        cmd = self._build_command(prompt)

        # Debug: print the command that will be executed
        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(f"{Fore.YELLOW}Debug - Command: {' '.join(cmd)}")

        try:
            # Use proper Windows process creation with CREATE_NEW_PROCESS_GROUP
            startup_info = subprocess.STARTUPINFO()
            startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup_info.wShowWindow = subprocess.SW_HIDE

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,  # Unbuffered for real-time streaming
                universal_newlines=True,
                startupinfo=startup_info,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )

            response_parts = []

            # Read output in real-time for streaming
            while True:
                # Use a small timeout to check for process completion
                try:
                    output = self.process.stdout.read(1)
                    if output == '' and self.process.poll() is not None:
                        break
                    if output:
                        response_parts.append(output)
                        if stream_callback:
                            stream_callback(output)
                        # Allow other async tasks to run
                        await asyncio.sleep(0)
                except Exception:
                    break

            # Wait for process completion
            return_code = self.process.wait()

            if return_code != 0:
                stderr_output = self.process.stderr.read()
                raise RuntimeError(f"Gemma process failed (code {return_code}): {stderr_output}")

            return ''.join(response_parts)

        except Exception as e:
            print(f"{Fore.RED}Error communicating with gemma: {e}")
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"{Fore.YELLOW}Debug - Command was: {' '.join(cmd)}")
            return f"Error: {str(e)}"
        finally:
            if self.process:
                try:
                    if self.process.poll() is None:
                        self.process.terminate()
                        self.process.wait(timeout=5)
                except:
                    pass
                self.process = None

    def stop_generation(self):
        """Stop current generation process."""
        if self.process:
            try:
                # On Windows, try to terminate gracefully first
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                self.process.kill()
                self.process.wait()
            except Exception:
                pass
            self.process = None


class GemmaCLI:
    """Main CLI application class."""

    def __init__(self, args):
        self.args = args
        self.conversation = ConversationManager(max_context_length=args.max_context)

        # Initialize Gemma interface with native executable
        try:
            self.gemma = GemmaInterface(
                model_path=args.model,
                tokenizer_path=args.tokenizer,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
        except FileNotFoundError as e:
            print(f"{Fore.RED}Error: {e}")
            print(f"{Fore.YELLOW}Please ensure gemma.exe is built and available")
            sys.exit(1)

        # Initialize RAG system if Redis is available
        self.rag_system = None
        self.rag_enabled = False
        if REDIS_AVAILABLE and args.enable_rag:
            self.rag_system = RAGRedisManager(
                redis_host=getattr(args, 'redis_host', 'localhost'),
                redis_port=getattr(args, 'redis_port', 6379),
                redis_db=getattr(args, 'redis_db', 0)
            )

        # Enable debug mode if requested
        if hasattr(args, 'debug') and args.debug:
            self.gemma.debug_mode = True

        self.running = True
        self.conversation_dir = Path.home() / ".gemma_conversations"
        self.conversation_dir.mkdir(exist_ok=True)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print(f"\n{Fore.YELLOW}Shutting down gracefully...")
        self.gemma.stop_generation()
        self.running = False
        sys.exit(0)

    def _print_header(self):
        """Print the CLI header."""
        print(f"{Fore.CYAN}{Style.BRIGHT}")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║                       Gemma CLI Wrapper                     ║")
        print("║              Chat interface for Gemma models                ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(f"{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Model: {self.args.model}")
        if self.args.tokenizer:
            print(f"{Fore.GREEN}Tokenizer: {self.args.tokenizer}")
        print(f"{Fore.GREEN}Max tokens: {self.args.max_tokens}, Temperature: {self.args.temperature}")
        print(f"{Fore.YELLOW}Type /help for commands, /quit to exit")
        print("-" * 60)

    def _print_help(self):
        """Print help information."""
        help_text = f"""
{Fore.CYAN}{Style.BRIGHT}Available Commands:{Style.RESET_ALL}
{Fore.GREEN}/help{Fore.WHITE}              - Show this help message
{Fore.GREEN}/clear{Fore.WHITE}             - Clear conversation history
{Fore.GREEN}/save [filename]{Fore.WHITE}   - Save conversation to file
{Fore.GREEN}/load [filename]{Fore.WHITE}   - Load conversation from file
{Fore.GREEN}/list{Fore.WHITE}              - List saved conversations
{Fore.GREEN}/status{Fore.WHITE}            - Show current session status
{Fore.GREEN}/settings{Fore.WHITE}          - Show current model settings
{Fore.GREEN}/quit{Fore.WHITE} or {Fore.GREEN}/exit{Fore.WHITE}    - Exit the application

{Fore.CYAN}{Style.BRIGHT}RAG Memory Commands:{Style.RESET_ALL}"""

        if self.rag_enabled:
            help_text += f"""
{Fore.GREEN}/store <text> [tier] [importance]{Fore.WHITE} - Store text in memory tier
{Fore.GREEN}/recall <query> [tier] [limit]{Fore.WHITE}   - Recall similar memories
{Fore.GREEN}/search <query> [tier] [min_importance]{Fore.WHITE} - Search memory by content
{Fore.GREEN}/ingest <file_path> [tier]{Fore.WHITE}       - Ingest document into memory
{Fore.GREEN}/memory_stats{Fore.WHITE}      - Show memory usage statistics
{Fore.GREEN}/cleanup{Fore.WHITE}           - Clean up expired memory entries

{Fore.CYAN}{Style.BRIGHT}Memory Tiers:{Style.RESET_ALL}
- working (15 min TTL, 15 items max)
- short_term (1 hour TTL, 100 items max)
- long_term (30 days TTL, 10K items max)
- episodic (7 days TTL, 5K items max)
- semantic (permanent, 50K items max)"""
        else:
            help_text += f"""
{Fore.YELLOW}RAG system not available. Start with --enable-rag and ensure Redis is running."""

        help_text += f"""

{Fore.CYAN}{Style.BRIGHT}Usage Tips:{Style.RESET_ALL}
- Type your message and press Enter to chat
- Use Ctrl+C to interrupt ongoing generation
- Conversations automatically maintain context
- Files saved to: {self.conversation_dir}
"""
        if self.rag_enabled:
            help_text += f"- RAG memories enhance responses with relevant context\n"

        print(help_text)

    def _handle_command(self, command: str) -> bool:
        """Handle special commands. Returns True if command was processed."""
        command = command.strip()

        if command in ["/quit", "/exit"]:
            print(f"{Fore.YELLOW}Goodbye!")
            return False

        elif command == "/help":
            self._print_help()

        elif command == "/clear":
            self.conversation.clear()
            print(f"{Fore.GREEN}Conversation history cleared.")

        elif command.startswith("/save"):
            parts = command.split(maxsplit=1)
            if len(parts) > 1:
                filename = parts[1]
            else:
                filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            filepath = self.conversation_dir / filename
            if self.conversation.save_to_file(filepath):
                print(f"{Fore.GREEN}Conversation saved to: {filepath}")

        elif command.startswith("/load"):
            parts = command.split(maxsplit=1)
            if len(parts) > 1:
                filename = parts[1]
                filepath = self.conversation_dir / filename
                if filepath.exists():
                    if self.conversation.load_from_file(filepath):
                        print(f"{Fore.GREEN}Conversation loaded from: {filepath}")
                        print(f"{Fore.CYAN}Loaded {len(self.conversation.messages)} messages")
                else:
                    print(f"{Fore.RED}File not found: {filepath}")
            else:
                print(f"{Fore.RED}Usage: /load <filename>")

        elif command == "/list":
            json_files = list(self.conversation_dir.glob("*.json"))
            if json_files:
                print(f"{Fore.CYAN}Saved conversations:")
                for file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True):
                    mtime = datetime.fromtimestamp(file.stat().st_mtime)
                    print(f"  {Fore.GREEN}{file.name}{Fore.WHITE} - {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"{Fore.YELLOW}No saved conversations found.")

        elif command == "/status":
            msg_count = len(self.conversation.messages)
            session_duration = datetime.now() - self.conversation.session_start
            print(f"{Fore.CYAN}Session Status:")
            print(f"  Messages in conversation: {msg_count}")
            print(f"  Session duration: {str(session_duration).split('.')[0]}")
            print(f"  Session started: {self.conversation.session_start.strftime('%Y-%m-%d %H:%M:%S')}")

        elif command == "/settings":
            print(f"{Fore.CYAN}Current Settings:")
            print(f"  Model: {self.gemma.model_path}")
            print(f"  Tokenizer: {self.gemma.tokenizer_path or 'Built-in'}")
            print(f"  Max tokens: {self.gemma.max_tokens}")
            print(f"  Temperature: {self.gemma.temperature}")
            print(f"  Max context length: {self.conversation.max_context_length}")
            print(f"  RAG enabled: {self.rag_enabled}")
            if self.rag_enabled:
                print(f"  Redis: {self.rag_system.redis_host}:{self.rag_system.redis_port}")

        # RAG Commands
        elif command.startswith("/store"):
            asyncio.create_task(self._handle_store_command(command))

        elif command.startswith("/recall"):
            asyncio.create_task(self._handle_recall_command(command))

        elif command.startswith("/search"):
            asyncio.create_task(self._handle_search_command(command))

        elif command.startswith("/ingest"):
            asyncio.create_task(self._handle_ingest_command(command))

        elif command == "/memory_stats":
            asyncio.create_task(self._handle_memory_stats_command())

        elif command == "/cleanup":
            asyncio.create_task(self._handle_cleanup_command())

        else:
            print(f"{Fore.RED}Unknown command: {command}")
            print(f"{Fore.YELLOW}Type /help for available commands.")

        return True

    async def _handle_store_command(self, command: str):
        """Handle /store command."""
        if not self.rag_enabled:
            print(f"{Fore.RED}RAG system not enabled. Start with --enable-rag")
            return

        try:
            parts = command.split(maxsplit=3)
            if len(parts) < 2:
                print(f"{Fore.RED}Usage: /store <text> [tier] [importance]")
                return

            text = parts[1]
            tier = parts[2] if len(parts) > 2 else MemoryTier.SHORT_TERM
            importance = float(parts[3]) if len(parts) > 3 else 0.5

            if tier not in [MemoryTier.WORKING, MemoryTier.SHORT_TERM, MemoryTier.LONG_TERM,
                           MemoryTier.EPISODIC, MemoryTier.SEMANTIC]:
                print(f"{Fore.RED}Invalid tier. Use: working, short_term, long_term, episodic, semantic")
                return

            entry_id = await self.rag_system.store_memory(text, tier, importance)
            if entry_id:
                print(f"{Fore.GREEN}Stored in {tier}: {entry_id[:8]}...")

        except Exception as e:
            print(f"{Fore.RED}Error storing memory: {e}")

    async def _handle_recall_command(self, command: str):
        """Handle /recall command."""
        if not self.rag_enabled:
            print(f"{Fore.RED}RAG system not enabled. Start with --enable-rag")
            return

        try:
            parts = command.split(maxsplit=3)
            if len(parts) < 2:
                print(f"{Fore.RED}Usage: /recall <query> [tier] [limit]")
                return

            query = parts[1]
            tier = parts[2] if len(parts) > 2 else None
            limit = int(parts[3]) if len(parts) > 3 else 5

            memories = await self.rag_system.recall_memories(query, tier, limit)

            if memories:
                print(f"{Fore.CYAN}Found {len(memories)} relevant memories:")
                for i, memory in enumerate(memories, 1):
                    score = getattr(memory, 'similarity_score', 0)
                    print(f"\n{Fore.YELLOW}[{i}] {memory.memory_type} (score: {score:.3f}, importance: {memory.importance:.2f})")
                    content_preview = memory.content[:200] + "..." if len(memory.content) > 200 else memory.content
                    print(f"{Fore.WHITE}{content_preview}")
            else:
                print(f"{Fore.YELLOW}No relevant memories found for: {query}")

        except Exception as e:
            print(f"{Fore.RED}Error recalling memories: {e}")

    async def _handle_search_command(self, command: str):
        """Handle /search command."""
        if not self.rag_enabled:
            print(f"{Fore.RED}RAG system not enabled. Start with --enable-rag")
            return

        try:
            parts = command.split(maxsplit=3)
            if len(parts) < 2:
                print(f"{Fore.RED}Usage: /search <query> [tier] [min_importance]")
                return

            query = parts[1]
            tier = parts[2] if len(parts) > 2 else None
            min_importance = float(parts[3]) if len(parts) > 3 else 0.0

            memories = await self.rag_system.search_memories(query, tier, min_importance)

            if memories:
                print(f"{Fore.CYAN}Found {len(memories)} matching memories:")
                for i, memory in enumerate(memories, 1):
                    print(f"\n{Fore.YELLOW}[{i}] {memory.memory_type} (importance: {memory.importance:.2f})")
                    print(f"{Fore.GREEN}Created: {memory.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
                    content_preview = memory.content[:200] + "..." if len(memory.content) > 200 else memory.content
                    print(f"{Fore.WHITE}{content_preview}")
            else:
                print(f"{Fore.YELLOW}No memories found matching: {query}")

        except Exception as e:
            print(f"{Fore.RED}Error searching memories: {e}")

    async def _handle_ingest_command(self, command: str):
        """Handle /ingest command."""
        if not self.rag_enabled:
            print(f"{Fore.RED}RAG system not enabled. Start with --enable-rag")
            return

        try:
            parts = command.split(maxsplit=2)
            if len(parts) < 2:
                print(f"{Fore.RED}Usage: /ingest <file_path> [tier]")
                return

            file_path = parts[1]
            tier = parts[2] if len(parts) > 2 else MemoryTier.LONG_TERM

            if tier not in [MemoryTier.SHORT_TERM, MemoryTier.LONG_TERM, MemoryTier.SEMANTIC]:
                print(f"{Fore.RED}Invalid tier for documents. Use: short_term, long_term, semantic")
                return

            print(f"{Fore.CYAN}Ingesting document: {file_path}")
            count = await self.rag_system.ingest_document(file_path, tier)

            if count > 0:
                print(f"{Fore.GREEN}Successfully ingested {count} chunks into {tier} memory")
            else:
                print(f"{Fore.RED}Failed to ingest document")

        except Exception as e:
            print(f"{Fore.RED}Error ingesting document: {e}")

    async def _handle_memory_stats_command(self):
        """Handle /memory_stats command."""
        if not self.rag_enabled:
            print(f"{Fore.RED}RAG system not enabled. Start with --enable-rag")
            return

        try:
            stats = await self.rag_system.get_memory_stats()

            print(f"{Fore.CYAN}{Style.BRIGHT}Memory Statistics:{Style.RESET_ALL}")
            print(f"{Fore.GREEN}Total entries: {stats.get('total', 0)}")
            print(f"{Fore.GREEN}Redis memory usage: {stats.get('redis_memory', 0)} bytes")

            print(f"\n{Fore.CYAN}Entries by tier:")
            for tier in [MemoryTier.WORKING, MemoryTier.SHORT_TERM, MemoryTier.LONG_TERM,
                        MemoryTier.EPISODIC, MemoryTier.SEMANTIC]:
                count = stats.get(tier, 0)
                config = self.rag_system.TIER_CONFIG[tier]
                max_size = config['max_size']
                ttl = config['ttl']
                ttl_str = f"{ttl}s" if ttl else "∞"
                print(f"  {Fore.YELLOW}{tier:12}: {Fore.WHITE}{count:4d}/{max_size} (TTL: {ttl_str})")

        except Exception as e:
            print(f"{Fore.RED}Error getting memory stats: {e}")

    async def _handle_cleanup_command(self):
        """Handle /cleanup command."""
        if not self.rag_enabled:
            print(f"{Fore.RED}RAG system not enabled. Start with --enable-rag")
            return

        try:
            print(f"{Fore.CYAN}Cleaning up expired memory entries...")
            cleaned = await self.rag_system.cleanup_expired()
            print(f"{Fore.GREEN}Cleanup complete: {cleaned} entries removed")

        except Exception as e:
            print(f"{Fore.RED}Error during cleanup: {e}")

    async def _get_response(self, user_input: str) -> str:
        """Get response from gemma with conversation context and RAG enhancement."""
        # Add user message to conversation
        self.conversation.add_message("user", user_input)

        # Get RAG context if enabled
        rag_context = ""
        if self.rag_enabled:
            try:
                # Store current conversation in working memory
                await self.rag_system.store_memory(
                    user_input, MemoryTier.WORKING, 0.6, ["conversation"]
                )

                # Recall relevant memories
                relevant_memories = await self.rag_system.recall_memories(user_input, limit=3)

                if relevant_memories:
                    rag_context = "\n[Relevant context from memory:\n"
                    for i, memory in enumerate(relevant_memories):
                        rag_context += f"{i+1}. {memory.content[:300]}...\n"
                    rag_context += "]\n"

            except Exception as e:
                print(f"{Fore.YELLOW}Warning: RAG context retrieval failed: {e}")

        # Build context-aware prompt
        conversation_context = self.conversation.get_context_prompt()
        full_prompt = f"{rag_context}{conversation_context}\nUser: {user_input}\nAssistant:"

        # Track streaming output
        response_parts = []
        print(f"{Fore.MAGENTA}Assistant: {Fore.WHITE}", end="", flush=True)

        def stream_callback(token):
            response_parts.append(token)
            print(token, end="", flush=True)

        # Generate response
        try:
            full_response = await self.gemma.generate_response(full_prompt, stream_callback)
            print()  # New line after streaming

            # Extract just the assistant's response (remove context echo)
            if "Assistant:" in full_response:
                assistant_response = full_response.split("Assistant:")[-1].strip()
            else:
                assistant_response = full_response.strip()

            # Add assistant response to conversation and RAG
            self.conversation.add_message("assistant", assistant_response)

            if self.rag_enabled:
                try:
                    # Store assistant response in episodic memory
                    await self.rag_system.store_memory(
                        f"Q: {user_input}\nA: {assistant_response}",
                        MemoryTier.EPISODIC, 0.7, ["conversation", "qa_pair"]
                    )
                except Exception as e:
                    print(f"{Fore.YELLOW}Warning: Failed to store response in memory: {e}")

            return assistant_response

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Generation interrupted by user.")
            return ""
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"\n{Fore.RED}{error_msg}")
            return ""

    async def run(self):
        """Main application loop."""
        self._print_header()

        # Initialize RAG system if enabled
        if self.rag_system:
            print(f"{Fore.CYAN}Initializing RAG-Redis system...")
            if await self.rag_system.initialize():
                self.rag_enabled = True
                print(f"{Fore.GREEN}RAG system ready with 5-tier memory architecture")
            else:
                print(f"{Fore.YELLOW}RAG system initialization failed, continuing without RAG")

        # Add system message for context
        system_msg = "You are a helpful AI assistant. Provide clear, concise, and helpful responses."
        if self.rag_enabled:
            system_msg += " Use relevant memories from your knowledge base to enhance responses."
        self.conversation.add_message("system", system_msg)

        while self.running:
            try:
                # Get user input
                user_input = input(f"{Fore.BLUE}You: {Fore.WHITE}").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if not self._handle_command(user_input):
                        break
                    continue

                # Generate response
                await self._get_response(user_input)
                print()  # Extra line for readability

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Use /quit to exit or continue chatting...")
            except EOFError:
                print(f"\n{Fore.YELLOW}Goodbye!")
                break
            except Exception as e:
                print(f"{Fore.RED}Unexpected error: {e}")


def validate_model_files(model_path: str, tokenizer_path: Optional[str]) -> bool:
    """Validate that model files exist and are accessible."""
    if not os.path.exists(model_path):
        print(f"{Fore.RED}Error: Model file not found: {model_path}")
        return False

    if tokenizer_path and not os.path.exists(tokenizer_path):
        print(f"{Fore.RED}Error: Tokenizer file not found: {tokenizer_path}")
        return False

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Gemma CLI Wrapper - Chat interface for Gemma models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gemma-cli.py --model C:\\codedev\\llm\\.models\\gemma-gemmacpp-2b-it-v3\\2b-it.sbs
  python gemma-cli.py --model path\\to\\model.sbs --tokenizer path\\to\\tokenizer.spm --temperature 0.8
  python gemma-cli.py --model path\\to\\model.sbs --max-tokens 4096 --max-context 16384
        """
    )

    # Model configuration
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to the Gemma model file (.sbs)"
    )
    parser.add_argument(
        "--tokenizer", "-t",
        help="Path to tokenizer file (.spm). Optional for single-file models."
    )

    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=8192,
        help="Maximum conversation context length (default: 8192)"
    )

    # Development and testing options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose output"
    )

    # RAG system options
    parser.add_argument(
        "--enable-rag",
        action="store_true",
        help="Enable RAG-Redis memory system"
    )
    parser.add_argument(
        "--redis-host",
        default="localhost",
        help="Redis server hostname (default: localhost)"
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis server port (default: 6379)"
    )
    parser.add_argument(
        "--redis-db",
        type=int,
        default=0,
        help="Redis database number (default: 0)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate model files
    if not validate_model_files(args.model, args.tokenizer):
        sys.exit(1)

    # Create and run CLI
    try:
        cli = GemmaCLI(args)
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted by user.")
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()