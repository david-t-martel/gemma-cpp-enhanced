"""
Gemma CLI - Enhanced CLI wrapper for Gemma inference with RAG-Redis integration.

This package provides a modern, user-friendly command-line interface for running
Gemma language models locally with advanced features including:
- RAG-enhanced memory system (5-tier architecture)
- MCP (Model Context Protocol) integration
- Rich terminal UI with progress indicators
- Model configuration presets
- Interactive conversation management
"""

__version__ = "2.0.0"
__author__ = "Gemma CLI Team"

from gemma_cli.core.conversation import ConversationManager
from gemma_cli.core.gemma import GemmaInterface

__all__ = [
    "ConversationManager",
    "GemmaInterface",
    "__version__",
]
