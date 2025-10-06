"""
Integration Test Suite for LLM Development Ecosystem

This comprehensive test suite validates the integrated operation of:
- Python stats framework with AI agents
- RAG-Redis memory system
- MCP server communication
- Gemma C++ inference engine
- Performance and stress testing
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "stats"))