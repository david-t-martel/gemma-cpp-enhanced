#!/usr/bin/env python
"""Script to modify hybrid_rag.py for optimization integration."""

import re
from pathlib import Path

# Read the file
hybrid_rag_path = Path("rag/hybrid_rag.py")
content = hybrid_rag_path.read_text()

# 1. Add use_optimized_rag parameter to __init__ signature
content = re.sub(
    r'(rust_mcp_server_path: Optional\[str\] = None,)\n(\s*\) -> None:)',
    r'\1\n        use_optimized_rag: bool = True,\n\2',
    content
)

# 2. Add parameter to docstring
content = re.sub(
    r'(rust_mcp_server_path: Path to Rust MCP server binary \(for \'rust\' backend\))\n(\s*""")',
    r'\1\n            use_optimized_rag: Use optimized RAG stores for better performance\n\2',
    content
)

# 3. Store as instance variable (after line ~37-40)
content = re.sub(
    r'(# Handle backward compatibility.*?\n.*?use_embedded = True.*?\n)',
    r'\1        \n        # Store optimization flag\n        self.use_optimized_rag = use_optimized_rag\n',
    content,
    flags=re.DOTALL
)

# 4. Update PythonRAGBackend instantiation at line ~58
content = re.sub(
    r'self\.python_backend = PythonRAGBackend\(use_embedded_store=use_embedded\)',
    r'self.python_backend = PythonRAGBackend(use_embedded_store=use_embedded, use_optimized_rag=self.use_optimized_rag)',
    content
)

# 5. Update fallback PythonRAGBackend instantiation at line ~76
content = re.sub(
    r'self\.python_backend = PythonRAGBackend\(use_embedded_store=True\)',
    r'self.python_backend = PythonRAGBackend(use_embedded_store=True, use_optimized_rag=self.use_optimized_rag)',
    content
)

# Write the modified content
hybrid_rag_path.write_text(content)

print("Successfully modified hybrid_rag.py:")
print("- Added use_optimized_rag parameter to __init__")
print("- Updated docstring")
print("- Stored optimization flag as instance variable")
print("- Updated PythonRAGBackend instantiation (both occurrences)")