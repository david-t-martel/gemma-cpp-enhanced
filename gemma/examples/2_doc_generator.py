#!/usr/bin/env python3
"""
Documentation Generation with Gemma via Ollama

This script automatically generates comprehensive documentation
from C++ code, including class descriptions, method documentation,
and usage examples.
"""

import subprocess
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

def run_ollama(prompt: str, model: str = "gemma2:2b") -> str:
    """Run Ollama with the specified prompt and model."""
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama: {e}")
        return ""
    except FileNotFoundError:
        print("Ollama is not installed. Please install from https://ollama.ai")
        sys.exit(1)

def extract_functions_and_classes(code: str) -> Dict[str, List[str]]:
    """Extract function and class signatures from C++ code."""
    result = {
        "classes": [],
        "functions": [],
        "structs": []
    }

    # Extract class definitions
    class_pattern = r'class\s+(\w+)\s*(?::\s*(?:public|private|protected)\s+[\w:]+)?'
    for match in re.finditer(class_pattern, code):
        result["classes"].append(match.group(1))

    # Extract struct definitions
    struct_pattern = r'struct\s+(\w+)'
    for match in re.finditer(struct_pattern, code):
        result["structs"].append(match.group(1))

    # Extract function signatures (simplified)
    func_pattern = r'(?:[\w:]+\s+)?(\w+)\s*\([^)]*\)\s*(?:const)?\s*(?:override)?\s*[{;]'
    for match in re.finditer(func_pattern, code):
        func_name = match.group(1)
        # Filter out common non-functions
        if func_name not in ['if', 'while', 'for', 'switch', 'class', 'struct']:
            result["functions"].append(func_name)

    return result

def generate_class_documentation(code: str, class_name: str) -> str:
    """Generate documentation for a specific class."""

    prompt = f"""Generate comprehensive documentation for this C++ class.
Include:
1. Class purpose and overview
2. Constructor documentation
3. Method descriptions with parameters and return values
4. Usage examples
5. Important notes or warnings

Class code:
```cpp
{code}
```

Focus on the class: {class_name}

Format the documentation in Markdown with proper headings and code examples."""

    return run_ollama(prompt)

def generate_api_documentation(code: str) -> str:
    """Generate API documentation for the entire file."""

    prompt = f"""You are a technical writer creating API documentation.
Generate complete API documentation for this C++ code.

Include:
1. Overview section
2. Class/struct descriptions
3. Function documentation with:
   - Purpose
   - Parameters (with types and descriptions)
   - Return values
   - Example usage
4. Constants and enums
5. Error handling
6. Thread safety notes (if applicable)

Code:
```cpp
{code}
```

Format as professional API documentation in Markdown."""

    return run_ollama(prompt)

def generate_readme_section(code: str, project_context: str = "") -> str:
    """Generate a README section for the code."""

    prompt = f"""Generate a README.md section for this C++ code module.

Include:
1. What this module does
2. Key features
3. Quick start example
4. API highlights
5. Dependencies
6. Building instructions (if apparent from code)

Project context: {project_context}

Code:
```cpp
{code}
```

Write in a clear, developer-friendly style with code examples."""

    return run_ollama(prompt)

def generate_inline_comments(code: str) -> str:
    """Generate inline documentation comments for code."""

    prompt = f"""Add comprehensive inline documentation comments to this C++ code.
Use Doxygen-style comments for classes and methods.

Rules:
1. Add @brief, @param, @return tags
2. Document complex algorithms
3. Explain non-obvious design decisions
4. Add usage examples for public APIs
5. Note any assumptions or prerequisites

Original code:
```cpp
{code}
```

Return the code with documentation comments added."""

    return run_ollama(prompt)

def save_documentation(doc: str, output_path: Path, format: str = "markdown") -> None:
    """Save generated documentation to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "markdown":
        output_path = output_path.with_suffix('.md')
    elif format == "html":
        # Convert markdown to HTML (basic conversion)
        doc = markdown_to_html(doc)
        output_path = output_path.with_suffix('.html')

    output_path.write_text(doc, encoding='utf-8')
    print(f"Documentation saved to: {output_path}")

def markdown_to_html(markdown: str) -> str:
    """Basic markdown to HTML conversion."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Generated Documentation</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               line-height: 1.6; padding: 20px; max-width: 900px; margin: 0 auto; }}
        pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
        h1, h2, h3 {{ color: #333; }}
        blockquote {{ border-left: 4px solid #ddd; margin: 0; padding-left: 16px; }}
    </style>
</head>
<body>
{markdown}
</body>
</html>"""
    return html

def process_gemma_file(filepath: Path) -> None:
    """Process a gemma.cpp file and generate documentation."""
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return

    code = filepath.read_text(encoding='utf-8')
    print(f"\nGenerating documentation for {filepath.name}...")

    # Generate different types of documentation
    api_doc = generate_api_documentation(code)

    # Save the documentation
    output_dir = Path("C:/codedev/llm/gemma/examples/generated_docs")
    output_file = output_dir / f"{filepath.stem}_api_doc"
    save_documentation(api_doc, output_file)

def main():
    """Main function with example usage."""

    print("Gemma Documentation Generator Demo")
    print("=" * 60)

    # Example 1: Generate documentation for sample code
    sample_code = """
#pragma once
#include <vector>
#include <memory>
#include <functional>

namespace gemma {

template<typename T>
class KVCache {
public:
    struct Config {
        size_t max_seq_len;
        size_t num_layers;
        size_t num_heads;
        size_t head_dim;
    };

    explicit KVCache(const Config& config);
    ~KVCache();

    void Reset();

    void Store(size_t layer, size_t pos, const T* k, const T* v);

    void Retrieve(size_t layer, size_t start, size_t len, T* k, T* v) const;

    size_t GetUsedTokens() const { return used_tokens_; }

    bool IsFull() const { return used_tokens_ >= config_.max_seq_len; }

private:
    Config config_;
    std::vector<T> keys_;
    std::vector<T> values_;
    size_t used_tokens_;

    size_t GetOffset(size_t layer, size_t pos) const;
};

class Attention {
public:
    using ScoreFunction = std::function<float(float)>;

    struct Options {
        bool use_flash_attention = false;
        float dropout_rate = 0.0f;
        ScoreFunction score_fn = nullptr;
    };

    static void MultiHeadAttention(
        const float* query,
        const float* key,
        const float* value,
        float* output,
        size_t seq_len,
        size_t num_heads,
        size_t head_dim,
        const Options& opts = {}
    );

    static void RotaryPositionalEncoding(
        float* data,
        size_t seq_len,
        size_t head_dim,
        size_t pos_offset = 0
    );
};

}  // namespace gemma
"""

    print("\n1. Generating API Documentation...")
    api_doc = generate_api_documentation(sample_code)

    print("\n2. Generating README Section...")
    readme_doc = generate_readme_section(sample_code, "Gemma.cpp - Efficient C++ inference engine")

    print("\n3. Generating Inline Comments...")
    commented_code = generate_inline_comments(sample_code)

    # Save all documentation
    output_dir = Path("C:/codedev/llm/gemma/examples/generated_docs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save API documentation
    (output_dir / "kvcache_api.md").write_text(api_doc, encoding='utf-8')
    print(f"\n✅ API documentation saved to: {output_dir / 'kvcache_api.md'}")

    # Save README section
    (output_dir / "kvcache_readme.md").write_text(readme_doc, encoding='utf-8')
    print(f"✅ README section saved to: {output_dir / 'kvcache_readme.md'}")

    # Save commented code
    (output_dir / "kvcache_documented.h").write_text(commented_code, encoding='utf-8')
    print(f"✅ Documented code saved to: {output_dir / 'kvcache_documented.h'}")

    print("\n" + "="*60)
    print("Documentation generation complete!")
    print("\nYou can now use this script to document any C++ file:")
    print("  python 2_doc_generator.py <path_to_cpp_file>")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Process specified file
        filepath = Path(sys.argv[1])
        process_gemma_file(filepath)
    else:
        # Run demo
        main()