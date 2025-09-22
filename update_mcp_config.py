#!/usr/bin/env python3
"""
Update MCP Configuration Script
Properly adds Gemma server configuration to existing MCP config.
"""

import json
from pathlib import Path

def main():
    config_path = Path("/c/codedev/llm/stats/mcp.json")

    # Load existing configuration
    with open(config_path) as f:
        config = json.load(f)

    # Add Gemma server configuration
    gemma_config = {
        "command": "uv",
        "args": ["run", "python", "/c/codedev/llm/gemma/mcp-server/gemma_mcp_server.py"],
        "cwd": "/c/codedev/llm/gemma",
        "transport": ["stdio"],
        "environment": {
            "PYTHONPATH": "/c/codedev/llm/gemma",
            "GEMMA_MODELS_DIR": "/c/codedev/llm/.models",
            "GEMMA_LOG_LEVEL": "info"
        },
        "capabilities": [
            "text_generation",
            "chat_completion",
            "model_management"
        ]
    }

    # Add to mcpServers
    config["mcpServers"]["gemma"] = gemma_config

    # Write back with proper formatting
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("Successfully updated MCP configuration with Gemma server")

if __name__ == "__main__":
    main()