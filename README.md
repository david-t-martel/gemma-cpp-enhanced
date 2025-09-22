# LLM Agent Workspace

This workspace contains a collection of projects for building and running a powerful LLM agent with RAG capabilities.

## Projects

*  **gemma**: A C++ implementation of the Gemma model, along with a Python CLI wrapper.
*  **stats**: A Python-based LLM agent framework with a ReAct agent, tool calling, and integration with the RAG system.
*  **rag-redis**: A high-performance RAG system built in Rust, with a Python bridge for integration with the `stats` agent.
*  **models**: A directory containing the downloaded LLM models.

## Architecture Overview

The `stats` agent is the central component of this workspace. It uses the `gemma` application as a backend for running the LLM and the `rag-redis` system for retrieval-augmented generation.

The `rag-redis` system is a Rust-based service that is accessed by the `stats` agent through a Python bridge, which runs as an MCP server.

## Getting Started

To get the entire system up and running, follow these steps:

### 1. Build the `gemma` executable

The `gemma` executable is built using WSL.

```bash
wsl
cd /mnt/c/codedev/llm/gemma/gemma.cpp
mkdir build_wsl && cd build_wsl
cmake ..
make -j4
```

This will create a `gemma` executable in the `build_wsl` directory.

### 2. Build the `rag-redis` system

The `rag-redis` system is a Rust project.

```bash
cd C:\codedev\llm\rag-redis
cargo build --release
```

### 3. Configure and run the `stats` agent

The `stats` agent is a Python project that uses `uv` for package management.

```bash
cd C:\codedev\llm\stats
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Before running the agent, make sure the `rag-redis` MCP server is running. The `stats` agent is configured to launch it automatically via the `mcp.json` file.

To run the agent in interactive mode:

```bash
uv run python main.py
```

## Integration Details

### `stats` agent and `gemma` backend

The `stats` agent can use the `gemma.cpp` executable as a backend. This is configured in the `stats/main.py` file. The `gemma-cli.py` script in the `gemma` project is also an example of how to interact with the `gemma` executable.

The `gemma-cli.py` script needs to be updated to use the WSL-built executable:

```python
# In gemma/gemma-cli.py

# Change this:
self.gemma_executable = os.path.normpath(gemma_executable)
# To this:
self.gemma_executable = ["wsl", "/mnt/c/codedev/llm/gemma/gemma.cpp/build_wsl/gemma"]

# And update the command building logic accordingly.
```

### `stats` agent and `rag-redis`

The `stats` agent communicates with the `rag-redis` system through an MCP server. The configuration for this is in the `stats/mcp.json` file. The `rag-redis/python-bridge` directory contains the Python code for the MCP server.

## Documentation for LLM Agents

This workspace is designed to be understood and used by LLM agents. The documentation in each subproject`s `README.md` file provides more detailed information about that specific component.

By following the instructions in this `README.md` and the subproject `README.md` files, an LLM agent should be able to build, run, and debug the entire system.
