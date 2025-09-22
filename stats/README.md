# stats LLM Agent Framework

This project is a high-performance LLM agent framework featuring Google Gemma/Phi-2 models, ReAct reasoning, and a production-ready RAG-Redis system built in Rust.

## Features

*   **ReAct Agent**: A reasoning and acting pattern with planning capabilities.
*   **Tool Calling**: A variety of built-in tools, including a calculator, file operations, and web search.
*   **Multi-Model Support**: Designed to work with Gemma and Phi-2 models.
*   **Async Architecture**: A scalable async/await inference pipeline.
*   **RAG-Redis System**: A high-performance RAG system built in Rust for retrieval-augmented generation.

## Getting Started

### Prerequisites

*   Python 3.11+ with `uv`
*   Rust toolchain
*   Redis server

### Installation

1.  **Set up the environment:**

    ```bash
    cd C:\codedev\llm\stats
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    uv pip install -r requirements.txt
    ```

2.  **Build Rust extensions:**

    The `stats` agent uses a Rust-based RAG system. To build it, follow the instructions in the `rag-redis` project's `README.md` file.

3.  **Download a model:**

    ```bash
    uv run python src/gcp/gemma_download.py --auto
    ```

### Running the Agent

#### Interactive CLI Mode

```bash
uv run python main.py
```

#### HTTP Server Mode

```bash
uv run python -m src.server.main
```

## Integration with `gemma.cpp`

The `stats` agent can use the `gemma.cpp` executable as a backend for inference. To enable this, you need to build the `gemma_extensions` Python bindings.

### Building `gemma_extensions`

*Instructions for building the Python bindings for `gemma.cpp` need to be added here. This will likely involve using `pyo3` and `maturin`.*

Once the bindings are built and installed in the `stats` agent's virtual environment, you can run the agent with the `cpp` backend:

```bash
uv run python main.py --backend cpp
```

## Documentation for LLM Agents

This `README.md` file, along with the other documentation in this workspace, is intended to be used by LLM agents to understand, build, and debug the system.