# GEMINI Project: Gemma C++ Engine and Python CLI

## Project Overview

This project consists of a high-performance C++ implementation of Google's Gemma model family, paired with a feature-rich Python command-line interface (CLI) that acts as a user-friendly wrapper. The core C++ engine is optimized for performance and can be built with various hardware acceleration backends, while the Python CLI provides a conversational chat experience.

**Main Technologies:**

*   **Core Engine:** C++
*   **CLI:** Python
*   **Build System:** CMake
*   **Dependencies:** Highway, SentencePiece, and potentially hardware-specific libraries like CUDA and oneAPI.

**Architecture:**

The project follows a dual-component architecture:

1.  **C++ Backend (`gemma.cpp`):** This is the core of the project, responsible for running the Gemma model. It's designed to be compiled into a standalone executable (`gemma` or `gemma.exe`).
2.  **Python Frontend (`gemma-cli.py`):** This script provides the user interface. It calls the C++ executable in the background, managing input and output to create a seamless chat experience. It also includes advanced features like conversation management and a RAG system for context augmentation.

## Building and Running

### Building the C++ Engine

The primary method for building the C++ engine is using CMake within the Windows Subsystem for Linux (WSL).

1.  **Navigate to the `gemma.cpp` directory in a WSL terminal:**
    ```bash
    cd /mnt/c/codedev/llm/gemma/gemma.cpp
    ```
2.  **Create and enter a build directory:**
    ```bash
    mkdir build_wsl && cd build_wsl
    ```
3.  **Configure the build with CMake:**
    ```bash
    cmake ..
    ```
4.  **Compile the executable:**
    ```bash
    make -j4
    ```
This will create the `gemma` executable in the `gemma.cpp/build_wsl` directory.

### Running the Python CLI

The `gemma-cli.py` script is the recommended way to interact with the model.

1.  **Ensure you have Python 3.7+ installed on Windows.**
2.  **Run the following command from the project root directory:**
    ```bash
    python gemma-cli.py --model C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs
    ```
    *(Replace the `--model` path with the actual path to your Gemma model file.)*

## Development Conventions

*   **Build System:** CMake is the standard for building the C++ project. The `CMakeLists.txt` file is heavily commented and provides numerous options for customizing the build.
*   **C++ Standard:** The project uses C++20.
*   **Testing:** The project includes a `tests` directory, indicating a commitment to testing. The `pytest.ini` file suggests that `pytest` is used for Python testing.
*   **CLI:** The `gemma-cli.py` script is well-structured, with clear separation of concerns between the `ConversationManager`, `GemmaInterface`, and `GemmaCLI` classes. It also includes type hinting and error handling.
*   **RAG System:** The CLI has an experimental Retrieval-Augmented Generation (RAG) system that can be enabled with the `--enable-rag` flag. This system uses Redis for memory and `sentence-transformers` for embeddings.
