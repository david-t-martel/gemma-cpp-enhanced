# gemma Project

This project contains a C++ implementation of the Gemma model and a Python CLI wrapper for interacting with it.

## Building the `gemma` executable

The `gemma` executable is built using the Windows Subsystem for Linux (WSL). Follow these steps:

1.  Open a WSL terminal.
2.  Navigate to the `gemma.cpp` directory:
    ```bash
    cd /mnt/c/codedev/llm/gemma/gemma.cpp
    ```
3.  Create a build directory and navigate into it:
    ```bash
    mkdir build_wsl && cd build_wsl
    ```
4.  Configure the build with CMake:
    ```bash
    cmake ..
    ```
5.  Compile the executable:
    ```bash
    make -j4
    ```

This will create a `gemma` executable in the `build_wsl` directory.

## Documentation

The C++ code is documented using Doxygen. To generate the documentation, you will need to have Doxygen installed. You can then run the following command from the `gemma` directory:

```bash
doxygen Doxyfile
```

This will generate HTML documentation in the `html` directory. You can view the documentation by opening the `index.html` file in your web browser.

The documentation is also automatically generated and deployed to GitHub Pages on every push to the `main` branch.

## Using the `gemma-cli.py` wrapper

The `gemma-cli.py` script provides a convenient way to interact with the `gemma` executable from Windows. It manages conversation history and provides a chat-like interface.

To use the CLI, you need to have Python 3.7+ installed on Windows.

### Running the CLI

To run the CLI, execute the following command from the `gemma` directory:

```bash
python gemma-cli.py --model C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3\2b-it.sbs
```

### `gemma-cli.py` Configuration

The `gemma-cli.py` script has been configured to use the WSL-built `gemma` executable. It uses the `wsl.exe` command to run the Linux executable from Windows.

See the `GEMMA_CLI_USAGE.md` file for more details on the available command-line options and interactive commands.