"""Main entry point for the LLM agent system."""

import argparse
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up standardized logging
from src.shared.logging import LogLevel, get_logger, setup_logging

setup_logging(level=LogLevel.INFO, console=True)
logger = get_logger(__name__)

from src.agent.gemma_agent import AgentMode, create_gemma_agent
from src.agent.tools import tool_registry

# Try to import gemma.cpp extensions with graceful fallback
GEMMA_CPP_AVAILABLE = False
try:
    from gemma_extensions import GemmaCpp

    GEMMA_CPP_AVAILABLE = True
    logger.info("gemma.cpp extensions available")
except ImportError:
    logger.info("gemma.cpp extensions not available - using PyTorch backend only")


def check_model_existence(model_name: str = "google/gemma-2b-it") -> bool:
    """Check if the required model files exist locally.

    Args:
        model_name: The model name to check for

    Returns:
        True if model exists, False otherwise
    """
    project_root = Path(__file__).parent

    # Check for models directory (direct download structure)
    models_dir = project_root / "models" / "gemma-2b"
    if models_dir.exists() and any(models_dir.glob("*")):
        logger.info(f"Found model files in: {models_dir}")
        return True

    # Check for models_cache directory (Hugging Face cache structure)
    cache_dir = project_root / "models_cache"
    if cache_dir.exists():
        # Look for actual model files, not just metadata
        model_files = (
            list(cache_dir.rglob("*.bin"))
            + list(cache_dir.rglob("*.safetensors"))
            + list(cache_dir.rglob("pytorch_model.bin"))
        )
        if model_files:
            logger.info(f"Found cached model files in: {cache_dir}")
            return True

    # Check system-wide Hugging Face cache
    try:
        from transformers.utils import TRANSFORMERS_CACHE

        hf_cache = Path(TRANSFORMERS_CACHE)
        if hf_cache.exists():
            # Look for model-specific cache directory
            for cache_subdir in hf_cache.iterdir():
                if cache_subdir.is_dir() and "gemma" in cache_subdir.name.lower():
                    model_files = list(cache_subdir.rglob("*.bin")) + list(
                        cache_subdir.rglob("*.safetensors")
                    )
                    if model_files:
                        logger.info(f"Found model files in HF cache: {cache_subdir}")
                        return True
    except ImportError:
        pass  # transformers not installed yet

    return False


def print_model_download_instructions():
    """Print clear instructions for downloading required models."""
    print("\n" + "=" * 60)
    print("❌ MODEL FILES NOT FOUND")
    print("=" * 60)
    print("\nThe Gemma model files are required but not found locally.")
    print("\nTo download the model, run:")
    print("   uv run python -m src.gcp.gemma_download gemma-2b")
    print("\nNote: You may need a Hugging Face token for Gemma models.")
    print("If you don't have one:")
    print("1. Create account at https://huggingface.co")
    print("2. Generate token at https://huggingface.co/settings/tokens")
    print("3. Accept Gemma license at https://huggingface.co/google/gemma-2b-it")
    print("4. Login with: huggingface-cli login")
    print("\nAlternatively, try the lightweight mode which may download automatically:")
    print("   uv run python main.py --lightweight")
    print("=" * 60)


def interactive_chat(agent):
    """Run an interactive chat session with the agent."""
    print("\n" + "=" * 60)
    print("🤖 LLM Agent with Tool Calling")
    print("=" * 60)
    print("\nAvailable tools:")
    for tool in tool_registry.list_tools():
        print(f"  • {tool.name}: {tool.description}")

    print("\n" + "-" * 60)
    print("Commands:")
    print("  /help    - Show this help message")
    print("  /tools   - List available tools")
    print("  /history - Show conversation history")
    print("  /save    - Save conversation history")
    print("  /load    - Load conversation history")
    print("  /reset   - Clear conversation history")
    print("  /exit    - Exit the chat")
    print("-" * 60)

    while True:
        try:
            # Get user input
            print("\n" + "=" * 40)
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower().split()[0]

                if command == "/exit":
                    print("👋 Goodbye!")
                    break

                elif command == "/help":
                    print("\nCommands:")
                    print("  /help    - Show this help message")
                    print("  /tools   - List available tools")
                    print("  /history - Show conversation history")
                    print("  /save    - Save conversation history")
                    print("  /load    - Load conversation history")
                    print("  /reset   - Clear conversation history")
                    print("  /exit    - Exit the chat")

                elif command == "/tools":
                    print("\nAvailable tools:")
                    for tool in tool_registry.list_tools():
                        print(f"\n• {tool.name}")
                        print(f"  {tool.description}")
                        if tool.parameters:
                            print("  Parameters:")
                            for param in tool.parameters:
                                req = "required" if param.required else "optional"
                                print(
                                    f"    - {param.name} ({param.type}, {req}): {param.description}"
                                )

                elif command == "/history":
                    print("\nConversation History:")
                    for msg in agent.history.get_messages():
                        print(f"\n[{msg.role.upper()}]: {msg.content[:200]}...")

                elif command == "/save":
                    parts = user_input.split(maxsplit=1)
                    path = parts[1] if len(parts) > 1 else "conversation_history.json"
                    agent.save_history(path)

                elif command == "/load":
                    parts = user_input.split(maxsplit=1)
                    path = parts[1] if len(parts) > 1 else "conversation_history.json"
                    try:
                        agent.load_history(path)
                    except FileNotFoundError:
                        print(f"❌ File not found: {path}")

                elif command == "/reset":
                    agent.reset()

                else:
                    print(f"❌ Unknown command: {command}")

                continue

            # Get agent response
            print("\n" + "-" * 40)
            response = agent.chat(user_input)
            print(f"\n🤖 Assistant: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            print(f"\n❌ Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LLM Agent with Tool Calling")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2b-it",
        help="Model name to use (default: google/gemma-2b-it)",
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use lightweight pipeline version (recommended for quick start)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512, help="Maximum tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p sampling parameter (default: 0.9)"
    )
    parser.add_argument(
        "--8bit", action="store_true", help="Use 8-bit quantization (requires CUDA)"
    )
    parser.add_argument("--no-tools", action="store_true", help="Disable tool calling")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbose output")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["pytorch", "cpp"],
        default="pytorch",
        help="Backend to use for inference (default: pytorch)",
    )

    args = parser.parse_args()

    # Check if model files exist before proceeding
    if not args.lightweight:  # Skip check for lightweight mode which may auto-download
        if not check_model_existence(args.model):
            print_model_download_instructions()
            sys.exit(1)

    # Validate backend choice
    if args.backend == "cpp" and not GEMMA_CPP_AVAILABLE:
        print("\n❌ Error: C++ backend requested but gemma.cpp extensions not available")
        print("\nTo build gemma.cpp support:")
        print("1. Follow the instructions in `gemma/README.md` to build the `gemma` executable using WSL.")
        print("2. Build the Python bindings for `gemma.cpp` (instructions to be added).")
        print("3. Install the bindings in this project's virtual environment.")
        print("\nFalling back to PyTorch backend...")
        args.backend = "pytorch"

    # Create agent
    print("\n🚀 Initializing LLM Agent...")
    print(f"   Model: {args.model}")
    agent_mode = AgentMode.LIGHTWEIGHT if args.lightweight else AgentMode.FULL
    print(f"   Mode: {'Lightweight Pipeline' if args.lightweight else 'Full Model'}")
    print(f"   Backend: {args.backend.upper()}")

    try:
        if args.backend == "cpp" and GEMMA_CPP_AVAILABLE:

            class GemmaCppAgent:
                def __init__(self, model_path):
                    self.model = GemmaCpp(model_path)
                    self.history = []

                def chat(self, message):
                    # The GemmaCpp object is expected to have a method that takes a prompt and returns a response.
                    # The exact method name and signature may need to be adjusted based on the actual bindings.
                    response = self.model.generate(message) # Assuming a `generate` method
                    self.history.append({"role": "user", "content": message})
                    self.history.append({"role": "assistant", "content": response})
                    return response

                def get_history(self):
                    return self.history

                def reset(self):
                    self.history = []

            # The model path for the C++ backend needs to be determined.
            # Assuming it's in the .models directory with a .bin extension.
            model_path = f"C:/codedev/llm/.models/{args.model.split('/')[-1]}.bin"
            agent = GemmaCppAgent(model_path)

        else:
            # Use PyTorch backend
            agent = create_gemma_agent(
                mode=agent_mode,
                model_name=args.model,
                tool_registry=None if args.no_tools else tool_registry,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                use_8bit=args._8bit if agent_mode == AgentMode.FULL else False,
                verbose=not args.quiet,
            )

        print("\n✅ Agent initialized successfully!")

        # Run interactive chat
        interactive_chat(agent)

    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have installed all requirements:")
        print("   uv pip install -r requirements.txt")
        print("2. For Gemma models, you may need to authenticate with Hugging Face:")
        print("   huggingface-cli login")
        print("3. Try using the lightweight mode:")
        print("   python main.py --lightweight")
        sys.exit(1)


if __name__ == "__main__":
    main()
