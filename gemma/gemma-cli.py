#!/usr/bin/env python3
"""
Gemma CLI Wrapper - Windows chat interface for native gemma.exe
Provides a chat-like interface similar to claude or gemini-cli with conversation management.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import signal

try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    # Fallback for no colorama
    COLORS_AVAILABLE = False
    class MockFore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class MockStyle:
        BRIGHT = DIM = RESET_ALL = ""
    Fore = MockFore()
    Style = MockStyle()


class ConversationManager:
    """Manages conversation history and context for the chat session."""

    def __init__(self, max_context_length: int = 8192):
        self.messages: List[Dict[str, str]] = []
        self.max_context_length = max_context_length
        self.session_start = datetime.now()

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_context()

    def _trim_context(self) -> None:
        """Trim conversation to stay within context length."""
        total_length = sum(len(msg["content"]) for msg in self.messages)
        while total_length > self.max_context_length and len(self.messages) > 1:
            # Remove oldest messages but keep system message if present
            if self.messages[0].get("role") == "system":
                if len(self.messages) > 2:
                    self.messages.pop(1)
                else:
                    break
            else:
                self.messages.pop(0)
            total_length = sum(len(msg["content"]) for msg in self.messages)

    def get_context_prompt(self) -> str:
        """Generate a context-aware prompt for gemma."""
        if not self.messages:
            return ""

        # Build conversation context
        context_parts = []
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                context_parts.append(f"Assistant: {content}")
            elif role == "system":
                context_parts.append(f"System: {content}")

        return "\n".join(context_parts)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        self.session_start = datetime.now()

    def save_to_file(self, filepath: Path) -> bool:
        """Save conversation to JSON file."""
        try:
            data = {
                "session_start": self.session_start.isoformat(),
                "messages": self.messages,
                "saved_at": datetime.now().isoformat()
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"{Fore.RED}Error saving conversation: {e}")
            return False

    def load_from_file(self, filepath: Path) -> bool:
        """Load conversation from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.messages = data.get("messages", [])
            session_start_str = data.get("session_start")
            if session_start_str:
                self.session_start = datetime.fromisoformat(session_start_str)
            return True
        except Exception as e:
            print(f"{Fore.RED}Error loading conversation: {e}")
            return False


class GemmaInterface:
    """Interface for communicating with the WSL-built gemma executable."""

    def __init__(self,
                 model_path: str,
                 tokenizer_path: Optional[str] = None,
                 gemma_executable: str = "/mnt/c/codedev/llm/gemma/gemma.cpp/build_wsl/gemma",
                 max_tokens: int = 2048,
                 temperature: float = 0.7):
        self.model_path = self._to_wsl_path(model_path)
        self.tokenizer_path = self._to_wsl_path(tokenizer_path) if tokenizer_path else None
        self.gemma_executable = gemma_executable
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.process: Optional[subprocess.Popen] = None

    def _to_wsl_path(self, windows_path: str) -> str:
        """Converts a Windows path to a WSL path."""
        if not windows_path:
            return None
        # First, normalize the path to use forward slashes, which WSL can often handle.
        path = windows_path.replace('\\', '/')
        # Then, replace C: with /mnt/c, etc.
        path = re.sub(r'^([A-Za-z]):', lambda m: f'/mnt/{m.group(1).lower()}', path)
        return path

    def _build_command(self, prompt: str) -> List[str]:
        """Build the WSL command to execute the gemma executable."""
        cmd = ["wsl", self.gemma_executable, "--weights", self.model_path]

        if self.tokenizer_path:
            cmd.extend(["--tokenizer", self.tokenizer_path])

        # Add generation parameters
        cmd.extend([
            "--max_generated_tokens", str(self.max_tokens),
            "--temperature", str(self.temperature),
            "--prompt", prompt
        ])

        return cmd

    async def generate_response(self, prompt: str, stream_callback=None) -> str:
        """Generate response from gemma with optional streaming callback."""
        cmd = self._build_command(prompt)

        # Debug: print the command that will be executed
        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(f"{Fore.YELLOW}Debug - Command: {' '.join(cmd)}")

        try:
            # Use proper Windows process creation with CREATE_NEW_PROCESS_GROUP
            startup_info = subprocess.STARTUPINFO()
            startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup_info.wShowWindow = subprocess.SW_HIDE

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,  # Unbuffered for real-time streaming
                universal_newlines=True,
                startupinfo=startup_info,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )

            response_parts = []

            # Read output in real-time for streaming
            while True:
                # Use a small timeout to check for process completion
                try:
                    output = self.process.stdout.read(1)
                    if output == '' and self.process.poll() is not None:
                        break
                    if output:
                        response_parts.append(output)
                        if stream_callback:
                            stream_callback(output)
                        # Allow other async tasks to run
                        await asyncio.sleep(0)
                except Exception:
                    break

            # Wait for process completion
            return_code = self.process.wait()

            if return_code != 0:
                stderr_output = self.process.stderr.read()
                raise RuntimeError(f"Gemma process failed (code {return_code}): {stderr_output}")

            return ''.join(response_parts)

        except Exception as e:
            print(f"{Fore.RED}Error communicating with gemma: {e}")
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"{Fore.YELLOW}Debug - Command was: {' '.join(cmd)}")
            return f"Error: {str(e)}"
        finally:
            if self.process:
                try:
                    if self.process.poll() is None:
                        self.process.terminate()
                        self.process.wait(timeout=5)
                except:
                    pass
                self.process = None

    def stop_generation(self):
        """Stop current generation process."""
        if self.process:
            try:
                # On Windows, try to terminate gracefully first
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                self.process.kill()
                self.process.wait()
            except Exception:
                pass
            self.process = None


class GemmaCLI:
    """Main CLI application class."""

    def __init__(self, args):
        self.args = args
        self.conversation = ConversationManager(max_context_length=args.max_context)
        self.gemma = GemmaInterface(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )

        # Enable debug mode if requested
        if hasattr(args, 'debug') and args.debug:
            self.gemma.debug_mode = True

        self.running = True
        self.conversation_dir = Path.home() / ".gemma_conversations"
        self.conversation_dir.mkdir(exist_ok=True)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        print(f"\n{Fore.YELLOW}Shutting down gracefully...")
        self.gemma.stop_generation()
        self.running = False
        sys.exit(0)

    def _print_header(self):
        """Print the CLI header."""
        print(f"{Fore.CYAN}{Style.BRIGHT}")
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║                       Gemma CLI Wrapper                     ║")
        print("║              Chat interface for Gemma models                ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(f"{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Model: {self.args.model}")
        if self.args.tokenizer:
            print(f"{Fore.GREEN}Tokenizer: {self.args.tokenizer}")
        print(f"{Fore.GREEN}Max tokens: {self.args.max_tokens}, Temperature: {self.args.temperature}")
        print(f"{Fore.YELLOW}Type /help for commands, /quit to exit")
        print("-" * 60)

    def _print_help(self):
        """Print help information."""
        help_text = f"""
{Fore.CYAN}{Style.BRIGHT}Available Commands:{Style.RESET_ALL}
{Fore.GREEN}/help{Fore.WHITE}              - Show this help message
{Fore.GREEN}/clear{Fore.WHITE}             - Clear conversation history
{Fore.GREEN}/save [filename]{Fore.WHITE}   - Save conversation to file
{Fore.GREEN}/load [filename]{Fore.WHITE}   - Load conversation from file
{Fore.GREEN}/list{Fore.WHITE}              - List saved conversations
{Fore.GREEN}/status{Fore.WHITE}            - Show current session status
{Fore.GREEN}/settings{Fore.WHITE}          - Show current model settings
{Fore.GREEN}/quit{Fore.WHITE} or {Fore.GREEN}/exit{Fore.WHITE}    - Exit the application

{Fore.CYAN}{Style.BRIGHT}Usage Tips:{Style.RESET_ALL}
- Type your message and press Enter to chat
- Use Ctrl+C to interrupt ongoing generation
- Conversations automatically maintain context
- Files saved to: {self.conversation_dir}
"""
        print(help_text)

    def _handle_command(self, command: str) -> bool:
        """Handle special commands. Returns True if command was processed."""
        command = command.strip()

        if command in ["/quit", "/exit"]:
            print(f"{Fore.YELLOW}Goodbye!")
            return False

        elif command == "/help":
            self._print_help()

        elif command == "/clear":
            self.conversation.clear()
            print(f"{Fore.GREEN}Conversation history cleared.")

        elif command.startswith("/save"):
            parts = command.split(maxsplit=1)
            if len(parts) > 1:
                filename = parts[1]
            else:
                filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            filepath = self.conversation_dir / filename
            if self.conversation.save_to_file(filepath):
                print(f"{Fore.GREEN}Conversation saved to: {filepath}")

        elif command.startswith("/load"):
            parts = command.split(maxsplit=1)
            if len(parts) > 1:
                filename = parts[1]
                filepath = self.conversation_dir / filename
                if filepath.exists():
                    if self.conversation.load_from_file(filepath):
                        print(f"{Fore.GREEN}Conversation loaded from: {filepath}")
                        print(f"{Fore.CYAN}Loaded {len(self.conversation.messages)} messages")
                else:
                    print(f"{Fore.RED}File not found: {filepath}")
            else:
                print(f"{Fore.RED}Usage: /load <filename>")

        elif command == "/list":
            json_files = list(self.conversation_dir.glob("*.json"))
            if json_files:
                print(f"{Fore.CYAN}Saved conversations:")
                for file in sorted(json_files, key=lambda x: x.stat().st_mtime, reverse=True):
                    mtime = datetime.fromtimestamp(file.stat().st_mtime)
                    print(f"  {Fore.GREEN}{file.name}{Fore.WHITE} - {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"{Fore.YELLOW}No saved conversations found.")

        elif command == "/status":
            msg_count = len(self.conversation.messages)
            session_duration = datetime.now() - self.conversation.session_start
            print(f"{Fore.CYAN}Session Status:")
            print(f"  Messages in conversation: {msg_count}")
            print(f"  Session duration: {str(session_duration).split('.')[0]}")
            print(f"  Session started: {self.conversation.session_start.strftime('%Y-%m-%d %H:%M:%S')}")

        elif command == "/settings":
            print(f"{Fore.CYAN}Current Settings:")
            print(f"  Model: {self.gemma.model_path}")
            print(f"  Tokenizer: {self.gemma.tokenizer_path or 'Built-in'}")
            print(f"  Max tokens: {self.gemma.max_tokens}")
            print(f"  Temperature: {self.gemma.temperature}")
            print(f"  Max context length: {self.conversation.max_context_length}")

        else:
            print(f"{Fore.RED}Unknown command: {command}")
            print(f"{Fore.YELLOW}Type /help for available commands.")

        return True

    async def _get_response(self, user_input: str) -> str:
        """Get response from gemma with conversation context."""
        # Add user message to conversation
        self.conversation.add_message("user", user_input)

        # Build context-aware prompt
        context = self.conversation.get_context_prompt()
        full_prompt = f"{context}\nUser: {user_input}\nAssistant:"

        # Track streaming output
        response_parts = []
        print(f"{Fore.MAGENTA}Assistant: {Fore.WHITE}", end="", flush=True)

        def stream_callback(token):
            response_parts.append(token)
            print(token, end="", flush=True)

        # Generate response
        try:
            full_response = await self.gemma.generate_response(full_prompt, stream_callback)
            print()  # New line after streaming

            # Extract just the assistant's response (remove context echo)
            if "Assistant:" in full_response:
                assistant_response = full_response.split("Assistant:")[-1].strip()
            else:
                assistant_response = full_response.strip()

            # Add assistant response to conversation
            self.conversation.add_message("assistant", assistant_response)

            return assistant_response

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Generation interrupted by user.")
            return ""
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(f"\n{Fore.RED}{error_msg}")
            return ""

    async def run(self):
        """Main application loop."""
        self._print_header()

        # Add system message for context
        system_msg = "You are a helpful AI assistant. Provide clear, concise, and helpful responses."
        self.conversation.add_message("system", system_msg)

        while self.running:
            try:
                # Get user input
                user_input = input(f"{Fore.BLUE}You: {Fore.WHITE}").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if not self._handle_command(user_input):
                        break
                    continue

                # Generate response
                await self._get_response(user_input)
                print()  # Extra line for readability

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Use /quit to exit or continue chatting...")
            except EOFError:
                print(f"\n{Fore.YELLOW}Goodbye!")
                break
            except Exception as e:
                print(f"{Fore.RED}Unexpected error: {e}")


def validate_model_files(model_path: str, tokenizer_path: Optional[str]) -> bool:
    """Validate that model files exist and are accessible."""
    if not os.path.exists(model_path):
        print(f"{Fore.RED}Error: Model file not found: {model_path}")
        return False

    if tokenizer_path and not os.path.exists(tokenizer_path):
        print(f"{Fore.RED}Error: Tokenizer file not found: {tokenizer_path}")
        return False

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Gemma CLI Wrapper - Chat interface for Gemma models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gemma-cli.py --model C:\\codedev\\llm\\.models\\gemma-gemmacpp-2b-it-v3\\2b-it.sbs
  python gemma-cli.py --model path\\to\\model.sbs --tokenizer path\\to\\tokenizer.spm --temperature 0.8
  python gemma-cli.py --model path\\to\\model.sbs --max-tokens 4096 --max-context 16384
        """
    )

    # Model configuration
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to the Gemma model file (.sbs)"
    )
    parser.add_argument(
        "--tokenizer", "-t",
        help="Path to tokenizer file (.spm). Optional for single-file models."
    )

    # Generation parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=8192,
        help="Maximum conversation context length (default: 8192)"
    )

    # Development and testing options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose output"
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate model files
    if not validate_model_files(args.model, args.tokenizer):
        sys.exit(1)

    # Create and run CLI
    try:
        cli = GemmaCLI(args)
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted by user.")
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()