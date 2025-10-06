#!/usr/bin/env python3
"""
MCP Gemma Server - Main executable for the MCP server.

Usage:
    python main.py --mode stdio                           # Start in stdio mode (default)
    python main.py --mode http --host 0.0.0.0 --port 8080 # Start HTTP server
    python main.py --mode websocket --port 8081           # Start WebSocket server
    python main.py --mode all                             # Start all transports
"""

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.base import GemmaConfig, GemmaServer
from server.transports import HTTPTransport, StdioTransport, WebSocketTransport


class MCPGemmaApp:
    """Main MCP Gemma application."""

    def __init__(self, config: GemmaConfig, args):
        self.config = config
        self.args = args
        self.server = GemmaServer(config)
        self.transports = []
        self.running = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logging.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.stop())

    async def start(self):
        """Start the application with specified transports."""
        self.running = True
        logging.info("Starting MCP Gemma Server")

        try:
            # Initialize transports based on mode
            if self.args.mode in ["stdio", "all"]:
                stdio_transport = StdioTransport(self.server)
                self.transports.append(stdio_transport)

            if self.args.mode in ["http", "all"]:
                http_transport = HTTPTransport(
                    self.server, host=self.args.host, port=self.args.port
                )
                self.transports.append(http_transport)

            if self.args.mode in ["websocket", "all"]:
                ws_port = self.args.ws_port or (self.args.port + 1)
                websocket_transport = WebSocketTransport(
                    self.server, host=self.args.host, port=ws_port
                )
                self.transports.append(websocket_transport)

            # Start all transports
            if self.args.mode == "stdio":
                # For stdio mode, run synchronously
                await self.transports[0].start()
            else:
                # For other modes, start all transports concurrently
                tasks = []
                for transport in self.transports:
                    tasks.append(asyncio.create_task(transport.start()))

                if self.args.mode == "all":
                    # Start stdio in background for all mode
                    stdio_task = asyncio.create_task(self._run_stdio_background())
                    tasks.append(stdio_task)

                # Wait for all tasks
                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logging.error(f"Error starting server: {e}")
            await self.stop()
            raise

    async def _run_stdio_background(self):
        """Run stdio transport in background mode."""
        stdio_transport = StdioTransport(self.server)
        try:
            await stdio_transport.start()
        except Exception as e:
            logging.error(f"Stdio transport error: {e}")

    async def stop(self):
        """Stop the application and cleanup."""
        if not self.running:
            return

        self.running = False
        logging.info("Stopping MCP Gemma Server")

        # Stop all transports
        for transport in self.transports:
            try:
                await transport.stop()
            except Exception as e:
                logging.error(f"Error stopping transport: {e}")

        logging.info("MCP Gemma Server stopped")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MCP Gemma Server - Model Context Protocol server for gemma.cpp"
    )

    # Server mode
    parser.add_argument(
        "--mode",
        choices=["stdio", "http", "websocket", "all"],
        default="stdio",
        help="Server mode (default: stdio)",
    )

    # Network settings
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to for HTTP/WebSocket servers (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument(
        "--ws-port", type=int, help="Port for WebSocket server (default: http-port + 1)"
    )

    # Model settings
    parser.add_argument("--model", required=True, help="Path to the Gemma model file")
    parser.add_argument(
        "--tokenizer", help="Path to the tokenizer file (optional for single-file models)"
    )
    parser.add_argument(
        "--gemma-executable",
        default="/mnt/c/codedev/llm/gemma/gemma.cpp/build_wsl/gemma",
        help="Path to the gemma executable",
    )

    # Generation settings
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Maximum tokens to generate (default: 2048)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max-context", type=int, default=8192, help="Maximum context length (default: 8192)"
    )

    # Redis settings
    parser.add_argument("--no-redis", action="store_true", help="Disable Redis memory features")
    parser.add_argument("--redis-host", default="localhost", help="Redis host (default: localhost)")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port (default: 6379)")
    parser.add_argument(
        "--redis-db", type=int, default=0, help="Redis database number (default: 0)"
    )

    # Logging
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-file", help="Log file path (logs to console if not specified)")

    return parser.parse_args()


def setup_logging(args):
    """Setup logging configuration."""
    level = logging.DEBUG if args.debug else logging.INFO
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if args.log_file:
        logging.basicConfig(level=level, format=format_string, filename=args.log_file, filemode="a")
    else:
        logging.basicConfig(level=level, format=format_string)


def validate_model_path(model_path: str) -> bool:
    """Validate that the model path exists and is accessible."""
    path = Path(model_path)
    if not path.exists():
        logging.error(f"Model file not found: {model_path}")
        return False

    if not path.is_file():
        logging.error(f"Model path is not a file: {model_path}")
        return False

    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    logging.info(f"Model file size: {size_mb:.2f} MB")

    return True


async def main():
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args)

    # Validate model path
    if not validate_model_path(args.model):
        sys.exit(1)

    # Create configuration
    config = GemmaConfig(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        gemma_executable=args.gemma_executable,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_context=args.max_context,
        enable_redis=not args.no_redis,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        debug=args.debug,
    )

    # Create and start application
    app = MCPGemmaApp(config, args)

    try:
        await app.start()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"Application error: {e}")
        sys.exit(1)
    finally:
        await app.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
