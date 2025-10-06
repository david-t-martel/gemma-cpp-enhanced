#!/usr/bin/env python3
"""
MCP Gemma Server - Refactored main executable following SOLID principles.

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
from typing import List, Optional

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.core import ConfigurationBuilder, ServerFactory, TransportFactory
from server.transports import CompositeTransport, TransportStrategy


class Application:
    """Main application class following Single Responsibility Principle."""

    def __init__(self, args):
        self.args = args
        self.server = None
        self.transports: List[TransportStrategy] = []
        self.running = False
        self.logger = logging.getLogger(__name__)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.stop())

    async def initialize(self):
        """Initialize the application."""
        # Build configuration
        config = ConfigurationBuilder.from_args(self.args)

        # Create server using factory
        self.server = ServerFactory.create_from_config(config)
        await self.server.initialize()

        # Create transports based on mode
        self._create_transports()

    def _create_transports(self):
        """Create transport strategies based on mode."""
        if self.args.mode == "stdio":
            transport = TransportFactory.create_stdio_transport(self.server)
            self.transports.append(transport)

        elif self.args.mode == "http":
            transport = TransportFactory.create_http_transport(
                self.server, self.args.host, self.args.port
            )
            self.transports.append(transport)

        elif self.args.mode == "websocket":
            ws_port = self.args.ws_port or (self.args.port + 1)
            transport = TransportFactory.create_websocket_transport(
                self.server, self.args.host, ws_port
            )
            self.transports.append(transport)

        elif self.args.mode == "all":
            # Create composite transport with all strategies
            composite = CompositeTransport(self.server)

            # Add stdio
            stdio = TransportFactory.create_stdio_transport(self.server)
            composite.add_transport(stdio)

            # Add HTTP
            http = TransportFactory.create_http_transport(
                self.server, self.args.host, self.args.port
            )
            composite.add_transport(http)

            # Add WebSocket
            ws_port = self.args.ws_port or (self.args.port + 1)
            websocket = TransportFactory.create_websocket_transport(
                self.server, self.args.host, ws_port
            )
            composite.add_transport(websocket)

            self.transports.append(composite)

    async def start(self):
        """Start the application."""
        self.running = True
        self.logger.info("Starting MCP Gemma Server")

        try:
            # Initialize the application
            await self.initialize()

            # Start all transports
            if len(self.transports) == 1 and self.args.mode == "stdio":
                # For stdio, run directly
                await self.transports[0].start()
            else:
                # For other modes, run concurrently
                tasks = []
                for transport in self.transports:
                    tasks.append(asyncio.create_task(transport.start()))

                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            self.logger.error(f"Application error: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop the application."""
        if not self.running:
            return

        self.running = False
        self.logger.info("Stopping MCP Gemma Server")

        # Stop all transports
        for transport in self.transports:
            try:
                await transport.stop()
            except Exception as e:
                self.logger.error(f"Error stopping transport: {e}")

        # Shutdown server
        if self.server:
            await self.server.shutdown()

        self.logger.info("MCP Gemma Server stopped")


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

    # Memory settings
    parser.add_argument("--no-memory", action="store_true", help="Disable memory features")
    parser.add_argument(
        "--memory-backend",
        choices=["redis", "inmemory"],
        default="redis",
        help="Memory backend (default: redis)",
    )
    parser.add_argument("--redis-host", default="localhost", help="Redis host (default: localhost)")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port (default: 6379)")
    parser.add_argument(
        "--redis-db", type=int, default=0, help="Redis database number (default: 0)"
    )

    # Metrics
    parser.add_argument("--no-metrics", action="store_true", help="Disable metrics collection")

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


def validate_configuration(args) -> bool:
    """Validate configuration before starting."""
    # Check model path
    model_path = Path(args.model)
    if not model_path.exists():
        logging.error(f"Model file not found: {args.model}")
        return False

    if not model_path.is_file():
        logging.error(f"Model path is not a file: {args.model}")
        return False

    # Log model file size
    size_mb = model_path.stat().st_size / (1024 * 1024)
    logging.info(f"Model file size: {size_mb:.2f} MB")

    # Check tokenizer if provided
    if args.tokenizer:
        tokenizer_path = Path(args.tokenizer)
        if not tokenizer_path.exists():
            logging.error(f"Tokenizer file not found: {args.tokenizer}")
            return False

    return True


async def main():
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args)

    # Validate configuration
    if not validate_configuration(args):
        sys.exit(1)

    # Adjust arguments for configuration builder
    if args.no_memory:
        args.memory_backend = "disabled"
    if args.no_metrics:
        args.metrics_enabled = False
    else:
        args.metrics_enabled = True

    # Create and start application
    app = Application(args)

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
