"""Client for the RAG server.

This module provides a client for communicating with the RAG server.
"""

import subprocess
import json

class RagClient:
    """A client for the RAG server."""

    def __init__(self, server_path: str):
        self.server_path = server_path
        self.process: subprocess.Popen | None = None

    def start_server(self) -> None:
        """Start the RAG server."""
        self.process = subprocess.Popen(
            [self.server_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def stop_server(self) -> None:
        """Stop the RAG server."""
        if self.process:
            self.process.terminate()

    def send_message(self, message: dict) -> None:
        """Send a message to the RAG server."""
        if self.process and self.process.stdin:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()

    def receive_message(self) -> dict | None:
        """Receive a message from the RAG server."""
        if self.process and self.process.stdout:
            line = self.process.stdout.readline()
            if line:
                return json.loads(line)
        return None

    def rag_command(self, command: str, **kwargs) -> dict | None:
        """Send a command to the RAG server and get the response."""
        message = {"command": command, "kwargs": kwargs}
        self.send_message(message)
        return self.receive_message()
