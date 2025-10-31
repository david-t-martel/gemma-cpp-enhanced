"""RAG commands for the Gemma CLI.

This module provides commands for interacting with the RAG system.
"""

from typing import Annotated

import typer

from ..utils import get_console, handle_exceptions
from ..core.rag import RagClient

# Create RAG subcommand app
rag_app = typer.Typer(name="rag", help="🚀 RAG commands", rich_markup_mode="rich")

console = get_console()


@rag_app.command("ingest")
@handle_exceptions(console)
def ingest(
    path: Annotated[str, typer.Argument(help="Path to the file to ingest")],
) -> None:
    """Ingest a file into the RAG system."""
    console.print(f"Ingesting file: {path}")
    rag_client = RagClient("C:\\codedev\\llm\\rag-redis\\rag-binaries\\bin\\rag-server.exe")
    rag_client.start_server()
    response = rag_client.rag_command("ingest", path=path)
    console.print(response)
    rag_client.stop_server()

