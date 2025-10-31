"""Client for the Gemma CLI.

This module provides a client for communicating with the Gemma CLI server.
"""

import asyncio
import aiohttp

class GemmaCLIClient:
    """A client for the Gemma CLI server."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session: aiohttp.ClientSession | None = None

    async def connect(self) -> None:
        """Connect to the server."""
        self.session = aiohttp.ClientSession()

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self.session:
            await self.session.close()

    async def hello(self) -> str:
        """Send a "hello" message to the server."""
        if not self.session:
            raise ConnectionError("Not connected to the server.")

        async with self.session.get(f"{self.base_url}/") as response:
            return await response.text()

async def main():
    """A simple main function to test the client."""
    client = GemmaCLIClient("http://localhost:8000")
    await client.connect()
    try:
        message = await client.hello()
        print(message)
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
