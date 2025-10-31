"""Server for the Gemma CLI.

This module provides a simple server for the Gemma CLI.
"""

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    """A simple endpoint that returns a "Hello, World!" message."""
    return {"message": "Hello, World!"}
