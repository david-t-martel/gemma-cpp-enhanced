"""Conversation management for Gemma CLI."""

import json
from .enums import Role
from datetime import datetime
from pathlib import Path
from typing import Any
import logging

import aiofiles

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation history and context for the chat session."""

    # Security constants
    VALID_ROLES = {role.value for role in Role}
    MAX_MESSAGE_LENGTH = 100_000  # 100KB per message

    def __init__(self, max_context_length: int = 8192) -> None:
        """
        Initialize the conversation manager.

        Args:
            max_context_length: Maximum total character length for conversation context
        """
        self.messages: list[dict[str, str]] = []
        self.max_context_length = max_context_length
        self.session_start = datetime.now()
        self._total_length = 0  # Track total length for O(1) access

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Message role ("user", "assistant", or "system")
            content: Message content

        Raises:
            ValueError: If role is invalid or content exceeds maximum length
        """
        # Input validation
        if role not in self.VALID_ROLES:
            raise ValueError(
                f"Invalid role: {role}. Must be one of {self.VALID_ROLES}"
            )

        if len(content) > self.MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"Message content exceeds maximum length of {self.MAX_MESSAGE_LENGTH} characters"
            )

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        self.messages.append(message)
        self._total_length += len(content)
        logger.debug(f"Message added: role={role}, content_len={len(content)}, total_len={self._total_length}")
        self._trim_context()

    def _trim_context(self) -> None:
        """Trim conversation to stay within context length."""
        if self._total_length <= self.max_context_length:
            return

        # Find the index of the first non-system message
        first_message_to_remove = 0
        if self.messages and self.messages[0].get("role") == Role.SYSTEM.value:
            first_message_to_remove = 1

        # Remove messages until the context is within the limit
        while self._total_length > self.max_context_length and len(self.messages) > first_message_to_remove:
            removed = self.messages.pop(first_message_to_remove)
            self._total_length -= len(removed["content"])
            logger.debug(f"Trimmed message: role={removed["role"]}, content_len={len(removed["content"])}, new_total_len={self._total_length}")

    def get_context_prompt(self) -> str:
        """
        Generate a context-aware prompt for gemma.

        Returns:
            Formatted conversation context string
        """
        if not self.messages:
            return ""

        # Build conversation context
        context_parts: list[str] = []
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]

            if role == Role.USER.value:
                context_parts.append(f"User: {content}")
            elif role == Role.ASSISTANT.value:
                context_parts.append(f"Assistant: {content}")
            elif role == Role.SYSTEM.value:
                context_parts.append(f"System: {content}")

        return "\n".join(context_parts)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        self._total_length = 0
        self.session_start = datetime.now()
        logger.info("Conversation history cleared.")

    async def save_to_file(self, filepath: Path) -> bool:
        """
        Save conversation to JSON file asynchronously.

        Args:
            filepath: Path to save the conversation

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            data = {
                "session_start": self.session_start.isoformat(),
                "messages": self.messages,
                "saved_at": datetime.now().isoformat(),
            }

            async with aiofiles.open(filepath, mode="w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))

            return True
        except (OSError, json.JSONEncodeError, ValueError) as e:
            logger.error(f"Error saving conversation: {e}")
            return False

    async def load_from_file(self, filepath: Path) -> bool:
        """
        Load conversation from JSON file asynchronously.

        Args:
            filepath: Path to load the conversation from

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            async with aiofiles.open(filepath, mode="r", encoding="utf-8") as f:
                content = await f.read()
                data = json.loads(content)

            self.messages = data.get("messages", [])
            session_start_str = data.get("session_start")
            if session_start_str:
                self.session_start = datetime.fromisoformat(session_start_str)

            return True
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error loading conversation: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """
        Get conversation statistics.

        Returns:
            Dictionary containing conversation statistics
        """
        return {
            "message_count": len(self.messages),
            "session_duration": datetime.now() - self.session_start,
            "session_start": self.session_start,
            "total_characters": self._total_length,
            "context_utilization": self._total_length / self.max_context_length if self.max_context_length > 0 else 0,
        }
