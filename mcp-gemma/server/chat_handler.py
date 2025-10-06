"""
Chat and conversation handler for MCP Gemma server.
Adds conversation state management and chat capabilities.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union


class ConversationHandler:
    """Handler for chat conversations with state management."""

    def __init__(self, server):
        self.server = server
        self.logger = logging.getLogger(self.__class__.__name__)
        self.active_conversations: Dict[str, Dict[str, Any]] = {}

    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Handle chat interaction with conversation state."""
        # Generate conversation ID if not provided
        if conversation_id is None:
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"

        # Default system prompt
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant."

        # Get or create conversation
        if conversation_id not in self.active_conversations:
            self.active_conversations[conversation_id] = {
                "messages": [{"role": "system", "content": system_prompt}],
                "created_at": time.time(),
                "last_activity": time.time(),
            }

        conversation = self.active_conversations[conversation_id]
        conversation["last_activity"] = time.time()
        conversation["messages"].append({"role": "user", "content": message})

        # Format conversation for Gemma
        prompt_parts = []
        for msg in conversation["messages"]:
            if msg["role"] == "system":
                prompt_parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")

        prompt_parts.append("Assistant:")
        full_prompt = "\n\n".join(prompt_parts)

        # Generate response using the existing generation handler
        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.8)

        # Use the server's generation handler if available, otherwise fall back to gemma_interface
        if hasattr(self.server, "generation_handler"):
            response_text = await self.server.generation_handler.generate(
                full_prompt, max_tokens=max_tokens, temperature=temperature
            )
        else:
            # Direct interface fallback
            original_max_tokens = self.server.gemma_interface.max_tokens
            original_temperature = self.server.gemma_interface.temperature

            try:
                self.server.gemma_interface.max_tokens = max_tokens
                self.server.gemma_interface.temperature = temperature
                response_text = await self.server.gemma_interface.generate_response(full_prompt)
            finally:
                # Restore original parameters
                self.server.gemma_interface.max_tokens = original_max_tokens
                self.server.gemma_interface.temperature = original_temperature

        # Clean up response (remove "Assistant:" prefix if present)
        response_text = response_text.strip()
        if response_text.startswith("Assistant:"):
            response_text = response_text[10:].strip()

        # Add to conversation
        conversation["messages"].append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "conversation_id": conversation_id,
            "message_count": len(conversation["messages"]),
            "created_at": conversation["created_at"],
            "last_activity": conversation["last_activity"],
        }

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation by ID."""
        return self.active_conversations.get(conversation_id)

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all active conversations."""
        conversations = []
        for conv_id, conv_data in self.active_conversations.items():
            conversations.append(
                {
                    "conversation_id": conv_id,
                    "message_count": len(conv_data["messages"]),
                    "created_at": conv_data["created_at"],
                    "last_activity": conv_data["last_activity"],
                    "last_message": (
                        conv_data["messages"][-1]["content"][:100] + "..."
                        if len(conv_data["messages"]) > 1
                        else "No messages"
                    ),
                }
            )
        return sorted(conversations, key=lambda x: x["last_activity"], reverse=True)

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a specific conversation."""
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
            return True
        return False

    def clear_old_conversations(self, max_age_hours: int = 24) -> int:
        """Clear conversations older than specified hours."""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)

        old_conversations = [
            conv_id
            for conv_id, conv_data in self.active_conversations.items()
            if conv_data["last_activity"] < cutoff_time
        ]

        for conv_id in old_conversations:
            del self.active_conversations[conv_id]

        return len(old_conversations)

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about active conversations."""
        total_conversations = len(self.active_conversations)
        total_messages = sum(len(conv["messages"]) for conv in self.active_conversations.values())

        if total_conversations == 0:
            return {
                "total_conversations": 0,
                "total_messages": 0,
                "average_messages_per_conversation": 0,
                "oldest_conversation": None,
                "newest_conversation": None,
            }

        oldest_time = min(conv["created_at"] for conv in self.active_conversations.values())
        newest_time = max(conv["created_at"] for conv in self.active_conversations.values())

        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "average_messages_per_conversation": total_messages / total_conversations,
            "oldest_conversation": oldest_time,
            "newest_conversation": newest_time,
        }


class LegacyCompatibilityHandler:
    """Handler to maintain compatibility with simple MCP server functionality."""

    def __init__(self, server, conversation_handler):
        self.server = server
        self.conversation_handler = conversation_handler
        self.logger = logging.getLogger(self.__class__.__name__)

    async def handle_gemma_generate(self, prompt: str, **kwargs) -> str:
        """Handle legacy gemma_generate calls."""
        if hasattr(self.server, "generation_handler"):
            return await self.server.generation_handler.generate(prompt, **kwargs)
        else:
            # Direct interface fallback
            max_tokens = kwargs.get("max_tokens", self.server.config.max_tokens)
            temperature = kwargs.get("temperature", self.server.config.temperature)

            original_max_tokens = self.server.gemma_interface.max_tokens
            original_temperature = self.server.gemma_interface.temperature

            try:
                self.server.gemma_interface.max_tokens = max_tokens
                self.server.gemma_interface.temperature = temperature
                return await self.server.gemma_interface.generate_response(prompt)
            finally:
                # Restore original parameters
                self.server.gemma_interface.max_tokens = original_max_tokens
                self.server.gemma_interface.temperature = original_temperature

    async def handle_gemma_chat(self, message: str, **kwargs) -> Dict[str, Any]:
        """Handle legacy gemma_chat calls."""
        return await self.conversation_handler.chat(message, **kwargs)

    async def handle_models_list(self) -> List[Dict[str, Any]]:
        """Handle legacy gemma_models_list calls."""
        models = []

        # Try to get models directory from config
        models_dir = getattr(self.server.config, "models_dir", None)
        if models_dir is None:
            # Fallback to standard location
            from pathlib import Path

            models_dir = Path(__file__).parent.parent.parent.parent / ".models"

        if models_dir and models_dir.exists():
            for model_file in models_dir.glob("*.sbs"):
                model_name = model_file.stem
                model_size = model_file.stat().st_size
                models.append(
                    {
                        "name": model_name,
                        "file": str(model_file),
                        "size_mb": round(model_size / (1024 * 1024), 2),
                    }
                )

        return models

    async def handle_model_info(self, model: str) -> Dict[str, Any]:
        """Handle legacy gemma_model_info calls."""
        # Try to get models directory from config
        models_dir = getattr(self.server.config, "models_dir", None)
        if models_dir is None:
            # Fallback to standard location
            from pathlib import Path

            models_dir = Path(__file__).parent.parent.parent.parent / ".models"

        if not models_dir or not models_dir.exists():
            return {"error": f"Models directory not found: {models_dir}"}

        model_file = models_dir / f"{model}.sbs"
        if not model_file.exists():
            return {"error": f"Model {model} not found in {models_dir}"}

        stat = model_file.stat()
        return {
            "model": model,
            "file_path": str(model_file),
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": stat.st_mtime,
            "format": "SFP (Single File Pack)",
        }
