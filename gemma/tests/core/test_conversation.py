
import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from gemma_cli.core.conversation import ConversationManager
from gemma_cli.core.enums import Role


@pytest.fixture
def conversation_manager():
    return ConversationManager(max_context_length=100)


def test_add_message(conversation_manager):
    conversation_manager.add_message(Role.USER.value, "Hello")
    assert len(conversation_manager.messages) == 1
    assert conversation_manager.messages[0]["role"] == Role.USER.value
    assert conversation_manager.messages[0]["content"] == "Hello"
    assert conversation_manager._total_length == 5


def test_add_message_invalid_role(conversation_manager):
    with pytest.raises(ValueError):
        conversation_manager.add_message("invalid_role", "Hello")


def test_add_message_too_long(conversation_manager):
    with pytest.raises(ValueError):
        conversation_manager.add_message(Role.USER.value, "a" * 100_001)

def test_trim_context(conversation_manager):
    conversation_manager.add_message(Role.SYSTEM.value, "System message")
    conversation_manager.add_message(Role.USER.value, "a" * 50)
    conversation_manager.add_message(Role.ASSISTANT.value, "b" * 50)
    assert len(conversation_manager.messages) == 2
    assert conversation_manager.messages[0]["role"] == Role.SYSTEM.value

def test_get_context_prompt(conversation_manager):
    conversation_manager.add_message(Role.USER.value, "Hello")
    conversation_manager.add_message(Role.ASSISTANT.value, "Hi there")
    prompt = conversation_manager.get_context_prompt()
    assert prompt == "User: Hello\nAssistant: Hi there"

def test_clear(conversation_manager):
    conversation_manager.add_message(Role.USER.value, "Hello")
    conversation_manager.clear()
    assert len(conversation_manager.messages) == 0
    assert conversation_manager._total_length == 0


@pytest.mark.asyncio
async def test_save_and_load_from_file(conversation_manager, tmp_path):
    conversation_manager.add_message(Role.USER.value, "Hello")
    filepath = tmp_path / "conversation.json"

    await conversation_manager.save_to_file(filepath)
    assert filepath.exists()

    new_conversation_manager = ConversationManager()
    await new_conversation_manager.load_from_file(filepath)

    assert len(new_conversation_manager.messages) == 1
    assert new_conversation_manager.messages[0]["role"] == Role.USER.value
    assert new_conversation_manager.messages[0]["content"] == "Hello"

def test_get_stats(conversation_manager):
    conversation_manager.add_message(Role.USER.value, "Hello")
    stats = conversation_manager.get_stats()
    assert stats["message_count"] == 1
    assert stats["total_characters"] == 5
