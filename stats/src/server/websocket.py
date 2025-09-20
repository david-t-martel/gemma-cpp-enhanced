"""WebSocket manager for real-time chat functionality.

This module provides WebSocket connection management, message broadcasting,
and real-time chat capabilities with proper error handling and connection pooling.
"""

import asyncio
import json
import time
import uuid
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi import status
from pydantic import ValidationError

from ..application.inference.service import InferenceService
from ..domain.models.chat import ChatSession
from ..domain.models.chat import MessageRole as DomainMessageRole
from ..shared.exceptions import InferenceException
from ..shared.exceptions import ValidationException
from ..shared.logging import get_logger
from .api.schemas import WebSocketChatMessage
from .api.schemas import WebSocketMessage
from .api.schemas import WebSocketResponse

logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""

    def __init__(self):
        # Active connections by connection ID
        self.active_connections: dict[str, WebSocket] = {}

        # Sessions by session ID
        self.active_sessions: dict[str, ChatSession] = {}

        # Connection to session mapping
        self.connection_sessions: dict[str, str] = {}

        # Session to connections mapping (multiple connections per session)
        self.session_connections: dict[str, set[str]] = defaultdict(set)

        # Connection metadata
        self.connection_metadata: dict[str, dict[str, any]] = {}

        # Message queue for offline delivery
        self.message_queue: dict[str, list[WebSocketMessage]] = defaultdict(list)

        # Statistics
        self.total_connections = 0
        self.total_messages = 0
        self.connection_start_times: dict[str, float] = {}

    async def connect(self, websocket: WebSocket, session_id: str | None = None) -> str:
        """Accept a WebSocket connection and return connection ID.

        Args:
            websocket: WebSocket connection
            session_id: Optional existing session ID to join

        Returns:
            Connection ID
        """
        await websocket.accept()

        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        self.connection_start_times[connection_id] = time.time()
        self.total_connections += 1

        # Handle session assignment
        if session_id and session_id in self.active_sessions:
            # Join existing session
            self.connection_sessions[connection_id] = session_id
            self.session_connections[session_id].add(connection_id)
        else:
            # Create new session
            new_session = ChatSession()
            session_id = new_session.id
            self.active_sessions[session_id] = new_session
            self.connection_sessions[connection_id] = session_id
            self.session_connections[session_id].add(connection_id)

        # Initialize metadata
        self.connection_metadata[connection_id] = {
            "session_id": session_id,
            "connected_at": time.time(),
            "last_activity": time.time(),
            "message_count": 0,
        }

        logger.info(f"WebSocket connected: {connection_id} -> session {session_id}")

        # Send welcome message
        welcome_msg = WebSocketResponse(
            type="connection",
            session_id=session_id,
            data={
                "connection_id": connection_id,
                "session_id": session_id,
                "status": "connected",
                "message": "Connected to Gemma Chatbot",
            },
        )
        await self.send_personal_message(welcome_msg.json(), websocket)

        return connection_id

    async def disconnect(self, connection_id: str):
        """Disconnect a WebSocket connection.

        Args:
            connection_id: Connection ID to disconnect
        """
        if connection_id not in self.active_connections:
            return

        # Get session info
        session_id = self.connection_sessions.get(connection_id)

        # Remove from active connections
        del self.active_connections[connection_id]
        del self.connection_start_times[connection_id]

        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]

        if session_id:
            # Remove from session connections
            if session_id in self.session_connections:
                self.session_connections[session_id].discard(connection_id)

                # Clean up empty sessions
                if not self.session_connections[session_id]:
                    del self.session_connections[session_id]
                    if session_id in self.active_sessions:
                        del self.active_sessions[session_id]
                    if session_id in self.message_queue:
                        del self.message_queue[session_id]

            del self.connection_sessions[connection_id]

        logger.info(f"WebSocket disconnected: {connection_id}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection.

        Args:
            message: Message to send
            websocket: Target WebSocket connection
        """
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    async def send_to_connection(self, message: str, connection_id: str):
        """Send a message to a specific connection by ID.

        Args:
            message: Message to send
            connection_id: Target connection ID
        """
        websocket = self.active_connections.get(connection_id)
        if websocket:
            await self.send_personal_message(message, websocket)
        else:
            logger.warning(f"Connection {connection_id} not found")

    async def broadcast_to_session(
        self, message: str, session_id: str, exclude_connection: str | None = None
    ):
        """Broadcast a message to all connections in a session.

        Args:
            message: Message to broadcast
            session_id: Target session ID
            exclude_connection: Connection ID to exclude from broadcast
        """
        if session_id not in self.session_connections:
            return

        connection_ids = self.session_connections[session_id].copy()
        if exclude_connection:
            connection_ids.discard(exclude_connection)

        disconnected = []
        for connection_id in connection_ids:
            websocket = self.active_connections.get(connection_id)
            if websocket:
                try:
                    await websocket.send_text(message)
                    # Update activity
                    if connection_id in self.connection_metadata:
                        self.connection_metadata[connection_id]["last_activity"] = time.time()
                except Exception as e:
                    logger.error(f"Error broadcasting to {connection_id}: {e}")
                    disconnected.append(connection_id)
            else:
                disconnected.append(connection_id)

        # Clean up disconnected connections
        for connection_id in disconnected:
            await self.disconnect(connection_id)

    async def broadcast_to_all(self, message: str):
        """Broadcast a message to all active connections.

        Args:
            message: Message to broadcast
        """
        disconnected = []
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                disconnected.append(connection_id)

        # Clean up disconnected connections
        for connection_id in disconnected:
            await self.disconnect(connection_id)

    def get_session(self, session_id: str) -> ChatSession | None:
        """Get a chat session by ID.

        Args:
            session_id: Session ID

        Returns:
            ChatSession or None if not found
        """
        return self.active_sessions.get(session_id)

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)

    def get_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self.active_sessions)

    def get_statistics(self) -> dict[str, any]:
        """Get connection manager statistics."""
        current_time = time.time()
        total_uptime = sum(
            current_time - start_time for start_time in self.connection_start_times.values()
        )
        avg_connection_time = (
            total_uptime / len(self.connection_start_times) if self.connection_start_times else 0
        )

        return {
            "active_connections": len(self.active_connections),
            "active_sessions": len(self.active_sessions),
            "total_connections": self.total_connections,
            "total_messages": self.total_messages,
            "average_connection_time": avg_connection_time,
            "connections_per_session": (
                len(self.active_connections) / len(self.active_sessions)
                if self.active_sessions
                else 0
            ),
        }


# Global connection manager
manager = ConnectionManager()


class WebSocketHandler:
    """Handles WebSocket message processing and chat operations."""

    def __init__(self, inference_service: InferenceService):
        self.inference_service = inference_service
        self.logger = get_logger(f"{__name__}.WebSocketHandler")

    async def handle_connection(self, websocket: WebSocket, session_id: str | None = None):
        """Handle a new WebSocket connection.

        Args:
            websocket: WebSocket connection
            session_id: Optional session ID to join
        """
        connection_id = None
        try:
            connection_id = await manager.connect(websocket, session_id)

            # Message handling loop
            while True:
                # Receive message
                message_text = await websocket.receive_text()
                await self.process_message(message_text, connection_id, websocket)

        except WebSocketDisconnect:
            self.logger.info(f"WebSocket disconnected normally: {connection_id}")
        except Exception as e:
            self.logger.error(f"WebSocket error for {connection_id}: {e}")
            # Send error message if connection is still active
            if connection_id and connection_id in manager.active_connections:
                error_response = WebSocketResponse(
                    type="error",
                    session_id=manager.connection_sessions.get(connection_id, ""),
                    data={"error": str(e)},
                )
                await manager.send_to_connection(error_response.json(), connection_id)
        finally:
            if connection_id:
                await manager.disconnect(connection_id)

    async def process_message(self, message_text: str, connection_id: str, websocket: WebSocket):
        """Process a received WebSocket message.

        Args:
            message_text: Raw message text
            connection_id: Connection ID
            websocket: WebSocket connection
        """
        try:
            # Parse message
            message_data = json.loads(message_text)
            message = WebSocketMessage(**message_data)

            # Update statistics
            manager.total_messages += 1
            if connection_id in manager.connection_metadata:
                manager.connection_metadata[connection_id]["message_count"] += 1
                manager.connection_metadata[connection_id]["last_activity"] = time.time()

            # Route message based on type
            if message.type == "chat":
                await self.handle_chat_message(message, connection_id)
            elif message.type == "ping":
                await self.handle_ping(connection_id)
            elif message.type == "join_session":
                await self.handle_join_session(message, connection_id)
            elif message.type == "leave_session":
                await self.handle_leave_session(connection_id)
            elif message.type == "get_history":
                await self.handle_get_history(message, connection_id)
            else:
                await self.send_error(connection_id, f"Unknown message type: {message.type}")

        except json.JSONDecodeError as e:
            self.logger.warning(f"Invalid JSON from {connection_id}: {e}")
            await self.send_error(connection_id, "Invalid JSON message")
        except ValidationError as e:
            self.logger.warning(f"Invalid message format from {connection_id}: {e}")
            await self.send_error(connection_id, f"Invalid message format: {e}")
        except Exception as e:
            self.logger.error(f"Error processing message from {connection_id}: {e}")
            await self.send_error(connection_id, "Error processing message")

    async def handle_chat_message(self, message: WebSocketMessage, connection_id: str):
        """Handle a chat message.

        Args:
            message: WebSocket message
            connection_id: Connection ID
        """
        session_id = manager.connection_sessions.get(connection_id)
        if not session_id:
            await self.send_error(connection_id, "No active session")
            return

        session = manager.get_session(session_id)
        if not session:
            await self.send_error(connection_id, "Session not found")
            return

        try:
            # Parse chat message
            chat_data = message.data
            user_message = chat_data.get("message", "")
            stream = chat_data.get("stream", True)

            if not user_message.strip():
                await self.send_error(connection_id, "Empty message")
                return

            # Broadcast user message to session
            user_msg_response = WebSocketResponse(
                type="user_message",
                session_id=session_id,
                data={
                    "message": user_message,
                    "user_id": chat_data.get("user_id"),
                },
            )
            await manager.broadcast_to_session(user_msg_response.json(), session_id)

            # Generate AI response
            if stream:
                await self.generate_streaming_response(session, user_message, session_id)
            else:
                await self.generate_response(session, user_message, session_id)

        except Exception as e:
            self.logger.error(f"Error handling chat message: {e}")
            await self.send_error(connection_id, f"Error generating response: {e}")

    async def generate_streaming_response(
        self, session: ChatSession, user_message: str, session_id: str
    ):
        """Generate a streaming AI response.

        Args:
            session: Chat session
            user_message: User's message
            session_id: Session ID
        """
        try:
            # Start streaming response
            start_response = WebSocketResponse(
                type="assistant_message_start",
                session_id=session_id,
                data={"streaming": True},
            )
            await manager.broadcast_to_session(start_response.json(), session_id)

            # Generate streaming response
            full_content = ""
            async for chunk in self.inference_service.generate_streaming_response(
                session, user_message
            ):
                full_content += chunk.content

                # Send chunk
                chunk_response = WebSocketResponse(
                    type="assistant_message_chunk",
                    session_id=session_id,
                    data={
                        "content": chunk.content,
                        "full_content": full_content,
                        "is_complete": chunk.is_complete,
                        "token_count": chunk.token_count,
                    },
                )
                await manager.broadcast_to_session(chunk_response.json(), session_id)

                if chunk.is_complete:
                    break

            # Send completion
            end_response = WebSocketResponse(
                type="assistant_message_complete",
                session_id=session_id,
                data={
                    "content": full_content,
                    "streaming": False,
                },
            )
            await manager.broadcast_to_session(end_response.json(), session_id)

        except Exception as e:
            self.logger.error(f"Error in streaming response: {e}")
            error_response = WebSocketResponse(
                type="error",
                session_id=session_id,
                data={"error": f"Streaming response failed: {e}"},
            )
            await manager.broadcast_to_session(error_response.json(), session_id)

    async def generate_response(self, session: ChatSession, user_message: str, session_id: str):
        """Generate a non-streaming AI response.

        Args:
            session: Chat session
            user_message: User's message
            session_id: Session ID
        """
        try:
            # Generate response
            response_message = await self.inference_service.generate_response(session, user_message)

            # Send response
            response = WebSocketResponse(
                type="assistant_message",
                session_id=session_id,
                data={
                    "content": response_message.content,
                    "streaming": False,
                    "token_usage": (
                        {
                            "prompt_tokens": (
                                response_message.token_usage.prompt_tokens
                                if response_message.token_usage
                                else 0
                            ),
                            "completion_tokens": (
                                response_message.token_usage.completion_tokens
                                if response_message.token_usage
                                else 0
                            ),
                            "total_tokens": (
                                response_message.token_usage.total_tokens
                                if response_message.token_usage
                                else 0
                            ),
                        }
                        if response_message.token_usage
                        else None
                    ),
                },
            )
            await manager.broadcast_to_session(response.json(), session_id)

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            error_response = WebSocketResponse(
                type="error",
                session_id=session_id,
                data={"error": f"Response generation failed: {e}"},
            )
            await manager.broadcast_to_session(error_response.json(), session_id)

    async def handle_ping(self, connection_id: str):
        """Handle a ping message.

        Args:
            connection_id: Connection ID
        """
        pong_response = WebSocketResponse(
            type="pong",
            session_id="",
            data={"timestamp": time.time()},
        )
        await manager.send_to_connection(pong_response.json(), connection_id)

    async def handle_join_session(self, message: WebSocketMessage, connection_id: str):
        """Handle joining a session.

        Args:
            message: WebSocket message
            connection_id: Connection ID
        """
        target_session_id = message.data.get("session_id")
        if not target_session_id:
            await self.send_error(connection_id, "Session ID required")
            return

        # Implementation would require more complex session switching logic
        await self.send_error(connection_id, "Session switching not implemented yet")

    async def handle_leave_session(self, connection_id: str):
        """Handle leaving a session.

        Args:
            connection_id: Connection ID
        """
        # Implementation would create a new session for the connection
        await self.send_error(connection_id, "Session leaving not implemented yet")

    async def handle_get_history(self, message: WebSocketMessage, connection_id: str):
        """Handle getting session history.

        Args:
            message: WebSocket message
            connection_id: Connection ID
        """
        session_id = manager.connection_sessions.get(connection_id)
        if not session_id:
            await self.send_error(connection_id, "No active session")
            return

        session = manager.get_session(session_id)
        if not session:
            await self.send_error(connection_id, "Session not found")
            return

        # Send history
        history_response = WebSocketResponse(
            type="history",
            session_id=session_id,
            data={
                "messages": [
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                        "timestamp": msg.created_at.isoformat(),
                    }
                    for msg in session.messages
                ],
                "message_count": len(session.messages),
            },
        )
        await manager.send_to_connection(history_response.json(), connection_id)

    async def send_error(self, connection_id: str, error_message: str):
        """Send an error message to a connection.

        Args:
            connection_id: Connection ID
            error_message: Error message
        """
        error_response = WebSocketResponse(
            type="error",
            session_id=manager.connection_sessions.get(connection_id, ""),
            data={"error": error_message},
        )
        await manager.send_to_connection(error_response.json(), connection_id)
