#!/usr/bin/env python3
"""
Multi-Agent Coordination Client for RAG-Redis System
Provides Python interface for coordinating multiple AI agents through Redis pub/sub
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from datetime import datetime

import redis.asyncio as aioredis


class AgentState(Enum):
    """Agent states for coordination tracking"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class MessageType(Enum):
    """Message types for inter-agent communication"""
    HEARTBEAT = "heartbeat"
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    COORDINATION = "coordination"
    STATUS_UPDATE = "status_update"
    MEMORY_UPDATE = "memory_update"
    SHUTDOWN = "shutdown"


@dataclass
class AgentInfo:
    """Information about an agent in the system"""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    state: AgentState
    last_heartbeat: float
    metadata: Dict[str, Any]


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    message_id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    correlation_id: Optional[str] = None


class MultiAgentCoordinator:
    """Coordinates multiple AI agents through Redis pub/sub and shared memory"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        agent_id: Optional[str] = None,
        agent_type: str = "worker",
        capabilities: Optional[List[str]] = None
    ):
        self.redis_url = redis_url
        self.agent_id = agent_id or f"{agent_type}_{uuid.uuid4().hex[:8]}"
        self.agent_type = agent_type
        self.capabilities = capabilities or []

        # Redis connections
        self.redis_client: Optional[aioredis.Redis] = None
        self.pubsub: Optional[aioredis.client.PubSub] = None

        # Agent state
        self.state = AgentState.INACTIVE
        self.registered_agents: Dict[str, AgentInfo] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.is_running = False

        # Redis keys
        self.registry_key = "agents:registry"
        self.memory_key = "agents:shared_memory"
        self.heartbeat_key = "agents:heartbeats"
        self.channel_all = "agents:broadcast"
        self.channel_personal = f"agents:personal:{self.agent_id}"

        # Configuration
        self.heartbeat_interval = 30.0  # seconds
        self.agent_timeout = 90.0  # seconds

        # Logging
        self.logger = logging.getLogger(f"agent.{self.agent_id}")

        # Default message handlers
        self._setup_default_handlers()

    def _setup_default_handlers(self):
        """Set up default message handlers"""
        self.message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self.message_handlers[MessageType.STATUS_UPDATE] = self._handle_status_update
        self.message_handlers[MessageType.SHUTDOWN] = self._handle_shutdown

    async def start(self) -> None:
        """Start the agent coordination system"""
        try:
            self.logger.info(f"Starting agent coordinator: {self.agent_id}")

            # Connect to Redis
            self.redis_client = aioredis.from_url(self.redis_url, decode_responses=True)
            self.pubsub = self.redis_client.pubsub()

            # Test connection
            await self.redis_client.ping()
            self.logger.info("Connected to Redis successfully")

            # Register agent
            await self._register_agent()

            # Subscribe to channels
            await self.pubsub.subscribe(self.channel_all, self.channel_personal)

            # Start background tasks
            self.is_running = True
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._message_loop())
            asyncio.create_task(self._cleanup_loop())

            self.state = AgentState.ACTIVE
            self.logger.info(f"Agent {self.agent_id} is now active")

        except Exception as e:
            self.logger.error(f"Failed to start coordinator: {e}")
            self.state = AgentState.ERROR
            raise

    async def stop(self) -> None:
        """Stop the agent coordination system"""
        self.logger.info(f"Stopping agent coordinator: {self.agent_id}")
        self.state = AgentState.SHUTTING_DOWN
        self.is_running = False

        # Send shutdown message
        await self.broadcast_message(MessageType.SHUTDOWN, {"reason": "normal_shutdown"})

        # Unregister agent
        await self._unregister_agent()

        # Close connections
        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()

        if self.redis_client:
            await self.redis_client.close()

        self.state = AgentState.INACTIVE
        self.logger.info(f"Agent {self.agent_id} stopped")

    async def _register_agent(self) -> None:
        """Register this agent in the system"""
        agent_info = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "state": self.state.value,
            "last_heartbeat": time.time(),
            "metadata": {
                "started_at": datetime.utcnow().isoformat(),
                "version": "1.0.0"
            }
        }

        await self.redis_client.hset(
            self.registry_key,
            self.agent_id,
            json.dumps(agent_info)
        )

        self.logger.info(f"Registered agent {self.agent_id} in system")

    async def _unregister_agent(self) -> None:
        """Unregister this agent from the system"""
        await self.redis_client.hdel(self.registry_key, self.agent_id)
        await self.redis_client.hdel(self.heartbeat_key, self.agent_id)
        self.logger.info(f"Unregistered agent {self.agent_id} from system")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats"""
        while self.is_running:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)

    async def _send_heartbeat(self) -> None:
        """Send heartbeat to keep agent alive"""
        timestamp = time.time()
        await self.redis_client.hset(self.heartbeat_key, self.agent_id, timestamp)

        # Update state in registry
        agent_data = await self.redis_client.hget(self.registry_key, self.agent_id)
        if agent_data:
            agent_info = json.loads(agent_data)
            agent_info["last_heartbeat"] = timestamp
            agent_info["state"] = self.state.value
            await self.redis_client.hset(
                self.registry_key,
                self.agent_id,
                json.dumps(agent_info)
            )

    async def _message_loop(self) -> None:
        """Process incoming messages"""
        while self.is_running:
            try:
                message = await self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    await self._process_message(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in message loop: {e}")

    async def _process_message(self, raw_message: Dict) -> None:
        """Process an incoming message"""
        try:
            data = json.loads(raw_message['data'])
            message = AgentMessage(**data)

            # Skip our own messages
            if message.sender_id == self.agent_id:
                return

            # Handle message based on type
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                self.logger.warning(f"No handler for message type: {message.message_type}")

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    async def _cleanup_loop(self) -> None:
        """Clean up dead agents periodically"""
        while self.is_running:
            try:
                await self._cleanup_dead_agents()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_dead_agents(self) -> None:
        """Remove agents that haven't sent heartbeats"""
        current_time = time.time()
        heartbeats = await self.redis_client.hgetall(self.heartbeat_key)

        for agent_id, last_heartbeat in heartbeats.items():
            if current_time - float(last_heartbeat) > self.agent_timeout:
                await self.redis_client.hdel(self.registry_key, agent_id)
                await self.redis_client.hdel(self.heartbeat_key, agent_id)
                self.logger.info(f"Cleaned up dead agent: {agent_id}")

    async def broadcast_message(
        self,
        message_type: MessageType,
        content: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> None:
        """Broadcast a message to all agents"""
        message = AgentMessage(
            message_id=uuid.uuid4().hex,
            sender_id=self.agent_id,
            receiver_id=None,
            message_type=message_type,
            content=content,
            timestamp=time.time(),
            correlation_id=correlation_id
        )

        await self.redis_client.publish(
            self.channel_all,
            json.dumps(message.__dict__)
        )

    async def send_message(
        self,
        target_agent_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> None:
        """Send a message to a specific agent"""
        message = AgentMessage(
            message_id=uuid.uuid4().hex,
            sender_id=self.agent_id,
            receiver_id=target_agent_id,
            message_type=message_type,
            content=content,
            timestamp=time.time(),
            correlation_id=correlation_id
        )

        await self.redis_client.publish(
            f"agents:personal:{target_agent_id}",
            json.dumps(message.__dict__)
        )

    async def get_active_agents(self) -> List[AgentInfo]:
        """Get list of all active agents"""
        agents_data = await self.redis_client.hgetall(self.registry_key)
        agents = []

        for agent_id, data in agents_data.items():
            try:
                info = json.loads(data)
                agents.append(AgentInfo(**info))
            except Exception as e:
                self.logger.error(f"Error parsing agent data for {agent_id}: {e}")

        return agents

    async def update_shared_memory(self, key: str, value: Any) -> None:
        """Update shared memory accessible by all agents"""
        await self.redis_client.hset(self.memory_key, key, json.dumps(value))

    async def get_shared_memory(self, key: str) -> Optional[Any]:
        """Get value from shared memory"""
        value = await self.redis_client.hget(self.memory_key, key)
        return json.loads(value) if value else None

    async def get_all_shared_memory(self) -> Dict[str, Any]:
        """Get all shared memory"""
        data = await self.redis_client.hgetall(self.memory_key)
        return {k: json.loads(v) for k, v in data.items()}

    def register_message_handler(
        self,
        message_type: MessageType,
        handler: Callable[[AgentMessage], None]
    ) -> None:
        """Register a custom message handler"""
        self.message_handlers[message_type] = handler

    async def _handle_heartbeat(self, message: AgentMessage) -> None:
        """Handle heartbeat messages"""
        self.logger.debug(f"Received heartbeat from {message.sender_id}")

    async def _handle_status_update(self, message: AgentMessage) -> None:
        """Handle status update messages"""
        self.logger.info(f"Status update from {message.sender_id}: {message.content}")

    async def _handle_shutdown(self, message: AgentMessage) -> None:
        """Handle shutdown messages"""
        self.logger.info(f"Shutdown signal from {message.sender_id}")
        if self.agent_type == "coordinator":
            # Coordinator handles system shutdown
            await self.stop()


class AgentCoordinatorManager:
    """Manages multiple agent coordinators for testing"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.coordinators: Dict[str, MultiAgentCoordinator] = {}

    async def create_agent(
        self,
        agent_type: str,
        capabilities: Optional[List[str]] = None
    ) -> MultiAgentCoordinator:
        """Create and start a new agent"""
        coordinator = MultiAgentCoordinator(
            redis_url=self.redis_url,
            agent_type=agent_type,
            capabilities=capabilities or []
        )

        await coordinator.start()
        self.coordinators[coordinator.agent_id] = coordinator
        return coordinator

    async def stop_all_agents(self) -> None:
        """Stop all managed agents"""
        for coordinator in self.coordinators.values():
            await coordinator.stop()
        self.coordinators.clear()

    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        if not self.coordinators:
            return {"status": "no_agents", "agents": []}

        # Use any coordinator to get system view
        coordinator = next(iter(self.coordinators.values()))
        agents = await coordinator.get_active_agents()

        return {
            "status": "active",
            "total_agents": len(agents),
            "agents": [
                {
                    "id": agent.agent_id,
                    "type": agent.agent_type,
                    "state": agent.state.value,
                    "capabilities": agent.capabilities
                }
                for agent in agents
            ],
            "shared_memory_keys": list((await coordinator.get_all_shared_memory()).keys())
        }


# Example usage and testing
async def main():
    """Example usage of the multi-agent coordination system"""
    manager = AgentCoordinatorManager()

    try:
        # Create coordinator agent
        coordinator = await manager.create_agent("coordinator", ["orchestration", "monitoring"])

        # Create worker agents
        worker1 = await manager.create_agent("worker", ["data_processing", "analysis"])
        worker2 = await manager.create_agent("worker", ["text_generation", "summarization"])

        # Create memory agent
        memory_agent = await manager.create_agent("memory", ["rag_operations", "knowledge_management"])

        # Test coordination
        await coordinator.broadcast_message(
            MessageType.COORDINATION,
            {"task": "system_initialization", "priority": "high"}
        )

        # Update shared memory
        await coordinator.update_shared_memory("system_config", {
            "max_workers": 10,
            "memory_limit": "2GB",
            "features": ["rag", "coordination", "monitoring"]
        })

        # Show system status
        status = await manager.get_system_status()
        print(f"System Status: {json.dumps(status, indent=2)}")

        # Wait a bit to see messages
        await asyncio.sleep(5)

    finally:
        await manager.stop_all_agents()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())