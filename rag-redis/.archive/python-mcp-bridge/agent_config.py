#!/usr/bin/env python3
"""
Configuration for Multi-Agent Coordination System
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import os


@dataclass
class RedisConfig:
    """Redis connection configuration"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 20
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


@dataclass
class AgentPoolConfig:
    """Configuration for agent pools"""
    coordinator_count: int = 1
    worker_count: int = 3
    memory_agent_count: int = 1
    max_agents: int = 10
    heartbeat_interval: float = 30.0
    agent_timeout: float = 90.0
    cleanup_interval: float = 60.0


@dataclass
class MemoryConfig:
    """Configuration for shared memory management"""
    max_memory_size: int = 1024 * 1024 * 100  # 100MB
    memory_cleanup_interval: float = 300.0  # 5 minutes
    memory_retention_days: int = 7
    enable_persistence: bool = True
    enable_compression: bool = True


@dataclass
class MessagingConfig:
    """Configuration for inter-agent messaging"""
    max_message_size: int = 1024 * 1024  # 1MB
    message_ttl: int = 3600  # 1 hour
    enable_message_encryption: bool = False
    batch_size: int = 100
    queue_max_size: int = 1000


@dataclass
class MultiAgentSystemConfig:
    """Main configuration for the multi-agent system"""
    redis: RedisConfig
    agent_pool: AgentPoolConfig
    memory: MemoryConfig
    messaging: MessagingConfig

    # System settings
    system_name: str = "RAG-Redis-MultiAgent"
    version: str = "1.0.0"
    debug_mode: bool = False
    log_level: str = "INFO"

    # Performance settings
    enable_metrics: bool = True
    metrics_interval: float = 60.0
    enable_profiling: bool = False

    # Security settings
    enable_auth: bool = False
    auth_token: Optional[str] = None

    # Feature flags
    enable_memory_consolidation: bool = True
    enable_auto_scaling: bool = False
    enable_distributed_mode: bool = False

    @classmethod
    def default(cls) -> "MultiAgentSystemConfig":
        """Create default configuration"""
        return cls(
            redis=RedisConfig(),
            agent_pool=AgentPoolConfig(),
            memory=MemoryConfig(),
            messaging=MessagingConfig()
        )

    @classmethod
    def from_env(cls) -> "MultiAgentSystemConfig":
        """Create configuration from environment variables"""
        config = cls.default()

        # Redis configuration from environment
        config.redis.host = os.getenv("REDIS_HOST", config.redis.host)
        config.redis.port = int(os.getenv("REDIS_PORT", config.redis.port))
        config.redis.password = os.getenv("REDIS_PASSWORD", config.redis.password)
        config.redis.db = int(os.getenv("REDIS_DB", config.redis.db))

        # Agent pool configuration
        config.agent_pool.coordinator_count = int(os.getenv("COORDINATOR_COUNT", config.agent_pool.coordinator_count))
        config.agent_pool.worker_count = int(os.getenv("WORKER_COUNT", config.agent_pool.worker_count))
        config.agent_pool.memory_agent_count = int(os.getenv("MEMORY_AGENT_COUNT", config.agent_pool.memory_agent_count))

        # System settings
        config.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        config.log_level = os.getenv("LOG_LEVEL", config.log_level)
        config.enable_auth = os.getenv("ENABLE_AUTH", "false").lower() == "true"
        config.auth_token = os.getenv("AUTH_TOKEN", config.auth_token)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db,
                "max_connections": self.redis.max_connections,
                "socket_timeout": self.redis.socket_timeout,
                "socket_connect_timeout": self.redis.socket_connect_timeout
            },
            "agent_pool": {
                "coordinator_count": self.agent_pool.coordinator_count,
                "worker_count": self.agent_pool.worker_count,
                "memory_agent_count": self.agent_pool.memory_agent_count,
                "max_agents": self.agent_pool.max_agents,
                "heartbeat_interval": self.agent_pool.heartbeat_interval,
                "agent_timeout": self.agent_pool.agent_timeout,
                "cleanup_interval": self.agent_pool.cleanup_interval
            },
            "memory": {
                "max_memory_size": self.memory.max_memory_size,
                "memory_cleanup_interval": self.memory.memory_cleanup_interval,
                "memory_retention_days": self.memory.memory_retention_days,
                "enable_persistence": self.memory.enable_persistence,
                "enable_compression": self.memory.enable_compression
            },
            "messaging": {
                "max_message_size": self.messaging.max_message_size,
                "message_ttl": self.messaging.message_ttl,
                "enable_message_encryption": self.messaging.enable_message_encryption,
                "batch_size": self.messaging.batch_size,
                "queue_max_size": self.messaging.queue_max_size
            },
            "system": {
                "system_name": self.system_name,
                "version": self.version,
                "debug_mode": self.debug_mode,
                "log_level": self.log_level,
                "enable_metrics": self.enable_metrics,
                "metrics_interval": self.metrics_interval,
                "enable_profiling": self.enable_profiling,
                "enable_auth": self.enable_auth,
                "enable_memory_consolidation": self.enable_memory_consolidation,
                "enable_auto_scaling": self.enable_auto_scaling,
                "enable_distributed_mode": self.enable_distributed_mode
            }
        }


# Predefined agent capabilities
AGENT_CAPABILITIES = {
    "coordinator": [
        "orchestration",
        "monitoring",
        "task_distribution",
        "resource_management",
        "performance_tracking",
        "error_handling"
    ],
    "worker": [
        "data_processing",
        "text_generation",
        "analysis",
        "computation",
        "file_operations",
        "api_calls"
    ],
    "memory": [
        "rag_operations",
        "knowledge_management",
        "memory_consolidation",
        "search_operations",
        "indexing",
        "data_storage"
    ],
    "specialist": [
        "domain_expertise",
        "advanced_reasoning",
        "specialized_tasks",
        "quality_assurance",
        "validation",
        "optimization"
    ]
}


# Message templates for common operations
MESSAGE_TEMPLATES = {
    "task_assignment": {
        "task_id": "",
        "task_type": "",
        "priority": "normal",
        "deadline": None,
        "requirements": [],
        "context": {}
    },
    "status_report": {
        "agent_id": "",
        "status": "",
        "progress": 0.0,
        "current_task": None,
        "metrics": {},
        "errors": []
    },
    "memory_update": {
        "operation": "",  # store, retrieve, update, delete
        "key": "",
        "value": None,
        "metadata": {},
        "tags": []
    },
    "coordination_request": {
        "request_type": "",
        "participants": [],
        "objective": "",
        "constraints": {},
        "timeout": 300
    }
}


def create_production_config() -> MultiAgentSystemConfig:
    """Create production-ready configuration"""
    config = MultiAgentSystemConfig.default()

    # Production Redis settings
    config.redis.max_connections = 50
    config.redis.socket_timeout = 10.0

    # Production agent pool settings
    config.agent_pool.worker_count = 5
    config.agent_pool.max_agents = 20
    config.agent_pool.heartbeat_interval = 15.0
    config.agent_pool.agent_timeout = 45.0

    # Production memory settings
    config.memory.max_memory_size = 1024 * 1024 * 500  # 500MB
    config.memory.memory_cleanup_interval = 180.0  # 3 minutes
    config.memory.enable_compression = True

    # Production messaging settings
    config.messaging.max_message_size = 1024 * 512  # 512KB
    config.messaging.batch_size = 50
    config.messaging.queue_max_size = 5000

    # Production system settings
    config.log_level = "WARNING"
    config.enable_metrics = True
    config.enable_auth = True
    config.enable_memory_consolidation = True

    return config


def create_development_config() -> MultiAgentSystemConfig:
    """Create development configuration"""
    config = MultiAgentSystemConfig.default()

    # Development settings
    config.debug_mode = True
    config.log_level = "DEBUG"
    config.enable_profiling = True
    config.agent_pool.worker_count = 2
    config.agent_pool.max_agents = 5
    config.memory.max_memory_size = 1024 * 1024 * 50  # 50MB

    return config


def create_test_config() -> MultiAgentSystemConfig:
    """Create test configuration"""
    config = MultiAgentSystemConfig.default()

    # Test settings
    config.redis.db = 15  # Use separate DB for tests
    config.debug_mode = True
    config.log_level = "DEBUG"
    config.agent_pool.coordinator_count = 1
    config.agent_pool.worker_count = 2
    config.agent_pool.memory_agent_count = 1
    config.agent_pool.max_agents = 5
    config.agent_pool.heartbeat_interval = 5.0
    config.agent_pool.agent_timeout = 15.0
    config.memory.max_memory_size = 1024 * 1024 * 10  # 10MB
    config.messaging.message_ttl = 300  # 5 minutes

    return config


if __name__ == "__main__":
    # Example usage
    import json

    print("=== Default Configuration ===")
    default_config = MultiAgentSystemConfig.default()
    print(json.dumps(default_config.to_dict(), indent=2))

    print("\n=== Production Configuration ===")
    prod_config = create_production_config()
    print(json.dumps(prod_config.to_dict(), indent=2))

    print("\n=== Development Configuration ===")
    dev_config = create_development_config()
    print(json.dumps(dev_config.to_dict(), indent=2))