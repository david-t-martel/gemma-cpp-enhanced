"""Configuration templates for different use cases."""

from typing import Any

# Configuration templates for common use cases
TEMPLATES: dict[str, dict[str, Any]] = {
    "minimal": {
        "name": "Minimal Setup",
        "description": "Basic configuration for quick start with CPU-only inference",
        "config": {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "pool_size": 5,
                "connection_timeout": 5,
                "enable_fallback": True,
            },
            "memory": {
                "working_ttl": 900,
                "short_term_ttl": 3600,
                "working_capacity": 10,
                "short_term_capacity": 50,
                "enable_background_tasks": False,
                "auto_consolidate": False,
            },
            "embedding": {
                "provider": "local",
                "model": "all-MiniLM-L6-v2",
                "batch_size": 16,
                "cache_embeddings": True,
            },
            "document": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "chunking_method": "token",
            },
            "mcp": {
                "enabled": False,
            },
            "ui": {
                "theme": "default",
                "show_memory_stats": False,
                "show_performance": False,
                "show_status_bar": True,
                "color_scheme": "auto",
            },
            "conversation": {
                "max_context_length": 4096,
                "max_history_messages": 20,
                "auto_save": False,
            },
            "system": {
                "enable_rag_context": False,
                "max_rag_context_tokens": 1000,
            },
            "monitoring": {
                "enabled": False,
            },
        },
    },
    "developer": {
        "name": "Developer Setup",
        "description": "Full features with MCP, RAG, and monitoring enabled",
        "config": {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "pool_size": 10,
                "connection_timeout": 5,
                "command_timeout": 10,
                "max_retries": 3,
                "retry_delay": 0.1,
                "enable_fallback": True,
            },
            "memory": {
                "working_ttl": 900,
                "short_term_ttl": 3600,
                "long_term_ttl": 2592000,
                "episodic_ttl": 604800,
                "semantic_ttl": 0,
                "working_capacity": 15,
                "short_term_capacity": 100,
                "long_term_capacity": 10000,
                "episodic_capacity": 5000,
                "semantic_capacity": 50000,
                "consolidation_threshold": 0.75,
                "importance_decay_rate": 0.1,
                "cleanup_interval": 300,
                "enable_background_tasks": True,
                "auto_consolidate": True,
            },
            "embedding": {
                "provider": "local",
                "model": "all-MiniLM-L6-v2",
                "dimension": 384,
                "batch_size": 32,
                "cache_embeddings": True,
            },
            "vector_store": {
                "dimension": 384,
                "distance_metric": "cosine",
                "index_type": "hnsw",
                "hnsw_m": 16,
                "hnsw_ef_construction": 200,
                "hnsw_ef_search": 50,
            },
            "document": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "min_chunk_size": 100,
                "max_chunk_size": 1000,
                "chunking_method": "token",
                "supported_formats": ["txt", "md", "html", "json", "pdf"],
                "max_file_size": 52428800,
            },
            "mcp": {
                "enabled": True,
                "servers_config": "config/mcp_servers.toml",
                "tool_cache_ttl": 3600,
                "connection_timeout": 10,
                "retry_count": 3,
            },
            "ui": {
                "theme": "default",
                "show_memory_stats": True,
                "show_performance": True,
                "show_status_bar": True,
                "progress_style": "rich",
                "color_scheme": "auto",
            },
            "conversation": {
                "max_context_length": 8192,
                "max_history_messages": 50,
                "save_directory": "~/.gemma_conversations",
                "auto_save": True,
                "auto_save_interval": 300,
            },
            "system": {
                "prompt_file": "config/prompts/GEMMA.md",
                "enable_rag_context": True,
                "max_rag_context_tokens": 2000,
            },
            "logging": {
                "level": "INFO",
                "file": "~/.gemma_cli/gemma.log",
                "max_size": 10485760,
                "backup_count": 5,
            },
            "monitoring": {
                "enabled": True,
                "track_latency": True,
                "track_memory": True,
                "track_token_usage": True,
                "report_interval": 60,
            },
        },
    },
    "performance": {
        "name": "Performance Optimized",
        "description": "Optimized for speed and throughput with minimal overhead",
        "config": {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "pool_size": 20,
                "connection_timeout": 3,
                "command_timeout": 5,
                "max_retries": 2,
                "retry_delay": 0.05,
                "enable_fallback": False,
            },
            "memory": {
                "working_ttl": 600,
                "short_term_ttl": 1800,
                "long_term_ttl": 604800,
                "working_capacity": 20,
                "short_term_capacity": 200,
                "long_term_capacity": 5000,
                "consolidation_threshold": 0.85,
                "cleanup_interval": 600,
                "enable_background_tasks": True,
                "auto_consolidate": True,
            },
            "embedding": {
                "provider": "local",
                "model": "all-MiniLM-L6-v2",
                "dimension": 384,
                "batch_size": 64,
                "cache_embeddings": True,
            },
            "vector_store": {
                "dimension": 384,
                "distance_metric": "cosine",
                "index_type": "hnsw",
                "hnsw_m": 32,
                "hnsw_ef_construction": 400,
                "hnsw_ef_search": 100,
            },
            "document": {
                "chunk_size": 256,
                "chunk_overlap": 25,
                "min_chunk_size": 50,
                "max_chunk_size": 500,
                "chunking_method": "token",
            },
            "mcp": {
                "enabled": True,
                "tool_cache_ttl": 7200,
                "connection_timeout": 5,
                "retry_count": 2,
            },
            "ui": {
                "theme": "default",
                "show_memory_stats": False,
                "show_performance": True,
                "show_status_bar": False,
                "progress_style": "simple",
                "color_scheme": "auto",
            },
            "conversation": {
                "max_context_length": 4096,
                "max_history_messages": 30,
                "auto_save": False,
            },
            "system": {
                "enable_rag_context": True,
                "max_rag_context_tokens": 1000,
            },
            "logging": {
                "level": "WARNING",
            },
            "monitoring": {
                "enabled": True,
                "track_latency": True,
                "track_memory": False,
                "track_token_usage": True,
                "report_interval": 300,
            },
        },
    },
}


def get_template(name: str) -> dict[str, Any]:
    """
    Get configuration template by name.

    Args:
        name: Template name (minimal, developer, performance)

    Returns:
        Template configuration dictionary

    Raises:
        KeyError: If template name doesn't exist
    """
    if name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise KeyError(
            f"Template '{name}' not found. Available templates: {available}"
        )

    return TEMPLATES[name].copy()


def customize_template(
    template: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    """
    Customize template with user overrides.

    Args:
        template: Base template configuration
        overrides: Dictionary with override values

    Returns:
        Customized configuration dictionary
    """
    import copy

    config = copy.deepcopy(template["config"])

    # Deep merge overrides into config
    def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge override dict into base dict."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    return deep_merge(config, overrides)


def list_templates() -> list[tuple[str, str, str]]:
    """
    Get list of available templates with descriptions.

    Returns:
        List of tuples (key, name, description)
    """
    return [
        (key, template["name"], template["description"])
        for key, template in TEMPLATES.items()
    ]
