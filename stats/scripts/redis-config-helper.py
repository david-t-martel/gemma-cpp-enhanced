#!/usr/bin/env python3
"""Redis configuration helper script.

This script helps configure Redis settings for different environments,
validates connections, and manages configuration files.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from redis.exceptions import (
    ConnectionError as RedisConnectionError,
    TimeoutError as RedisTimeoutError,
)

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.shared.config.redis_config import RedisConfig, create_redis_config


def detect_environment() -> str:
    """Detect the current environment."""
    if os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER"):
        return "docker"
    elif os.name == "nt":
        return "windows"
    else:
        return "unix"


def test_redis_connection(config: RedisConfig) -> dict[str, Any]:
    """Test Redis connection and return status."""
    result = {
        "success": False,
        "config": {
            "host": config.host,
            "port": config.port,
            "db": config.db,
        },
        "info": {},
        "error": None,
    }

    try:
        client = config.create_client()

        # Test basic connectivity
        ping_result = client.ping()
        result["success"] = ping_result

        if ping_result:
            # Get Redis server info
            info = client.info()
            result["info"] = {
                "redis_version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
                "keyspace": {k: v for k, v in info.items() if k.startswith("db")},
            }

            # Test basic operations
            test_key = f"{config.key_prefix}connection_test"
            client.set(test_key, "test_value", ex=60)
            get_result = client.get(test_key)
            client.delete(test_key)

            if get_result != "test_value":
                result["error"] = "Failed basic read/write test"
                result["success"] = False

    except (RedisConnectionError, RedisTimeoutError) as e:
        result["error"] = f"Connection failed: {e}"
    except Exception as e:
        result["error"] = f"Unexpected error: {e}"

    return result


def generate_config_for_environment(env: str) -> RedisConfig:
    """Generate Redis configuration for specific environment."""
    if env == "docker":
        return RedisConfig.get_default_docker_config()
    elif env == "windows":
        return RedisConfig.get_default_windows_config()
    else:
        return RedisConfig()


def update_env_file(env_path: Path, config: RedisConfig):
    """Update environment file with Redis configuration."""
    env_content = f"""# Redis Configuration (auto-generated)
REDIS_HOST={config.host}
REDIS_PORT={config.port}
REDIS_DB={config.db}
REDIS_MAX_CONNECTIONS={config.pool.max_connections}
REDIS_CONNECTION_TIMEOUT={config.timeouts.connection_timeout}
REDIS_SOCKET_TIMEOUT={config.timeouts.socket_timeout}
REDIS_MAX_RETRIES={config.retry.max_retries}
REDIS_RETRY_DELAY_MS={int(config.retry.retry_delay * 1000)}
REDIS_ENABLE_FALLBACK={str(config.fallback.enabled).lower()}
REDIS_KEY_PREFIX={config.key_prefix}
"""

    if config.username:
        env_content += f"REDIS_USERNAME={config.username}\n"
    if config.password:
        env_content += f"REDIS_PASSWORD={config.password}\n"

    env_path.write_text(env_content)
    print(f"✅ Environment file updated: {env_path}")


def validate_config() -> None:
    """Validate current Redis configuration."""
    print("🔍 Validating Redis configuration...")

    # Detect environment
    env = detect_environment()
    print(f"📍 Detected environment: {env}")

    # Load configuration
    config = create_redis_config()
    print("⚙️  Configuration loaded:")
    print(f"   Host: {config.host}")
    print(f"   Port: {config.port}")
    print(f"   Database: {config.db}")
    print(f"   Fallback: {config.fallback.enabled}")

    # Test connection
    print("\n🔗 Testing Redis connection...")
    result = test_redis_connection(config)

    if result["success"]:
        print("✅ Redis connection successful!")
        print(f"   Redis version: {result['info'].get('redis_version', 'unknown')}")
        print(f"   Memory usage: {result['info'].get('used_memory_human', 'unknown')}")
        print(f"   Connected clients: {result['info'].get('connected_clients', 'unknown')}")

        if result["info"]["keyspace"]:
            print(f"   Databases: {', '.join(result['info']['keyspace'].keys())}")
    else:
        print("❌ Redis connection failed!")
        print(f"   Error: {result['error']}")

        if config.fallback.enabled:
            print("⚠️  Fallback to in-memory cache is enabled")
        else:
            print("🚨 No fallback configured - application may fail to start")

    return result["success"]


def setup_environment(env: str, output_file: str | None = None) -> None:
    """Setup Redis configuration for specific environment."""
    print(f"🛠️  Setting up Redis configuration for {env} environment...")

    config = generate_config_for_environment(env)

    # Determine output file
    env_path = Path(output_file) if output_file else PROJECT_ROOT / f".env.redis.{env}"

    # Test configuration
    print("🔗 Testing generated configuration...")
    result = test_redis_connection(config)

    if result["success"]:
        print("✅ Configuration test successful!")
    else:
        print(f"⚠️  Configuration test failed: {result['error']}")
        print("💡 You may need to start Redis or adjust configuration")

    # Update environment file
    update_env_file(env_path, config)

    # Show usage instructions
    print("\n📋 To use this configuration:")
    print(f"1. Copy {env_path} to .env")
    print(f"2. Or set: export $(cat {env_path} | xargs)")
    print("3. Or source it in your application")


def migrate_configuration() -> None:
    """Migrate old Redis configuration to new centralized system."""
    print("🔄 Migrating Redis configuration...")

    old_configs = []

    # Check common configuration files for old Redis settings
    config_files = [
        PROJECT_ROOT / "mcp.json",
        PROJECT_ROOT / "docker-compose.yml",
        PROJECT_ROOT / "docker-compose.production.yml",
        PROJECT_ROOT / ".env",
    ]

    for config_file in config_files:
        if config_file.exists():
            content = config_file.read_text()
            if "6379" in content and "redis" in content.lower():
                old_configs.append(config_file)

    if old_configs:
        print("⚠️  Found files with old Redis configuration:")
        for file in old_configs:
            print(f"   - {file}")
        print("\n💡 These files should be updated to use environment variables")
        print("   Run with --setup to generate new configuration")
    else:
        print("✅ No old configuration files found")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Redis configuration helper for LLM Stats project")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Validate command
    subparsers.add_parser("validate", help="Validate current Redis configuration")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup Redis configuration for environment")
    setup_parser.add_argument(
        "environment", choices=["windows", "docker", "unix"], help="Target environment"
    )
    setup_parser.add_argument("--output", "-o", help="Output environment file path")

    # Migrate command
    subparsers.add_parser("migrate", help="Migrate old Redis configuration")

    # Test command
    subparsers.add_parser("test", help="Test Redis connection with current settings")

    # Info command
    subparsers.add_parser("info", help="Show Redis configuration info")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "validate":
            success = validate_config()
            sys.exit(0 if success else 1)

        elif args.command == "setup":
            setup_environment(args.environment, args.output)

        elif args.command == "migrate":
            migrate_configuration()

        elif args.command == "test":
            config = create_redis_config()
            result = test_redis_connection(config)

            print(json.dumps(result, indent=2))
            sys.exit(0 if result["success"] else 1)

        elif args.command == "info":
            config = create_redis_config()
            info = {
                "environment": detect_environment(),
                "config": {
                    "host": config.host,
                    "port": config.port,
                    "db": config.db,
                    "ssl_enabled": config.ssl.enabled,
                    "fallback_enabled": config.fallback.enabled,
                    "key_prefix": config.key_prefix,
                    "redis_url": config.get_redis_url(),
                },
            }
            print(json.dumps(info, indent=2))

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
