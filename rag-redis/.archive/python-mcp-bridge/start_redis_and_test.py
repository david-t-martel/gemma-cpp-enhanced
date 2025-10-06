#!/usr/bin/env python3
"""
Startup script for Redis server and multi-agent system testing
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
import signal
import platform


class RedisManager:
    """Manages Redis server lifecycle"""

    def __init__(self, port: int = 6379):
        self.port = port
        self.redis_process: Optional[subprocess.Popen] = None
        self.logger = logging.getLogger("redis_manager")

    def is_redis_running(self) -> bool:
        """Check if Redis is already running"""
        try:
            if platform.system() == "Windows":
                # Use netstat on Windows
                result = subprocess.run(
                    ["netstat", "-an"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                return f":{self.port}" in result.stdout
            else:
                # Use ss or netstat on Linux/WSL
                try:
                    result = subprocess.run(
                        ["ss", "-tlnp"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                except FileNotFoundError:
                    result = subprocess.run(
                        ["netstat", "-tlnp"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                return f":{self.port}" in result.stdout
        except Exception as e:
            self.logger.warning(f"Could not check if Redis is running: {e}")
            return False

    def find_redis_executable(self) -> Optional[str]:
        """Find Redis server executable"""
        possible_paths = [
            "redis-server",
            "/usr/bin/redis-server",
            "/usr/local/bin/redis-server",
            "/opt/redis/bin/redis-server",
            "C:\\Redis\\redis-server.exe",
            "C:\\Program Files\\Redis\\redis-server.exe"
        ]

        for path in possible_paths:
            try:
                if os.path.exists(path):
                    return path
                elif platform.system() != "Windows":
                    # Try to find in PATH
                    result = subprocess.run(
                        ["which", path.split("/")[-1]],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        return result.stdout.strip()
            except Exception:
                continue

        return None

    def start_redis(self) -> bool:
        """Start Redis server"""
        if self.is_redis_running():
            self.logger.info(f"Redis is already running on port {self.port}")
            return True

        redis_executable = self.find_redis_executable()
        if not redis_executable:
            self.logger.error("Redis server executable not found!")
            self.logger.info("Please install Redis server:")
            if platform.system() == "Windows":
                self.logger.info("- Download from https://github.com/microsoftarchive/redis/releases")
                self.logger.info("- Or use WSL: sudo apt-get install redis-server")
            else:
                self.logger.info("- Ubuntu/Debian: sudo apt-get install redis-server")
                self.logger.info("- CentOS/RHEL: sudo yum install redis")
                self.logger.info("- macOS: brew install redis")
            return False

        try:
            # Create basic Redis config
            config_content = f"""
port {self.port}
bind 127.0.0.1
daemonize no
save ""
appendonly no
maxmemory 256mb
maxmemory-policy allkeys-lru
timeout 0
tcp-keepalive 300
"""
            config_path = Path("redis-test.conf")
            config_path.write_text(config_content)

            # Start Redis server
            self.logger.info(f"Starting Redis server on port {self.port}...")
            self.redis_process = subprocess.Popen(
                [redis_executable, str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=None if platform.system() == "Windows" else os.setsid
            )

            # Wait for Redis to start
            for i in range(10):
                if self.is_redis_running():
                    self.logger.info("Redis server started successfully")
                    return True
                time.sleep(1)

            self.logger.error("Redis server failed to start within 10 seconds")
            self.stop_redis()
            return False

        except Exception as e:
            self.logger.error(f"Failed to start Redis server: {e}")
            return False

    def stop_redis(self):
        """Stop Redis server"""
        if self.redis_process:
            self.logger.info("Stopping Redis server...")
            try:
                if platform.system() == "Windows":
                    self.redis_process.terminate()
                else:
                    os.killpg(os.getpgid(self.redis_process.pid), signal.SIGTERM)
                self.redis_process.wait(timeout=5)
            except Exception as e:
                self.logger.warning(f"Error stopping Redis: {e}")
                try:
                    if platform.system() == "Windows":
                        self.redis_process.kill()
                    else:
                        os.killpg(os.getpgid(self.redis_process.pid), signal.SIGKILL)
                except Exception:
                    pass

            self.redis_process = None

    def __enter__(self):
        if self.start_redis():
            return self
        else:
            raise RuntimeError("Failed to start Redis server")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_redis()


async def install_dependencies():
    """Install required Python dependencies"""
    print("Installing Python dependencies...")

    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("requirements.txt not found, skipping dependency installation")
        return True

    try:
        # Try to install with pip
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True, capture_output=True, text=True)

        print("Dependencies installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        print("Please install manually with:")
        print(f"pip install -r {requirements_file}")
        return False


async def run_multi_agent_tests() -> Dict[str, Any]:
    """Run the multi-agent coordination tests"""
    try:
        # Import test module
        from test_multi_agent import run_comprehensive_test, run_performance_test

        print("Running comprehensive multi-agent tests...")
        comprehensive_results = await run_comprehensive_test()

        print("Running performance tests...")
        performance_results = await run_performance_test()

        return {
            "comprehensive": comprehensive_results,
            "performance": performance_results
        }

    except ImportError as e:
        print(f"Failed to import test modules: {e}")
        print("Make sure all dependencies are installed")
        return {"error": "import_failed", "details": str(e)}
    except Exception as e:
        print(f"Test execution failed: {e}")
        return {"error": "test_failed", "details": str(e)}


async def main():
    """Main function to start Redis and run tests"""
    print("=" * 60)
    print("RAG-Redis Multi-Agent Coordination System")
    print("Startup and Test Script")
    print("=" * 60)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check if dependencies need to be installed
    try:
        import redis.asyncio as aioredis
        print("✓ Dependencies already installed")
    except ImportError:
        if not await install_dependencies():
            print("❌ Failed to install dependencies")
            return False

    # Start Redis server
    redis_manager = RedisManager()

    try:
        with redis_manager:
            print("✓ Redis server is running")

            # Test Redis connection
            try:
                redis_client = aioredis.from_url("redis://localhost:6379")
                await redis_client.ping()
                await redis_client.close()
                print("✓ Redis connection test passed")
            except Exception as e:
                print(f"❌ Redis connection test failed: {e}")
                return False

            # Run multi-agent tests
            test_results = await run_multi_agent_tests()

            if "error" in test_results:
                print(f"❌ Tests failed: {test_results['error']}")
                return False

            # Print summary
            comprehensive = test_results.get("comprehensive", {})
            performance = test_results.get("performance", {})

            print("\n" + "=" * 60)
            print("FINAL RESULTS")
            print("=" * 60)

            if comprehensive:
                total = comprehensive.get("total_tests", 0)
                passed = comprehensive.get("passed_tests", 0)
                print(f"Comprehensive Tests: {passed}/{total} passed")

            if performance:
                agents = performance.get("agents_created", 0)
                total_time = performance.get("total_time", 0)
                print(f"Performance Test: {agents} agents in {total_time:.2f}s")

            success = (
                comprehensive.get("passed_tests", 0) == comprehensive.get("total_tests", 0) and
                performance.get("agents_created", 0) > 0
            )

            if success:
                print("✓ All tests passed successfully!")
                print("\nThe RAG-Redis multi-agent coordination system is ready for production use.")
            else:
                print("❌ Some tests failed")

            return success

    except RuntimeError as e:
        print(f"❌ Failed to start Redis: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Redis is installed")
        print("2. Check if port 6379 is available")
        print("3. Run as administrator/sudo if needed")
        return False
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())

    if success:
        print("\n🎉 RAG-Redis Multi-Agent System is ready!")
        print("You can now integrate it with your Python stats agents.")
    else:
        print("\n💥 Setup failed. Please check the errors above.")

    sys.exit(0 if success else 1)