#!/usr/bin/env python3
"""
Test script for Multi-Agent Coordination System
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any

from multi_agent_coordinator import (
    MultiAgentCoordinator,
    AgentCoordinatorManager,
    AgentMessage,
    MessageType,
    AgentState
)
from agent_config import create_test_config, AGENT_CAPABILITIES


class TestScenarios:
    """Test scenarios for multi-agent coordination"""

    def __init__(self, manager: AgentCoordinatorManager):
        self.manager = manager
        self.logger = logging.getLogger("test_scenarios")

    async def test_basic_coordination(self) -> Dict[str, Any]:
        """Test basic agent coordination and messaging"""
        self.logger.info("Testing basic coordination...")

        results = {"scenario": "basic_coordination", "success": False, "details": {}}

        try:
            # Create agents
            coordinator = await self.manager.create_agent(
                "coordinator",
                AGENT_CAPABILITIES["coordinator"]
            )
            worker1 = await self.manager.create_agent(
                "worker",
                AGENT_CAPABILITIES["worker"]
            )
            worker2 = await self.manager.create_agent(
                "worker",
                AGENT_CAPABILITIES["worker"]
            )

            # Wait for registration
            await asyncio.sleep(2)

            # Test broadcasting
            await coordinator.broadcast_message(
                MessageType.COORDINATION,
                {"test": "broadcast_message", "timestamp": time.time()}
            )

            # Test direct messaging
            await coordinator.send_message(
                worker1.agent_id,
                MessageType.TASK_REQUEST,
                {"task": "test_task", "priority": "high"}
            )

            # Wait for message processing
            await asyncio.sleep(2)

            # Check system status
            status = await self.manager.get_system_status()
            results["details"]["system_status"] = status
            results["details"]["active_agents"] = len(status["agents"])

            results["success"] = len(status["agents"]) >= 3

        except Exception as e:
            results["details"]["error"] = str(e)
            self.logger.error(f"Basic coordination test failed: {e}")

        return results

    async def test_shared_memory(self) -> Dict[str, Any]:
        """Test shared memory operations"""
        self.logger.info("Testing shared memory...")

        results = {"scenario": "shared_memory", "success": False, "details": {}}

        try:
            # Create memory agent
            memory_agent = await self.manager.create_agent(
                "memory",
                AGENT_CAPABILITIES["memory"]
            )

            # Test memory operations
            test_data = {
                "config": {"max_workers": 5, "timeout": 30},
                "metrics": {"cpu_usage": 45.2, "memory_usage": 67.8},
                "tasks": ["task1", "task2", "task3"]
            }

            # Store data
            for key, value in test_data.items():
                await memory_agent.update_shared_memory(key, value)

            # Retrieve data
            retrieved_data = {}
            for key in test_data.keys():
                retrieved_data[key] = await memory_agent.get_shared_memory(key)

            # Verify data integrity
            success = all(
                retrieved_data[key] == test_data[key]
                for key in test_data.keys()
            )

            results["details"]["stored_data"] = test_data
            results["details"]["retrieved_data"] = retrieved_data
            results["details"]["data_integrity"] = success
            results["success"] = success

        except Exception as e:
            results["details"]["error"] = str(e)
            self.logger.error(f"Shared memory test failed: {e}")

        return results

    async def test_agent_lifecycle(self) -> Dict[str, Any]:
        """Test agent lifecycle management"""
        self.logger.info("Testing agent lifecycle...")

        results = {"scenario": "agent_lifecycle", "success": False, "details": {}}

        try:
            # Track initial state
            initial_status = await self.manager.get_system_status()
            initial_count = initial_status["total_agents"]

            # Create new agent
            new_agent = await self.manager.create_agent(
                "worker",
                ["test_capability"]
            )

            # Wait for registration
            await asyncio.sleep(2)

            # Check agent was added
            mid_status = await self.manager.get_system_status()
            mid_count = mid_status["total_agents"]

            # Stop the agent
            await new_agent.stop()

            # Wait for cleanup
            await asyncio.sleep(2)

            # Check agent was removed
            final_status = await self.manager.get_system_status()
            final_count = final_status["total_agents"]

            results["details"]["initial_count"] = initial_count
            results["details"]["mid_count"] = mid_count
            results["details"]["final_count"] = final_count
            results["success"] = (
                mid_count == initial_count + 1 and
                final_count == initial_count
            )

        except Exception as e:
            results["details"]["error"] = str(e)
            self.logger.error(f"Agent lifecycle test failed: {e}")

        return results

    async def test_message_routing(self) -> Dict[str, Any]:
        """Test message routing between agents"""
        self.logger.info("Testing message routing...")

        results = {"scenario": "message_routing", "success": False, "details": {}}

        try:
            # Create agents with custom message handlers
            coordinator = await self.manager.create_agent(
                "coordinator",
                AGENT_CAPABILITIES["coordinator"]
            )
            worker = await self.manager.create_agent(
                "worker",
                AGENT_CAPABILITIES["worker"]
            )

            # Set up message tracking
            received_messages = []

            async def track_messages(message: AgentMessage):
                received_messages.append({
                    "sender": message.sender_id,
                    "receiver": message.receiver_id,
                    "type": message.message_type.value,
                    "content": message.content
                })

            # Register custom handler
            worker.register_message_handler(MessageType.TASK_REQUEST, track_messages)

            # Wait for setup
            await asyncio.sleep(1)

            # Send test messages
            test_messages = [
                {"type": MessageType.TASK_REQUEST, "content": {"task": "process_data"}},
                {"type": MessageType.STATUS_UPDATE, "content": {"status": "busy"}},
                {"type": MessageType.COORDINATION, "content": {"action": "synchronize"}}
            ]

            for msg in test_messages:
                await coordinator.send_message(
                    worker.agent_id,
                    msg["type"],
                    msg["content"]
                )

            # Wait for message processing
            await asyncio.sleep(2)

            results["details"]["sent_messages"] = len(test_messages)
            results["details"]["received_messages"] = len(received_messages)
            results["details"]["message_details"] = received_messages
            results["success"] = len(received_messages) >= 1  # At least task_request should be tracked

        except Exception as e:
            results["details"]["error"] = str(e)
            self.logger.error(f"Message routing test failed: {e}")

        return results

    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery"""
        self.logger.info("Testing error handling...")

        results = {"scenario": "error_handling", "success": False, "details": {}}

        try:
            # Create agent
            agent = await self.manager.create_agent(
                "worker",
                AGENT_CAPABILITIES["worker"]
            )

            # Test invalid message handling
            try:
                # This should not crash the system
                await agent.redis_client.publish(
                    agent.channel_personal,
                    "invalid_json_data"
                )
                await asyncio.sleep(1)
                results["details"]["invalid_message_handled"] = True
            except Exception:
                results["details"]["invalid_message_handled"] = False

            # Test shared memory with invalid data
            try:
                await agent.update_shared_memory("test_key", {"valid": "data"})
                invalid_retrieved = await agent.get_shared_memory("nonexistent_key")
                results["details"]["nonexistent_key_handled"] = invalid_retrieved is None
            except Exception:
                results["details"]["nonexistent_key_handled"] = False

            # Check agent is still responsive
            status = await self.manager.get_system_status()
            agent_active = any(
                a["id"] == agent.agent_id and a["state"] == "active"
                for a in status["agents"]
            )

            results["details"]["agent_still_active"] = agent_active
            results["success"] = (
                results["details"].get("invalid_message_handled", False) and
                results["details"].get("nonexistent_key_handled", False) and
                agent_active
            )

        except Exception as e:
            results["details"]["error"] = str(e)
            self.logger.error(f"Error handling test failed: {e}")

        return results


async def run_comprehensive_test() -> Dict[str, Any]:
    """Run comprehensive test suite"""
    print("=" * 60)
    print("Multi-Agent Coordination System - Comprehensive Test")
    print("=" * 60)

    # Configure test environment
    config = create_test_config()
    manager = AgentCoordinatorManager(config.redis.url)
    scenarios = TestScenarios(manager)

    test_results = {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "test_details": []
    }

    # List of test scenarios
    test_functions = [
        scenarios.test_basic_coordination,
        scenarios.test_shared_memory,
        scenarios.test_agent_lifecycle,
        scenarios.test_message_routing,
        scenarios.test_error_handling
    ]

    try:
        for test_func in test_functions:
            test_results["total_tests"] += 1
            print(f"\nRunning {test_func.__name__}...")

            try:
                result = await test_func()
                test_results["test_details"].append(result)

                if result["success"]:
                    test_results["passed_tests"] += 1
                    print(f"✓ {result['scenario']} - PASSED")
                else:
                    test_results["failed_tests"] += 1
                    print(f"✗ {result['scenario']} - FAILED")
                    if "error" in result["details"]:
                        print(f"  Error: {result['details']['error']}")

            except Exception as e:
                test_results["failed_tests"] += 1
                print(f"✗ {test_func.__name__} - CRASHED: {e}")
                test_results["test_details"].append({
                    "scenario": test_func.__name__,
                    "success": False,
                    "details": {"error": str(e)}
                })

            # Clean up between tests
            await manager.stop_all_agents()
            await asyncio.sleep(1)

    finally:
        # Final cleanup
        await manager.stop_all_agents()

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"Passed: {test_results['passed_tests']}")
    print(f"Failed: {test_results['failed_tests']}")
    print(f"Success Rate: {(test_results['passed_tests'] / test_results['total_tests'] * 100):.1f}%")

    return test_results


async def run_performance_test() -> Dict[str, Any]:
    """Run performance test with multiple agents"""
    print("\n" + "=" * 60)
    print("Performance Test - Multiple Concurrent Agents")
    print("=" * 60)

    config = create_test_config()
    manager = AgentCoordinatorManager(config.redis.url)

    start_time = time.time()
    agents = []

    try:
        # Create multiple agents
        print("Creating agents...")
        for i in range(5):
            agent = await manager.create_agent(
                "worker",
                [f"capability_{i}", "performance_test"]
            )
            agents.append(agent)

        creation_time = time.time() - start_time
        print(f"Agent creation time: {creation_time:.2f} seconds")

        # Test message throughput
        print("Testing message throughput...")
        message_start = time.time()

        coordinator = agents[0]  # Use first agent as coordinator

        # Send messages between agents
        for i in range(10):
            await coordinator.broadcast_message(
                MessageType.COORDINATION,
                {"test_message": i, "timestamp": time.time()}
            )

        message_time = time.time() - message_start
        print(f"Message sending time: {message_time:.2f} seconds")

        # Test shared memory performance
        print("Testing shared memory performance...")
        memory_start = time.time()

        for i in range(20):
            await coordinator.update_shared_memory(
                f"perf_test_{i}",
                {"data": f"test_data_{i}", "index": i}
            )

        memory_time = time.time() - memory_start
        print(f"Memory operations time: {memory_time:.2f} seconds")

        # Get final status
        status = await manager.get_system_status()
        total_time = time.time() - start_time

        return {
            "agents_created": len(agents),
            "creation_time": creation_time,
            "message_time": message_time,
            "memory_time": memory_time,
            "total_time": total_time,
            "final_status": status
        }

    finally:
        await manager.stop_all_agents()


async def main():
    """Main test function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Run comprehensive tests
        test_results = await run_comprehensive_test()

        # Run performance test
        perf_results = await run_performance_test()

        # Save results
        results = {
            "comprehensive_tests": test_results,
            "performance_tests": perf_results,
            "timestamp": time.time()
        }

        print(f"\nPerformance Results:")
        print(f"- Agents created: {perf_results['agents_created']}")
        print(f"- Total time: {perf_results['total_time']:.2f}s")
        print(f"- Creation time: {perf_results['creation_time']:.2f}s")
        print(f"- Message time: {perf_results['message_time']:.2f}s")
        print(f"- Memory time: {perf_results['memory_time']:.2f}s")

        # Write results to file
        with open("test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nDetailed results saved to test_results.json")

        return test_results["passed_tests"] == test_results["total_tests"]

    except Exception as e:
        print(f"Test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)