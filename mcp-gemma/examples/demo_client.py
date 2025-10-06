#!/usr/bin/env python3
"""
Demonstration of MCP Gemma client usage.

This script shows how to use the MCP Gemma client for various tasks.
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from client.stdio_client import GemmaStdioClient
from client.http_client import GemmaHTTPClient
from client.base_client import GenerationRequest, MemoryEntry

async def demo_stdio_client():
    """Demonstrate stdio client usage."""
    print("=== Stdio Client Demo ===")

    # Note: This requires a valid model path
    model_path = "/c/codedev/llm/.models/gemma2-2b-it-sfp.sbs"

    print(f"Using model: {model_path}")
    print("Note: This demo requires a valid Gemma model file.")

    try:
        async with GemmaStdioClient(model_path=model_path, debug=True) as client:
            print("✓ Connected to Gemma via stdio")

            # Simple text generation
            print("\n--- Simple Text Generation ---")
            response = await client.simple_generate(
                "Hello! Please introduce yourself briefly.",
                max_tokens=50
            )
            print(f"Response: {response}")

            # Chat with context
            print("\n--- Chat with Context ---")
            context = [
                {"role": "user", "content": "My name is Alice and I like cats."}
            ]
            response = await client.chat("What's my name and what do I like?", context)
            print(f"Context-aware response: {response}")

            # Memory operations
            print("\n--- Memory Operations ---")
            memory_id = await client.store_memory(MemoryEntry(
                key="demo_fact",
                content="The user prefers concise responses",
                metadata={"type": "preference", "importance": "high"}
            ))
            print(f"Stored memory with ID: {memory_id}")

            retrieved = await client.retrieve_memory("demo_fact")
            if retrieved:
                print(f"Retrieved memory: {retrieved.content}")
            else:
                print("Memory not found")

            # Search memory
            search_results = await client.search_memory("preference", limit=5)
            print(f"Found {len(search_results)} memories matching 'preference'")

            # Get metrics
            print("\n--- Server Metrics ---")
            metrics = await client.get_metrics()
            print(f"Server uptime: {metrics.get('performance', {}).get('uptime_seconds', 'unknown')} seconds")
            print(f"Total requests: {metrics.get('performance', {}).get('total_requests', 'unknown')}")

    except Exception as e:
        print(f"Error in stdio demo: {e}")
        print("This is expected if no model is available or server dependencies are missing.")

async def demo_http_client():
    """Demonstrate HTTP client usage."""
    print("\n=== HTTP Client Demo ===")

    # Note: This requires a running HTTP server
    base_url = "http://localhost:8080"

    print(f"Connecting to: {base_url}")
    print("Note: This requires a running MCP Gemma HTTP server.")

    try:
        async with GemmaHTTPClient(base_url=base_url, debug=True) as client:
            print("✓ Connected to HTTP server")

            # Health check
            print("\n--- Health Check ---")
            health = await client.health_check()
            print(f"Server status: {health.get('status', 'unknown')}")

            # List tools
            print("\n--- Available Tools ---")
            tools = await client.list_tools()
            print(f"Available tools: {[tool['name'] for tool in tools]}")

            # Simple text generation
            print("\n--- Text Generation ---")
            response = await client.simple_generate(
                "What is the capital of France?",
                max_tokens=30
            )
            print(f"Response: {response}")

            # Get metrics
            print("\n--- Server Metrics ---")
            metrics = await client.get_metrics()
            print(f"Model: {metrics.get('model_info', {}).get('model_path', 'unknown')}")
            print(f"Performance: {metrics.get('performance', {})}")

    except Exception as e:
        print(f"Error in HTTP demo: {e}")
        print("This is expected if no HTTP server is running.")

async def demo_batch_operations():
    """Demonstrate batch operations."""
    print("\n=== Batch Operations Demo ===")

    model_path = "/c/codedev/llm/.models/gemma2-2b-it-sfp.sbs"

    try:
        from client.base_client import BatchGemmaClient

        client = GemmaStdioClient(model_path=model_path)
        batch_client = BatchGemmaClient(client, max_concurrent=3)

        await client.connect()

        print("--- Batch Text Generation ---")
        prompts = [
            "What is AI?",
            "Explain machine learning.",
            "What is deep learning?"
        ]

        responses = await batch_client.generate_simple_batch(
            prompts,
            max_tokens=30
        )

        for prompt, response in zip(prompts, responses):
            print(f"Q: {prompt}")
            print(f"A: {response[:100]}...")
            print()

        await client.disconnect()

    except Exception as e:
        print(f"Error in batch demo: {e}")
        print("This is expected if no model is available.")

async def demo_integration():
    """Demonstrate framework integration."""
    print("\n=== Framework Integration Demo ===")

    try:
        from integration.stats_integration import GemmaMCPAgent

        print("--- Stats Framework Integration ---")
        async with GemmaMCPAgent(client_type="stdio") as agent:
            print("✓ Created integrated agent")

            # Generate with memory context
            response = await agent.generate_response("Hello, I'm testing the integration.")
            print(f"Agent response: {response[:100]}...")

            # Store and retrieve memory
            memory_id = await agent.store_memory(
                "integration_test",
                "This is a test of the integration system",
                {"demo": True}
            )
            print(f"Stored memory: {memory_id}")

            retrieved = await agent.retrieve_memory("integration_test")
            if retrieved:
                print(f"Retrieved: {retrieved['content']}")

    except Exception as e:
        print(f"Error in integration demo: {e}")
        print("This is expected if integration dependencies are not available.")

async def main():
    """Run all demonstrations."""
    print("🚀 MCP Gemma Client Demonstration")
    print("=" * 50)

    demos = [
        demo_stdio_client,
        demo_http_client,
        demo_batch_operations,
        demo_integration
    ]

    for demo in demos:
        try:
            await demo()
        except KeyboardInterrupt:
            print("\n⚠ Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")

        print("\n" + "-" * 50)

    print("\n✅ Demonstration completed!")
    print("\nNext steps:")
    print("1. Start the MCP server: python server/main.py --model <path-to-model>")
    print("2. Try the HTTP demo with a running server")
    print("3. Explore the integration examples")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)