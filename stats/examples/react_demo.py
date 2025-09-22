"""Demonstration of ReAct agent solving complex problems."""

import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
import json
import random

from src.agent.gemma_agent import AgentMode
from src.agent.react_agent import create_react_agent
from src.agent.tools import ToolDefinition, ToolParameter, ToolRegistry


# Custom tools for demonstration
def weather_tool(city: str) -> str:
    """Simulated weather tool for demo."""
    # In real implementation, this would call a weather API
    weather_conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "foggy"]
    temps = {
        "New York": random.randint(30, 85),
        "Los Angeles": random.randint(60, 90),
        "Chicago": random.randint(25, 80),
        "Miami": random.randint(70, 95),
        "Seattle": random.randint(40, 75),
        "Boston": random.randint(30, 85),
        "Denver": random.randint(20, 85),
        "San Francisco": random.randint(50, 75),
    }

    temp = temps.get(city, random.randint(40, 80))
    condition = random.choice(weather_conditions)

    return f"Weather in {city}: {temp}°F, {condition}. Humidity: {random.randint(30, 80)}%. Wind: {random.randint(5, 20)} mph."


def stock_price_tool(symbol: str) -> str:
    """Simulated stock price tool."""
    # Simulated stock data
    stocks = {
        "AAPL": {"price": 175.43, "change": 2.15, "percent": 1.24},
        "GOOGL": {"price": 138.92, "change": -0.83, "percent": -0.59},
        "MSFT": {"price": 378.91, "change": 3.42, "percent": 0.91},
        "AMZN": {"price": 127.74, "change": 1.23, "percent": 0.97},
        "TSLA": {"price": 238.83, "change": -5.21, "percent": -2.14},
    }

    if symbol in stocks:
        data = stocks[symbol]
        change_symbol = "+" if data["change"] > 0 else ""
        return f"{symbol}: ${data['price']} ({change_symbol}{data['change']} / {change_symbol}{data['percent']}%)"
    else:
        return f"Stock data for {symbol} not available"


def news_search_tool(topic: str, max_results: int = 3) -> str:
    """Simulated news search tool."""
    # Simulated news articles
    news_templates = [
        {
            "title": f"Breaking: Major developments in {topic}",
            "source": "TechNews",
            "time": "2 hours ago",
        },
        {
            "title": f"Analysis: What {topic} means for the future",
            "source": "BusinessDaily",
            "time": "5 hours ago",
        },
        {
            "title": f"Expert opinion on {topic} trends",
            "source": "IndustryWatch",
            "time": "1 day ago",
        },
        {
            "title": f"How {topic} is changing the landscape",
            "source": "GlobalReport",
            "time": "3 hours ago",
        },
        {
            "title": f"New research reveals insights about {topic}",
            "source": "ScienceToday",
            "time": "6 hours ago",
        },
    ]

    selected_news = random.sample(news_templates, min(max_results, len(news_templates)))

    result = f"News results for '{topic}':\n\n"
    for i, article in enumerate(selected_news, 1):
        result += f"{i}. {article['title']}\n"
        result += f"   Source: {article['source']} - {article['time']}\n"
        result += f"   Summary: Latest updates and analysis on {topic}...\n\n"

    return result


def database_query_tool(query: str, table: str = "users") -> str:
    """Simulated database query tool."""
    # Simulated database responses
    if "count" in query.lower():
        return f"Query result: 1,247 records found in {table}"
    elif "select" in query.lower():
        return f"""Query result from {table}:
ID | Name        | Email              | Created
1  | John Doe    | john@example.com   | 2024-01-15
2  | Jane Smith  | jane@example.com   | 2024-01-16
3  | Bob Wilson  | bob@example.com    | 2024-01-17
(Showing first 3 of 1,247 results)"""
    else:
        return f"Query executed successfully on {table}"


def create_demo_tool_registry() -> ToolRegistry:
    """Create a tool registry with demo tools."""
    registry = ToolRegistry()

    # Add custom tools
    registry.register(
        ToolDefinition(
            name="weather",
            description="Get current weather information for a city",
            parameters=[ToolParameter(name="city", type="string", description="City name")],
            function=weather_tool,
        )
    )

    registry.register(
        ToolDefinition(
            name="stock_price",
            description="Get current stock price and change information",
            parameters=[
                ToolParameter(
                    name="symbol", type="string", description="Stock symbol (e.g., AAPL, GOOGL)"
                )
            ],
            function=stock_price_tool,
        )
    )

    registry.register(
        ToolDefinition(
            name="news_search",
            description="Search for recent news articles on a topic",
            parameters=[
                ToolParameter(name="topic", type="string", description="Topic to search for"),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum results",
                    required=False,
                ),
            ],
            function=news_search_tool,
        )
    )

    registry.register(
        ToolDefinition(
            name="database_query",
            description="Execute a database query",
            parameters=[
                ToolParameter(name="query", type="string", description="SQL query to execute"),
                ToolParameter(
                    name="table", type="string", description="Table name", required=False
                ),
            ],
            function=database_query_tool,
        )
    )

    return registry


def demo_simple_task():
    """Demonstrate ReAct agent on a simple task."""
    print("=" * 80)
    print("DEMO 1: Simple Task - Weather Query")
    print("=" * 80)

    # Create agent
    agent = create_react_agent(
        mode=AgentMode.LIGHTWEIGHT,
        tool_registry=create_demo_tool_registry(),
        verbose=True,
        enable_planning=False,  # Simple task doesn't need planning
        enable_reflection=False,
    )

    # Simple query
    result = agent.solve("What's the weather like in New York?")

    print(f"\n📊 Final Result: {result}")
    print("=" * 80)


def demo_comparison_task():
    """Demonstrate ReAct agent comparing multiple items."""
    print("\n" + "=" * 80)
    print("DEMO 2: Comparison Task - Weather in Multiple Cities")
    print("=" * 80)

    # Create agent
    agent = create_react_agent(
        mode=AgentMode.LIGHTWEIGHT,
        tool_registry=create_demo_tool_registry(),
        verbose=True,
        enable_planning=True,
        enable_reflection=True,
    )

    # Comparison query
    result = agent.solve(
        "Compare the weather in New York, Los Angeles, and Chicago. "
        "Which city has the best weather today?"
    )

    print(f"\n📊 Final Result: {result}")

    # Show trace summary
    print("\n📈 Trace Summary:")
    print(agent.get_trace_summary())
    print("=" * 80)


def demo_complex_analysis():
    """Demonstrate ReAct agent on complex multi-step analysis."""
    print("\n" + "=" * 80)
    print("DEMO 3: Complex Analysis - Market Research")
    print("=" * 80)

    # Create agent
    agent = create_react_agent(
        mode=AgentMode.LIGHTWEIGHT,
        tool_registry=create_demo_tool_registry(),
        verbose=True,
        enable_planning=True,
        enable_reflection=True,
        max_iterations=15,
    )

    # Complex query requiring multiple tools and reasoning
    result = agent.solve(
        "I need a comprehensive analysis of the tech sector. "
        "Check the stock prices for AAPL, GOOGL, and MSFT. "
        "Search for recent news about 'technology stocks'. "
        "Based on the stock performance and news, provide an investment recommendation."
    )

    print(f"\n📊 Final Result: {result}")

    # Show detailed trace
    print("\n📈 Detailed Trace Summary:")
    print(agent.get_trace_summary())
    print("=" * 80)


def demo_data_processing():
    """Demonstrate ReAct agent on data processing task."""
    print("\n" + "=" * 80)
    print("DEMO 4: Data Processing - Multi-Source Integration")
    print("=" * 80)

    # Create agent
    agent = create_react_agent(
        mode=AgentMode.LIGHTWEIGHT,
        tool_registry=create_demo_tool_registry(),
        verbose=True,
        enable_planning=True,
        enable_reflection=True,
    )

    # Data processing query
    result = agent.solve(
        "Analyze user growth by: "
        "1) Query the database for total user count, "
        "2) Search for news about 'user growth trends', "
        "3) Calculate if we're on track for 10,000 users by year end, "
        "4) Provide recommendations for improving growth."
    )

    print(f"\n📊 Final Result: {result}")
    print("=" * 80)


def demo_error_recovery():
    """Demonstrate ReAct agent recovering from errors."""
    print("\n" + "=" * 80)
    print("DEMO 5: Error Recovery - Handling Failures")
    print("=" * 80)

    # Create agent with a registry that has a failing tool
    registry = create_demo_tool_registry()

    # Add a tool that will fail
    def failing_tool(param: str) -> str:
        raise Exception("Simulated tool failure!")

    registry.register(
        ToolDefinition(
            name="unstable_api",
            description="An API that might fail",
            parameters=[ToolParameter(name="param", type="string", description="Parameter")],
            function=failing_tool,
        )
    )

    agent = create_react_agent(
        mode=AgentMode.LIGHTWEIGHT,
        tool_registry=registry,
        verbose=True,
        enable_planning=True,
        enable_reflection=True,
    )

    # Query that might trigger the failing tool
    result = agent.solve(
        "Try to get data from the unstable API, "
        "but if it fails, get weather for Seattle instead as a backup plan."
    )

    print(f"\n📊 Final Result: {result}")
    print("=" * 80)


def interactive_demo():
    """Interactive demo where user can input queries."""
    print("\n" + "=" * 80)
    print("INTERACTIVE DEMO - ReAct Agent")
    print("=" * 80)
    print("\nAvailable tools:")
    print("- weather (city)")
    print("- stock_price (symbol)")
    print("- news_search (topic)")
    print("- database_query (query)")
    print("- calculator (expression)")
    print("- system_info ()")
    print("\nType 'quit' to exit")
    print("=" * 80)

    # Create agent
    agent = create_react_agent(
        mode=AgentMode.LIGHTWEIGHT,
        tool_registry=create_demo_tool_registry(),
        verbose=True,
        enable_planning=True,
        enable_reflection=True,
    )

    while True:
        try:
            query = input("\n💬 Enter your query: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not query:
                continue

            print(f"\n🎯 Processing: {query}\n")
            result = agent.solve(query)

            print(f"\n✅ Result: {result}")

            # Ask if user wants to see trace
            show_trace = input("\nShow reasoning trace? (y/n): ").strip().lower()
            if show_trace == "y":
                print(agent.get_trace_summary())

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different query.")


def main():
    """Run all demos."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║             ReAct Agent Demonstration                             ║
║     Reasoning and Acting with Planning Capabilities               ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # Check if running specific demo
    if len(sys.argv) > 1:
        demo_num = sys.argv[1]
        if demo_num == "1":
            demo_simple_task()
        elif demo_num == "2":
            demo_comparison_task()
        elif demo_num == "3":
            demo_complex_analysis()
        elif demo_num == "4":
            demo_data_processing()
        elif demo_num == "5":
            demo_error_recovery()
        elif demo_num == "interactive":
            interactive_demo()
        else:
            print(f"Unknown demo: {demo_num}")
            print("Available: 1, 2, 3, 4, 5, interactive")
    else:
        # Run all demos
        print("\nRunning all demos...\n")

        try:
            # Simple demo
            demo_simple_task()
            input("\nPress Enter to continue to next demo...")

            # Comparison demo
            demo_comparison_task()
            input("\nPress Enter to continue to next demo...")

            # Complex analysis
            demo_complex_analysis()
            input("\nPress Enter to continue to next demo...")

            # Data processing
            demo_data_processing()
            input("\nPress Enter to continue to next demo...")

            # Error recovery
            demo_error_recovery()

            print("\n" + "=" * 80)
            print("All demos completed!")
            print("=" * 80)

            # Ask if user wants interactive mode
            interactive = input("\nWould you like to try interactive mode? (y/n): ").strip().lower()
            if interactive == "y":
                interactive_demo()

        except KeyboardInterrupt:
            print("\n\nDemos interrupted by user.")

    print("\n✨ Thank you for trying the ReAct Agent!")


if __name__ == "__main__":
    main()
