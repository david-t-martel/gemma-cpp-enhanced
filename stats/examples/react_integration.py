"""Integration example showing ReAct agent in a real application context."""

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.react_agent import create_react_agent
from src.agent.tools import ToolDefinition, ToolParameter, ToolRegistry


class TaskManager:
    """Example application using ReAct agent for task management."""

    def __init__(self, verbose: bool = True):
        """Initialize the task manager with ReAct agent."""
        self.agent = self._create_agent(verbose)
        self.tasks_file = "tasks.json"
        self.completed_tasks = []
        self.pending_tasks = []
        self._load_tasks()

    def _create_agent(self, verbose: bool):
        """Create ReAct agent with custom tools."""
        registry = ToolRegistry()

        # Add custom task management tools
        registry.register(
            ToolDefinition(
                name="add_task",
                description="Add a new task to the task list",
                parameters=[
                    ToolParameter(name="title", type="string", description="Task title"),
                    ToolParameter(
                        name="priority",
                        type="string",
                        description="Priority (high/medium/low)",
                        required=False,
                    ),
                ],
                function=self._add_task_tool,
            )
        )

        registry.register(
            ToolDefinition(
                name="list_tasks",
                description="List all pending tasks",
                parameters=[],
                function=self._list_tasks_tool,
            )
        )

        registry.register(
            ToolDefinition(
                name="complete_task",
                description="Mark a task as completed",
                parameters=[
                    ToolParameter(name="task_id", type="integer", description="Task ID to complete")
                ],
                function=self._complete_task_tool,
            )
        )

        registry.register(
            ToolDefinition(
                name="analyze_productivity",
                description="Analyze productivity metrics",
                parameters=[],
                function=self._analyze_productivity_tool,
            )
        )

        return create_react_agent(
            lightweight=True,
            tool_registry=registry,
            verbose=verbose,
            enable_planning=True,
            enable_reflection=True,
        )

    def _load_tasks(self):
        """Load tasks from file."""
        try:
            with open(self.tasks_file) as f:
                data = json.load(f)
                self.pending_tasks = data.get("pending", [])
                self.completed_tasks = data.get("completed", [])
        except FileNotFoundError:
            self.pending_tasks = []
            self.completed_tasks = []

    def _save_tasks(self):
        """Save tasks to file."""
        data = {"pending": self.pending_tasks, "completed": self.completed_tasks}
        with open(self.tasks_file, "w") as f:
            json.dump(data, f, indent=2)

    def _add_task_tool(self, title: str, priority: str = "medium") -> str:
        """Tool function to add a task."""
        task = {
            "id": len(self.pending_tasks) + len(self.completed_tasks) + 1,
            "title": title,
            "priority": priority,
            "created": datetime.now().isoformat(),
            "status": "pending",
        }
        self.pending_tasks.append(task)
        self._save_tasks()
        return f"Task added: {title} (ID: {task['id']}, Priority: {priority})"

    def _list_tasks_tool(self) -> str:
        """Tool function to list tasks."""
        if not self.pending_tasks:
            return "No pending tasks"

        result = "Pending Tasks:\n"
        for task in self.pending_tasks:
            result += f"- [{task['id']}] {task['title']} (Priority: {task['priority']})\n"
        return result

    def _complete_task_tool(self, task_id: int) -> str:
        """Tool function to complete a task."""
        for i, task in enumerate(self.pending_tasks):
            if task["id"] == task_id:
                task["status"] = "completed"
                task["completed"] = datetime.now().isoformat()
                self.completed_tasks.append(task)
                self.pending_tasks.pop(i)
                self._save_tasks()
                return f"Task {task_id} completed: {task['title']}"
        return f"Task {task_id} not found"

    def _analyze_productivity_tool(self) -> str:
        """Tool function to analyze productivity."""
        total = len(self.pending_tasks) + len(self.completed_tasks)
        if total == 0:
            return "No tasks to analyze"

        completion_rate = (len(self.completed_tasks) / total) * 100

        high_priority = sum(1 for t in self.pending_tasks if t.get("priority") == "high")
        med_priority = sum(1 for t in self.pending_tasks if t.get("priority") == "medium")
        low_priority = sum(1 for t in self.pending_tasks if t.get("priority") == "low")

        analysis = f"""Productivity Analysis:
- Total tasks: {total}
- Completed: {len(self.completed_tasks)}
- Pending: {len(self.pending_tasks)}
- Completion rate: {completion_rate:.1f}%

Pending tasks by priority:
- High: {high_priority}
- Medium: {med_priority}
- Low: {low_priority}

Recommendation: {"Focus on high-priority tasks" if high_priority > 0 else "Good progress! Keep it up!"}"""

        return analysis

    def process_request(self, request: str) -> str:
        """Process a natural language request about tasks."""
        return self.agent.solve(request)


class DataAnalyzer:
    """Example application using ReAct agent for data analysis."""

    def __init__(self, verbose: bool = True):
        """Initialize the data analyzer."""
        self.agent = self._create_agent(verbose)
        self.data_cache = {}

    def _create_agent(self, verbose: bool):
        """Create ReAct agent with data analysis tools."""
        registry = ToolRegistry()

        # Add data analysis tools
        registry.register(
            ToolDefinition(
                name="load_data",
                description="Load data from a CSV file",
                parameters=[
                    ToolParameter(name="filename", type="string", description="CSV filename")
                ],
                function=self._load_data_tool,
            )
        )

        registry.register(
            ToolDefinition(
                name="calculate_statistics",
                description="Calculate basic statistics for loaded data",
                parameters=[
                    ToolParameter(
                        name="column", type="string", description="Column name to analyze"
                    )
                ],
                function=self._calculate_stats_tool,
            )
        )

        registry.register(
            ToolDefinition(
                name="generate_report",
                description="Generate analysis report",
                parameters=[
                    ToolParameter(
                        name="report_type",
                        type="string",
                        description="Type of report (summary/detailed)",
                    )
                ],
                function=self._generate_report_tool,
            )
        )

        return create_react_agent(
            lightweight=True,
            tool_registry=registry,
            verbose=verbose,
            enable_planning=True,
            enable_reflection=True,
        )

    def _load_data_tool(self, filename: str) -> str:
        """Tool to simulate loading data."""
        # Simulated data loading
        self.data_cache[filename] = {
            "rows": 1000,
            "columns": ["id", "value", "category", "date"],
            "sample": [
                {"id": 1, "value": 100, "category": "A", "date": "2024-01-01"},
                {"id": 2, "value": 150, "category": "B", "date": "2024-01-02"},
            ],
        }
        return f"Loaded {filename}: 1000 rows, 4 columns"

    def _calculate_stats_tool(self, column: str) -> str:
        """Tool to calculate statistics."""
        if not self.data_cache:
            return "No data loaded"

        # Simulated statistics
        stats = {"mean": 125.5, "median": 120.0, "std": 45.2, "min": 10, "max": 300}

        return f"""Statistics for '{column}':
- Mean: {stats["mean"]}
- Median: {stats["median"]}
- Std Dev: {stats["std"]}
- Min: {stats["min"]}
- Max: {stats["max"]}"""

    def _generate_report_tool(self, report_type: str) -> str:
        """Tool to generate reports."""
        if not self.data_cache:
            return "No data loaded"

        if report_type == "summary":
            return """Summary Report:
- Dataset: sales_data.csv
- Total records: 1000
- Date range: 2024-01-01 to 2024-12-31
- Key findings:
  * Average value increased by 15%
  * Category A shows highest growth
  * Peak activity in Q3"""
        else:
            return """Detailed Report:
[Full analysis with charts and tables would be generated here]
- Executive Summary
- Methodology
- Data Analysis
- Key Findings
- Recommendations
- Appendices"""

    def analyze(self, query: str) -> str:
        """Analyze data based on natural language query."""
        return self.agent.solve(query)


def demo_task_manager():
    """Demonstrate task manager with ReAct agent."""
    print("=" * 60)
    print("Task Manager Demo")
    print("=" * 60)

    manager = TaskManager(verbose=True)

    # Example requests
    requests = [
        "Add a task to review the quarterly report with high priority",
        "Add a task to call the client with medium priority",
        "Show me all my pending tasks",
        "Complete task 1",
        "Analyze my productivity and give recommendations",
    ]

    for request in requests:
        print(f"\n📝 Request: {request}")
        print("-" * 40)
        result = manager.process_request(request)
        print("-" * 40)
        print(f"✅ Result: {result}\n")


def demo_data_analyzer():
    """Demonstrate data analyzer with ReAct agent."""
    print("\n" + "=" * 60)
    print("Data Analyzer Demo")
    print("=" * 60)

    analyzer = DataAnalyzer(verbose=True)

    # Complex analysis request
    request = """
    I need a complete data analysis:
    1. Load the sales_data.csv file
    2. Calculate statistics for the 'value' column
    3. Generate a summary report
    4. Based on the analysis, provide business recommendations
    """

    print(f"📊 Analysis Request: {request}")
    print("-" * 40)
    result = analyzer.analyze(request)
    print("-" * 40)
    print(f"✅ Analysis Result: {result}\n")


def demo_customer_support():
    """Demonstrate customer support bot with ReAct agent."""
    print("\n" + "=" * 60)
    print("Customer Support Bot Demo")
    print("=" * 60)

    # Create agent with customer support tools
    registry = ToolRegistry()

    def check_order_status(order_id: str) -> str:
        """Check order status."""
        # Simulated order lookup
        statuses = {
            "12345": "Shipped - Expected delivery: 2 days",
            "67890": "Processing - Will ship within 24 hours",
            "11111": "Delivered - Delivered on 2024-01-10",
        }
        return statuses.get(order_id, "Order not found")

    def process_return(order_id: str, reason: str) -> str:
        """Process a return request."""
        return f"Return initiated for order {order_id}. Reason: {reason}. Return label will be emailed within 24 hours."

    def check_warranty(product: str) -> str:
        """Check warranty information."""
        warranties = {
            "laptop": "2 years from purchase date",
            "phone": "1 year from purchase date",
            "tablet": "1 year from purchase date",
        }
        return f"Warranty for {product}: {warranties.get(product.lower(), 'Product not found')}"

    # Register tools
    registry.register(
        ToolDefinition(
            name="check_order",
            description="Check the status of an order",
            parameters=[ToolParameter(name="order_id", type="string", description="Order ID")],
            function=check_order_status,
        )
    )

    registry.register(
        ToolDefinition(
            name="process_return",
            description="Process a return request",
            parameters=[
                ToolParameter(name="order_id", type="string", description="Order ID"),
                ToolParameter(name="reason", type="string", description="Return reason"),
            ],
            function=process_return,
        )
    )

    registry.register(
        ToolDefinition(
            name="check_warranty",
            description="Check warranty information for a product",
            parameters=[ToolParameter(name="product", type="string", description="Product type")],
            function=check_warranty,
        )
    )

    # Create support agent
    support_agent = create_react_agent(
        lightweight=True, tool_registry=registry, verbose=True, enable_planning=True
    )

    # Customer queries
    queries = [
        "I want to check the status of my order 12345",
        "I need to return order 67890 because it's defective",
        "What's the warranty on laptops?",
    ]

    for query in queries:
        print(f"\n👤 Customer: {query}")
        print("-" * 40)
        response = support_agent.solve(query)
        print("-" * 40)
        print(f"🤖 Support: {response}\n")


def main():
    """Run all integration demos."""
    print("""
╔══════════════════════════════════════════════════════════╗
║        ReAct Agent Integration Examples                   ║
║         Real-World Application Scenarios                  ║
╚══════════════════════════════════════════════════════════╝
    """)

    demos = [
        ("Task Manager", demo_task_manager),
        ("Data Analyzer", demo_data_analyzer),
        ("Customer Support Bot", demo_customer_support),
    ]

    for name, demo_func in demos:
        try:
            input(f"\nPress Enter to run {name} demo (Ctrl+C to skip)...")
            demo_func()
        except KeyboardInterrupt:
            print(f"\nSkipping {name} demo")
        except Exception as e:
            print(f"\nError in {name} demo: {e}")

    print("\n" + "=" * 60)
    print("Integration demos completed!")
    print("=" * 60)
    print("\nThese examples show how ReAct agents can be integrated into:")
    print("- Task management systems")
    print("- Data analysis pipelines")
    print("- Customer support applications")
    print("- And many other real-world scenarios!")


if __name__ == "__main__":
    main()
