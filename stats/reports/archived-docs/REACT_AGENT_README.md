# ReAct Agent Implementation

## Overview

The ReAct (Reasoning and Acting) agent implementation enhances the Gemma agent with systematic problem-solving capabilities through a structured thought-action-observation loop with integrated planning and reflection.

## Architecture

### Core Components

1. **ReAct Agent** (`src/agent/react_agent.py`)
   - Main agent class that orchestrates the reasoning process
   - Implements the thought → action → observation → reflection loop
   - Supports both heavyweight and lightweight Gemma models
   - Includes error recovery and self-correction capabilities

2. **Planner Module** (`src/agent/planner.py`)
   - Task complexity analysis (Simple/Medium/Complex/Very Complex)
   - Step-by-step plan generation with dependencies
   - Plan revision and progress tracking
   - Contingency planning for error scenarios

3. **Prompting Templates** (`src/agent/prompts.py`)
   - ReAct system prompts with tool integration
   - Planning prompts for task decomposition
   - Reflection prompts for progress assessment
   - Error recovery prompts for handling failures

4. **Demo Application** (`examples/react_demo.py`)
   - Interactive demonstrations of ReAct capabilities
   - Multiple complexity levels showcased
   - Custom tools for realistic scenarios

## Key Features

### 1. Reasoning Process

The agent follows a structured reasoning pattern:

```
THOUGHT → PLAN → ACTION → OBSERVATION → REFLECTION → (loop or ANSWER)
```

- **THOUGHT**: Analyzes the problem and current context
- **PLAN**: Breaks down complex tasks into manageable steps
- **ACTION**: Executes specific tool calls or operations
- **OBSERVATION**: Captures results from actions
- **REFLECTION**: Assesses progress and adjusts approach
- **ANSWER**: Provides final solution when complete

### 2. Planning Capabilities

- **Automatic Complexity Assessment**: Analyzes tasks to determine if planning is needed
- **Step Dependencies**: Manages execution order based on prerequisites
- **Progress Tracking**: Monitors completion percentage and current step
- **Contingency Handling**: Provides fallback strategies for failures

### 3. Self-Reflection

- **Periodic Progress Assessment**: Reviews accomplishments every few iterations
- **Challenge Identification**: Recognizes obstacles and incorrect assumptions
- **Strategy Adjustment**: Modifies approach based on observations
- **Error Learning**: Adapts behavior after encountering failures

### 4. Tool Integration

Seamlessly integrates with the existing tool system:
- Calculator for mathematical operations
- File operations (read/write)
- Web search and URL fetching
- System information queries
- Custom tools can be easily added

### 5. Trace Management

Complete logging of reasoning process:
- All thoughts, actions, and observations recorded
- Exportable traces for analysis
- Detailed summaries of problem-solving approach
- JSON serialization for persistence

## Usage

### Basic Usage

```python
from src.agent.react_agent import create_react_agent
from src.agent.tools import ToolRegistry

# Create agent with default settings
agent = create_react_agent(
    lightweight=True,  # Use pipeline for easier setup
    enable_planning=True,
    enable_reflection=True
)

# Solve a task
result = agent.solve("Compare the weather in New York and Los Angeles")
print(result)

# Get reasoning trace
print(agent.get_trace_summary())
```

### With Custom Tools

```python
from src.agent.tools import ToolDefinition, ToolParameter

# Define custom tool
def custom_tool(param: str) -> str:
    return f"Processed: {param}"

# Register tool
registry = ToolRegistry()
registry.register(ToolDefinition(
    name="custom",
    description="Custom processing tool",
    parameters=[
        ToolParameter(name="param", type="string", description="Input parameter")
    ],
    function=custom_tool
))

# Create agent with custom tools
agent = create_react_agent(tool_registry=registry)
```

### Complex Task Example

```python
# Complex multi-step task
task = """
Analyze the tech sector by:
1. Checking stock prices for major tech companies
2. Searching for recent news
3. Providing investment recommendations
"""

# Agent will automatically:
# - Create a plan
# - Execute steps in order
# - Reflect on progress
# - Handle any errors
# - Provide comprehensive answer

result = agent.solve(task, max_iterations=15)
```

## Configuration Options

### Agent Parameters

- `model_name`: Gemma model to use (default: "google/gemma-2b-it")
- `max_iterations`: Maximum reasoning iterations (default: 10)
- `enable_planning`: Use planning for complex tasks (default: True)
- `enable_reflection`: Use periodic reflection (default: True)
- `verbose`: Print detailed reasoning steps (default: True)

### Planning Configuration

- Automatic complexity assessment
- Customizable success criteria
- Dependency management
- Progress tracking

## Running Demos

The demo script provides multiple examples:

```bash
# Run all demos
python examples/react_demo.py

# Run specific demo
python examples/react_demo.py 1  # Simple task
python examples/react_demo.py 2  # Comparison task
python examples/react_demo.py 3  # Complex analysis
python examples/react_demo.py 4  # Data processing
python examples/react_demo.py 5  # Error recovery

# Interactive mode
python examples/react_demo.py interactive
```

## Implementation Details

### Thought Process Flow

1. **Initial Analysis**: Understand the problem and requirements
2. **Complexity Assessment**: Determine if planning is needed
3. **Plan Generation**: Create step-by-step approach for complex tasks
4. **Iterative Execution**:
   - Generate thought about current state
   - Decide on action to take
   - Execute action and observe results
   - Reflect on progress periodically
5. **Completion Check**: Determine if goal is achieved
6. **Final Answer**: Synthesize results into coherent response

### Error Handling

- Automatic retry on tool failures
- Graceful degradation for missing capabilities
- Error recovery strategies
- Maximum error threshold to prevent infinite loops

### Memory Management

- Conversation history tracking
- Recent context window for efficiency
- Trace history for analysis
- Configurable history limits

## Advanced Features

### Plan Revision

The agent can revise plans based on:
- Unexpected observations
- Tool failures
- New information discovered
- User feedback

### Multi-Step Tool Coordination

- Sequential tool execution with dependencies
- Parallel tool calls when possible
- Result aggregation and synthesis
- Tool output validation

### Self-Correction

- Identifies mistakes in reasoning
- Backtracks when necessary
- Adjusts approach based on feedback
- Learns from errors within session

## Performance Considerations

- Lightweight mode uses pipeline for faster inference
- Planning overhead only for complex tasks
- Efficient token usage through context management
- Configurable iteration limits

## Future Enhancements

Potential improvements:
1. Persistent memory across sessions
2. Learning from past problem-solving experiences
3. Collaborative multi-agent reasoning
4. Visual reasoning with image tools
5. Code generation and execution capabilities
6. Integration with external knowledge bases

## Troubleshooting

### Common Issues

1. **Agent gets stuck in loops**
   - Increase max_iterations
   - Enable reflection for better self-awareness
   - Check tool implementations

2. **Poor planning quality**
   - Ensure model has sufficient context
   - Adjust complexity thresholds
   - Provide clearer task descriptions

3. **Tool execution failures**
   - Verify tool registry setup
   - Check tool parameter types
   - Enable verbose mode for debugging

## API Reference

### ReActAgent Class

```python
class ReActAgent:
    def __init__(
        model_name: str = "google/gemma-2b-it",
        tool_registry: Optional[ToolRegistry] = None,
        max_iterations: int = 10,
        verbose: bool = True,
        enable_planning: bool = True,
        enable_reflection: bool = True,
        **kwargs
    )

    def solve(
        task: str,
        context: str = "",
        max_iterations: Optional[int] = None
    ) -> str

    def get_trace_summary(trace_index: int = -1) -> str

    def save_trace(filepath: str, trace_index: int = -1)
```

### Planner Class

```python
class Planner:
    def analyze_complexity(task: str, context: str = "") -> TaskComplexity

    def create_plan(
        task: str,
        context: str = "",
        tools_schemas: Optional[List[Dict]] = None,
        model_response: Optional[str] = None
    ) -> Plan

    def revise_plan(feedback: str, error: Optional[str] = None) -> Plan
```

## Conclusion

The ReAct agent implementation provides a robust framework for systematic problem-solving with the Gemma model. Through its combination of reasoning, planning, and reflection capabilities, it can handle complex, multi-step tasks while maintaining transparency in its decision-making process.
