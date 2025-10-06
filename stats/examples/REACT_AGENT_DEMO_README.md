# ReAct Agent with Gemma - Coding Demonstration Guide

This directory contains comprehensive demonstrations of the Python ReAct agent solving coding problems using the Gemma LLM.

## 📁 Files Overview

1. **`coding_agent_demo.py`** - Complete standalone demonstration script
2. **`react_agent_coding_notebook.ipynb`** - Interactive Jupyter notebook
3. **`test_agent_setup.py`** - Quick test to verify setup
4. **`REACT_AGENT_DEMO_README.md`** - This guide

## 🚀 Quick Start

### Step 1: Verify Setup

First, ensure the agent is properly configured:

```bash
cd /c/codedev/llm/stats
uv run python examples/test_agent_setup.py
```

### Step 2: Download Models (if needed)

If models are not present:

```bash
# Download Gemma 2B model (recommended for demos)
uv run python -m src.gcp.gemma_download gemma-2b-it

# Or auto-select based on hardware
uv run python -m src.gcp.gemma_download --auto
```

### Step 3: Start Redis (for RAG features)

```bash
redis-server
```

### Step 4: Run the Demonstration

#### Option A: Command Line Demo

```bash
# Run in lightweight mode (faster)
uv run python examples/coding_agent_demo.py --mode lightweight

# Run with specific problems
uv run python examples/coding_agent_demo.py --problems 3

# Run in full mode with all features
uv run python examples/coding_agent_demo.py --mode full --verbose
```

#### Option B: Jupyter Notebook (Interactive)

```bash
# Start Jupyter
uv run jupyter notebook

# Open: examples/react_agent_coding_notebook.ipynb
```

## 🎯 Demonstration Features

### 1. Code Generation with Reflection
- **Problem**: Generate efficient algorithms (Fibonacci, sorting, etc.)
- **Agent Actions**:
  - Plans approach using algorithmic knowledge
  - Implements solution with proper documentation
  - Tests implementation
  - Reflects on performance and improvements

### 2. Bug Fixing with Step-by-Step Reasoning
- **Problem**: Fix bugs in provided code
- **Agent Actions**:
  - Analyzes code to identify issues
  - Understands failure scenarios
  - Implements fixes
  - Validates corrections

### 3. Code Review and Optimization
- **Problem**: Optimize slow or inefficient code
- **Agent Actions**:
  - Analyzes time/space complexity
  - Identifies bottlenecks
  - Proposes optimizations
  - Implements and compares performance

### 4. RAG-Enhanced Problem Solving
- **Problem**: Complex implementations requiring context
- **Agent Actions**:
  - Retrieves relevant documentation
  - Accesses design patterns
  - Synthesizes information
  - Creates production-ready solutions

## 🧠 Agent Architecture

### ReAct Loop
```
1. Thought → 2. Action → 3. Observation → 4. Reflection → (Repeat)
```

### Memory Tiers
- **Working Memory**: Current problem context (10 items)
- **Short-term**: Recent code and results (100 items)
- **Long-term**: Learned patterns (10K items)
- **Episodic**: Action sequences
- **Semantic**: Conceptual knowledge

### Available Tools
- `execute_code`: Run Python code safely
- `analyze_code`: Static analysis and metrics
- `retrieve_context`: RAG-based context retrieval
- `generate_tests`: Create test cases
- `calculate`: Mathematical operations
- `search_web`: Web search (simulated)

## 📊 Performance Metrics

The demonstrations track and report:

- **Reasoning Steps**: Number of thoughts generated
- **Tool Usage**: Which tools were used and how often
- **Time Taken**: Execution time per problem
- **Success Rate**: Problems solved successfully
- **Memory Usage**: RAM consumption
- **Token Count**: LLM tokens used

## 🔧 Configuration Options

### Agent Modes

```python
# Lightweight Mode (default)
AgentMode.LIGHTWEIGHT
- Faster responses
- Lower resource usage
- Good for simple problems

# Full Mode
AgentMode.FULL
- Deeper reasoning
- More comprehensive solutions
- Better for complex problems
```

### Planning & Reflection

```python
# Enable/disable features
enable_planning=True   # Strategic problem approach
enable_reflection=True # Self-improvement
```

### Model Selection

```python
# Available models
model_name="gemma-2b"    # Fast, lightweight
model_name="gemma-7b"    # Better quality
model_name="codegemma-2b" # Specialized for code
```

## 📈 Example Output

```
================================================================================
 PYTHON REACT AGENT WITH GEMMA - CODING DEMONSTRATION
================================================================================
Mode: lightweight
Problems to solve: 4
================================================================================

[1/4] Starting problem: Binary Search Implementation
============================================================
Solving: Binary Search Implementation
============================================================

THOUGHT: Analyzing the problem requirements...
ACTION: retrieve_context({"query": "binary search algorithm"})
OBSERVATION: Binary search is a divide-and-conquer algorithm...
THOUGHT: Planning implementation approach...
ACTION: execute_code({"code": "def binary_search(arr, target):..."})
OBSERVATION: Code executed successfully
REFLECTION: The implementation handles edge cases correctly...

✓ SOLVED
Time: 8.43 seconds
Reasoning Steps: 12
Tools Used: retrieve_context, execute_code, analyze_code
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure dependencies are installed
   cd /c/codedev/llm/stats
   uv sync --all-groups
   ```

2. **Model Not Found**
   ```bash
   # Download required model
   uv run python -m src.gcp.gemma_download gemma-2b-it
   ```

3. **Redis Connection Failed**
   ```bash
   # Start Redis server
   redis-server
   # Or disable RAG features in lightweight mode
   ```

4. **Out of Memory**
   ```bash
   # Use lightweight mode
   python examples/coding_agent_demo.py --mode lightweight
   # Or use smaller model
   ```

## 🎓 Learning Resources

### Understanding the Agent

1. **ReAct Pattern**: Reasoning + Acting in interleaved fashion
2. **Tool Use**: How the agent decides which tools to use
3. **Memory Management**: Multi-tier memory for context retention
4. **Planning**: Breaking complex problems into steps
5. **Reflection**: Learning from successes and failures

### Customization Points

- **Add Tools**: Register new tools in `setup_tools()`
- **Custom Problems**: Add to `create_coding_problems()`
- **Modify Prompts**: Edit system prompts in agent initialization
- **Adjust Parameters**: Temperature, max_iterations, etc.

## 📝 Example Problems Included

1. **Binary Search** - Algorithm implementation
2. **Rate Limiter** - System design with concurrency
3. **Memory Leak Fix** - Debugging and correction
4. **Database Query Optimization** - Performance improvement
5. **LRU Cache** - Complex data structure with thread safety

## 🚦 Next Steps

After running the demonstrations:

1. **Experiment**: Modify problems and observe agent behavior
2. **Extend**: Add new tools and problem types
3. **Integrate**: Use the agent in your own applications
4. **Fine-tune**: Adjust parameters for your use case
5. **Deploy**: Create an API service for the agent

## 📊 Performance Benchmarks

Typical performance on standard hardware:

| Problem Type | Lightweight Mode | Full Mode |
|-------------|-----------------|-----------|
| Simple Generation | 3-5 seconds | 5-10 seconds |
| Bug Fixing | 5-8 seconds | 10-15 seconds |
| Optimization | 8-12 seconds | 15-25 seconds |
| Complex w/ RAG | 10-15 seconds | 20-30 seconds |

## 🤝 Contributing

To add new demonstrations:

1. Create new problem in `create_coding_problems()`
2. Add custom tools if needed
3. Run and verify the solution
4. Document the new feature

## 📧 Support

For issues or questions:
- Check the troubleshooting section
- Review the agent logs for detailed errors
- Ensure all dependencies are correctly installed