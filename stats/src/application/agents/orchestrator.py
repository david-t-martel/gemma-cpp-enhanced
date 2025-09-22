"""Parallel agent orchestration system for the LLM framework."""

import asyncio
import ast
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field

from src.domain.tools.base import ToolExecutionContext
from src.domain.tools.base import ToolRegistry
from src.domain.tools.base import ToolResult
from src.infrastructure.sandbox.docker import DockerSandbox
from src.infrastructure.sandbox.process import ProcessSandbox
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


class AgentState(str, Enum):
    """Agent execution states."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutionMode(str, Enum):
    """Execution modes."""

    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    PIPELINE = "pipeline"
    CONDITIONAL = "conditional"


@dataclass
class AgentCapability:
    """Agent capability definition."""

    name: str
    description: str
    input_types: list[str]
    output_types: list[str]
    max_parallel: int = 1
    timeout: int = 300
    retry_count: int = 3
    resources_required: dict[str, Any] = None


class AgentTask(BaseModel):
    """Task for agent execution."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: int | None = None
    retry_count: int = 3
    dependencies: list[str] = Field(default_factory=list)
    context: ToolExecutionContext | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class AgentResult(BaseModel):
    """Result from agent execution."""

    task_id: str
    agent_id: str
    success: bool
    result: ToolResult | None = None
    error: str | None = None
    execution_time: float
    retries_used: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    completed_at: float = Field(default_factory=time.time)


class Agent(BaseModel):
    """Agent definition."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    capabilities: list[AgentCapability] = Field(default_factory=list)
    max_concurrent_tasks: int = 1
    state: AgentState = AgentState.IDLE
    current_tasks: list[str] = Field(default_factory=list)
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_execution_time: float = 0
    last_activity: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def can_execute(self, task: AgentTask) -> bool:
        """Check if agent can execute a task."""
        if self.state in [AgentState.FAILED, AgentState.CANCELLED]:
            return False

        if len(self.current_tasks) >= self.max_concurrent_tasks:
            return False

        # Check capabilities
        for capability in self.capabilities:
            if capability.name == task.tool_name:
                return len(self.current_tasks) < capability.max_parallel

        return False

    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to the agent."""
        self.capabilities.append(capability)

    def get_capability(self, name: str) -> AgentCapability | None:
        """Get capability by name."""
        for capability in self.capabilities:
            if capability.name == name:
                return capability
        return None


class WorkflowStep(BaseModel):
    """Workflow execution step."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    tasks: list[AgentTask]
    mode: ExecutionMode = ExecutionMode.PARALLEL
    condition: str | None = None  # Python expression
    depends_on: list[str] = Field(default_factory=list)
    timeout: int | None = None
    retry_on_failure: bool = False
    continue_on_failure: bool = False


class Workflow(BaseModel):
    """Agent workflow definition."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    steps: list[WorkflowStep]
    timeout: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def validate_workflow(self) -> list[str]:
        """Validate workflow structure."""
        errors = []
        step_ids = {step.id for step in self.steps}

        for step in self.steps:
            # Check dependencies
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(f"Step {step.name} depends on non-existent step {dep}")

        # Check for circular dependencies
        if self._has_circular_dependencies():
            errors.append("Workflow has circular dependencies")

        return errors

    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies in workflow."""
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()

        def dfs(step_id: str) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)

            # Find step by id
            step = next((s for s in self.steps if s.id == step_id), None)
            if not step:
                return False

            for dep in step.depends_on:
                if dep not in visited:
                    if dfs(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(step_id)
            return False

        return any(step.id not in visited and dfs(step.id) for step in self.steps)


class ExecutionPlan(BaseModel):
    """Execution plan for tasks and workflows."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    tasks: list[AgentTask] = Field(default_factory=list)
    workflow: Workflow | None = None
    mode: ExecutionMode = ExecutionMode.PARALLEL
    max_concurrent: int = 10
    timeout: int | None = None
    priority_order: bool = True
    created_at: float = Field(default_factory=time.time)

    def add_task(self, task: AgentTask) -> None:
        """Add a task to the execution plan."""
        self.tasks.append(task)

    def get_ready_tasks(self, completed_task_ids: set) -> list[AgentTask]:
        """Get tasks that are ready to execute."""
        ready_tasks = []

        for task in self.tasks:
            if task.id in completed_task_ids:
                continue

            # Check dependencies
            dependencies_met = all(dep in completed_task_ids for dep in task.dependencies)

            if dependencies_met:
                ready_tasks.append(task)

        # Sort by priority if enabled
        if self.priority_order:
            priority_order = {
                TaskPriority.CRITICAL: 0,
                TaskPriority.HIGH: 1,
                TaskPriority.NORMAL: 2,
                TaskPriority.LOW: 3,
            }
            ready_tasks.sort(key=lambda t: priority_order.get(t.priority, 2))

        return ready_tasks


class AgentOrchestrator:
    """Orchestrate parallel agent execution."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        max_agents: int = 10,
        docker_sandbox: DockerSandbox | None = None,
        process_sandbox: ProcessSandbox | None = None,
    ):
        self.tool_registry = tool_registry
        self.max_agents = max_agents
        self.docker_sandbox = docker_sandbox
        self.process_sandbox = process_sandbox

        self._agents: dict[str, Agent] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._result_queue: asyncio.Queue = asyncio.Queue()
        self._active_executions: dict[str, asyncio.Task] = {}
        self._execution_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        self._stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0,
            "agents_created": 0,
            "workflows_executed": 0,
        }

    async def initialize(self) -> None:
        """Initialize the orchestrator."""
        # Initialize sandboxes
        if self.docker_sandbox:
            await self.docker_sandbox.initialize()

        if self.process_sandbox:
            await self.process_sandbox.initialize()

        # Create default agents based on available tools
        await self._create_default_agents()

        logger.info(f"Agent orchestrator initialized with {len(self._agents)} agents")

    async def _create_default_agents(self) -> None:
        """Create default agents for available tools."""
        tool_schemas = self.tool_registry.get_schemas()

        # Group tools by category
        tool_categories = {}
        for schema in tool_schemas:
            category = schema.category
            if category not in tool_categories:
                tool_categories[category] = []
            tool_categories[category].append(schema)

        # Create agents for each category
        for category, schemas in tool_categories.items():
            agent = Agent(
                name=f"{category.value}_agent",
                description=f"Agent specialized in {category.value} operations",
                max_concurrent_tasks=3,
            )

            # Add capabilities for each tool
            for schema in schemas:
                capability = AgentCapability(
                    name=schema.name,
                    description=schema.description,
                    input_types=[param.type for param in schema.parameters],
                    output_types=["any"],  # Simplified
                    max_parallel=2 if schema.async_capable else 1,
                    timeout=schema.resource_limits.max_execution_time_seconds or 300,
                    retry_count=3,
                )
                agent.add_capability(capability)

            await self.add_agent(agent)

    async def add_agent(self, agent: Agent) -> None:
        """Add an agent to the orchestrator."""
        async with self._execution_lock:
            if len(self._agents) >= self.max_agents:
                raise ValueError(f"Maximum number of agents ({self.max_agents}) reached")

            self._agents[agent.id] = agent
            self._stats["agents_created"] += 1
            logger.info(f"Added agent: {agent.name} ({agent.id})")

    async def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the orchestrator."""
        async with self._execution_lock:
            if agent_id not in self._agents:
                return False

            agent = self._agents[agent_id]

            # Cancel running tasks
            for task_id in agent.current_tasks:
                if task_id in self._active_executions:
                    self._active_executions[task_id].cancel()

            agent.state = AgentState.CANCELLED
            del self._agents[agent_id]
            logger.info(f"Removed agent: {agent.name} ({agent_id})")
            return True

    async def execute_task(self, task: AgentTask) -> AgentResult:
        """Execute a single task."""
        return await self._execute_single_task(task)

    async def execute_tasks(self, tasks: list[AgentTask]) -> list[AgentResult]:
        """Execute multiple tasks in parallel."""
        plan = ExecutionPlan(tasks=tasks, mode=ExecutionMode.PARALLEL)
        return await self.execute_plan(plan)

    async def execute_workflow(self, workflow: Workflow) -> list[AgentResult]:
        """Execute a workflow."""
        # Validate workflow
        errors = workflow.validate_workflow()
        if errors:
            raise ValueError(f"Invalid workflow: {errors}")

        self._stats["workflows_executed"] += 1
        results = []

        try:
            # Execute steps in dependency order
            completed_steps = set()
            step_results = {}

            while len(completed_steps) < len(workflow.steps):
                # Find ready steps
                ready_steps = []
                for step in workflow.steps:
                    if step.id in completed_steps:
                        continue

                    dependencies_met = all(dep in completed_steps for dep in step.depends_on)
                    if dependencies_met:
                        # Check condition if specified
                        if step.condition:
                            try:
                                # Safe condition evaluation using simpleeval-like logic
                                condition_met = self._safe_evaluate_condition(step.condition, {"results": step_results})
                                if not condition_met:
                                    completed_steps.add(step.id)
                                    continue
                            except Exception as e:
                                logger.warning(
                                    f"Failed to evaluate condition for step {step.name}: {e}"
                                )
                                continue

                        ready_steps.append(step)

                if not ready_steps:
                    # No more steps can be executed
                    remaining_steps = [
                        s.name for s in workflow.steps if s.id not in completed_steps
                    ]
                    logger.warning(f"Workflow stalled. Remaining steps: {remaining_steps}")
                    break

                # Execute ready steps
                step_tasks = []
                for step in ready_steps:
                    if step.mode == ExecutionMode.PARALLEL:
                        step_tasks.extend(step.tasks)
                    else:  # Sequential
                        # Execute tasks one by one
                        for task in step.tasks:
                            try:
                                result = await self._execute_single_task(task)
                                results.append(result)
                            except Exception as e:
                                if not step.continue_on_failure:
                                    raise
                                logger.warning(f"Task failed but continuing: {e}")

                # Execute parallel tasks if any
                if step_tasks:
                    try:
                        step_results_list = await self._execute_tasks_parallel(step_tasks)
                        results.extend(step_results_list)

                        # Store step results for condition evaluation
                        for step in ready_steps:
                            step_results[step.id] = step_results_list

                    except Exception as e:
                        # Check if any ready step allows failure
                        should_continue = any(step.continue_on_failure for step in ready_steps)
                        if not should_continue:
                            raise
                        logger.warning(f"Step failed but continuing: {e}")

                # Mark steps as completed
                completed_steps.update(step.id for step in ready_steps)

            return results

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise

    async def execute_plan(self, plan: ExecutionPlan) -> list[AgentResult]:
        """Execute an execution plan."""
        if plan.workflow:
            return await self.execute_workflow(plan.workflow)
        else:
            return await self._execute_plan_tasks(plan)

    async def _execute_plan_tasks(self, plan: ExecutionPlan) -> list[AgentResult]:
        """Execute tasks from an execution plan."""
        if plan.mode == ExecutionMode.SEQUENTIAL:
            return await self._execute_tasks_sequential(plan.tasks)
        else:  # PARALLEL or PIPELINE
            return await self._execute_tasks_parallel(plan.tasks)

    async def _execute_tasks_sequential(self, tasks: list[AgentTask]) -> list[AgentResult]:
        """Execute tasks sequentially."""
        results = []
        for task in tasks:
            try:
                result = await self._execute_single_task(task)
                results.append(result)

                # Stop on first failure unless configured otherwise
                if not result.success:
                    logger.warning(f"Sequential execution stopped due to task failure: {task.id}")
                    break

            except Exception as e:
                logger.error(f"Sequential task execution failed: {e}")
                break

        return results

    async def _execute_tasks_parallel(self, tasks: list[AgentTask]) -> list[AgentResult]:
        """Execute tasks in parallel."""
        # Create tasks for parallel execution
        execution_tasks = []
        for task in tasks:
            execution_tasks.append(asyncio.create_task(self._execute_single_task(task)))

        try:
            # Wait for all tasks to complete
            results = await asyncio.gather(*execution_tasks, return_exceptions=True)

            # Process results
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Create error result
                    error_result = AgentResult(
                        task_id=tasks[i].id,
                        agent_id="unknown",
                        success=False,
                        error=str(result),
                        execution_time=0,
                    )
                    final_results.append(error_result)
                else:
                    final_results.append(result)

            return final_results

        except Exception as e:
            logger.error(f"Parallel task execution failed: {e}")
            raise

    async def _execute_single_task(self, task: AgentTask) -> AgentResult:
        """Execute a single task."""
        start_time = time.time()
        retries_used = 0

        # Find suitable agent
        agent = await self._find_agent_for_task(task)
        if not agent:
            return AgentResult(
                task_id=task.id,
                agent_id="none",
                success=False,
                error="No suitable agent found",
                execution_time=time.time() - start_time,
            )

        # Execute with retries
        last_error = None
        for attempt in range(task.retry_count + 1):
            try:
                # Update agent state
                agent.state = AgentState.RUNNING
                agent.current_tasks.append(task.id)
                agent.last_activity = time.time()

                # Create execution context if not provided
                if not task.context:
                    task.context = ToolExecutionContext(
                        agent_id=agent.id,
                        timeout=task.timeout,
                        sandbox_mode=True,
                        metadata={"orchestrator": True, **task.metadata},
                    )

                # Execute the tool
                tool_result = await self.tool_registry.execute_tool(
                    task.tool_name, task.context, **task.parameters
                )

                # Update agent state
                agent.current_tasks.remove(task.id)
                if tool_result.success:
                    agent.completed_tasks += 1
                    agent.state = AgentState.IDLE
                    self._stats["tasks_completed"] += 1
                else:
                    agent.failed_tasks += 1
                    if attempt == task.retry_count:
                        agent.state = AgentState.IDLE
                        self._stats["tasks_failed"] += 1

                execution_time = time.time() - start_time
                agent.total_execution_time += execution_time
                self._stats["total_execution_time"] += execution_time

                return AgentResult(
                    task_id=task.id,
                    agent_id=agent.id,
                    success=tool_result.success,
                    result=tool_result,
                    error=tool_result.error if not tool_result.success else None,
                    execution_time=execution_time,
                    retries_used=retries_used,
                )

            except Exception as e:
                last_error = str(e)
                retries_used += 1
                logger.warning(f"Task execution attempt {attempt + 1} failed: {e}")

                if attempt < task.retry_count:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    # Final failure
                    agent.current_tasks.remove(task.id)
                    agent.failed_tasks += 1
                    agent.state = AgentState.FAILED
                    self._stats["tasks_failed"] += 1

        execution_time = time.time() - start_time
        return AgentResult(
            task_id=task.id,
            agent_id=agent.id,
            success=False,
            error=last_error or "Unknown error",
            execution_time=execution_time,
            retries_used=retries_used,
        )

    async def _find_agent_for_task(self, task: AgentTask) -> Agent | None:
        """Find a suitable agent for executing a task."""
        suitable_agents = []

        for agent in self._agents.values():
            if agent.can_execute(task):
                capability = agent.get_capability(task.tool_name)
                if capability:
                    # Score agent based on current load and capability match
                    load_factor = len(agent.current_tasks) / agent.max_concurrent_tasks
                    score = 1.0 - load_factor  # Lower load = higher score

                    suitable_agents.append((agent, score))

        if not suitable_agents:
            return None

        # Sort by score (highest first) and return best agent
        suitable_agents.sort(key=lambda x: x[1], reverse=True)
        return suitable_agents[0][0]

    def _safe_evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate a condition string without using eval().

        This implements a simple expression evaluator that only supports
        basic comparison operations and prevents arbitrary code execution.

        Args:
            condition: Condition string to evaluate (e.g., "results['step1']['success'] == True")
            context: Context dictionary for variable resolution

        Returns:
            Boolean result of condition evaluation

        Raises:
            ValueError: If condition contains unsafe operations
        """

        # Clean the condition string
        condition = condition.strip()

        # Basic security check - reject obviously dangerous patterns
        dangerous_patterns = [
            r'__import__', r'exec\s*\(', r'eval\s*\(', r'open\s*\(',
            r'file\s*\(', r'input\s*\(', r'raw_input\s*\(',
            r'compile\s*\(', r'globals\s*\(', r'locals\s*\(',
            r'vars\s*\(', r'dir\s*\(', r'hasattr\s*\(', r'getattr\s*\(',
            r'setattr\s*\(', r'delattr\s*\(', r'callable\s*\(',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, condition, re.IGNORECASE):
                raise ValueError(f"Unsafe operation detected in condition: {condition}")

        # Try to parse as a literal expression first
        try:
            # For simple literals like "True", "False", "123", etc.
            return ast.literal_eval(condition)
        except (ValueError, SyntaxError):
            pass

        # Handle simple variable references and comparisons
        try:
            # Parse the condition as an AST
            parsed = ast.parse(condition, mode='eval')

            # Evaluate using a restricted evaluator
            return self._evaluate_ast_node(parsed.body, context)

        except Exception as e:
            raise ValueError(f"Failed to safely evaluate condition '{condition}': {e}")

    def _evaluate_ast_node(self, node: ast.AST, context: Dict[str, Any]) -> Any:
        """Evaluate an AST node safely.

        Args:
            node: AST node to evaluate
            context: Context for variable resolution

        Returns:
            Evaluated value

        Raises:
            ValueError: If node type is not supported
        """
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in context:
                return context[node.id]
            raise NameError(f"Name '{node.id}' is not defined")
        elif isinstance(node, ast.Subscript):
            # Handle dictionary/list access like results['step1']
            value = self._evaluate_ast_node(node.value, context)
            slice_value = self._evaluate_ast_node(node.slice, context)
            return value[slice_value]
        elif isinstance(node, ast.Compare):
            # Handle comparisons like ==, !=, <, >, etc.
            left = self._evaluate_ast_node(node.left, context)

            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ValueError("Only simple comparisons supported")

            op = node.ops[0]
            right = self._evaluate_ast_node(node.comparators[0], context)

            if isinstance(op, ast.Eq):
                return left == right
            elif isinstance(op, ast.NotEq):
                return left != right
            elif isinstance(op, ast.Lt):
                return left < right
            elif isinstance(op, ast.LtE):
                return left <= right
            elif isinstance(op, ast.Gt):
                return left > right
            elif isinstance(op, ast.GtE):
                return left >= right
            elif isinstance(op, ast.Is):
                return left is right
            elif isinstance(op, ast.IsNot):
                return left is not right
            elif isinstance(op, ast.In):
                return left in right
            elif isinstance(op, ast.NotIn):
                return left not in right
            else:
                raise ValueError(f"Unsupported comparison operator: {type(op)}")
        elif isinstance(node, ast.BoolOp):
            # Handle boolean operations like 'and', 'or'
            if isinstance(node.op, ast.And):
                return all(self._evaluate_ast_node(value, context) for value in node.values)
            elif isinstance(node.op, ast.Or):
                return any(self._evaluate_ast_node(value, context) for value in node.values)
            else:
                raise ValueError(f"Unsupported boolean operator: {type(node.op)}")
        elif isinstance(node, ast.UnaryOp):
            # Handle unary operations like 'not'
            operand = self._evaluate_ast_node(node.operand, context)
            if isinstance(node.op, ast.Not):
                return not operand
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")
        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    async def get_agent_status(
        self, agent_id: str | None = None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Get status of agents."""
        if agent_id:
            agent = self._agents.get(agent_id)
            if not agent:
                return {}

            return {
                "id": agent.id,
                "name": agent.name,
                "state": agent.state,
                "current_tasks": len(agent.current_tasks),
                "completed_tasks": agent.completed_tasks,
                "failed_tasks": agent.failed_tasks,
                "success_rate": (
                    agent.completed_tasks / (agent.completed_tasks + agent.failed_tasks)
                    if (agent.completed_tasks + agent.failed_tasks) > 0
                    else 0
                ),
                "total_execution_time": agent.total_execution_time,
                "last_activity": agent.last_activity,
                "capabilities": [cap.name for cap in agent.capabilities],
            }
        else:
            return [await self.get_agent_status(agent_id) for agent_id in self._agents]

    async def get_system_stats(self) -> dict[str, Any]:
        """Get system statistics."""
        active_agents = sum(
            1 for agent in self._agents.values() if agent.state == AgentState.RUNNING
        )
        total_current_tasks = sum(len(agent.current_tasks) for agent in self._agents.values())

        return {
            **self._stats,
            "total_agents": len(self._agents),
            "active_agents": active_agents,
            "current_tasks": total_current_tasks,
            "success_rate": (
                self._stats["tasks_completed"]
                / (self._stats["tasks_completed"] + self._stats["tasks_failed"])
                if (self._stats["tasks_completed"] + self._stats["tasks_failed"]) > 0
                else 0
            ),
            "average_execution_time": (
                self._stats["total_execution_time"] / self._stats["tasks_completed"]
                if self._stats["tasks_completed"] > 0
                else 0
            ),
        }

    async def shutdown(self) -> None:
        """Shutdown the orchestrator."""
        logger.info("Shutting down agent orchestrator")
        self._shutdown_event.set()

        # Cancel all active executions
        for task in self._active_executions.values():
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._active_executions.values(), return_exceptions=True)

        # Set all agents to cancelled state
        for agent in self._agents.values():
            agent.state = AgentState.CANCELLED
            agent.current_tasks.clear()

        # Cleanup sandboxes
        if self.docker_sandbox:
            await self.docker_sandbox.cleanup()

        if self.process_sandbox:
            await self.process_sandbox.cleanup()

        logger.info("Agent orchestrator shutdown complete")
