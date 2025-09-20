"""ReAct (Reasoning and Acting) agent implementation with planning capabilities."""

import json
import re
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any

from ..shared.logging import get_logger
from .gemma_agent import AgentMode
from .gemma_agent import UnifiedGemmaAgent
from .planner import Plan
from .planner import Planner
from .planner import TaskComplexity
from .prompts import create_error_recovery_prompt
from .prompts import create_planning_prompt
from .prompts import create_react_system_prompt
from .prompts import create_reflection_prompt
from .tools import ToolRegistry

logger = get_logger(__name__)


class ThoughtType(Enum):
    """Types of thoughts in the reasoning process."""

    ANALYSIS = "analysis"
    PLANNING = "planning"
    ACTION = "action"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    ERROR = "error"
    ANSWER = "answer"


@dataclass
class ThoughtStep:
    """Represents a single step in the thought process."""

    type: ThoughtType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str | None = None


@dataclass
class ReActTrace:
    """Complete trace of ReAct reasoning process."""

    goal: str
    thoughts: list[ThoughtStep] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    plan: Plan | None = None
    final_answer: str | None = None
    success: bool = False

    def add_thought(self, thought_type: ThoughtType, content: str, **metadata: Any) -> None:
        """Add a thought to the trace."""
        self.thoughts.append(ThoughtStep(type=thought_type, content=content, metadata=metadata))

    def get_recent_context(self, n: int = 3) -> str:
        """Get recent context from trace."""
        context_parts = []

        # Get last n thoughts
        context_parts.extend(
            [f"{thought.type.value.upper()}: {thought.content}" for thought in self.thoughts[-n:]]
        )

        # Get last n actions and observations
        for action, observation in zip(self.actions[-n:], self.observations[-n:], strict=False):
            context_parts.append(f"ACTION: {json.dumps(action)}")
            context_parts.append(f"OBSERVATION: {observation}")

        return "\n".join(context_parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            "goal": self.goal,
            "thoughts": [
                {"type": t.type.value, "content": t.content, "metadata": t.metadata}
                for t in self.thoughts
            ],
            "actions": self.actions,
            "observations": self.observations,
            "reflections": self.reflections,
            "plan": self.plan.to_dict() if self.plan else None,
            "final_answer": self.final_answer,
            "success": self.success,
        }


class UnifiedReActAgent(UnifiedGemmaAgent):
    """Unified ReAct agent supporting both full and lightweight modes."""

    def __init__(
        self,
        model_name: str | None = None,
        mode: AgentMode = AgentMode.LIGHTWEIGHT,
        tool_registry: ToolRegistry | None = None,
        max_iterations: int = 10,
        verbose: bool = True,
        enable_planning: bool = True,
        enable_reflection: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize unified ReAct agent.

        Args:
            model_name: Name of the Gemma model to use
            mode: Agent operation mode (full or lightweight)
            tool_registry: Registry of available tools
            max_iterations: Maximum reasoning iterations
            verbose: Whether to print verbose output
            enable_planning: Whether to use planning for complex tasks
            enable_reflection: Whether to use reflection
            **kwargs: Additional arguments for UnifiedGemmaAgent
        """
        # Initialize base agent with ReAct system prompt
        super().__init__(
            model_name=model_name,
            mode=mode,
            tool_registry=tool_registry,
            system_prompt=None,  # Will be set later
            max_iterations=max_iterations,
            verbose=verbose,
            **kwargs,
        )

        # ReAct specific attributes
        self.enable_planning = enable_planning
        self.enable_reflection = enable_reflection
        self.planner = Planner(verbose=verbose)
        self.current_trace: ReActTrace | None = None
        self.trace_history: list[ReActTrace] = []

        # Update system prompt for ReAct
        tools_schemas = self.tool_registry.get_tool_schemas()
        self.system_prompt = create_react_system_prompt(tools_schemas)

    def think(self, prompt: str, trace: ReActTrace) -> str:
        """Generate a thought about the current situation.

        Args:
            prompt: Current context/prompt
            trace: Current trace

        Returns:
            Generated thought
        """
        thinking_prompt = f"""Given the current situation, provide your reasoning.

Current Goal: {trace.goal}

Recent Context:
{trace.get_recent_context()}

Current Prompt: {prompt}

Provide a THOUGHT about what needs to be done next. Be specific and analytical."""

        response = self.generate_response(thinking_prompt)

        # Extract thought from response
        thought_match = re.search(r"THOUGHT:\s*(.+?)(?:\n|$)", response, re.DOTALL)
        if thought_match:
            return thought_match.group(1).strip()
        return response.strip()

    def plan(self, task: str, context: str = "") -> Plan:
        """Create a plan for solving a task.

        Args:
            task: Task to plan for
            context: Additional context

        Returns:
            Generated plan
        """
        if self.verbose:
            print("📋 Generating plan...")

        # Get model to generate a plan
        tools_schemas = self.tool_registry.get_tool_schemas()
        planning_prompt = create_planning_prompt(task, context, tools_schemas)

        plan_response = self.generate_response(planning_prompt)

        # Create plan from response
        plan = self.planner.create_plan(
            task=task, context=context, tools_schemas=tools_schemas, model_response=plan_response
        )

        return plan

    def act(self, thought: str, trace: ReActTrace) -> tuple[dict[str, Any] | None, str | None]:
        """Decide on and execute an action based on thought.

        Args:
            thought: Current thought
            trace: Current trace

        Returns:
            Tuple of (action, observation)
        """
        # Generate action based on thought
        action_prompt = f"""Based on your reasoning, decide on an action.

Goal: {trace.goal}
Current Thought: {thought}

Recent Context:
{trace.get_recent_context()}

Available Tools:
{json.dumps(self.tool_registry.get_tool_schemas(), indent=2)}

Provide an ACTION in this exact JSON format:
{{"name": "tool_name", "arguments": {{"param": "value"}}}}

Or if no tool is needed, respond with:
{{"name": "none", "arguments": {{}}}}"""

        response = self.generate_response(action_prompt)

        # Parse action from response
        action = self._parse_action(response)

        if not action or action.get("name") == "none":
            return None, None

        # Execute action
        if self.verbose:
            print(f"🔧 Executing action: {action['name']}")

        result = self.tool_registry.execute_tool(action["name"], action.get("arguments", {}))

        observation = result.output if result.success else f"Error: {result.error}"

        return action, observation

    def reflect(self, trace: ReActTrace) -> str:
        """Reflect on progress and determine next steps.

        Args:
            trace: Current trace

        Returns:
            Reflection text
        """
        if not self.enable_reflection:
            return "Continuing with task..."

        reflection_prompt = create_reflection_prompt(
            current_state=trace.get_recent_context(),
            recent_actions=json.dumps(trace.actions[-3:]) if trace.actions else "No actions yet",
            observations=(
                "\n".join(trace.observations[-3:]) if trace.observations else "No observations yet"
            ),
            goal=trace.goal,
        )

        response = self.generate_response(reflection_prompt)

        # Extract reflection
        reflection_match = re.search(r"NEXT STEPS:\s*(.+?)$", response, re.DOTALL)
        if reflection_match:
            return reflection_match.group(1).strip()

        return response.strip()

    def _parse_action(self, response: str) -> dict[str, Any] | None:
        """Parse action from model response.

        Args:
            response: Model response

        Returns:
            Parsed action dictionary or None
        """
        # Look for ACTION marker
        action_match = re.search(r"ACTION:\s*(\{.+?\})", response, re.DOTALL)
        if action_match:
            try:
                parsed_action: dict[str, Any] = json.loads(action_match.group(1))
                return parsed_action
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse action JSON: {action_match.group(1)}")

        # Try to find any JSON in response
        json_match = re.search(r'\{.*?"name".*?:.*?".*?".*?\}', response, re.DOTALL)
        if json_match:
            try:
                parsed_json: dict[str, Any] = json.loads(json_match.group(0))
                return parsed_json
            except json.JSONDecodeError:
                logger.debug(f"Failed to parse JSON: {json_match.group(0)}")

        return None

    def solve(self, task: str, context: str = "", max_iterations: int | None = None) -> str:
        """Solve a task using ReAct approach.

        Args:
            task: Task to solve
            context: Additional context
            max_iterations: Maximum iterations (overrides default)

        Returns:
            Final answer
        """
        max_iter = max_iterations or self.max_iterations
        trace = ReActTrace(goal=task)
        self.current_trace = trace

        if self.verbose:
            print(f"\n🎯 Goal: {task}\n")

        # Step 1: Initial thought
        if self.verbose:
            print("💭 THINKING about the problem...")

        initial_thought = self.think(task, trace)
        trace.add_thought(ThoughtType.ANALYSIS, initial_thought)

        if self.verbose:
            print(
                f"   {initial_thought[:200]}..."
                if len(initial_thought) > 200
                else f"   {initial_thought}"
            )

        # Step 2: Planning (if enabled and task is complex)
        complexity = self.planner.analyze_complexity(task, context)
        if self.enable_planning and complexity in [
            TaskComplexity.COMPLEX,
            TaskComplexity.VERY_COMPLEX,
        ]:
            if self.verbose:
                print(f"\n📋 PLANNING (complexity: {complexity.value})...")

            plan = self.plan(task, context)
            trace.plan = plan
            trace.add_thought(ThoughtType.PLANNING, f"Created plan with {len(plan.steps)} steps")

            if self.verbose:
                print(f"   Generated {len(plan.steps)} step plan")

        # Step 3: Main reasoning loop
        iteration = 0
        error_count = 0
        max_errors = 3

        while iteration < max_iter:
            iteration += 1

            if self.verbose:
                print(f"\n🔄 Iteration {iteration}/{max_iter}")

            try:
                # Think about current situation
                current_thought = self.think(
                    f"Continue working on: {task}\nIteration: {iteration}", trace
                )
                trace.add_thought(ThoughtType.ANALYSIS, current_thought)

                if self.verbose:
                    print(f"💭 THOUGHT: {current_thought[:150]}...")

                # Check if we have an answer
                if "ANSWER:" in current_thought or "final answer" in current_thought.lower():
                    answer_match = re.search(r"ANSWER:\s*(.+?)$", current_thought, re.DOTALL)
                    if answer_match:
                        final_answer = answer_match.group(1).strip()
                    else:
                        final_answer = current_thought

                    trace.final_answer = final_answer
                    trace.success = True
                    trace.add_thought(ThoughtType.ANSWER, final_answer)

                    if self.verbose:
                        print(f"\n✅ ANSWER: {final_answer}")

                    break

                # Decide on action
                action, observation = self.act(current_thought, trace)

                if action:
                    trace.actions.append(action)
                    trace.add_thought(ThoughtType.ACTION, f"Executing {action['name']}")

                    if observation:
                        trace.observations.append(observation)
                        trace.add_thought(ThoughtType.OBSERVATION, observation[:200])

                        if self.verbose:
                            print(f"👁️ OBSERVATION: {observation[:150]}...")

                # Reflect on progress
                if iteration % 3 == 0 and self.enable_reflection:
                    if self.verbose:
                        print("🤔 REFLECTING on progress...")

                    reflection = self.reflect(trace)
                    trace.reflections.append(reflection)
                    trace.add_thought(ThoughtType.REFLECTION, reflection)

                    if self.verbose:
                        print(f"   {reflection[:150]}...")

                # Check if task seems complete
                if trace.plan and trace.plan.is_complete():
                    if self.verbose:
                        print("\n✅ Plan completed!")

                    # Generate final answer
                    final_prompt = f"""The plan for '{task}' is complete.

Results from executed steps:
{json.dumps([s.result for s in trace.plan.steps if s.result], indent=2)}

Provide a final ANSWER that summarizes the solution."""

                    final_response = self.generate_response(final_prompt)
                    answer_match = re.search(r"ANSWER:\s*(.+?)$", final_response, re.DOTALL)

                    if answer_match:
                        trace.final_answer = answer_match.group(1).strip()
                    else:
                        trace.final_answer = final_response

                    trace.success = True
                    break

            except Exception as e:
                error_count += 1
                logger.error(f"Error in iteration {iteration}: {e}")
                trace.add_thought(ThoughtType.ERROR, str(e))

                if error_count >= max_errors:
                    trace.final_answer = (
                        f"Failed to complete task after {error_count} errors. Last error: {e}"
                    )
                    trace.success = False
                    break

                # Try error recovery
                if self.verbose:
                    print("❌ Error occurred, attempting recovery...")

                recovery_prompt = create_error_recovery_prompt(
                    error_details=str(e),
                    context=trace.get_recent_context(),
                    recent_actions=json.dumps(trace.actions[-2:]) if trace.actions else "[]",
                )

                recovery_response = self.generate_response(recovery_prompt)
                trace.add_thought(
                    ThoughtType.REFLECTION, f"Error recovery: {recovery_response[:200]}"
                )

        # Finalize
        if not trace.final_answer:
            # Generate final answer based on trace
            summary_prompt = f"""Task: {task}

Based on the following work, provide a final answer:

Thoughts: {len(trace.thoughts)}
Actions taken: {len(trace.actions)}
Observations: {len(trace.observations)}

Recent context:
{trace.get_recent_context(5)}

Provide a concise ANSWER:"""

            final_response = self.generate_response(summary_prompt)
            answer_match = re.search(r"ANSWER:\s*(.+?)$", final_response, re.DOTALL)

            if answer_match:
                trace.final_answer = answer_match.group(1).strip()
            else:
                trace.final_answer = final_response

            trace.success = iteration < max_iter

        # Save trace
        self.trace_history.append(trace)

        if self.verbose:
            print(f"\n{'✅' if trace.success else '⚠️'} Task completed in {iteration} iterations")
            print(f"   Total thoughts: {len(trace.thoughts)}")
            print(f"   Total actions: {len(trace.actions)}")
            print(f"   Success: {trace.success}")

        return trace.final_answer or "Unable to complete task"

    def chat(self, user_input: str, use_tools: bool = True) -> str:
        """Override chat to use ReAct solving approach.

        Args:
            user_input: User input
            use_tools: Whether to use tools

        Returns:
            Agent response
        """
        if use_tools:
            return self.solve(user_input)
        else:
            # Fall back to base chat without ReAct
            return super().chat(user_input, use_tools=False)

    def get_trace_summary(self, trace_index: int = -1) -> str:
        """Get a summary of a reasoning trace.

        Args:
            trace_index: Index of trace to summarize (-1 for most recent)

        Returns:
            Formatted summary
        """
        if not self.trace_history:
            return "No traces available"

        trace = self.trace_history[trace_index]

        summary = f"""
ReAct Trace Summary
==================
Goal: {trace.goal}
Success: {trace.success}
Final Answer: {trace.final_answer or "No answer generated"}

Reasoning Process:
-----------------
Thoughts: {len(trace.thoughts)}
Actions: {len(trace.actions)}
Observations: {len(trace.observations)}
Reflections: {len(trace.reflections)}

Thought Types:
"""
        thought_counts: dict[str, int] = {}
        for thought in trace.thoughts:
            thought_counts[thought.type.value] = thought_counts.get(thought.type.value, 0) + 1

        for thought_type, count in thought_counts.items():
            summary += f"  - {thought_type}: {count}\n"

        if trace.plan:
            summary += f"""
Plan Details:
------------
Steps: {len(trace.plan.steps)}
Completed: {sum(1 for s in trace.plan.steps if s.completed)}
Progress: {trace.plan.get_progress():.1f}%
"""

        return summary

    def save_trace(self, filepath: str, trace_index: int = -1) -> None:
        """Save a reasoning trace to file.

        Args:
            filepath: Path to save file
            trace_index: Index of trace to save (-1 for most recent)
        """
        if not self.trace_history:
            raise ValueError("No traces to save")

        trace = self.trace_history[trace_index]
        trace_dict = trace.to_dict()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace_dict, f, indent=2)

        if self.verbose:
            print(f"Trace saved to {filepath}")


# Legacy classes for backwards compatibility
class ReActAgent(UnifiedReActAgent):
    """Legacy ReActAgent - use UnifiedReActAgent with mode=AgentMode.FULL instead."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(mode=AgentMode.FULL, **kwargs)


class LightweightReActAgent(UnifiedReActAgent):
    """Legacy LightweightReActAgent - use UnifiedReActAgent with mode=AgentMode.LIGHTWEIGHT instead."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(mode=AgentMode.LIGHTWEIGHT, **kwargs)


def create_react_agent(
    lightweight: bool = True,
    model_name: str | None = None,
    mode: AgentMode | None = None,
    enable_planning: bool = True,
    enable_reflection: bool = True,
    **kwargs: Any,
) -> UnifiedReActAgent:
    """Factory function to create a ReAct agent.

    Args:
        lightweight: Whether to use lightweight pipeline version (deprecated, use mode instead)
        model_name: Model name to use (auto-selected based on mode if None)
        mode: Agent operation mode (overrides lightweight parameter)
        enable_planning: Whether to enable planning
        enable_reflection: Whether to enable reflection
        **kwargs: Additional arguments

    Returns:
        Configured ReAct agent
    """
    # Determine mode
    if mode is not None:
        agent_mode = mode
    else:
        agent_mode = AgentMode.LIGHTWEIGHT if lightweight else AgentMode.FULL

    return UnifiedReActAgent(
        model_name=model_name,
        mode=agent_mode,
        enable_planning=enable_planning,
        enable_reflection=enable_reflection,
        **kwargs,
    )
