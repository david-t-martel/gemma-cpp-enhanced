"""Planning module for task decomposition and strategy generation."""

import re
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any


class TaskComplexity(Enum):
    """Task complexity levels."""

    SIMPLE = "simple"  # Single action, direct answer
    MEDIUM = "medium"  # 2-3 steps, some coordination
    COMPLEX = "complex"  # Multiple steps, dependencies
    VERY_COMPLEX = "very_complex"  # Many steps, planning required


@dataclass
class Step:
    """Represents a single step in a plan."""

    id: int
    description: str
    action: str | None = None
    tools_needed: list[str] = field(default_factory=list)
    dependencies: list[int] = field(default_factory=list)
    expected_outcome: str = ""
    completed: bool = False
    result: Any | None = None


@dataclass
class Plan:
    """Represents a complete plan for solving a task."""

    goal: str
    steps: list[Step]
    current_step: int = 0
    contingencies: dict[str, str] = field(default_factory=dict)
    success_criteria: list[str] = field(default_factory=list)
    complexity: TaskComplexity = TaskComplexity.MEDIUM
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_current_step(self) -> Step | None:
        """Get the current step to execute."""
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    def mark_step_complete(self, result: Any = None) -> None:
        """Mark the current step as complete and move to next."""
        if current := self.get_current_step():
            current.completed = True
            current.result = result
            self.current_step += 1

    def get_next_actionable_step(self) -> Step | None:
        """Get the next step that can be executed (dependencies met)."""
        for step in self.steps[self.current_step :]:
            if not step.completed and self._dependencies_met(step):
                return step
        return None

    def _dependencies_met(self, step: Step) -> bool:
        """Check if all dependencies for a step are met."""
        for dep_id in step.dependencies:
            if dep_id < len(self.steps) and not self.steps[dep_id].completed:
                return False
        return True

    def is_complete(self) -> bool:
        """Check if the plan is complete."""
        return all(step.completed for step in self.steps)

    def get_progress(self) -> float:
        """Get plan completion percentage."""
        if not self.steps:
            return 0.0
        completed = sum(1 for step in self.steps if step.completed)
        return (completed / len(self.steps)) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert plan to dictionary format."""
        return {
            "goal": self.goal,
            "complexity": self.complexity.value,
            "progress": self.get_progress(),
            "current_step": self.current_step,
            "steps": [
                {
                    "id": step.id,
                    "description": step.description,
                    "completed": step.completed,
                    "tools_needed": step.tools_needed,
                    "dependencies": step.dependencies,
                }
                for step in self.steps
            ],
            "contingencies": self.contingencies,
            "success_criteria": self.success_criteria,
        }


class Planner:
    """Strategic planner for task decomposition and plan generation."""

    def __init__(self, verbose: bool = True):
        """Initialize the planner.

        Args:
            verbose: Whether to print planning details
        """
        self.verbose = verbose
        self.current_plan: Plan | None = None
        self.plan_history: list[Plan] = []

    def analyze_complexity(self, task: str, context: str = "") -> TaskComplexity:
        """Analyze the complexity of a task.

        Args:
            task: Task description
            context: Additional context

        Returns:
            TaskComplexity level
        """
        # Simple heuristics for complexity assessment
        task_lower = task.lower()

        # Check for complexity indicators
        complex_keywords = ["analyze", "compare", "research", "implement", "design", "optimize"]
        multi_step_keywords = ["and then", "after that", "followed by", "multiple", "several"]
        simple_keywords = ["what is", "calculate", "find", "get", "show", "display"]

        # Count potential steps
        sentences = re.split(r"[.!?]", task)
        num_sentences = len([s for s in sentences if s.strip()])

        # Analyze
        if any(keyword in task_lower for keyword in complex_keywords):
            if any(keyword in task_lower for keyword in multi_step_keywords) or num_sentences > 3:
                return TaskComplexity.VERY_COMPLEX
            return TaskComplexity.COMPLEX
        elif any(keyword in task_lower for keyword in multi_step_keywords) or num_sentences > 2:
            return TaskComplexity.MEDIUM
        elif any(keyword in task_lower for keyword in simple_keywords) or num_sentences == 1:
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.MEDIUM

    def create_plan(
        self,
        task: str,
        context: str = "",
        tools_schemas: list[dict[str, Any]] | None = None,
        model_response: str | None = None,
    ) -> Plan:
        """Create a plan for solving a task.

        Args:
            task: Task to plan for
            context: Additional context
            tools_schemas: Available tool schemas
            model_response: Optional model-generated plan

        Returns:
            Generated Plan object
        """
        complexity = self.analyze_complexity(task, context)

        if self.verbose:
            print(f"📋 Creating plan for task (complexity: {complexity.value})")

        if model_response:
            # Parse model-generated plan
            plan = self._parse_model_plan(model_response, task, complexity)
        else:
            # Create a basic plan based on complexity
            plan = self._create_basic_plan(task, complexity, tools_schemas)

        self.current_plan = plan
        self.plan_history.append(plan)

        if self.verbose:
            print(f"✅ Plan created with {len(plan.steps)} steps")
            self._print_plan(plan)

        return plan

    def _parse_model_plan(self, response: str, task: str, complexity: TaskComplexity) -> Plan:
        """Parse a model-generated plan from response text.

        Args:
            response: Model response containing plan
            task: Original task
            complexity: Task complexity

        Returns:
            Parsed Plan object
        """
        plan = Plan(goal=task, steps=[], complexity=complexity)

        # Extract goal
        goal_match = re.search(r"GOAL:\s*(.+?)(?:\n|$)", response)
        if goal_match:
            plan.goal = goal_match.group(1).strip()

        # Extract steps
        steps_section = re.search(
            r"STEPS:(.*?)(?:CONTINGENCIES:|SUCCESS CRITERIA:|$)", response, re.DOTALL
        )
        if steps_section:
            step_pattern = (
                r"(\d+)\.\s*(.+?)"
                r"(?:- Expected outcome:\s*(.+?))?"
                r"(?:- Tools needed:\s*(.+?))?"
                r"(?:- Dependencies:\s*(.+?))?"
                r"(?=\d+\.|CONTINGENCIES:|SUCCESS CRITERIA:|$)"
            )
            matches = re.finditer(step_pattern, steps_section.group(1), re.DOTALL)

            for match in matches:
                step_id = int(match.group(1)) - 1
                description = match.group(2).strip()
                expected = match.group(3).strip() if match.group(3) else ""
                tools = [t.strip() for t in match.group(4).split(",")] if match.group(4) else []
                deps = []
                if match.group(5):
                    deps_text = match.group(5).strip()
                    dep_numbers = re.findall(r"\d+", deps_text)
                    deps = [int(d) - 1 for d in dep_numbers]

                step = Step(
                    id=step_id,
                    description=description,
                    tools_needed=tools,
                    dependencies=deps,
                    expected_outcome=expected,
                )
                plan.steps.append(step)

        # Extract contingencies
        contingencies_section = re.search(
            r"CONTINGENCIES:(.*?)(?:SUCCESS CRITERIA:|$)", response, re.DOTALL
        )
        if contingencies_section:
            contingency_pattern = r"- If (.+?), then (.+?)(?:\n|$)"
            for match in re.finditer(contingency_pattern, contingencies_section.group(1)):
                condition = match.group(1).strip()
                action = match.group(2).strip()
                plan.contingencies[condition] = action

        # Extract success criteria
        criteria_section = re.search(r"SUCCESS CRITERIA:(.*?)$", response, re.DOTALL)
        if criteria_section:
            criteria_pattern = r"- (.+?)(?:\n|$)"
            for match in re.finditer(criteria_pattern, criteria_section.group(1)):
                plan.success_criteria.append(match.group(1).strip())

        # If no steps were parsed, create a basic step
        if not plan.steps:
            plan.steps.append(
                Step(
                    id=0,
                    description=f"Execute task: {task}",
                    expected_outcome="Task completed successfully",
                )
            )

        return plan

    def _create_basic_plan(
        self,
        task: str,
        complexity: TaskComplexity,
        tools_schemas: list[dict[str, Any]] | None = None,
    ) -> Plan:
        """Create a basic plan based on task complexity.

        Args:
            task: Task description
            complexity: Task complexity level
            tools_schemas: Available tools

        Returns:
            Basic Plan object
        """
        plan = Plan(goal=task, steps=[], complexity=complexity)

        if complexity == TaskComplexity.SIMPLE:
            # Single step plan
            plan.steps.append(
                Step(id=0, description=f"Execute task: {task}", expected_outcome="Task completed")
            )
        elif complexity == TaskComplexity.MEDIUM:
            # 2-3 step plan
            plan.steps.extend(
                [
                    Step(
                        id=0,
                        description="Analyze requirements",
                        expected_outcome="Clear understanding of task",
                    ),
                    Step(
                        id=1,
                        description="Execute main action",
                        dependencies=[0],
                        expected_outcome="Primary goal achieved",
                    ),
                    Step(
                        id=2,
                        description="Verify and summarize results",
                        dependencies=[1],
                        expected_outcome="Complete solution provided",
                    ),
                ]
            )
        else:  # COMPLEX or VERY_COMPLEX
            # Multi-step plan
            plan.steps.extend(
                [
                    Step(
                        id=0,
                        description="Analyze and decompose task",
                        expected_outcome="Task broken into subtasks",
                    ),
                    Step(
                        id=1,
                        description="Gather necessary information",
                        dependencies=[0],
                        expected_outcome="All required data collected",
                    ),
                    Step(
                        id=2,
                        description="Process and analyze data",
                        dependencies=[1],
                        expected_outcome="Data analyzed and insights gained",
                    ),
                    Step(
                        id=3,
                        description="Generate solution",
                        dependencies=[2],
                        expected_outcome="Solution created",
                    ),
                    Step(
                        id=4,
                        description="Validate and refine",
                        dependencies=[3],
                        expected_outcome="Solution validated",
                    ),
                    Step(
                        id=5,
                        description="Present final results",
                        dependencies=[4],
                        expected_outcome="Complete solution delivered",
                    ),
                ]
            )

        # Add success criteria
        plan.success_criteria = [
            "Task objective achieved",
            "Results are accurate and complete",
            "Solution is clearly presented",
        ]

        return plan

    def revise_plan(self, feedback: str, error: str | None = None) -> Plan:
        """Revise the current plan based on feedback or errors.

        Args:
            feedback: Feedback or observations
            error: Error message if applicable

        Returns:
            Revised Plan object
        """
        if not self.current_plan:
            raise ValueError("No current plan to revise")

        if self.verbose:
            print("🔄 Revising plan based on feedback")

        # Create a copy of the current plan
        revised_plan = Plan(
            goal=self.current_plan.goal,
            steps=self.current_plan.steps.copy(),
            current_step=self.current_plan.current_step,
            contingencies=self.current_plan.contingencies.copy(),
            success_criteria=self.current_plan.success_criteria.copy(),
            complexity=self.current_plan.complexity,
        )

        if error:
            # Add error recovery step
            recovery_step = Step(
                id=len(revised_plan.steps),
                description=f"Recover from error: {error[:100]}",
                expected_outcome="Error resolved, continue with task",
            )
            revised_plan.steps.insert(revised_plan.current_step, recovery_step)

        # Update plan metadata
        revised_plan.metadata["revision_count"] = revised_plan.metadata.get("revision_count", 0) + 1
        revised_plan.metadata["last_revision_reason"] = feedback[:200]

        self.current_plan = revised_plan
        self.plan_history.append(revised_plan)

        return revised_plan

    def _print_plan(self, plan: Plan) -> None:
        """Print plan details for debugging."""
        print(f"\n📋 Plan: {plan.goal}")
        print(f"   Complexity: {plan.complexity.value}")
        print(f"   Steps ({len(plan.steps)}):")
        for step in plan.steps:
            status = "✅" if step.completed else "⏳"
            deps = f" (depends on: {step.dependencies})" if step.dependencies else ""
            print(f"   {status} {step.id + 1}. {step.description}{deps}")
        if plan.contingencies:
            print(f"   Contingencies: {len(plan.contingencies)}")
        print(f"   Progress: {plan.get_progress():.1f}%\n")

    def get_next_action(self) -> tuple[str, Step] | None:
        """Get the next action to take based on current plan.

        Returns:
            Tuple of (action_description, step) or None
        """
        if not self.current_plan:
            return None

        next_step = self.current_plan.get_next_actionable_step()
        if next_step:
            action = f"Step {next_step.id + 1}: {next_step.description}"
            return action, next_step

        return None

    def update_step_result(self, step_id: int, result: Any) -> None:
        """Update the result of a completed step.

        Args:
            step_id: Step ID
            result: Step execution result
        """
        if self.current_plan and 0 <= step_id < len(self.current_plan.steps):
            step = self.current_plan.steps[step_id]
            step.completed = True
            step.result = result

            # Move to next step if this was the current one
            if step_id == self.current_plan.current_step:
                self.current_plan.current_step += 1

            if self.verbose:
                print(f"✅ Step {step_id + 1} completed: {step.description[:50]}")
                print(f"   Progress: {self.current_plan.get_progress():.1f}%")
