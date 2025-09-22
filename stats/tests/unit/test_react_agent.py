"""Comprehensive unit tests for the ReAct agent implementation."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Any

from src.agent.react_agent import (
    UnifiedReActAgent,
    ReActTrace,
    ThoughtStep,
    ThoughtType,
    ReActAgent,
    LightweightReActAgent,
    create_react_agent
)
from src.agent.gemma_agent import AgentMode
from src.agent.planner import Plan, Step, TaskComplexity, Planner
from src.agent.tools import ToolRegistry, ToolResult


class TestThoughtStep:
    """Test the ThoughtStep dataclass."""

    def test_thought_step_creation(self):
        """Test creating a thought step."""
        step = ThoughtStep(
            type=ThoughtType.ANALYSIS,
            content="Test thought",
            metadata={"key": "value"},
            timestamp="2023-01-01"
        )

        assert step.type == ThoughtType.ANALYSIS
        assert step.content == "Test thought"
        assert step.metadata == {"key": "value"}
        assert step.timestamp == "2023-01-01"

    def test_thought_step_minimal(self):
        """Test creating a minimal thought step."""
        step = ThoughtStep(
            type=ThoughtType.ACTION,
            content="Execute action"
        )

        assert step.type == ThoughtType.ACTION
        assert step.content == "Execute action"
        assert step.metadata == {}
        assert step.timestamp is None


class TestReActTrace:
    """Test the ReActTrace class."""

    def test_trace_initialization(self):
        """Test trace initialization."""
        trace = ReActTrace(goal="Test goal")

        assert trace.goal == "Test goal"
        assert trace.thoughts == []
        assert trace.actions == []
        assert trace.observations == []
        assert trace.reflections == []
        assert trace.plan is None
        assert trace.final_answer is None
        assert trace.success is False

    def test_add_thought(self):
        """Test adding thoughts to trace."""
        trace = ReActTrace(goal="Test")

        trace.add_thought(ThoughtType.ANALYSIS, "First thought", key="value")

        assert len(trace.thoughts) == 1
        assert trace.thoughts[0].type == ThoughtType.ANALYSIS
        assert trace.thoughts[0].content == "First thought"
        assert trace.thoughts[0].metadata == {"key": "value"}

    def test_get_recent_context(self):
        """Test getting recent context from trace."""
        trace = ReActTrace(goal="Test")

        # Add some thoughts
        trace.add_thought(ThoughtType.ANALYSIS, "Thought 1")
        trace.add_thought(ThoughtType.PLANNING, "Thought 2")
        trace.add_thought(ThoughtType.ACTION, "Thought 3")

        # Add some actions and observations
        trace.actions.append({"name": "action1", "args": {}})
        trace.observations.append("Observation 1")
        trace.actions.append({"name": "action2", "args": {}})
        trace.observations.append("Observation 2")

        context = trace.get_recent_context(2)

        assert "PLANNING: Thought 2" in context
        assert "ACTION: Thought 3" in context
        assert '"name": "action2"' in context
        assert "OBSERVATION: Observation 2" in context

    def test_get_recent_context_empty(self):
        """Test getting context from empty trace."""
        trace = ReActTrace(goal="Test")

        context = trace.get_recent_context(3)

        assert context == ""

    def test_to_dict(self):
        """Test converting trace to dictionary."""
        trace = ReActTrace(goal="Test goal")
        trace.add_thought(ThoughtType.ANALYSIS, "Test thought")
        trace.actions.append({"name": "test_action"})
        trace.observations.append("Test observation")
        trace.reflections.append("Test reflection")
        trace.final_answer = "Test answer"
        trace.success = True

        # Create a mock plan
        mock_plan = Mock()
        mock_plan.to_dict.return_value = {"plan": "data"}
        trace.plan = mock_plan

        result = trace.to_dict()

        assert result["goal"] == "Test goal"
        assert len(result["thoughts"]) == 1
        assert result["thoughts"][0]["type"] == "analysis"
        assert result["thoughts"][0]["content"] == "Test thought"
        assert result["actions"] == [{"name": "test_action"}]
        assert result["observations"] == ["Test observation"]
        assert result["reflections"] == ["Test reflection"]
        assert result["plan"] == {"plan": "data"}
        assert result["final_answer"] == "Test answer"
        assert result["success"] is True

    def test_to_dict_no_plan(self):
        """Test to_dict with no plan."""
        trace = ReActTrace(goal="Test")

        result = trace.to_dict()

        assert result["plan"] is None


class TestUnifiedReActAgent:
    """Test the UnifiedReActAgent class."""

    @pytest.fixture
    def mock_tool_registry(self):
        """Create a mock tool registry."""
        registry = Mock(spec=ToolRegistry)
        registry.get_tool_schemas.return_value = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        ]
        registry.execute_tool.return_value = ToolResult(success=True, output="Tool result")
        return registry

    @pytest.fixture
    def mock_agent(self, mock_tool_registry):
        """Create a mock ReAct agent."""
        with patch('src.agent.react_agent.UnifiedGemmaAgent.__init__') as mock_init:
            mock_init.return_value = None
            agent = UnifiedReActAgent(
                model_name="test_model",
                tool_registry=mock_tool_registry,
                verbose=False
            )

            # Set up required attributes
            agent.tool_registry = mock_tool_registry
            agent.enable_planning = True
            agent.enable_reflection = True
            agent.planner = Mock(spec=Planner)
            agent.current_trace = None
            agent.trace_history = []
            agent.max_iterations = 5
            agent.verbose = False
            agent.system_prompt = "Test system prompt"

            return agent

    def test_initialization(self, mock_tool_registry):
        """Test agent initialization."""
        with patch('src.agent.react_agent.UnifiedGemmaAgent.__init__') as mock_init, \
             patch('src.agent.react_agent.create_react_system_prompt') as mock_prompt:
            mock_init.return_value = None
            mock_prompt.return_value = "ReAct system prompt"

            agent = UnifiedReActAgent(
                model_name="test_model",
                mode=AgentMode.LIGHTWEIGHT,
                tool_registry=mock_tool_registry,
                max_iterations=10,
                verbose=True,
                enable_planning=False,
                enable_reflection=False
            )

            # Verify initialization parameters
            mock_init.assert_called_once()
            assert agent.enable_planning is False
            assert agent.enable_reflection is False
            assert isinstance(agent.planner, Planner)
            assert agent.trace_history == []

    def test_think(self, mock_agent):
        """Test the think method."""
        trace = ReActTrace(goal="Test goal")
        trace.add_thought(ThoughtType.ANALYSIS, "Previous thought")

        with patch.object(mock_agent, 'generate_response') as mock_generate:
            mock_generate.return_value = "THOUGHT: This is my reasoning about the situation."

            thought = mock_agent.think("Current prompt", trace)

            assert thought == "This is my reasoning about the situation."
            mock_generate.assert_called_once()

    def test_think_fallback(self, mock_agent):
        """Test think method fallback when no THOUGHT marker found."""
        trace = ReActTrace(goal="Test goal")

        with patch.object(mock_agent, 'generate_response') as mock_generate:
            mock_generate.return_value = "Just some general response without THOUGHT marker"

            thought = mock_agent.think("Current prompt", trace)

            assert thought == "Just some general response without THOUGHT marker"

    def test_plan(self, mock_agent):
        """Test the plan method."""
        mock_plan = Mock(spec=Plan)
        mock_agent.planner.create_plan.return_value = mock_plan

        with patch.object(mock_agent, 'generate_response') as mock_generate:
            mock_generate.return_value = "Plan response"

            result = mock_agent.plan("Test task", "Test context")

            assert result == mock_plan
            mock_agent.planner.create_plan.assert_called_once_with(
                task="Test task",
                context="Test context",
                tools_schemas=mock_agent.tool_registry.get_tool_schemas(),
                model_response="Plan response"
            )

    def test_act_with_tool_call(self, mock_agent):
        """Test the act method with successful tool execution."""
        trace = ReActTrace(goal="Test goal")

        with patch.object(mock_agent, 'generate_response') as mock_generate:
            mock_generate.return_value = 'ACTION: {"name": "test_tool", "arguments": {"arg": "value"}}'

            action, observation = mock_agent.act("Test thought", trace)

            assert action == {"name": "test_tool", "arguments": {"arg": "value"}}
            assert observation == "Tool result"
            mock_agent.tool_registry.execute_tool.assert_called_once_with(
                "test_tool", {"arg": "value"}
            )

    def test_act_no_tool_needed(self, mock_agent):
        """Test act method when no tool is needed."""
        trace = ReActTrace(goal="Test goal")

        with patch.object(mock_agent, 'generate_response') as mock_generate:
            mock_generate.return_value = 'ACTION: {"name": "none", "arguments": {}}'

            action, observation = mock_agent.act("Test thought", trace)

            assert action is None
            assert observation is None

    def test_act_tool_failure(self, mock_agent):
        """Test act method with tool execution failure."""
        trace = ReActTrace(goal="Test goal")
        mock_agent.tool_registry.execute_tool.return_value = ToolResult(
            success=False, output=None, error="Tool failed"
        )

        with patch.object(mock_agent, 'generate_response') as mock_generate:
            mock_generate.return_value = 'ACTION: {"name": "test_tool", "arguments": {}}'

            action, observation = mock_agent.act("Test thought", trace)

            assert action == {"name": "test_tool", "arguments": {}}
            assert observation == "Error: Tool failed"

    def test_parse_action_with_action_marker(self, mock_agent):
        """Test parsing action with ACTION marker."""
        response = 'Here is my action: ACTION: {"name": "test_tool", "arguments": {"param": "value"}}'

        action = mock_agent._parse_action(response)

        assert action == {"name": "test_tool", "arguments": {"param": "value"}}

    def test_parse_action_with_json_pattern(self, mock_agent):
        """Test parsing action with JSON pattern."""
        response = 'I need to call {"name": "test_tool", "arguments": {"param": "value"}} to solve this.'

        action = mock_agent._parse_action(response)

        assert action == {"name": "test_tool", "arguments": {"param": "value"}}

    def test_parse_action_invalid_json(self, mock_agent):
        """Test parsing action with invalid JSON."""
        response = 'ACTION: {invalid json}'

        with patch('src.agent.react_agent.logger') as mock_logger:
            action = mock_agent._parse_action(response)

            assert action is None
            mock_logger.debug.assert_called()

    def test_parse_action_no_match(self, mock_agent):
        """Test parsing action with no valid patterns."""
        response = 'Just some text without any action'

        action = mock_agent._parse_action(response)

        assert action is None

    def test_reflect_enabled(self, mock_agent):
        """Test reflection when enabled."""
        trace = ReActTrace(goal="Test goal")
        trace.actions.append({"name": "action1"})
        trace.observations.append("Observation 1")

        with patch.object(mock_agent, 'generate_response') as mock_generate:
            mock_generate.return_value = "Analysis of progress. NEXT STEPS: Continue with step 2."

            reflection = mock_agent.reflect(trace)

            assert reflection == "Continue with step 2."

    def test_reflect_disabled(self, mock_agent):
        """Test reflection when disabled."""
        mock_agent.enable_reflection = False
        trace = ReActTrace(goal="Test goal")

        reflection = mock_agent.reflect(trace)

        assert reflection == "Continuing with task..."

    def test_reflect_fallback(self, mock_agent):
        """Test reflection fallback when no NEXT STEPS marker found."""
        trace = ReActTrace(goal="Test goal")

        with patch.object(mock_agent, 'generate_response') as mock_generate:
            mock_generate.return_value = "Just some reflection without next steps marker"

            reflection = mock_agent.reflect(trace)

            assert reflection == "Just some reflection without next steps marker"

    def test_solve_simple_task(self, mock_agent):
        """Test solving a simple task."""
        mock_agent.planner.analyze_complexity.return_value = TaskComplexity.SIMPLE

        responses = [
            "THOUGHT: I understand the task.",
            "ANSWER: This is the final answer."
        ]

        with patch.object(mock_agent, 'generate_response', side_effect=responses):
            result = mock_agent.solve("Simple task")

            assert result == "This is the final answer."
            assert mock_agent.current_trace is not None
            assert mock_agent.current_trace.success is True
            assert len(mock_agent.trace_history) == 1

    def test_save_trace(self, mock_agent):
        """Test saving trace to file."""
        trace = ReActTrace(goal="Test goal")
        trace.to_dict = Mock(return_value={"test": "data"})
        mock_agent.trace_history.append(trace)

        with patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump:

            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            mock_agent.save_trace("/test/path.json")

            mock_open.assert_called_once_with("/test/path.json", "w", encoding="utf-8")
            mock_json_dump.assert_called_once_with({"test": "data"}, mock_file, indent=2)

    def test_save_trace_no_traces(self, mock_agent):
        """Test saving trace with no traces available."""
        with pytest.raises(ValueError, match="No traces to save"):
            mock_agent.save_trace("/test/path.json")


class TestLegacyClasses:
    """Test legacy compatibility classes."""

    def test_react_agent_legacy(self):
        """Test legacy ReActAgent class."""
        with patch('src.agent.react_agent.UnifiedReActAgent.__init__') as mock_init:
            mock_init.return_value = None

            agent = ReActAgent(test_param="value")

            mock_init.assert_called_once_with(mode=AgentMode.FULL, test_param="value")

    def test_lightweight_react_agent_legacy(self):
        """Test legacy LightweightReActAgent class."""
        with patch('src.agent.react_agent.UnifiedReActAgent.__init__') as mock_init:
            mock_init.return_value = None

            agent = LightweightReActAgent(test_param="value")

            mock_init.assert_called_once_with(mode=AgentMode.LIGHTWEIGHT, test_param="value")


class TestCreateReActAgent:
    """Test the create_react_agent factory function."""

    def test_create_react_agent_lightweight_default(self):
        """Test creating lightweight agent by default."""
        with patch('src.agent.react_agent.UnifiedReActAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            result = create_react_agent(
                model_name="test_model",
                enable_planning=False,
                test_param="value"
            )

            assert result == mock_agent
            mock_agent_class.assert_called_once_with(
                model_name="test_model",
                mode=AgentMode.LIGHTWEIGHT,
                enable_planning=False,
                enable_reflection=True,
                test_param="value"
            )

    def test_create_react_agent_explicit_mode(self):
        """Test creating agent with explicit mode."""
        with patch('src.agent.react_agent.UnifiedReActAgent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            result = create_react_agent(
                lightweight=False,  # This should be ignored
                mode=AgentMode.FULL,
                enable_reflection=False
            )

            assert result == mock_agent
            mock_agent_class.assert_called_once_with(
                model_name=None,
                mode=AgentMode.FULL,
                enable_planning=True,
                enable_reflection=False
            )