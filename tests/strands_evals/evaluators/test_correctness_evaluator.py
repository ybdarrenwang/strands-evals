from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators import CorrectnessEvaluator
from strands_evals.evaluators.correctness_evaluator import (
    CorrectnessRating,
    CorrectnessReferenceRating,
    CorrectnessReferenceScore,
    CorrectnessScore,
)
from strands_evals.types import EvaluationData
from strands_evals.types.trace import (
    AgentInvocationSpan,
    EvaluationLevel,
    Session,
    SpanInfo,
    ToolCall,
    ToolConfig,
    ToolExecutionSpan,
    ToolResult,
    Trace,
)


@pytest.fixture
def evaluation_data():
    now = datetime.now()
    span_info = SpanInfo(session_id="test-session", start_time=now, end_time=now)

    tool_config = ToolConfig(name="calculator", description="Evaluate mathematical expressions")

    agent_span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="What is 2 + 2?",
        agent_response="The answer is 4.",
        available_tools=[tool_config],
    )

    tool_span = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="calculator", arguments={"expression": "2+2"}, tool_call_id="1"),
        tool_result=ToolResult(content="4", tool_call_id="1"),
    )

    trace = Trace(spans=[agent_span, tool_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")

    return EvaluationData(
        input="What is 2 + 2?",
        actual_output="The answer is 4.",
        actual_trajectory=session,
        name="test",
    )


@pytest.fixture
def evaluation_data_with_reference():
    now = datetime.now()
    span_info = SpanInfo(session_id="test-session", start_time=now, end_time=now)

    tool_config = ToolConfig(name="calculator", description="Evaluate mathematical expressions")

    agent_span = AgentInvocationSpan(
        span_info=span_info,
        user_prompt="What is 2 + 2?",
        agent_response="The answer is 4.",
        available_tools=[tool_config],
    )

    tool_span = ToolExecutionSpan(
        span_info=span_info,
        tool_call=ToolCall(name="calculator", arguments={"expression": "2+2"}, tool_call_id="1"),
        tool_result=ToolResult(content="4", tool_call_id="1"),
    )

    trace = Trace(spans=[agent_span, tool_span], trace_id="trace1", session_id="test-session")
    session = Session(traces=[trace], session_id="test-session")

    return EvaluationData(
        input="What is 2 + 2?",
        actual_output="The answer is 4.",
        expected_output="4",
        actual_trajectory=session,
        name="test-reference",
        metadata={"additional_context": "Simple arithmetic question."},
    )


# --- Initialization tests ---


def test_init_with_defaults():
    evaluator = CorrectnessEvaluator()

    assert evaluator.version == "v0"
    assert evaluator.model is None
    assert evaluator.system_prompt is not None
    assert evaluator.reference_system_prompt is not None
    assert evaluator.system_prompt != evaluator.reference_system_prompt
    assert evaluator.evaluation_level == EvaluationLevel.TRACE_LEVEL


def test_init_with_custom_values():
    evaluator = CorrectnessEvaluator(version="v0", model="gpt-4", system_prompt="Custom")

    assert evaluator.version == "v0"
    assert evaluator.model == "gpt-4"
    assert evaluator.system_prompt == "Custom"


# --- Detection tests ---


def test_has_expected_response_true(evaluation_data_with_reference):
    evaluator = CorrectnessEvaluator()
    assert evaluator._has_expected_response(evaluation_data_with_reference) is True


def test_has_expected_response_false(evaluation_data):
    evaluator = CorrectnessEvaluator()
    assert evaluator._has_expected_response(evaluation_data) is False


# --- Basic mode tests ---


@patch("strands_evals.evaluators.correctness_evaluator.Agent")
def test_evaluate_basic_perfectly_correct(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = CorrectnessRating(
        reasoning="The response correctly states 2+2=4", score=CorrectnessScore.PERFECTLY_CORRECT
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = CorrectnessEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "The response correctly states 2+2=4"
    assert result[0].label == CorrectnessScore.PERFECTLY_CORRECT


@pytest.mark.parametrize(
    "score,expected_value,expected_pass",
    [
        (CorrectnessScore.PERFECTLY_CORRECT, 1.0, True),
        (CorrectnessScore.PARTIALLY_CORRECT, 0.5, False),
        (CorrectnessScore.INCORRECT, 0.0, False),
    ],
)
@patch("strands_evals.evaluators.correctness_evaluator.Agent")
def test_basic_score_mapping(mock_agent_class, evaluation_data, score, expected_value, expected_pass):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = CorrectnessRating(reasoning="Test", score=score)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = CorrectnessEvaluator()

    result = evaluator.evaluate(evaluation_data)

    assert len(result) == 1
    assert result[0].score == expected_value
    assert result[0].test_pass == expected_pass
    assert result[0].label == score


@patch("strands_evals.evaluators.correctness_evaluator.Agent")
def test_evaluate_basic_uses_basic_system_prompt(mock_agent_class, evaluation_data):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = CorrectnessRating(
        reasoning="Test", score=CorrectnessScore.PERFECTLY_CORRECT
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = CorrectnessEvaluator()

    evaluator.evaluate(evaluation_data)

    mock_agent_class.assert_called_once_with(
        model=None, system_prompt=evaluator.system_prompt, callback_handler=None
    )


# --- Reference mode tests ---


@patch("strands_evals.evaluators.correctness_evaluator.Agent")
def test_evaluate_with_reference_correct(mock_agent_class, evaluation_data_with_reference):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = CorrectnessReferenceRating(
        reasoning="The agent response matches the expected answer of 4",
        verdict=CorrectnessReferenceScore.CORRECT,
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = CorrectnessEvaluator()

    result = evaluator.evaluate(evaluation_data_with_reference)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "The agent response matches the expected answer of 4"
    assert result[0].label == CorrectnessReferenceScore.CORRECT


@patch("strands_evals.evaluators.correctness_evaluator.Agent")
def test_evaluate_with_reference_incorrect(mock_agent_class, evaluation_data_with_reference):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = CorrectnessReferenceRating(
        reasoning="The agent response does not match", verdict=CorrectnessReferenceScore.INCORRECT
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = CorrectnessEvaluator()

    result = evaluator.evaluate(evaluation_data_with_reference)

    assert len(result) == 1
    assert result[0].score == 0.0
    assert result[0].test_pass is False
    assert result[0].label == CorrectnessReferenceScore.INCORRECT


@pytest.mark.parametrize(
    "verdict,expected_value,expected_pass",
    [
        (CorrectnessReferenceScore.CORRECT, 1.0, True),
        (CorrectnessReferenceScore.INCORRECT, 0.0, False),
    ],
)
@patch("strands_evals.evaluators.correctness_evaluator.Agent")
def test_reference_score_mapping(
    mock_agent_class, evaluation_data_with_reference, verdict, expected_value, expected_pass
):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = CorrectnessReferenceRating(reasoning="Test", verdict=verdict)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = CorrectnessEvaluator()

    result = evaluator.evaluate(evaluation_data_with_reference)

    assert len(result) == 1
    assert result[0].score == expected_value
    assert result[0].test_pass == expected_pass
    assert result[0].label == verdict


@patch("strands_evals.evaluators.correctness_evaluator.Agent")
def test_evaluate_with_reference_uses_reference_system_prompt(mock_agent_class, evaluation_data_with_reference):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = CorrectnessReferenceRating(
        reasoning="Test", verdict=CorrectnessReferenceScore.CORRECT
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    evaluator = CorrectnessEvaluator()

    evaluator.evaluate(evaluation_data_with_reference)

    mock_agent_class.assert_called_once_with(
        model=None, system_prompt=evaluator.reference_system_prompt, callback_handler=None
    )


# --- Async tests ---


@pytest.mark.asyncio
@patch("strands_evals.evaluators.correctness_evaluator.Agent")
async def test_evaluate_async_basic(mock_agent_class, evaluation_data):
    mock_agent = Mock()

    async def mock_invoke_async(*args, **kwargs):
        mock_result = Mock()
        mock_result.structured_output = CorrectnessRating(
            reasoning="Correct answer", score=CorrectnessScore.PERFECTLY_CORRECT
        )
        return mock_result

    mock_agent.invoke_async = mock_invoke_async
    mock_agent_class.return_value = mock_agent
    evaluator = CorrectnessEvaluator()

    result = await evaluator.evaluate_async(evaluation_data)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "Correct answer"
    assert result[0].label == CorrectnessScore.PERFECTLY_CORRECT


@pytest.mark.asyncio
@patch("strands_evals.evaluators.correctness_evaluator.Agent")
async def test_evaluate_async_with_reference(mock_agent_class, evaluation_data_with_reference):
    mock_agent = Mock()

    async def mock_invoke_async(*args, **kwargs):
        mock_result = Mock()
        mock_result.structured_output = CorrectnessReferenceRating(
            reasoning="Matches expected response", verdict=CorrectnessReferenceScore.CORRECT
        )
        return mock_result

    mock_agent.invoke_async = mock_invoke_async
    mock_agent_class.return_value = mock_agent
    evaluator = CorrectnessEvaluator()

    result = await evaluator.evaluate_async(evaluation_data_with_reference)

    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True
    assert result[0].reason == "Matches expected response"
    assert result[0].label == CorrectnessReferenceScore.CORRECT
