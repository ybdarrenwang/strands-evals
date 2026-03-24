from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model
from typing_extensions import Union

from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.trace import EvaluationLevel, SessionLevelInput
from .evaluator import Evaluator
from .prompt_templates.goal_success_rate import get_assertion_template, get_template


class GoalSuccessScore(str, Enum):
    """Binary goal success ratings."""

    YES = "Yes"
    NO = "No"


class GoalSuccessRating(BaseModel):
    """Structured output for goal success evaluation."""

    reasoning: str = Field(description="Step by step reasoning to derive the final score")
    score: GoalSuccessScore = Field(description="Score should be one of 'Yes' or 'No'")


class GoalSuccessAssertionScore(str, Enum):
    """Binary assertion-based goal success ratings."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class GoalSuccessAssertionRating(BaseModel):
    """Structured output for assertion-based goal success evaluation."""

    reasoning: str = Field(description="Brief explanation of the evaluation")
    verdict: GoalSuccessAssertionScore = Field(description="Verdict should be one of 'SUCCESS' or 'FAILURE'")


class GoalSuccessRateEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates whether all user goals were successfully achieved in a conversation.

    Supports two modes:
    - **Basic mode**: Evaluates goal success based on conversation analysis alone.
      Uses a Yes/No scoring rubric (Yes=1.0, No=0.0).
    - **Assertion mode**: When assertions are provided via ``metadata["assertions"]``,
      evaluates whether the agent's behavior satisfies the specified success assertions.
      Uses a SUCCESS/FAILURE scoring rubric (SUCCESS=1.0, FAILURE=0.0).
      Optionally accepts ``metadata["additional_context"]`` for extra evaluation context.
    """

    evaluation_level = EvaluationLevel.SESSION_LEVEL

    _score_mapping = {
        GoalSuccessScore.YES: 1.0,
        GoalSuccessScore.NO: 0.0,
    }

    _assertion_score_mapping = {
        GoalSuccessAssertionScore.SUCCESS: 1.0,
        GoalSuccessAssertionScore.FAILURE: 0.0,
    }

    def __init__(
        self,
        version: str = "v0",
        model: Union[Model, str, None] = None,
        system_prompt: str | None = None,
        assertion_system_prompt: str | None = None,
    ):
        super().__init__()
        self.system_prompt = system_prompt if system_prompt is not None else get_template(version).SYSTEM_PROMPT
        self.assertion_system_prompt = (
            assertion_system_prompt
            if assertion_system_prompt is not None
            else get_assertion_template(version).SYSTEM_PROMPT
        )
        self.version = version
        self.model = model

    def _has_assertions(self, evaluation_case: EvaluationData[InputT, OutputT]) -> bool:
        """Check if the evaluation case contains assertions in metadata."""
        if evaluation_case.metadata and evaluation_case.metadata.get("assertions"):
            return True
        return False

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        session_input = self._parse_trajectory(evaluation_case)

        if self._has_assertions(evaluation_case):
            return self._evaluate_with_assertions(session_input, evaluation_case)

        return self._evaluate_basic(session_input)

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        session_input = self._parse_trajectory(evaluation_case)

        if self._has_assertions(evaluation_case):
            return await self._evaluate_with_assertions_async(session_input, evaluation_case)

        return await self._evaluate_basic_async(session_input)

    def _evaluate_basic(self, session_input: SessionLevelInput) -> list[EvaluationOutput]:
        """Evaluate goal success using the basic prompt (no assertions)."""
        prompt = self._format_prompt(session_input)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = evaluator_agent(prompt, structured_output_model=GoalSuccessRating)
        rating = cast(GoalSuccessRating, result.structured_output)
        normalized_score = self._score_mapping[rating.score]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 1.0,
                reason=rating.reasoning,
                label=rating.score,
            )
        ]

    async def _evaluate_basic_async(self, session_input: SessionLevelInput) -> list[EvaluationOutput]:
        """Evaluate goal success using the basic prompt asynchronously."""
        prompt = self._format_prompt(session_input)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = await evaluator_agent.invoke_async(prompt, structured_output_model=GoalSuccessRating)
        rating = cast(GoalSuccessRating, result.structured_output)
        normalized_score = self._score_mapping[rating.score]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 1.0,
                reason=rating.reasoning,
                label=rating.score,
            )
        ]

    def _evaluate_with_assertions(
        self,
        session_input: SessionLevelInput,
        evaluation_case: EvaluationData[InputT, OutputT],
    ) -> list[EvaluationOutput]:
        """Evaluate goal success using assertion-based prompt."""
        prompt = self._format_assertion_prompt(session_input, evaluation_case)
        evaluator_agent = Agent(model=self.model, system_prompt=self.assertion_system_prompt, callback_handler=None)
        result = evaluator_agent(prompt, structured_output_model=GoalSuccessAssertionRating)
        rating = cast(GoalSuccessAssertionRating, result.structured_output)
        normalized_score = self._assertion_score_mapping[rating.verdict]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 1.0,
                reason=rating.reasoning,
                label=rating.verdict,
            )
        ]

    async def _evaluate_with_assertions_async(
        self,
        session_input: SessionLevelInput,
        evaluation_case: EvaluationData[InputT, OutputT],
    ) -> list[EvaluationOutput]:
        """Evaluate goal success using assertion-based prompt asynchronously."""
        prompt = self._format_assertion_prompt(session_input, evaluation_case)
        evaluator_agent = Agent(model=self.model, system_prompt=self.assertion_system_prompt, callback_handler=None)
        result = await evaluator_agent.invoke_async(prompt, structured_output_model=GoalSuccessAssertionRating)
        rating = cast(GoalSuccessAssertionRating, result.structured_output)
        normalized_score = self._assertion_score_mapping[rating.verdict]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 1.0,
                reason=rating.reasoning,
                label=rating.verdict,
            )
        ]

    def _format_prompt(self, session_input: SessionLevelInput) -> str:
        """Format evaluation prompt from session-level input."""
        parts = []

        if session_input.available_tools:
            parts.append(f"# Available tools\n{self._format_tools(session_input.available_tools)}")

        if session_input.session_history:
            parts.append(f"# Conversation record\n{self._format_session_history(session_input.session_history)}")

        return "\n\n".join(parts)

    def _format_assertion_prompt(
        self,
        session_input: SessionLevelInput,
        evaluation_case: EvaluationData[InputT, OutputT],
    ) -> str:
        """Format evaluation prompt for assertion-based evaluation."""
        metadata = evaluation_case.metadata or {}
        assertions = metadata.get("assertions", "")
        additional_context = metadata.get("additional_context", "N/A")

        parts = []

        if session_input.session_history:
            parts.append(f"CONVERSATION RECORD:\n{self._format_session_history(session_input.session_history)}")

        parts.append(f"SUCCESS ASSERTIONS:\n{assertions}")
        parts.append(f"ADDITIONAL CONTEXT:\n{additional_context}")

        return "\n\n".join(parts)
