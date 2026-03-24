from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model
from typing_extensions import Union

from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.trace import EvaluationLevel, TraceLevelInput
from .evaluator import Evaluator
from .prompt_templates.correctness import get_reference_template, get_template


class CorrectnessScore(str, Enum):
    """Categorical correctness ratings."""

    PERFECTLY_CORRECT = "Perfectly Correct"
    PARTIALLY_CORRECT = "Partially Correct"
    INCORRECT = "Incorrect"


class CorrectnessRating(BaseModel):
    """Structured output for correctness evaluation."""

    reasoning: str = Field(
        description="Step by step reasoning to derive the final answer, using no more than 250 words"
    )
    score: CorrectnessScore = Field(
        description="Score should be one of 'Perfectly Correct', 'Partially Correct' or 'Incorrect'"
    )


class CorrectnessReferenceScore(str, Enum):
    """Binary correctness ratings for reference-based evaluation."""

    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"


class CorrectnessReferenceRating(BaseModel):
    """Structured output for reference-based correctness evaluation."""

    reasoning: str = Field(description="Explanation of the evaluation focusing on core information accuracy")
    verdict: CorrectnessReferenceScore = Field(description="Verdict should be one of 'CORRECT' or 'INCORRECT'")


class CorrectnessEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates the correctness of the assistant's response.

    Supports two modes:
    - **Basic mode**: Evaluates correctness based on the conversation context and the
      assistant's response alone. Uses a 3-level scoring rubric
      (Perfectly Correct=1.0, Partially Correct=0.5, Incorrect=0.0).
    - **Reference mode**: When an expected output is provided via ``expected_output``,
      evaluates whether the agent's response is correct by comparing it to the reference.
      Uses a binary CORRECT/INCORRECT scoring rubric (CORRECT=1.0, INCORRECT=0.0).
      Optionally accepts ``metadata["additional_context"]`` for extra evaluation context.
    """

    evaluation_level = EvaluationLevel.TRACE_LEVEL

    _score_mapping = {
        CorrectnessScore.PERFECTLY_CORRECT: 1.0,
        CorrectnessScore.PARTIALLY_CORRECT: 0.5,
        CorrectnessScore.INCORRECT: 0.0,
    }

    _reference_score_mapping = {
        CorrectnessReferenceScore.CORRECT: 1.0,
        CorrectnessReferenceScore.INCORRECT: 0.0,
    }

    def __init__(
        self,
        version: str = "v0",
        model: Union[Model, str, None] = None,
        system_prompt: str | None = None,
    ):
        super().__init__()
        self.system_prompt = system_prompt if system_prompt is not None else get_template(version).SYSTEM_PROMPT
        self.reference_system_prompt = get_reference_template(version).SYSTEM_PROMPT
        self.version = version
        self.model = model

    def _has_expected_response(self, evaluation_case: EvaluationData[InputT, OutputT]) -> bool:
        """Check if the evaluation case contains an expected output for reference-based evaluation."""
        return evaluation_case.expected_output is not None

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        parsed_input = self._get_last_turn(evaluation_case)

        if self._has_expected_response(evaluation_case):
            return self._evaluate_with_reference(parsed_input, evaluation_case)

        return self._evaluate_basic(parsed_input)

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        parsed_input = self._get_last_turn(evaluation_case)

        if self._has_expected_response(evaluation_case):
            return await self._evaluate_with_reference_async(parsed_input, evaluation_case)

        return await self._evaluate_basic_async(parsed_input)

    def _evaluate_basic(self, parsed_input: TraceLevelInput) -> list[EvaluationOutput]:
        """Evaluate correctness using the basic 3-level prompt."""
        prompt = self._format_prompt(parsed_input)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = evaluator_agent(prompt, structured_output_model=CorrectnessRating)
        rating = cast(CorrectnessRating, result.structured_output)
        normalized_score = self._score_mapping[rating.score]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 1.0,
                reason=rating.reasoning,
                label=rating.score,
            )
        ]

    async def _evaluate_basic_async(self, parsed_input: TraceLevelInput) -> list[EvaluationOutput]:
        """Evaluate correctness using the basic 3-level prompt asynchronously."""
        prompt = self._format_prompt(parsed_input)
        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        result = await evaluator_agent.invoke_async(prompt, structured_output_model=CorrectnessRating)
        rating = cast(CorrectnessRating, result.structured_output)
        normalized_score = self._score_mapping[rating.score]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 1.0,
                reason=rating.reasoning,
                label=rating.score,
            )
        ]

    def _evaluate_with_reference(
        self,
        parsed_input: TraceLevelInput,
        evaluation_case: EvaluationData[InputT, OutputT],
    ) -> list[EvaluationOutput]:
        """Evaluate correctness using reference-based prompt."""
        prompt = self._format_reference_prompt(parsed_input, evaluation_case)
        evaluator_agent = Agent(model=self.model, system_prompt=self.reference_system_prompt, callback_handler=None)
        result = evaluator_agent(prompt, structured_output_model=CorrectnessReferenceRating)
        rating = cast(CorrectnessReferenceRating, result.structured_output)
        normalized_score = self._reference_score_mapping[rating.verdict]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 1.0,
                reason=rating.reasoning,
                label=rating.verdict,
            )
        ]

    async def _evaluate_with_reference_async(
        self,
        parsed_input: TraceLevelInput,
        evaluation_case: EvaluationData[InputT, OutputT],
    ) -> list[EvaluationOutput]:
        """Evaluate correctness using reference-based prompt asynchronously."""
        prompt = self._format_reference_prompt(parsed_input, evaluation_case)
        evaluator_agent = Agent(model=self.model, system_prompt=self.reference_system_prompt, callback_handler=None)
        result = await evaluator_agent.invoke_async(prompt, structured_output_model=CorrectnessReferenceRating)
        rating = cast(CorrectnessReferenceRating, result.structured_output)
        normalized_score = self._reference_score_mapping[rating.verdict]
        return [
            EvaluationOutput(
                score=normalized_score,
                test_pass=normalized_score >= 1.0,
                reason=rating.reasoning,
                label=rating.verdict,
            )
        ]

    def _format_prompt(self, parsed_input: TraceLevelInput) -> str:
        """Format evaluation prompt for basic correctness evaluation."""
        parts = []

        # Format conversation context
        parts.append(f"# Context\n{self._format_trace_level_prompt(parsed_input)}")

        # Format the candidate response (the assistant's last response)
        parts.append(f"# Candidate Response\n{parsed_input.agent_response.text}")

        return "\n\n".join(parts)

    def _format_reference_prompt(
        self,
        parsed_input: TraceLevelInput,
        evaluation_case: EvaluationData[InputT, OutputT],
    ) -> str:
        """Format evaluation prompt for reference-based correctness evaluation."""
        metadata = evaluation_case.metadata or {}
        additional_context = metadata.get("additional_context", "N/A")

        # Extract user prompt from the last user message in session history
        user_prompt = self._extract_user_prompt(parsed_input)

        parts = []
        parts.append(f"# User query\n{user_prompt}")
        parts.append(f"# Additional context\n{additional_context}")
        parts.append(f"# Agent response\n{parsed_input.agent_response.text}")
        parts.append(f"# Expected response\n{evaluation_case.expected_output}")

        return "\n\n".join(parts)

    def _extract_user_prompt(self, parsed_input: TraceLevelInput) -> str:
        """Extract the user prompt from the session history.

        Looks for the last UserMessage in the session history to find the user's query.

        Args:
            parsed_input: Trace-level input containing session history

        Returns:
            User prompt text, or empty string if not available
        """
        if not parsed_input.session_history:
            return ""

        # Walk backwards through session history to find the last user message
        for msg in reversed(parsed_input.session_history):
            if not isinstance(msg, list) and self._has_text_content(msg):
                from ..types.trace import Role

                if hasattr(msg, "role") and msg.role == Role.USER:
                    return self._extract_text_content(msg)

        return ""
