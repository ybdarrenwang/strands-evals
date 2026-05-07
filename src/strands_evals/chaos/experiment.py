"""Chaos Experiment.

Extends the base Experiment to run test cases across multiple chaos scenarios,
providing deterministic evaluation of agent resilience under tool failures.
"""

import logging
from collections.abc import Callable
from typing import Any, Optional

from ..case import Case
from ..evaluators.evaluator import Evaluator
from ..experiment import Experiment
from ..types.evaluation_report import EvaluationReport
from .aggregator import ChaosScenarioAggregator
from .aggregator_types import ChaosAggregationReport
from .plugin import ChaosPlugin
from .scenario import ChaosScenario

logger = logging.getLogger(__name__)


class ChaosExperiment(Experiment):
    """Extends Experiment to run cases × chaos scenarios.

    For each scenario, sets it as active on the ChaosPlugin, runs all cases
    through the evaluators, then clears. Reports are tagged by scenario name.

    Optionally includes a baseline run (no chaos) for comparison.

    Example::

        from strands_evals.chaos import (
            ChaosExperiment,
            ChaosPlugin,
            ChaosScenario,
            ChaosScenarioAggregator,
            ToolChaosEffect,
        )

        experiment = ChaosExperiment(
            chaos_plugin=chaos,
            chaos_scenarios=scenarios,
            cases=cases,
            evaluators=[GoalSuccessRateEvaluator(), RecoveryStrategyEvaluator()],
            aggregator=ChaosScenarioAggregator(),
        )

        reports = experiment.run_evaluations(task=my_task)
        aggregation_report = experiment.aggregate_evaluations()
        aggregation_report.run_display()
        aggregation_report.to_file("chaos_report.json")
    """

    def __init__(
        self,
        chaos_plugin: ChaosPlugin,
        chaos_scenarios: list[ChaosScenario],
        cases: Optional[list[Case]] = None,
        evaluators: Optional[list[Evaluator]] = None,
        include_baseline: bool = True,
        baseline_assertion: Optional[str] = None,
        aggregator: Optional[ChaosScenarioAggregator] = None,
    ):
        """Initialize a ChaosExperiment.

        Args:
            chaos_plugin: The ChaosPlugin instance attached to the agent.
            chaos_scenarios: List of scenarios to evaluate. Each scenario runs
                all cases independently.
            cases: Test cases to evaluate (same as base Experiment).
            evaluators: Evaluators to assess results (same as base Experiment).
            include_baseline: If True, runs all cases with no chaos first for comparison.
            baseline_assertion: Optional assertion string for baseline evaluation.
            aggregator: Optional ChaosScenarioAggregator for cross-scenario analysis.
                If provided, aggregate_evaluations() can be called after run_evaluations().
                The aggregator auto-derives known_tools from chaos_scenarios.
        """
        super().__init__(cases=cases, evaluators=evaluators)
        self.chaos_plugin = chaos_plugin
        self.chaos_scenarios = chaos_scenarios
        self.include_baseline = include_baseline
        self.baseline_assertion = baseline_assertion
        self._aggregator = aggregator
        self._last_reports: list[EvaluationReport] = []

        # Auto-populate aggregator's known_tools from scenarios
        if self._aggregator is not None and not self._aggregator.known_tools:
            tools: set[str] = set()
            for scenario in chaos_scenarios:
                tools.update(scenario.tool_effects.keys())
            self._aggregator.known_tools = sorted(tools)

    def run_evaluations(
        self,
        task: Callable[[Case], Any],
        **kwargs,
    ) -> list[EvaluationReport]:
        """Run evaluations across all scenarios (and optionally baseline).

        Executes the task for each (scenario, case) pair:
        1. If include_baseline=True, runs all cases with no chaos active.
        2. For each scenario, activates it on the plugin, runs all cases,
           then clears.

        Results are stored internally for use by aggregate_evaluations().

        Args:
            task: The task function to evaluate. Takes a Case and returns output.
            **kwargs: Additional kwargs passed to the base run_evaluations.

        Returns:
            List of EvaluationReport objects covering all scenarios.
        """
        all_reports: list[EvaluationReport] = []

        # Baseline run (no chaos)
        if self.include_baseline:
            logger.info("Running baseline evaluation (no chaos)")
            self.chaos_plugin.set_active_scenario(None)
            baseline_cases = self._tag_cases_with_scenario("baseline")
            original_cases = self._cases
            self._cases = baseline_cases
            try:
                reports = super().run_evaluations(task, **kwargs)
                all_reports.extend(reports)
            finally:
                self._cases = original_cases

        # Chaos scenario runs
        for scenario in self.chaos_scenarios:
            logger.info(f"Running chaos scenario: {scenario.name}")
            self.chaos_plugin.set_active_scenario(scenario)
            scenario_cases = self._tag_cases_with_scenario(scenario.name, scenario)
            original_cases = self._cases
            self._cases = scenario_cases
            try:
                reports = super().run_evaluations(task, **kwargs)
                all_reports.extend(reports)
            finally:
                self._cases = original_cases

        # Clear active scenario after all runs
        self.chaos_plugin.set_active_scenario(None)
        logger.info(
            f"Chaos experiment complete: {len(all_reports)} reports "
            f"({1 if self.include_baseline else 0} baseline + "
            f"{len(self.chaos_scenarios)} scenarios)"
        )

        # Store for aggregate_evaluations()
        self._last_reports = all_reports
        return all_reports

    def aggregate_evaluations(self) -> ChaosAggregationReport:
        """Aggregate the last run's evaluation reports into a ChaosAggregationReport.

        Must be called after run_evaluations(). Uses the aggregator passed to __init__.

        Returns:
            ChaosAggregationReport with .run_display() and .to_file() methods.

        Raises:
            RuntimeError: If no aggregator was configured or run_evaluations() hasn't been called.
        """
        if self._aggregator is None:
            raise RuntimeError(
                "No aggregator configured. Pass aggregator=ChaosScenarioAggregator() "
                "to ChaosExperiment.__init__()."
            )
        if not self._last_reports:
            raise RuntimeError(
                "No evaluation reports available. Call run_evaluations() first."
            )

        report = self._aggregator.aggregate(self._last_reports)
        # Store the raw reports on the aggregation report for run_display()
        report._reports = self._last_reports
        return report

    def _tag_cases_with_scenario(
        self, scenario_name: str, scenario: Optional[ChaosScenario] = None
    ) -> list[Case]:
        """Create copies of cases with scenario name injected into metadata.

        Args:
            scenario_name: The scenario name to tag.
            scenario: Optional ChaosScenario object to extract tool_effects from.

        Returns:
            Deep copies of all cases with metadata["chaos_scenario"] set.
            If a scenario is provided, also sets metadata["chaos_tool_effects"]
            for downstream aggregation.
        """
        tagged_cases = []
        for case in self._cases:
            tagged = case.model_copy(deep=True)
            if tagged.metadata is None:
                tagged.metadata = {}
            tagged.metadata["chaos_scenario"] = scenario_name

            # Store structured tool_effects for aggregator consumption
            if scenario is not None and scenario.tool_effects:
                tool_effects_serialized = {}
                for tool_name, effect_spec in scenario.tool_effects.items():
                    if isinstance(effect_spec, str):
                        tool_effects_serialized[tool_name] = effect_spec
                    elif hasattr(effect_spec, "value"):
                        tool_effects_serialized[tool_name] = effect_spec.value
                    elif hasattr(effect_spec, "effect"):
                        tool_effects_serialized[tool_name] = effect_spec.effect.value
                    else:
                        tool_effects_serialized[tool_name] = str(effect_spec)
                tagged.metadata["chaos_tool_effects"] = tool_effects_serialized

            # Update case name to include scenario for report clarity
            if tagged.name:
                tagged.name = f"{tagged.name} [{scenario_name}]"
            else:
                tagged.name = f"[{scenario_name}]"
            tagged_cases.append(tagged)
        return tagged_cases
