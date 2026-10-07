from numbers import Real
from typing import Any

from scanning.scanning_orchestrator.components.scanning_results import ScanningResults


class BaseEvaluator:
    """
    Base class for the lightweight probe evaluators.

    Subclasses inspect one scan result and return a single evaluation signal
    under ``self.category``.
    """

    def __init__(
        self,
        category: str = "Attack Success",
        failure_threshold: float = 0.5,
    ):
        self.category = category
        self.failure_threshold = failure_threshold

    def normalize_evaluator_results(self, evaluator_results: dict) -> dict:
        """Convert an evaluator payload to the stored scoring format."""
        category = self.category
        value = evaluator_results.get(self.category)
        failure_threshold = self.failure_threshold

        return {
            "category": category,
            "value": value,
            "failure_threshold": failure_threshold,
        }

    @staticmethod
    def _get_attack_metadata(scanning_results: ScanningResults) -> dict[str, Any]:
        malicious_node_spec = getattr(scanning_results, "malicious_node_spec", None)
        metadata = getattr(malicious_node_spec, "metadata", None) or {}
        attack_metadata = metadata.get("instance_info", {})
        return attack_metadata if isinstance(attack_metadata, dict) else {}

    @staticmethod
    def _normalize_trace_index(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _get_trace_selection_index(
        self,
        scanning_results: ScanningResults,
        attack_metadata: dict[str, Any] | None = None,
    ) -> int | None:
        attack_metadata = (
            attack_metadata
            if isinstance(attack_metadata, dict)
            else self._get_attack_metadata(scanning_results)
        )

        for key in (
            "trace_to_eval",
            "trace_index",
            "trace_index_to_eval",
            "trace_index_to_evaluate",
            "trace_to_evaluate",
        ):
            trace_index = self._normalize_trace_index(attack_metadata.get(key))
            if trace_index is not None:
                return trace_index
        return None

    @classmethod
    def _normalize_execution_trace_entries(
        cls,
        trace_entries: Any,
    ) -> list[dict[str, Any]]:
        if not isinstance(trace_entries, (list, tuple)):
            return []

        normalized_entries: list[dict[str, Any]] = []
        for index, entry in enumerate(trace_entries):
            if not isinstance(entry, dict):
                continue

            trace_key = str(entry.get("trace_key") or "").strip()
            if not trace_key:
                continue

            execution_cmd = str(entry.get("execution_cmd") or "").strip()
            step_index = cls._normalize_trace_index(entry.get("step_index"))
            if step_index is None:
                step_index = index

            normalized_entries.append(
                {
                    "step_index": step_index,
                    "execution_cmd": execution_cmd,
                    "trace_key": trace_key,
                }
            )

        return normalized_entries

    def _get_execution_trace_entries(
        self,
        scanning_results: ScanningResults,
        attack_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        attack_metadata = (
            attack_metadata
            if isinstance(attack_metadata, dict)
            else self._get_attack_metadata(scanning_results)
        )
        return self._normalize_execution_trace_entries(
            attack_metadata.get("execution_trace_entries")
        )

    def _get_trace_key(
        self,
        scanning_results: ScanningResults,
        execution_cmd: str | None = None,
        attack_metadata: dict[str, Any] | None = None,
    ) -> str | None:
        attack_metadata = (
            attack_metadata
            if isinstance(attack_metadata, dict)
            else self._get_attack_metadata(scanning_results)
        )
        trace_entries = self._get_execution_trace_entries(
            scanning_results,
            attack_metadata,
        )
        if not trace_entries:
            return None

        trace_index = self._get_trace_selection_index(scanning_results, attack_metadata)
        if trace_index is not None:
            try:
                selected_entry = trace_entries[trace_index]
            except IndexError:
                selected_entry = None
            if selected_entry is not None:
                selected_cmd = str(selected_entry.get("execution_cmd") or "").strip()
                if execution_cmd is None or execution_cmd.strip() == selected_cmd:
                    return str(selected_entry.get("trace_key") or "").strip() or None

        if execution_cmd is not None:
            normalized_execution_cmd = execution_cmd.strip()
            for entry in trace_entries:
                if str(entry.get("execution_cmd") or "").strip() != normalized_execution_cmd:
                    continue
                return str(entry.get("trace_key") or "").strip() or None

        return str(trace_entries[0].get("trace_key") or "").strip() or None

    def _get_execution_cmd(
        self,
        scanning_results: ScanningResults,
        attack_metadata: dict[str, Any] | None = None,
    ) -> str | None:
        attack_metadata = (
            attack_metadata
            if isinstance(attack_metadata, dict)
            else self._get_attack_metadata(scanning_results)
        )
        execution_cmds = attack_metadata.get("execution_cmds", [])

        if isinstance(execution_cmds, str):
            execution_cmds = [execution_cmds]
        elif not isinstance(execution_cmds, (list, tuple)):
            execution_cmds = []

        execution_cmds = [cmd for cmd in execution_cmds if cmd is not None]
        if not execution_cmds:
            trace_entries = self._get_execution_trace_entries(
                scanning_results,
                attack_metadata,
            )
            execution_cmds = [
                entry.get("execution_cmd")
                for entry in trace_entries
                if entry.get("execution_cmd")
            ]
        if not execution_cmds:
            return None

        trace_index = self._get_trace_selection_index(scanning_results, attack_metadata)
        if trace_index is not None:
            try:
                return execution_cmds[trace_index]
            except IndexError:
                pass

        return execution_cmds[0]

    def _get_primary_trace(self, scanning_results: ScanningResults):
        """Return the preferred loaded trace, or ``None`` when unavailable."""
        traces_of_execution = scanning_results.traces_of_execution or {}
        if self._looks_like_trace(traces_of_execution):
            return traces_of_execution

        if not isinstance(traces_of_execution, dict) or not traces_of_execution:
            return None

        selected_trace_key = self._get_trace_key(scanning_results)
        if selected_trace_key:
            selected_trace = traces_of_execution.get(selected_trace_key)
            if selected_trace is not None:
                return selected_trace

        ordered_traces = list(traces_of_execution.values())
        trace_index = self._get_trace_selection_index(scanning_results)
        if trace_index is not None:
            try:
                selected_trace = ordered_traces[trace_index]
                if selected_trace is not None:
                    return selected_trace
            except IndexError:
                pass

        for trace in ordered_traces:
            if self._looks_like_trace(trace):
                return trace

        return ordered_traces[0]

    def _get_trace(self, scanning_results: ScanningResults, execution_cmd: str | None = None):
        """Return the selected trace from an execution result map.

        Executor results may be keyed either by the raw execution command
        (legacy behavior) or by a unique per-step trace key recorded under
        ``metadata.instance_info.execution_trace_entries``. When the probe
        metadata includes a trace-selection index such as ``trace_to_eval``,
        the matching step is selected before falling back to the first trace.
        """
        traces_of_execution = scanning_results.traces_of_execution or {}
        if self._looks_like_trace(traces_of_execution):
            return traces_of_execution

        if not isinstance(traces_of_execution, dict) or not traces_of_execution:
            return None

        attack_metadata = self._get_attack_metadata(scanning_results)
        trace_key = self._get_trace_key(
            scanning_results,
            execution_cmd=execution_cmd,
            attack_metadata=attack_metadata,
        )
        if trace_key:
            trace = traces_of_execution.get(trace_key)
            if trace is not None:
                return trace

        if execution_cmd is not None:
            trace = traces_of_execution.get(execution_cmd)
            if trace is not None:
                return trace

        return self._get_primary_trace(scanning_results)

    @staticmethod
    def _looks_like_trace(value) -> bool:
        """Detect trace/flow payloads with embedded events."""
        if not isinstance(value, dict):
            model_dump = getattr(value, "model_dump", None)
            if callable(model_dump):
                try:
                    value = model_dump()
                except TypeError:
                    value = None

        if not isinstance(value, dict):
            return False

        if isinstance(value.get("events"), list):
            return True

        nested_trace = value.get("trace")
        return isinstance(nested_trace, dict) and isinstance(nested_trace.get("events"), list)

    # def _normalize_score_value(self, value):
    #     """Normalize evaluator outputs to numbers in [0, 1] or None."""
    #     if value is None:
    #         return None
    #     if isinstance(value, bool):
    #         return int(value)
    #     if isinstance(value, Real):
    #         normalized = float(value)
    #         if normalized.is_integer():
    #             return int(normalized)
    #         return normalized
    #     return None

    def evaluate_results(
        self,
        scanning_results: ScanningResults,
    ) -> dict:
        """Evaluate one scan result using the subclass-specific check.

        Args:
            scanning_results (ScanningResults): Scan output containing the
                malicious node spec and execution traces.

        Returns:
            Subclasses generally return a dictionary keyed by
            ``self.category``. This base method documents the common contract.
        """
        malicious_node_spec = scanning_results.malicious_node_spec
        traces_of_execution = scanning_results.traces_of_execution
