"""Shared utilities for constructing and evaluating probe instances."""

from collections.abc import Callable
from typing import Any, Dict, List
import json
from pathlib import Path


from node_spec.structure_schema import FlowSpec, NodeSpec
from scanning.probes.utils.base_util import (
    change_external_mcp_to_emulated,
    extract_first_return_expression as extract_first_return_expression_util,
    get_first_executable_line,
    inject_instruction_into_tool_output as inject_instruction_into_tool_output_util,
    insert_line_in_content,
    normalize_flows,
    rewrite_return_statements as rewrite_return_statements_util,
)


class BaseProbe():
    """
    Base class for probe implementations.

    A probe represents one adversarial scenario to test against a system
    described by a `NodeSpec`. Concrete probes typically:

    1. Select the flows that are relevant for the scenario.
    2. Create a modified "malicious twin" of the original specification.
    3. Execute the modified specification elsewhere in the pipeline.
    4. Evaluate the resulting traces.

    This base class provides the common mechanics for storing probe metadata,
    cloning the source specification, loading trace payloads, and building a
    standard result object.
    """
    guidance_fields: list[str] = []

    def __init__(
        self,
        name: str = "Generic Threat",
        description: str = '',
        attack_surface: str ='',
        attack_suit: str ='',
        attacker_intent: str = '',
        requirements:list[str] = [],
    ):
        """
        Initialize a probe definition bound to a source ``NodeSpec``.

        Args:
            full_node_spec (NodeSpec): Original system specification that will be
                cloned and mutated by the probe.
            name (str, optional): Human-readable probe name.
            description (str, optional): Short explanation of the attack
                scenario represented by the probe.
            risk_type (str, optional): High-level risk category used for
                reporting or grouping probes.
            requirements (list[str], optional): Preconditions or assumptions for
                the probe. The base class stores this metadata but does not
                enforce it.
        """
        self.name = name
        self.description = description
        self.attack_suit = attack_suit
        self.attack_surface = attack_surface
        self.attacker_intent = attacker_intent
        self.requirements = requirements
        self.guidance_fields = list(self.guidance_fields)

    @staticmethod
    def filter_relevant_flows(
        full_node_spec: NodeSpec,
        flows: List[FlowSpec] = None,
    ) -> list[FlowSpec]:
        """
        Return the flows relevant to this probe.

        The default implementation treats every flow in ``full_node_spec``
        as relevant. Subclasses can override this method to filter the list and
        should return the flows they want to keep.

        Returns:
            list[FlowSpec]: Flows considered relevant for this probe.
        """
        return normalize_flows(full_node_spec, flows)

    @staticmethod
    def get_definition_code_reference(tool_node: Any) -> Any:
        """
        Return the definition code reference for the given tool.
        """
        for code in getattr(tool_node, "code_references", None) or []:
            if getattr(code, "kind", None) == "definition":
                return code
        raise ValueError(
            f"Could not find an definition code reference for tool '{tool_node.id}'."
        )

    @staticmethod
    def extract_first_return_expression(snippet: str) -> str | None:
        """
        Return the expression from the first return statement in the snippet.
        """
        return extract_first_return_expression_util(snippet)

    def assert_has_simple_return(self, attacked_tool_node: Any, snippet: str) -> str:
        """
        Validate that the tool definition contains a simple return expression.
        """
        return_expression = self.extract_first_return_expression(snippet)
        if return_expression is None:
            raise ValueError(
                f"Could not find a return expression in tool '{attacked_tool_node.id}'."
            )
        return return_expression

    def inject_instruction_into_tool_output(
        self,
        attacked_tool_node: Any,
        instruction: str,
        *,
        instruction_is_expression: bool = False,
        injection_place: str = "start",
    ) -> None:
        """
        Inject instruction content into the attacked tool return value.
        """
        inject_instruction_into_tool_output_util(
            attacked_tool_node=attacked_tool_node,
            instruction=instruction,
            instruction_is_expression=instruction_is_expression,
            injection_place=injection_place,
        )

    def rewrite_return_statements(
        self,
        attacked_tool_node: Any,
        replacement_builder: Callable[[str], str],
    ) -> None:
        """
        Rewrite return statements of a tool definition using a shared builder.
        """
        rewrite_return_statements_util(
            attacked_tool_node=attacked_tool_node,
            replacement_builder=replacement_builder,
        )

    def rewrite_first_return_statement(
        self,
        attacked_tool_node: Any,
        replacement_builder: Callable[[str], str],
    ) -> None:
        """
        Backwards-compatible alias for ``rewrite_return_statements``.
        """
        self.rewrite_return_statements(
            attacked_tool_node=attacked_tool_node,
            replacement_builder=replacement_builder,
        )

    def apply_trigger_and_output_injection(
        self,
        attacked_tool_node: Any,
        trigger: str,
        output_instruction: str | None = None,
        *,
        output_instruction_is_expression: bool = False,
        output_injection_place: str = "start",
        require_simple_return: bool = True,
    ) -> None:
        """
        Insert trigger logic into tool body and optionally patch its return line.
        """
        code = self.get_definition_code_reference(attacked_tool_node)
        if require_simple_return:
            self.assert_has_simple_return(attacked_tool_node, code.snippet)

        first_body_line = get_first_executable_line(code.snippet)
        if first_body_line is None:
            raise ValueError(
                f"Could not find an executable line in tool '{attacked_tool_node.id}'."
            )
        code.snippet = insert_line_in_content(code.snippet, first_body_line, trigger)

        if output_instruction is not None:
            self.inject_instruction_into_tool_output(
                attacked_tool_node=attacked_tool_node,
                instruction=output_instruction,
                instruction_is_expression=output_instruction_is_expression,
                injection_place=output_injection_place,
            )

    @staticmethod
    def _normalize_execution_cmds(execution_cmds: Any) -> list[str]:
        if isinstance(execution_cmds, (list, tuple)):
            return [str(cmd).strip() for cmd in execution_cmds if str(cmd).strip()]
        if isinstance(execution_cmds, str):
            normalized = execution_cmds.strip()
            return [normalized] if normalized else []
        return []

    @classmethod
    def infer_system_execution_count(
        cls,
        instance_info: Dict | None,
    ) -> int:
        normalized_info = dict(instance_info or {})
        explicit_value = normalized_info.get("system_execution_count")
        if isinstance(explicit_value, bool):
            explicit_count = 0
        else:
            try:
                explicit_count = int(explicit_value)
            except (TypeError, ValueError):
                explicit_count = 0
        if explicit_count > 0:
            return explicit_count

        final_execution_count = len(
            cls._normalize_execution_cmds(normalized_info.get("execution_cmds"))
        )
        attack_turns = normalized_info.get("attack_turns")
        attack_turn_count = len(attack_turns) if isinstance(attack_turns, list) else 0
        return max(0, attack_turn_count + final_execution_count)

    def _add_metadata(self, 
                      malicious_node_spec: NodeSpec, 
                      instance_info:Dict|None = None):
        """
        Attach probe bookkeeping metadata to a malicious twin ``NodeSpec``.

        Args:
            malicious_node_spec (NodeSpec): Cloned specification being prepared
                for execution or evaluation.
            instance_info (dict | None, optional): Probe-specific details about
                the generated instance, such as affected nodes or execution
                commands.

        Returns:
            NodeSpec: The same ``NodeSpec`` with updated ``metadata`` fields.
        """
        normalized_instance_info = dict(instance_info or {})
        attacked_tool_node_id = normalized_instance_info.get("attacked_tool_node_id")
        if attacked_tool_node_id:
            change_external_mcp_to_emulated(
                malicious_node_spec=malicious_node_spec,
                attack_tool_id=attacked_tool_node_id,
            )
            attacked_tool_node = malicious_node_spec.get_node(
                node_id=attacked_tool_node_id
            )
            if attacked_tool_node is not None:
                normalized_instance_info["attacked_tool_node_emulated"] = bool(
                    getattr(attacked_tool_node, "emulated", False)
                )
        normalized_instance_info["system_execution_count"] = (
            self.infer_system_execution_count(normalized_instance_info)
        )

        metadata = dict(malicious_node_spec.metadata or {})
        metadata["updated_by_prob"] = self._get_name()
        metadata["instance_info"] = normalized_instance_info
        malicious_node_spec.metadata = metadata
        return malicious_node_spec


    def malicious_twin_update(
        self,
        full_node_spec: NodeSpec,
        flow: FlowSpec,
        guidance: Dict|None = None,
        instance_info: Dict|None = None,
    ) -> NodeSpec:
        """
        Create the malicious twin used by this probe instance for one flow.

        The base implementation copies ``full_node_spec`` and records probe
        metadata. Concrete probes usually override this method to inject the
        actual adversarial change, then either call ``_add_metadata()`` or this
        base implementation to keep the metadata layout consistent.

        Args:
            flow (FlowSpec): Single flow instance to attack.
            guidance (dict | None, optional): Probe-specific configuration used
                when building the malicious twin.
            instance_info (dict | None, optional): Additional metadata to store
                under ``metadata["instance_info"]``.

        Returns:
            NodeSpec: A cloned ``NodeSpec`` carrying the probe metadata.
        """
        malicious_node_spec = full_node_spec.model_copy(deep=True)
        malicious_node_spec = self._add_metadata(malicious_node_spec, instance_info)

        return malicious_node_spec

    def create_instance(
        self,
        full_node_spec: NodeSpec,
        guidance: Dict | None = None,
        relevant_flow: FlowSpec | None = None,
    ):
        """
        Build a concrete probe instance from the supplied guidance.

        This helper only delegates to :meth:`malicious_twin_update`. It does not
        execute the system under test and does not evaluate the probe outcome.

        Args:
            guidance (dict): Probe-specific configuration forwarded to
                :meth:`malicious_twin_update`.
            relevant_flow (FlowSpec | None): Optional preselected flow for the probe.

        Returns:
            NodeSpec: The value returned by :meth:`malicious_twin_update`. In the
                base implementation this is the malicious twin ``NodeSpec``.
        """
        if relevant_flow is None:
            relevant_flows = self.filter_relevant_flows(full_node_spec)
            if not relevant_flows:
                raise ValueError("No relevant flows were found for this probe.")
            relevant_flow = relevant_flows[0]

        return self.malicious_twin_update(
            full_node_spec=full_node_spec,
            guidance=guidance or {},
            flow=relevant_flow,
        )

    @staticmethod
    def build_instance_info(**fields: Any) -> dict[str, Any]:
        """
        Build probe instance metadata from arbitrary named fields.

        This helper keeps metadata construction consistent across probe
        families while allowing each family to choose the exact schema it
        needs.
        """
        return dict(fields)

    @staticmethod
    def snapshot_code_reference_snippets(
        owner_node: Any,
    ) -> list[tuple[Any, Any]]:
        """
        Capture ``(kind, snippet)`` pairs for a node's code references.

        This snapshot can be compared after a mutation to determine which
        code-reference kinds were actually changed.
        """
        snapshot: list[tuple[Any, Any]] = []
        for code_reference in getattr(owner_node, "code_references", None) or []:
            snapshot.append(
                (
                    getattr(code_reference, "kind", None),
                    getattr(code_reference, "snippet", None),
                )
            )
        return snapshot

    @staticmethod
    def changed_code_reference_kinds(
        owner_node: Any,
        before_snapshot: list[tuple[Any, Any]],
    ) -> list[str]:
        """
        Return unique code-reference kinds whose entries changed since snapshot.
        """
        changed_kinds: list[str] = []
        current_code_references = getattr(owner_node, "code_references", None) or []

        for index, code_reference in enumerate(current_code_references):
            previous_kind = None
            previous_snippet = None
            if index < len(before_snapshot):
                previous_kind, previous_snippet = before_snapshot[index]

            current_kind = getattr(code_reference, "kind", None)
            current_snippet = getattr(code_reference, "snippet", None)
            if (
                current_kind == previous_kind
                and current_snippet == previous_snippet
            ):
                continue

            kind_value = current_kind or previous_kind
            if not kind_value:
                continue
            kind_text = str(kind_value)
            if kind_text not in changed_kinds:
                changed_kinds.append(kind_text)

        return changed_kinds

    @staticmethod
    def build_affected_nodes(
        node_id: str,
        changed_kinds: list[str] | None,
    ) -> list[dict[str, list[str]]]:
        """
        Build the ``affected_nodes`` payload as ``[{node_id: [kinds...]}]``.
        """
        if not node_id:
            return []

        normalized_kinds: list[str] = []
        for kind in changed_kinds or []:
            if not kind:
                continue
            if kind in normalized_kinds:
                continue
            normalized_kinds.append(kind)

        if not normalized_kinds:
            return []
        return [{node_id: normalized_kinds}]

    def _normalize_traces_payload(self, traces_of_execution: dict | None) -> dict:
        """
        Normalize executor output to a dictionary payload.

        Executors may return ``None`` when no traces were produced. The base
        behavior converts that case to an empty dictionary so downstream code
        can assume a mapping interface. Empty or whitespace-only trace paths are
        also discarded because they represent "no exported trace".

        Args:
            traces_of_execution (dict | None): Raw executor output.

        Returns:
            dict: Normalized trace mapping, or an empty dictionary when no
                traces are present.
        """
        if not traces_of_execution:
            return {}

        normalized_traces = {}
        for command, trace in traces_of_execution.items():
            if trace is None:
                continue

            if isinstance(trace, (str, Path)) and not str(trace).strip():
                continue

            normalized_traces[command] = trace

        return normalized_traces

    def _load_traces(self, traces_of_execution: dict) -> dict:
        """
        Load trace payloads from JSON files when paths are provided.

        Args:
            traces_of_execution (dict): Mapping of executed command to either an
                already-loaded trace payload or a filesystem path pointing to a
                JSON file.

        Returns:
            dict: Mapping of command to fully loaded trace payload.
        """
        loaded_traces = {}
        for command, trace in traces_of_execution.items():
            if isinstance(trace, (str, Path)):
                trace_path = str(trace).strip()
                if not trace_path:
                    continue

                with open(trace_path, "r", encoding="utf-8") as f:
                    loaded_traces[command] = json.load(f)
            else:
                loaded_traces[command] = trace
        return loaded_traces

    def _build_result(
        self, parsed_traces: dict, malicious_twin: NodeSpec
    ) -> dict:
        """
        Build the standard result payload returned by probe execution.

        The ``status`` flag is ``1`` when at least one trace is present and ``0``
        otherwise.

        Args:
            parsed_traces (dict): Loaded execution traces keyed by command.
            malicious_twin (NodeSpec): Modified specification used for the probe
                run.

        Returns:
            dict: Standard probe result payload with probe name, status,
                traces, and malicious twin.
        """
        return {
            "probe_name": self._get_name(),
            "status": 1 if parsed_traces else 0,
            "traces_of_execution": parsed_traces,
            "malicious_twin": malicious_twin,
        }

    def _get_name(self) -> str:
        """
        Return the fully qualified probe name used in metadata and results.

        Returns:
            str: ``<module>.<ClassName>`` for the concrete probe.
        """
        module = self.__class__.__module__
        if module.startswith("scanning.probes."):
            module = module.split(".")[-1]
        return f"{module}.{self.__class__.__name__}"
