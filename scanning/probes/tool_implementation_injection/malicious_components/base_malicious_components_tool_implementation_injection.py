from node_spec.structure_schema import FlowSpec, NodeSpec

from scanning.probes.tool_implementation_injection.base_tool_implementation_injection import (
    BaseToolImplementationInjection,
)


class BaseMaliciousComponentsToolImplementationInjection(
    BaseToolImplementationInjection
):
    """
    Shared base for malicious-components tool-implementation-injection probes.

    Concrete subclasses are responsible for defining their own
    ``filter_relevant_flows(...)`` logic.
    """

    @classmethod
    def attack_family(cls) -> str:
        return "malicious_components"
