from scanning.probes.tool_output_injection.base_tool_output_injection import (
    BaseToolOutputInjection,
)


class BaseInternalStatePoisoningToolOutputInjection(BaseToolOutputInjection):
    """
    Tool-output injection base for attacks that poison tools reading internal
    resources.
    """

    @staticmethod
    def is_attacked_tool_candidate(tool_node) -> bool:
        return bool(getattr(tool_node, "read_internal", False))
