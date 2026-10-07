# NOTE: This file is a subset of NodeSpec_schema.py.
# It is intended for graph building phase when working with only minimal fields (before node completion)
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from pydantic import BaseModel, Field, model_validator


class NodeType(BaseModel):
    """
    Specifies the functional type of a node, such as LLM, Tool, agent, or Graph.
    """
    type: Literal[
        "LLM",
        "Tool",
        "Database",
        "System",
        "Local_MCP_server",
        "External_MCP_server",
        "Agent",
        "Deterministic_controller",
        "other",
    ]
    # If type == "other", a human-readable explanation is required.
    other_description: Optional[str] = None

    @model_validator(mode="after")
    def validate_other(self):
        if self.type == "other" and not self.other_description:
            raise ValueError("NodeType.other_description required when type='other'")
        if self.type != "other" and self.other_description is not None:
            raise ValueError("NodeType.other_description must be None unless type='other'")
        return self


class CodeReference(BaseModel):
    """
    Represents a reference to a specific location in the codebase that is
    relevant to the definition or behavior of the node.
    """
    # CodeReference.kind is used to justify a node or edge by pointing to the code.
    kind: Literal[
        "source",                   # reference to a source import, can be local code or from a python package.
        "assignment",               # reference to the assignment statement.
        "definition",               # reference to definition of an instance.
        "external_mcp_constructor",      # reference to the exact external MCP constructor/configuration used to run the server.
        "system_prompt",            # reference to the system prompt in the code.
        "usage",                    # refernce to a usage of an instance.
        "input schema",             # reference to an input schema in the code.
        "output_schema",            # reference to an output schema in the code.
        "other"                     # any other refernce that is not one of the above. must include the following field with additional descriptions.
    ]
    # Only used when kind == "other".
    other_kind_description: Optional[str] = None
    file: str
    line: Union[int, Tuple[int, int]]
    snippet: str

    @model_validator(mode="after")
    def validate_other(self):
        if self.kind == "other" and not self.other_kind_description:
            raise ValueError("CodeReference.other_kind_description required when kind='other'")
        if self.kind != "other" and self.other_kind_description is not None:
            raise ValueError("CodeReference.other_kind_description must be None unless kind='other'")
        return self


class IOSlot(BaseModel):
    """
    A single named slot in the node's input/output interface.
    Used as a base for both inputs and outputs.
    """
    name: str                          # e.g. "messages", "tool_call"
    dtype: str                         # e.g. "Message[]", "string", "boolean", "ToolCall"
    description: Optional[str] = None  # human-readable explanation


class InputPort(IOSlot):
    """
    Input specification for a node.
    """
    required: bool = True              # whether the caller must provide it
    default: Optional[Any] = None      # used when required=False and not provided


class OutputPort(IOSlot):
    """
    Output specification for a node.
    """
    output_kind: Literal["data", "action", "data_and_action"] = "data"


class NodeSpec(BaseModel):
    """
    Represents a single node or graph in the agent system, including structure,
    I/O interface, and type-specific configuration.
    """

    # Identity and role (all nodes have)
    name: str
    node_type: NodeType
    description: str

    # Implementation references (evidence in code)
    code_references: Optional[List[CodeReference]] = None

    # I/O interface (optional if not known)
    inputs: List[InputPort] = Field(default_factory=list)
    outputs: List[OutputPort] = Field(default_factory=list)

    # Free-form extension point
    metadata: Optional[Dict[str, Any]] = None
