# NOTE: This file is a the canonical schema file of the NodeSpec object.
# It is well commented and intended for human reading and prompt context.

from typing import Any, Dict, List, Optional, Tuple, Union, Set, Literal, DefaultDict, ClassVar, Type, Iterator, TypeVar
from pathlib import Path
from pydantic import BaseModel, Field, model_validator, ValidationInfo
from collections import Counter, defaultdict
import json

T = TypeVar("T")

# ============================================================
# BASIC OBJECTS
# These are the building blocks that appear across all nodes.
# ============================================================
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
        "other"
    ]
    # If type == "other", a human-readable explanation is required.
    other_description: Optional[str] = None   # required when type == "other"

    @model_validator(mode="after")
    def validate_other(self):
        if self.type == "other":
            if not self.other_description:
                raise ValueError("NodeType: other_description is required when type='other'")
        else:
            if self.other_description is not None:
                raise ValueError("NodeType: other_description must be None unless type='other'")
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
        if self.kind == "other":
            if not self.other_kind_description:
                raise ValueError("CodeReference: other_kind_description is required when kind='other'")
        else:
            if self.other_kind_description is not None:
                raise ValueError("CodeReference: other_kind_description must be None unless kind='other'")
        return self

class RequiredKeys(BaseModel):
    # Used to represent required API keys or secrets for a node.
    enabled: bool = False
    keys: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_keys(self):
        if self.enabled:
            if not self.keys or len(self.keys) == 0:
                raise ValueError(
                    "keys must be provided and non-empty when enabled=True"
                )
        else:
            if self.keys:
                raise ValueError(
                    "keys must be None or empty when enabled=False"
                )
        return self

class Duplication(BaseModel):
    # Signals that multiple nodes share the same name (same source, multiple instances).
    exists: bool = False
    instances: Optional[List[Dict[str, Any]]] = None

    @model_validator(mode="after")
    def validate_keys(self):
        if self.exists:
            if not self.instances or len(self.instances) == 0:
                raise ValueError(
                    "instances must be provided and non-empty when exists=True (duplicates exist)"
                )
        else:
            if self.instances:
                raise ValueError(
                    "instances must be None or empty when exists=False (duplicates do not exist)"
                )
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
    output_kind: Literal[
        "data",        # direct informational output (e.g., text, messages, JSON)
        "action",      # side-effect/action only (no primary informational payload)
        "data_and_action"  # returns data and also performs side effects/actions.
    ] = "data"

class Connection(BaseModel):
    """
    Describes local adjacency for a node by listing the upstream and downstream
    node identifiers it is connected to.
    """
    parent: str
    in_: Union[str, List[str]]
    out: Union[str, List[str]]

class FrameworkType(BaseModel):
    """
    Identifies the framework used to construct this graph, such as LangGraph or LangChain.
    """
    framework: Literal["LangGraph", "AutoGen", "CrewAI", "LangChain", "other"]
    other_description: Optional[str] = None   # required when framework == "other"

    @model_validator(mode="after")
    def validate_other(self):
        if self.framework == "other" and not self.other_description:
            raise ValueError("other_description is required when framework='other'")
        return self

# ============================================================
# LLM-SPECIFIC OBJECTS
# LLM configuration is optional and only applies to LLM or Agent nodes.
# ============================================================
class LLMConfig(BaseModel):
    """
    Configuration for an LLM used within a node. Allows specifying the provider,
    model class, model name, and runtime generation parameters.
    """
    provider: Optional[str] = None
    class_name: Optional[str] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = None

# ============================================================
# TOOL-SPECIFIC OBJECTS
# Tool example pairs are used to show structured input/output examples.
# ============================================================
class ToolIOPair(BaseModel):
    """
    Represents a single example mapping structured input fields to structured
    output fields for a tool.
    """
    input: Dict[str, Any]
    output: Dict[str, Any]

# ============================================================
# GRAPH-SPECIFIC OBJECTS
# Edge encodes directed relationships between nodes in a graph.
# ============================================================
class Edge(BaseModel):
    """
    Represents a directed edge in the graph-level wiring, connecting one node
    to another and optionally annotating the transition with a condition or
    descriptive text.
    """
    from_: str
    to: str
    condition: Optional[str] = None
    description: Optional[str] = None

# ============================================================
# AGENT-SPECIFIC OBJECTS
# Agent type is a lightweight label set to identify the agent style.
# ============================================================
AgentTypeLabel = Literal[
    "ReAct",
    "CodeAct",
    "Orchestrator",
    "Planner",
    "Executor",
    "other",
]

class AgentType(BaseModel):
    """
    Describes the agent style or pattern implemented in the graph.
    """
    type: Set[AgentTypeLabel] = Field(default_factory=set)
    other_description: Optional[str] = None  # must be set iff {"other"}

    @model_validator(mode="after")
    def validate_other(self):
        has_other = "other" in self.type

        # "other" must be selected alone
        if has_other and len(self.type) != 1:
            raise ValueError("'other' must be the only selected agent type when present")

        # other_description iff "other"
        if has_other:
            if not self.other_description:
                raise ValueError("other_description is required when type={'other'}")
        else:
            if self.other_description is not None:
                raise ValueError("other_description must be None unless type includes 'other'")

        return self

# ============================================================
# SYSTEM-SPECIFIC OBJECTS
# System type captures high-level orchestration patterns.
# ============================================================
SystemTypeLabel = Literal["Sequential", "Orchestrator", "Router", "Loop", "Hierarchical", "Single_agent", "other"]

class SystemType(BaseModel):
    """
    Describes the system style or pattern implemented in the graph.
    """
    type: Set[SystemTypeLabel] = Field(default_factory=set)
    other_description: Optional[str] = None   # required when type == "other"

    @model_validator(mode="after")
    def validate_other(self):
        has_other = "other" in self.type

        # "other" must be selected alone
        if has_other and len(self.type) != 1:
            raise ValueError("'other' must be the only selected system type when present")

        # other_description iff "other"
        if has_other:
            if not self.other_description:
                raise ValueError("other_description is required when type={'other'}")
        else:
            if self.other_description is not None:
                raise ValueError("other_description must be None unless type includes 'other'")

        return self

# ============================================================
# TOP-LEVEL OBJECTS
# These are used on the root graph or to describe top-level metadata.
# ============================================================
class ExampleArgument(BaseModel):
    """
    Describes a single example argument for a usage example, specifying its
    name, type, and an example value.
    """
    name: str
    type: str
    required: bool = False
    example: str
    is_task_input: bool = False

class UsageExample(BaseModel):
    """
    Represents an example of how to invoke an agent. Includes the script path and example arguments.
    """
    script: str
    arguments: List[ExampleArgument]

class AgencyLevel(BaseModel):
    """
    Indicates the degree of autonomy using a numeric scale from 0 to 4.
    """
    level: Literal[0, 1, 2, 3, 4]

class FlowToolCallSpec(BaseModel):
    """
    Tool-call event payload extracted from traces for flow reconstruction.
    """
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    tool_id: str = ""


class FlowEventSpec(BaseModel):
    """
    A single event in a simple flow trace.
    """
    seq: int
    node_id: str = ""
    type: str
    name: str
    time_ns: Optional[int] = None
    content: Optional[str] = None
    tool_calls: Optional[List[FlowToolCallSpec]] = None


class FlowSpec(BaseModel):
    """
    Simple-flow artifact derived from runtime traces.
    """
    flow_id: int
    flow_index: Optional[int] = None
    flow_signature_ordered: List[str] = Field(default_factory=list)
    flow_signature_key: str = ""
    flow_signature_id: str = ""
    flow_unique_components: List[str] = Field(default_factory=list)
    parsed_trace_file: str = ""
    source_trace_file: str = ""
    input_args: Dict[str, Any] = Field(default_factory=dict)
    events: List[FlowEventSpec] = Field(default_factory=list)
    invoked_tools: List[str] = Field(default_factory=list)
    invoked_agents: List[str] = Field(default_factory=list)

# ============================================================
# MAIN NODE SPEC
# NodeSpec represents any runtime node (or graph) in the system.
# It is recursive: a graph is just a NodeSpec with child nodes and edges.
# ============================================================
class NodeSpec(BaseModel):
    """
    Represents a single node or graph in the agent system, including structure,
    I/O interface, and type-specific configuration.
    """
    # Hard coded version
    version: ClassVar[str] = "2.0"

    # Override model_dump so the ClassVar `version` (normally excluded) is stamped into every dump.
    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["version"] = self.version
        return data

    # Identity and role (all nodes have)
    name: str
    id: Optional[str] = None

    node_type: NodeType                # functional type (LLM, Tool, Graph, etc.)
    description: str
    code_execution: Optional[bool] = False        # whether this node/graph executes code

    # Agent-specific configuration
    agent_type: Optional[AgentType] = None       # e.g. ReAct
    # System-specific configuration
    system_type: Optional[SystemType] = None       # e.g. Sequential

    # Implementation references (evidence in code)
    code_references: Optional[List[CodeReference]] = None
    emulated_code_references: Optional[List[CodeReference]] = None

    # I/O interface (optional if not known)
    inputs: List[InputPort] = Field(default_factory=list)
    outputs: List[OutputPort] = Field(default_factory=list)

    # Local adjacency inside a graph (rarely used in generated output)
    external_connections: Optional[Union[Connection, List[Connection]]] = None

    # Important indicators (keys, duplication)
    required_keys: RequiredKeys = RequiredKeys()
    duplicates: Duplication = Duplication()
    is_graph: Optional[bool] = False   # "graph" or "node"
    emulated: Optional[bool] = False

    # Top-level configuration
    framework: Optional[FrameworkType] = None    # e.g. LangGraph
    agency_level: Optional[AgencyLevel] = None
    flows: List[FlowSpec] = Field(default_factory=list)
    entry_point_usage_example: Optional[UsageExample] = None
    system_summary: Optional[str] = None
    # Root-only: the induced runtime alias registry (agent/tool name aliases, conflicts, pointer profile) from the execution-validation stage.
    runtime_mapping: Optional[Dict[str, Any]] = None


    # LLM-specific configuration
    llm_config: Optional[LLMConfig] = None
    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None

    # MCP-specific configuration
    tool_list: Optional[List["NodeSpec"]] = None

    # Tool-specific configuration
    tool_example_pairs: Optional[List[ToolIOPair]] = None
    read_internal: Optional[bool] = False
    read_external: Optional[bool] = False
    write_internal: Optional[bool] = False
    write_external: Optional[bool] = False
    is_rag_tool: Optional[bool] = False

    # Graph-specific configuration
    nodes: Optional[List["NodeSpec"]] = None
    internal_edges: Optional[List[Edge]] = None

    # Database configuration
    data_path: Optional[str] = None

    # Free-form extension point
    metadata: Optional[Dict[str, Any]] = None

    # ============================================================
    # Internal methods
    # These helper methods are used for loading, exporting, and traversing the tree.
    # ============================================================
    @classmethod
    def from_json(cls: Type[T], *, data: Dict[str, Any] | None = None, path: str | None = None) -> T:
        """
        Load a NodeSpec from a JSON dict or from a file path.
        """
        if data is None:
            if path is None:
                raise ValueError("Either `data` or `path` must be provided")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        return cls(**data)


    def to_json(self, *, path: str | None = None) -> Dict[str, Any]:
        """
        Serialize this NodeSpec to a JSON-compatible dict.
        If `path` is provided, also writes it to disk.
        """
        data = self.model_dump(mode="json", exclude_none=True)

        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        return data


    def iter_children(self) -> List["NodeSpec"]:
        """Direct children of this node (nodes + tool_list)."""
        out: List["NodeSpec"] = []
        if self.nodes:
            out.extend(self.nodes)
        if self.tool_list:
            out.extend(self.tool_list)
        return out


    def iter_descendants(self, include_self: bool = True) -> Iterator["NodeSpec"]:
        """DFS over the subtree (nodes + tool_list)."""
        stack: List["NodeSpec"] = [self] if include_self else self.iter_children()
        while stack:
            cur = stack.pop()
            yield cur
            stack.extend(cur.iter_children())


    def get_node(self, node_id: str) -> Optional["NodeSpec"]:
        for n in self.iter_descendants(include_self=True):
            if n.id == node_id:
                return n
        return None


    def list_by_types(self, types: Set[str]) -> List[Dict[str, Any]]:
        """
        Return one entry per (name, type) where multiple nodes with the same name+type
        are merged into a single entry with ids=[...].
        """
        groups: dict[tuple[str, str], set[str]] = defaultdict(set)

        for n in self.iter_descendants(include_self=True):
            t = getattr(n.node_type, "type", None)
            if t in types:
                groups[(n.name, t)].add(n.id)

        results: List[Dict[str, Any]] = []
        for (name, t), ids in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):  # (type, name)
            results.append({"name": name, "type": t, "ids": sorted(ids)})

        return results


    def list_tools(self) -> List[Dict[str, Any]]:
        return self.list_by_types({"Tool"})


    def list_agents(self) -> List[Dict[str, Any]]:
        return self.list_by_types({"Agent"})


    def list_systems(self) -> List[Dict[str, Any]]:
        return self.list_by_types({"System"})


    def list_llms(self) -> List[Dict[str, Any]]:
        return self.list_by_types({"LLM"})


    def list_mcp_servers(self) -> List[Dict[str, Any]]:
        return self.list_by_types({"Local_MCP_server", "External_MCP_server"})


    def get_nodes_with_snipets_from_given_file(self, file_path: str) -> List["NodeSpec"]:
        res = []
        for n in self.iter_descendants(include_self=True):
            if n.code_references:
                for code_ref in n.code_references:
                    if code_ref.file == file_path:
                        res.append(n.id)
                        break

        return res

    # ============================================================
    # Internal validators
    # ============================================================
    @model_validator(mode="after")
    def validate_tool_examples(self, info: ValidationInfo):
        """
        Ensure that tool_example_pairs are structurally consistent with the
        tool's declared inputs when this node is a Tool.
        """

        node_type = self.node_type
        inputs = self.inputs or []
        examples = self.tool_example_pairs or []

        # tool_example_pairs is only valid on Tool nodes.
        if examples and (not node_type or node_type.type != "Tool"):
            raise ValueError(
                f"Node '{self.name}' has tool_example_pairs but node_type="
                f"'{getattr(node_type, 'type', None)}' (only allowed on Tool nodes)."
            )

        # Nothing to check for non-Tool nodes or Tool nodes without examples.
        if not node_type or node_type.type != "Tool" or not examples:
            return self

        # Example.input must be a dict with exact keys = input names
        required_names = {slot.name for slot in inputs if getattr(slot, "required", False)}
        allowed_names = {slot.name for slot in inputs}  # required + optional
        for ex in examples:
            if not isinstance(ex.input, dict):
                raise ValueError(
                    f"Tool node '{self.name}' expects example.inputs as dict "
                    f"with keys subset of {sorted(allowed_names)}, got non-dict: {type(ex.input)}"
                )

            actual_keys = set(ex.input.keys())
            missing_required = required_names - actual_keys
            if missing_required:
                raise ValueError(
                    f"Tool node '{self.name}' example.input is missing required keys "
                    f"{sorted(missing_required)} (required={sorted(required_names)})"
                )
            unknown_keys = actual_keys - allowed_names
            if unknown_keys:
                raise ValueError(
                    f"Tool node '{self.name}' example.input has unknown keys {sorted(unknown_keys)}; "
                    f"allowed keys are {sorted(allowed_names)}"
                )
        return self

    @model_validator(mode="after")
    def propagate_io_flags_bottom_up(self, info: ValidationInfo) -> "NodeSpec":
        """
        Bottom-up closure for I/O capability flags:
        - First closes children (nodes + tool_list)
        - If any direct child has a given flag=True, parent gets that flag=True
        - Applies recursively so True values flow to the root
        """

        # Recurse first so descendants are already closed.
        for child in (self.nodes or []):
            child.propagate_io_flags_bottom_up(info)
        for tool in (self.tool_list or []):
            tool.propagate_io_flags_bottom_up(info)

        kids = list(self.nodes or []) + list(self.tool_list or [])
        if not kids:
            return self

        if any(bool(ch.read_internal) for ch in kids):
            self.read_internal = True
        if any(bool(ch.read_external) for ch in kids):
            self.read_external = True
        if any(bool(ch.write_internal) for ch in kids):
            self.write_internal = True
        if any(bool(ch.write_external) for ch in kids):
            self.write_external = True

        return self

    @model_validator(mode="after")
    def propagate_code_execution_bottom_up(self, info: ValidationInfo) -> "NodeSpec":
        """
        Bottom-up closure for code_execution:
        - First closes children (nodes + tool_list)
        - If any direct child has code_execution=True, parent is set to True
        - Does not force False when no child executes code
        """

        # 1) Recurse first so descendants are already closed.
        for child in (self.nodes or []):
            child.propagate_code_execution_bottom_up(info)
        for tool in (self.tool_list or []):
            tool.propagate_code_execution_bottom_up(info)

        # 2) If any child/tool executes code, parent must also be marked True.
        child_exec_true = any(bool(ch.code_execution) for ch in (self.nodes or []))
        tool_exec_true = any(bool(t.code_execution) for t in (self.tool_list or []))
        if child_exec_true or tool_exec_true:
            self.code_execution = True

        return self

    @model_validator(mode="after")
    def validate_code_reference_uniqueness(self, info: ValidationInfo):

        # Kinds that must appear at most once per node
        unique_kinds = {"definition", "assignment", "system_prompt"}

        counts = Counter(cr.kind for cr in (self.code_references or []))
        duplicates = {k: v for k, v in counts.items() if k in unique_kinds and v > 1}

        if duplicates:
            # Helpful error message (include node id if you have it)
            node_id = getattr(self, "id", "<unknown_node>")
            details = ", ".join(f"{k}={v}" for k, v in sorted(duplicates.items()))
            raise ValueError(
                f"NodeSpec(id={node_id}) has multiple CodeReferences for unique kinds: {details}. "
                f"Keep at most one of each: {sorted(unique_kinds)}."
            )

        return self

    @model_validator(mode="after")
    def propagate_required_keys_bottom_up(self, info: ValidationInfo) -> "NodeSpec":
        """
        Bottom-up closure:
        - First closes children (nodes + tool_list)
        - Then unions required keys from self + all descendants
        - Stores the union on this node (enabling/disabling accordingly)
        """

        # Bottom-up recursion: close all children first
        for child in (self.nodes or []):
            child.propagate_required_keys_bottom_up(info)
        for tool in (self.tool_list or []):
            tool.propagate_required_keys_bottom_up(info)

        # Collect keys from self and children
        merged: Set[str] = set()

        # self keys
        if self.required_keys.enabled and self.required_keys.keys:
            merged.update(k for k in self.required_keys.keys if k)

        # children keys
        for child in (self.nodes or []):
            rk = child.required_keys
            if rk.enabled and rk.keys:
                merged.update(k for k in rk.keys if k)

        # tool children keys
        for tool in (self.tool_list or []):
            rk = tool.required_keys
            if rk.enabled and rk.keys:
                merged.update(k for k in rk.keys if k)

        # Write back in a way that respects RequiredKeys.validate_keys()
        if merged:
            self.required_keys.enabled = True
            self.required_keys.keys = sorted(merged)
        else:
            self.required_keys.enabled = False
            self.required_keys.keys = None

        return self

class NodeSpecTop(NodeSpec):
    """
    Root-only model that performs global normalization/validation across the whole tree:
    - auto-generate ids from names
    - rewrite edges and external connections
    - enforce uniqueness
    """

    # Root-only pass: auto-ids, edge rewrites, and duplicate marking.
    @model_validator(mode="after")
    def auto_ids_and_rewrite_refs(self, info: ValidationInfo) -> "NodeSpecTop":
        if not (info.context or {}).get("run_root_pass", False):
            return self

        seen: set[str] = set()
        shareable_alias_types = {"Tool", "LLM", "Local_MCP_server", "External_MCP_server", "Database"}

        def _norm(s: str | None) -> str:
            return "".join(ch for ch in str(s or "").lower() if ch.isalnum())

        # If the same tool instance is referenced by multiple parents, clone it so each parent
        # gets its own concrete NodeSpec instance (same content, distinct id).
        def split_shared_components(root: "NodeSpec") -> None:
            first_parent_by_obj: Dict[int, str] = {}

            def rec(parent: "NodeSpec") -> None:
                for fld in ("nodes", "tool_list"):
                    vals = getattr(parent, fld) if isinstance(getattr(parent, fld), list) else []
                    if not isinstance(vals, list):
                        continue
                    for i, ch in enumerate(vals):
                        if not isinstance(ch, NodeSpec):
                            continue
                        ctype = ch.node_type.type if ch.node_type else ""
                        if ctype in shareable_alias_types:
                            obj_key = id(ch)
                            first_parent = first_parent_by_obj.get(obj_key)
                            if first_parent is None:
                                first_parent_by_obj[obj_key] = parent.name
                            elif first_parent != parent.name:
                                cloned = ch.model_copy(deep=True)
                                vals[i] = cloned
                                first_parent_by_obj[id(cloned)] = parent.name
                        rec(vals[i])

            rec(root)

        # For aliasable components, keep only parent-specific external_connections.
        def filter_connections_for_parent(
            n: "NodeSpec",
            *,
            parent_name: str | None,
            parent_id: str | None,
            scope_map: dict[str, str] | None,
        ) -> None:
            if not parent_name:
                return
            ctype = n.node_type.type if n.node_type else ""
            if ctype not in shareable_alias_types:
                return
            conn_list = n.external_connections if isinstance(n.external_connections, list) else (
                [n.external_connections] if n.external_connections else []
            )
            if not conn_list:
                return
            # If parent is specified in the connections, select only this parent's entries.
            has_parent_tags = any(bool((c.parent or "").strip()) for c in conn_list if c is not None)
            kept: list[Connection] = []
            if has_parent_tags:
                pnorms = {_norm(parent_name), _norm(parent_id)}
                for c in conn_list:
                    if c is None:
                        continue
                    cparent = str(c.parent or "")
                    if _norm(cparent) in pnorms:
                        kept.append(c)
            else:
                kept = [c for c in conn_list if c is not None]

            if not kept:
                n.external_connections = None
                return

            # Scope sanitize: keep only refs valid in current parent scope.
            allowed = {"START", "END"}
            if scope_map:
                allowed.update(scope_map.keys())
                allowed.update(scope_map.values())
            if n.name:
                allowed.add(n.name)
            if n.id:
                allowed.add(n.id)

            pruned: list[Connection] = []
            for c in kept:
                ins = c.in_ if isinstance(c.in_, list) else [c.in_]
                outs = c.out if isinstance(c.out, list) else [c.out]
                in_f = [x for x in ins if isinstance(x, str) and x in allowed]
                out_f = [x for x in outs if isinstance(x, str) and x in allowed]
                if not in_f and not out_f:
                    continue
                pruned.append(
                    Connection(
                        parent=c.parent,
                        in_=in_f if len(in_f) != 1 else in_f[0],
                        out=out_f if len(out_f) != 1 else out_f[0],
                    )
                )
            n.external_connections = pruned if pruned else None

        def qualify(parent_id: str | None, nm: str) -> str:
            return nm if not parent_id else f"{parent_id}_{nm}"

        def rewrite(refs, scope_map: dict[str, str]):
            def one(x: str) -> str:
                return scope_map.get(x, x)  # START/END or already-qualified stay
            if isinstance(refs, str): return one(refs)
            return [one(x) for x in refs]

        def walk(n: "NodeSpec", parent_id: str | None, scope_map: dict[str, str] | None, parent_name: str | None):
            # set this node's id if missing/unqualified
            if n.id is None or n.id == n.name:
                n.id = qualify(parent_id, n.name)

            if n.id in seen:
                raise ValueError(f"duplicate qualified id '{n.id}'")
            seen.add(n.id)

            # build THIS node's scope map for resolving references to its children
            kids = n.iter_children()
            this_scope: dict[str, str] = {}
            for ch in kids:
                ch_id = ch.id if (ch.id is not None and ch.id != ch.name) else qualify(n.id, ch.name)
                this_scope[ch.name] = ch_id

            # rewrite THIS node's internal edges using THIS scope
            if n.internal_edges:
                for e in n.internal_edges:
                    if isinstance(getattr(e, "from_", None), str):
                        e.from_ = rewrite(e.from_, this_scope)
                    if isinstance(getattr(e, "to", None), str):
                        e.to = rewrite(e.to, this_scope)

            # rewrite THIS node's external connections using PARENT scope (siblings live there)
            if n.external_connections and scope_map is not None:
                conn_list = n.external_connections if isinstance(n.external_connections, list) else [n.external_connections]
                new_conn: list[Connection] = []
                for c in conn_list:
                    if c is None:
                        continue
                    parent_ref = c.parent or ""
                    parent_ref = rewrite(parent_ref, scope_map) if parent_ref else parent_ref
                    in_refs = rewrite(c.in_, scope_map)
                    out_refs = rewrite(c.out, scope_map)
                    new_conn.append(Connection(parent=parent_ref, in_=in_refs, out=out_refs))
                n.external_connections = new_conn
            filter_connections_for_parent(n, parent_name=parent_name, parent_id=parent_id, scope_map=scope_map)

            # recurse into children, passing THIS node's scope as their parent scope
            for ch in kids:
                walk(ch, n.id, this_scope, n.name)

        # root has no parent scope
        split_shared_components(self)
        walk(self, parent_id=None, scope_map=None, parent_name=None)

        # mark duplicates by exact name (now that IDs exist)
        all_nodes: list["NodeSpec"] = []
        stack: list["NodeSpec"] = [self]
        while stack:
            cur = stack.pop()
            all_nodes.append(cur)
            stack.extend(cur.iter_children())

        # reset
        for n in all_nodes:
            n.duplicates = Duplication(exists=False, instances=None)

        # group by exact name only for shareable runtime components
        groups = defaultdict(list)
        for n in all_nodes:
            ntype = n.node_type.type if n.node_type else ""
            if ntype in shareable_alias_types:
                groups[n.name].append(n)

        # mark
        for group in groups.values():
            if len(group) <= 1:
                continue
            instances = []
            for g in group:
                if not g.id:
                    continue
                parent_id = g.id.rsplit("_", 1)[0] if "_" in g.id else ""
                conn_list = g.external_connections if isinstance(g.external_connections, list) else (
                    [g.external_connections] if g.external_connections else []
                )
                conn_snap = []
                for c in conn_list:
                    if c is None:
                        continue
                    in_refs = c.in_ if isinstance(c.in_, list) else [c.in_]
                    out_refs = c.out if isinstance(c.out, list) else [c.out]
                    conn_snap.append({"parent": c.parent or "", "in_": in_refs, "out": out_refs})
                instances.append({"id": g.id, "parent_id": parent_id, "connections": conn_snap})
            if not instances:
                continue
            for g in group:
                others = [inst for inst in instances if str(inst.get("id") or "") != str(g.id or "")]
                if others:
                    g.duplicates = Duplication(exists=True, instances=others)
                else:
                    g.duplicates = Duplication(exists=False, instances=None)

        return self

    # Root-only pass over every graph node: (1) all edge/connection refs point at an existing
    # node (or START/END), and (2) parent internal_edges and child external_connections agree
    # bidirectionally (edge X->Y <=> Y in X.out and X in Y.in_). Gated to the root so
    # partially-built NodeSpecs during construction are never affected.
    @model_validator(mode="after")
    def validate_graph_references(self, info: ValidationInfo) -> "NodeSpecTop":
        if not (info.context or {}).get("run_root_pass", False):
            return self

        allowed_special = {"START", "END"}

        for graph in self.iter_descendants(include_self=True):
            kids = graph.iter_children()
            if not graph.is_graph or not kids:
                continue

            # Canonicalize any ref (child name or id) to the child's id within this scope.
            ref_to_id: Dict[str, str] = {}
            for k in kids:
                cid = k.id or k.name
                if isinstance(k.name, str) and k.name:
                    ref_to_id[k.name] = cid
                if isinstance(k.id, str) and k.id:
                    ref_to_id[k.id] = cid
            valid_refs = set(ref_to_id)

            def canon(ref: str) -> str:
                return ref if ref in allowed_special else ref_to_id.get(ref, ref)

            def _check_ref(ref: str, context: str):
                if ref not in valid_refs and ref not in allowed_special:
                    raise ValueError(
                        f"Invalid reference '{ref}' in {context}: "
                        f"not a node id/name in this graph (valid: {sorted(valid_refs)})"
                    )

            real_nodes = {canon(k.id or k.name) for k in kids}

            # (1) existence checks + (2) gather connection adjacency, split by side.
            out_pairs: Set[Tuple[str, str]] = set()   # (source, target) from source.out
            in_pairs: Set[Tuple[str, str]] = set()    # (source, target) from target.in_
            for node in kids:
                kid = canon(node.id or node.name)
                conn = node.external_connections
                if conn is None:
                    continue
                conn_list = conn if isinstance(conn, list) else [conn]
                for c in conn_list:
                    if c is None:
                        continue
                    ins = c.in_ if isinstance(c.in_, list) else [c.in_]
                    outs = c.out if isinstance(c.out, list) else [c.out]
                    for ref in ins:
                        _check_ref(ref, f"connections.in_ of node '{node.name}'")
                        if isinstance(ref, str):
                            in_pairs.add((canon(ref), kid))
                    for ref in outs:
                        _check_ref(ref, f"connections.out of node '{node.name}'")
                        if isinstance(ref, str):
                            out_pairs.add((kid, canon(ref)))

            # edges: existence checks + gather declared edge pairs.
            edge_pairs: Set[Tuple[str, str]] = set()
            for edge in (graph.internal_edges or []):
                _check_ref(edge.from_, f"edge.from_ in graph '{graph.name}'")
                _check_ref(edge.to, f"edge.to in graph '{graph.name}'")
                if isinstance(edge.from_, str) and isinstance(edge.to, str):
                    edge_pairs.add((canon(edge.from_), canon(edge.to)))

            # (2) bidirectional alignment. An edge needs an out-declaration only if its source
            # is a real node, and an in-declaration only if its target is a real node
            # (START has no out owner, END has no in_ owner).
            missing_out = {p for p in edge_pairs if p[0] in real_nodes} - out_pairs
            missing_in = {p for p in edge_pairs if p[1] in real_nodes} - in_pairs
            extra_out = out_pairs - edge_pairs
            extra_in = in_pairs - edge_pairs

            if missing_out or missing_in or extra_out or extra_in:
                raise ValueError(
                    f"Edge/connection misalignment in graph '{graph.id or graph.name}':\n"
                    f"  edges missing from source.out (Y not in X.out): {sorted(missing_out)}\n"
                    f"  edges missing from target.in_ (X not in Y.in_): {sorted(missing_in)}\n"
                    f"  connection.out entries with no matching edge: {sorted(extra_out)}\n"
                    f"  connection.in_ entries with no matching edge: {sorted(extra_in)}\n"
                    f"Each edge X->Y must have Y in X.out and X in Y.in_, and vice-versa."
                )

        return self

# IMPORTANT: enable recursive NodeSpec type
NodeSpec.model_rebuild()
