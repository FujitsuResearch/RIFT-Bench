from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Literal, DefaultDict
from pydantic import BaseModel, Field, model_validator, ValidationInfo
from collections import Counter, defaultdict
import json


# ============================================================
# BASIC OBJECTS
# ============================================================
class NodeType(BaseModel):
    """
    Specifies the functional type of a node, such as LLM, Tool, Router, or Graph.
    """

    type: Literal[
        "LLM",
        "Tool",
        "System",
        "Local_MCP_server",
        "External_MCP_server",
        "Agent",
        "Deterministic_controller",
        "other",
    ]
    other_description: Optional[str] = None  # required when type == "other"
    tool_type: Optional[
        Literal["api_tool", "Local_MCP_server", "External_MCP_server", "custom_tool"]
    ] = None  # required when type == "Tool"

    @model_validator(mode="after")
    def validate_other(self):
        if self.type == "other":
            if not self.other_description:
                raise ValueError(
                    "NodeType: other_description is required when type='other'"
                )
        elif self.type == "Tool":
            if not self.tool_type:
                raise ValueError("NodeType: tool_type is required when type='Tool'")
        else:
            if self.other_description is not None:
                raise ValueError(
                    "NodeType: other_description must be None unless type='other'"
                )
        return self


class CodeReference(BaseModel):
    """
    Represents a reference to a specific location in the codebase that is
    relevant to the definition or behavior of the node.
    """

    kind: Literal[
        # the ffollowing "kinds" are code references for instances (represented as nodes in our schema) such as an agent, a tool , an LLM, etc.
        "source",  # reference to a source import, can be local code or from a python package.
        "definition",  # reference to a definition of an instance.
        "implementation",  # reference to implementation of an instance.
        "initialization",  # reference to an initialization of an instance.
        "usage",  # refernce to a usage of an instance.
        "LLM_config",  # refernce to where the LLM configurations are defined.
        "system_prompt",  # reference to the system prompt in the code.
        "user_prompt_template",  # refernce to a user prompt template used as input to an LLM.
        "input_scheme",  # reference to an input schema in the code.
        "output_scheme",  # reference to an output schema in the code.
        "logic",  # refernce to deterministic logic implementation.
        "other",  # any other refernce that is not one of the above. must include the following field with additional descriptions.
    ]
    other_kind_description: Optional[str] = None  # Only used when kind == "other"
    file: str
    line: Union[int, Tuple[int, int]]
    snippet: str

    @model_validator(mode="after")
    def validate_other(self):
        if self.kind == "other":
            if not self.other_kind_description:
                raise ValueError(
                    "CodeReference: other_kind_description is required when kind='other'"
                )
        else:
            if self.other_kind_description is not None:
                raise ValueError(
                    "CodeReference: other_kind_description must be None unless kind='other'"
                )
        return self


class RequiredKeys(BaseModel):
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
                raise ValueError("keys must be None or empty when enabled=False")
        return self


class Duplication(BaseModel):
    exists: bool = False
    ids: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_keys(self):
        if self.exists:
            if not self.ids or len(self.ids) == 0:
                raise ValueError(
                    "Ids must be provided and non-empty when exists=True (duplicates exist)"
                )
        else:
            if self.ids:
                raise ValueError(
                    "Ids must be None or empty when exists=False (duplicates do not exist)"
                )
        return self


class IOSlot(BaseModel):
    """
    A single named slot in the node's input/output interface.
    Used as a base for both inputs and outputs.
    """

    name: str  # e.g. "messages", "tool_call"
    dtype: str  # e.g. "Message[]", "string", "boolean", "ToolCall"
    description: Optional[str] = None  # human-readable explanation


class InputPort(IOSlot):
    """
    Input specification for a node.
    """

    required: bool = True  # whether the caller must provide it
    default: Optional[Any] = None  # used when required=False and not provided


class OutputPort(IOSlot):
    """
    Output specification for a node.
    """

    output_kind: Literal[
        "data",  # direct informational output (e.g., text, messages, JSON)
        "action",  # any additional action / side effect.
    ] = "data"


class Connection(BaseModel):
    """
    Describes local adjacency for a node by listing the upstream and downstream
    node identifiers it is connected to.
    """

    in_: Union[str, List[str]]
    out: Union[str, List[str]]


# ============================================================
# LLM-SPECIFIC OBJECTS
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
# Router-SPECIFIC OBJECTS
# ============================================================
class RoutingTest(BaseModel):
    """
    A single boolean test on a routing indicator, e.g. has_tool_call == true.
    """

    indicator: str  # must match one of RoutingLogic.inputs[*].name
    condition: bool  # expected value


class RoutingCondition(BaseModel):
    """
    A boolean expression over routing tests, supporting nested AND/OR trees.
    """

    op: Literal["TEST", "AND", "OR"]
    test: Optional[RoutingTest] = None  # used when op == "TEST"
    children: Optional[List["RoutingCondition"]] = None  # used when op in {"AND", "OR"}


RoutingCondition.model_rebuild()  # To enable recursive RoutingCondition


class RoutingRule(BaseModel):
    """
    A single routing rule mapping a boolean condition tree to a target node.
    """

    condition: RoutingCondition
    target: str


class RoutingLogic(BaseModel):
    """
    Defines how a router chooses its next node. Each rule combines tests on
    routing inputs and maps the result to a target node identifier.
    """

    inputs: List[IOSlot]
    rules: List[RoutingRule]

    @model_validator(mode="after")
    def validate_rule_inputs(self):
        inputs = self.inputs or []
        rules = self.rules or []

        allowed = {slot.name for slot in inputs}

        for rule in rules:
            for test in self._iter_tests(rule.condition):
                if test.indicator not in allowed:
                    raise ValueError(
                        f"RoutingTest.indicator '{test.indicator}' is not defined "
                        f"in RoutingLogic.inputs (allowed: {sorted(allowed)})"
                    )
        return self

    @staticmethod
    def _iter_tests(condition: RoutingCondition) -> List[RoutingTest]:
        """
        Collect all RoutingTest leaves from a RoutingCondition tree.
        """
        if condition.op == "TEST" and condition.test is not None:
            return [condition.test]
        if condition.children:
            tests: List[RoutingTest] = []
            for child in condition.children:
                tests.extend(RoutingLogic._iter_tests(child))
            return tests
        return []


# ============================================================
# Tool-SPECIFIC OBJECTS
# ============================================================
class ToolIOPair(BaseModel):
    """
    Represents a single example mapping structured input fields to structured
    output fields for a tool.
    """

    input: Dict[str, Any]
    output: Dict[str, Any]


# ============================================================
# Graph-SPECIFIC OBJECTS
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
# Agent-SPECIFIC OBJECTS
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
            raise ValueError(
                "'other' must be the only selected agent type when present"
            )

        # other_description iff "other"
        if has_other:
            if not self.other_description:
                raise ValueError("other_description is required when type={'other'}")
        else:
            if self.other_description is not None:
                raise ValueError(
                    "other_description must be None unless type includes 'other'"
                )

        return self


# ============================================================
# System-SPECIFIC OBJECTS
# ============================================================
SystemTypeLabel = Literal[
    "Sequential", "Orchestrator", "Router", "Loop", "Hierarchical", "other"
]


class SystemType(BaseModel):
    """
    Describes the system style or pattern implemented in the graph.
    """

    type: Set[SystemTypeLabel] = Field(default_factory=set)
    other_description: Optional[str] = None  # required when type == "other"

    @model_validator(mode="after")
    def validate_other(self):
        has_other = "other" in self.type

        # Rule 1: "other" must be selected alone
        if has_other and len(self.type) != 1:
            raise ValueError(
                "'other' must be the only selected system type when present"
            )

        # Rule 2: other_description iff "other"
        if has_other:
            if not self.other_description:
                raise ValueError("other_description is required when type={'other'}")
        else:
            if self.other_description is not None:
                raise ValueError(
                    "other_description must be None unless type includes 'other'"
                )

        return self


# ============================================================
# Top-level SPECIFIC OBJECTS
# ============================================================
class FrameworkType(BaseModel):
    """
    Identifies the framework used to construct this graph, such as LangGraph or LangChain.
    """

    framework: Literal["LangGraph", "AutoGen", "CrewAI", "other"]
    other_description: Optional[str] = None  # required when framework == "other"

    @model_validator(mode="after")
    def validate_other(self):
        if self.framework == "other" and not self.other_description:
            raise ValueError("other_description is required when framework='other'")
        return self


class ExampleArgument(BaseModel):
    """
    Describes a single example argument for a usage example, specifying its
    name, type, and an example value.
    """

    name: str
    type: str
    example: str


class UsageExample(BaseModel):
    """
    Represents an example of how to invoke an agent. Includes the script path and example arguments.
    """

    script: str
    arguments: List[ExampleArgument]


class AgencyLevel(BaseModel):
    """
    Indicates the agency level using a numeric scale from 0 to 5, with tool-based
    and planning-based sub-levels.
    """

    level: Literal[0, 1, 2, 3, 4, 5]
    description: str
    tool_agency_level: Literal[0, 1, 2, 3, 4]
    planning_agency_level: Literal[0, 1, 2, 3, 4]

    @model_validator(mode="after")
    def validate_level(self):
        agency_level = self.level
        if agency_level is None:
            raise ValueError("agency level is required")
        return self

    @model_validator(mode="after")
    def validate_tool_agency_level(self):
        tool_agency_level = self.tool_agency_level
        if tool_agency_level is None:
            raise ValueError("tool_agency_level is required")
        return self

    @model_validator(mode="after")
    def validate_planning_agency_level(self):
        planning_agency_level = self.planning_agency_level
        if planning_agency_level is None:
            raise ValueError("planning_agency_level is required")
        return self


class FlowSpec(BaseModel):
    """
    A class for flow specifications. This defines a flow and associates it with a cluster_id. Each flow must activate atleast one system, agent or llm.
    """

    cluster_id: str
    sample: Dict[str, Any]
    tools_invoked: List[str] = []
    systems_invoked: List[str] = []
    agents_invoked: List[str] = []
    llms_invoked: List[str] = []
    others: List[str] = []

    @model_validator(mode="after")
    def validate_llm_or_agent(self):
        if (
            self.llms_invoked is None
            and self.agents_invoked is None
            and self.systems_invoked is None
        ):
            raise ValueError(
                "Atleast one of LLM, Agents or Systems must be present in a flow"
            )
        return self


# ============================================================
# MAIN NODE SPEC
# ============================================================
class NodeSpec(BaseModel):
    """
    Represents a single node or graph in the agent system, including structure,
    I/O interface, and type-specific configuration.
    """

    # Hard coded version
    version: ClassVar[str] = "0.1"

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["version"] = self.version
        return data

    # Identity and role (all nodes have)
    name: str
    id: Optional[str] = None
    is_graph: Optional[bool] = False  # "graph" or "node"
    emulated: Optional[bool] = False
    node_type: NodeType  # functional type (LLM, Tool, Graph, etc.)
    description: str
    code_execution: Optional[bool] = False  # whether this node/graph executes code

    # Implementation references
    code_references: Optional[List[CodeReference]] = None
    emulated_code_references: Optional[List[CodeReference]] = None

    # I/O interface
    inputs: List[InputPort] = Field(default_factory=list)
    outputs: List[OutputPort] = Field(default_factory=list)

    # Local adjacency inside a graph
    external_connections: Optional[Connection] = None

    # Important indicators
    required_keys: RequiredKeys = RequiredKeys()
    duplicates: Duplication = Duplication()

    # Top-level configuration
    framework: Optional[FrameworkType] = None  # e.g. LangGraph
    agency_level: Optional[AgencyLevel] = None
    flows: Optional[List[FlowSpec]] = None
    entry_point_usage_example: Optional[UsageExample] = None

    # Agent-specific configuration
    agent_type: Optional[AgentType] = None  # e.g. ReAct

    # System-specific configuration
    system_type: Optional[SystemType] = None  # e.g. Sequential

    # LLM-specific configuration
    llm_config: Optional[LLMConfig] = None
    system_prompt: Optional[str] = None
    user_prompt_template: Optional[str] = None

    # Router-specific configuration
    routing_logic: Optional[RoutingLogic] = None

    # MCP-specific configuration
    tool_list: Optional[List["NodeSpec"]] = None

    # Tool-specific configuration
    tool_example_pairs: Optional[List[ToolIOPair]] = None

    # Graph-specific configuration
    nodes: Optional[List["NodeSpec"]] = None
    internal_edges: Optional[List[Edge]] = None

    # Free-form extension point
    metadata: Optional[Dict[str, Any]] = None

    # ============================================================
    # Internal methods
    # ============================================================
    @classmethod
    def from_json(
        cls: Type[T], *, data: Dict[str, Any] | None = None, path: str | None = None
    ) -> T:
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
        data = self.model_dump(mode="json", exclude_none=False)

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

    def get_nodes_with_snipets_from_given_file(
        self, file_path: str
    ) -> List["NodeSpec"]:
        res = []
        for n in self.iter_descendants(include_self=True):
            if n.code_references:
                for code_ref in n.code_references:
                    if code_ref.file == file_path:
                        res.append(n.id)
                        break

        return res

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
        for (name, t), ids in sorted(
            groups.items(), key=lambda x: (x[0][1], x[0][0])
        ):  # (type, name)
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

    def get_desc(self) -> str:
        """
        Efficient raw-text description using ONLY merged inventories (list_* methods)
        and the invariant:
        id == <top>_<level1>_..._<current>
        No tree-walk required.
        """

        TOP = self.id or self.name  # top node name/id (they match in your scheme)

        # ---- pull merged inventories (name/type/ids) ----
        systems = self.list_systems()
        agents = self.list_agents()
        llms = self.list_llms()
        tools = self.list_tools()

        mcp_local = self.list_by_types({"Local_MCP_server"})
        mcp_external = self.list_by_types({"External_MCP_server"})

        # Convenience: flatten instance lists
        def flatten(entries: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
            # returns list of (name, id)
            out: List[Tuple[str, str]] = []
            for e in entries:
                nm = e["name"]
                for _id in e.get("ids", []) or []:
                    out.append((nm, _id))
            return out

        agent_instances = flatten(agents)  # (agent_name, agent_id)
        tool_instances = flatten(tools)  # (tool_name, tool_id)
        llm_instances = flatten(llms)  # (llm_name, llm_id)
        mcp_local_instances = flatten(mcp_local)  # (mcp_name, mcp_id)
        mcp_ext_instances = flatten(mcp_external)  # (mcp_name, mcp_id)

        # ---- totals (dedup by name already happened in list_*) ----
        total_systems = len(systems)
        total_agents = len(agents)
        total_llms = len(llms)
        total_tools = len(tools)
        total_mcp_local = len(mcp_local)
        total_mcp_ext = len(mcp_external)
        total_mcp = total_mcp_local + total_mcp_ext

        lines: List[str] = []
        lines.append("In total there are:")
        lines.append(f"- Systems: {total_systems}")
        lines.append(f"- Agents: {total_agents}")
        lines.append(f"- LLMs: {total_llms}")
        lines.append(f"- Tools: {total_tools}")
        lines.append(
            f"- MCP servers: {total_mcp} (local: {total_mcp_local}, external: {total_mcp_ext})"
        )
        lines.append("")

        # ---- build fast lookup: agent_id -> agent_name ----
        agent_id_to_name: Dict[str, str] = {
            aid: aname for aname, aid in agent_instances
        }

        # ---- derive agent_id from an instance id by taking first two segments: TOP + agent_name ----
        # invariant: <TOP>_<AgentName>_...
        def agent_id_from_instance_id(instance_id: str) -> str:
            # Find TOP_ prefix, then the next token is the agent name
            # instance_id is assumed valid and hierarchical.
            parts = instance_id.split("_")
            if not parts:
                return instance_id
            if parts[0] != TOP.split("_")[0]:
                # still try: agent id is first two tokens
                return "_".join(parts[:2]) if len(parts) >= 2 else instance_id
            # More robust: TOP may itself contain underscores, so match TOP prefix by string.
            # If instance startswith TOP_, strip it then take next token as agent name.
            prefix = TOP + "_"
            if instance_id.startswith(prefix):
                rest = instance_id[len(prefix) :]
                agent_name = rest.split("_", 1)[0]
                return f"{TOP}_{agent_name}"
            # fallback: first two tokens
            return "_".join(parts[:2]) if len(parts) >= 2 else instance_id

        # ---- per-agent aggregation (by instance ids) ----
        tools_by_agent: Dict[str, Set[str]] = defaultdict(
            set
        )  # agent_id -> set(tool_name)
        llms_by_agent: Dict[str, Set[str]] = defaultdict(
            set
        )  # agent_id -> set(llm_name)
        mcps_by_agent: Dict[str, Set[Tuple[str, str]]] = defaultdict(
            set
        )  # agent_id -> {(mcp_name, kind)}

        for tool_name, tool_id in tool_instances:
            a_id = agent_id_from_instance_id(tool_id)
            tools_by_agent[a_id].add(tool_name)

        for llm_name, llm_id in llm_instances:
            a_id = agent_id_from_instance_id(llm_id)
            llms_by_agent[a_id].add(llm_name)

        for mcp_name, mcp_id in mcp_local_instances:
            a_id = agent_id_from_instance_id(mcp_id)
            mcps_by_agent[a_id].add((mcp_name, "local"))

        for mcp_name, mcp_id in mcp_ext_instances:
            a_id = agent_id_from_instance_id(mcp_id)
            mcps_by_agent[a_id].add((mcp_name, "external"))

        # Agents in this system = all agent_ids that start with TOP_
        agent_ids_in_system = sorted(
            [aid for aid in agent_id_to_name.keys() if aid.startswith(TOP + "_")],
            key=lambda x: x,
        )

        # ---- per-system section (assume single top system id == TOP) ----
        lines.append(f"System: {TOP}")
        lines.append(f"- Agents ({len(agent_ids_in_system)}):")
        for aid in agent_ids_in_system:
            tnames = sorted(tools_by_agent.get(aid, set()))
            lnames = sorted(llms_by_agent.get(aid, set()))
            mcps = sorted(mcps_by_agent.get(aid, set()), key=lambda x: x[0])

            lines.append(f"  - {aid}")
            lines.append(
                f"    - Tools ({len(tnames)}): {', '.join(tnames) if tnames else ''}".rstrip()
            )
            lines.append(
                f"    - LLMs ({len(lnames)}): {', '.join(lnames) if lnames else ''}".rstrip()
            )
            if mcps:
                mcp_str = ", ".join([f"{n} [{k}]" for n, k in mcps])
                lines.append(f"    - MCP servers ({len(mcps)}): {mcp_str}")
            else:
                lines.append("    - MCP servers (0)")
        lines.append("")

        # ---- global summaries (already merged by name+type) ----
        def render_summary(
            title: str, entries: List[Dict[str, Any]], *, mcp_kind: str | None = None
        ) -> None:
            lines.append(f"{title}:")
            if not entries:
                lines.append("- (none)")
                lines.append("")
                return
            for e in sorted(entries, key=lambda x: x["name"]):
                name = e["name"]
                ids = sorted(e.get("ids", []) or [])
                if mcp_kind:
                    lines.append(
                        f"- {name} [{mcp_kind}]: {len(ids)} instance{'s' if len(ids) != 1 else ''}"
                    )
                else:
                    lines.append(
                        f"- {name}: {len(ids)} instance{'s' if len(ids) != 1 else ''}"
                    )
                for _id in ids:
                    lines.append(f"  - {_id}")
            lines.append("")

        render_summary("Tools summary", tools)
        render_summary("LLMs summary", llms)
        render_summary("MCP servers summary (local)", mcp_local, mcp_kind="local")
        render_summary(
            "MCP servers summary (external)", mcp_external, mcp_kind="external"
        )

        return "\n".join(lines)

    # ============================================================
    # Internal validators
    # ============================================================
    @model_validator(mode="after")
    def validate_graph_references(self, info: ValidationInfo):
        """
        Ensure that connections, edges, and routing rule targets only reference
        node IDs that exist in this graph (plus a small set of allowed specials).
        """
        if not (info.context or {}).get("run_root_pass", False):
            return self

        is_graph = self.is_graph
        nodes = self.nodes or []
        edges = self.internal_edges or []

        # Only enforce for graph nodes that actually have children
        if not is_graph or not nodes:
            return self

        # All valid node IDs in this graph
        valid_ids = {node.name for node in nodes}

        # Special non-node identifiers that are allowed as endpoints
        allowed_special = {"START", "END"}

        def _check_ref(ref: str, context: str):
            if ref not in valid_ids and ref not in allowed_special:
                raise ValueError(
                    f"Invalid reference '{ref}' in {context}: "
                    f"not a node id in this graph (valid: {sorted(valid_ids)})"
                )

        # 1) Validate each child node's connections (Connection.in_ / Connection.out)
        for node in nodes:
            conn = node.external_connections
            if conn is None:
                continue

            ins = conn.in_ if isinstance(conn.in_, list) else [conn.in_]
            outs = conn.out if isinstance(conn.out, list) else [conn.out]

            for ref in ins:
                _check_ref(ref, f"connections.in_ of node '{node.name}'")
            for ref in outs:
                _check_ref(ref, f"connections.out of node '{node.name}'")

        # 2) Validate graph-level edges (Edge.from_ / Edge.to)
        for edge in edges:
            _check_ref(edge.from_, f"edge.from_ in graph '{self.name}'")
            _check_ref(edge.to, f"edge.to in graph '{self.name}'")

        # 3) Validate router rule targets (RoutingRule.target) for router nodes
        # for node in nodes:
        #     logic = node.routing_logic
        #     if logic is None:
        #         continue
        #     for rule in logic.rules or []:
        #         _check_ref(rule.target, f"routing rule target of node '{node.name}'")

        return self

    @model_validator(mode="after")
    def validate_tool_examples(self, info: ValidationInfo):
        """
        Ensure that tool_example_pairs are structurally consistent with the
        tool's declared inputs when this node is a Tool.
        """

        node_type = self.node_type
        inputs = self.inputs or []
        examples = self.tool_example_pairs or []

        # Only apply to Tool nodes with examples
        if not node_type or node_type.type != "Tool" or not examples:
            return self

        input_names = [slot.name for slot in inputs]

        # If no inputs are declared, we cannot validate structure
        if not input_names:
            raise ValueError(
                f"Tool node '{self.id}' has tool_example_pairs but no inputs defined."
            )

        # Example.input must be a dict with exact keys = input_names
        required_names = {
            slot.name for slot in inputs if getattr(slot, "required", False)
        }
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
    def validate_code_execution(self, info: ValidationInfo):
        """
        code_execution may be True only for Agent or Tool node types.
        """

        if self.code_execution:
            allowed = {"Agent", "Tool", "System", "other"}
            nt = self.node_type.type
            if nt not in allowed:
                raise ValueError(
                    f"Node '{self.name}' has code_execution=True, but node_type='{self.node_type.type}' is not allowed. Only ['Agent', 'Tool'] may enable code execution."
                )
        return self

    @model_validator(mode="after")
    def validate_code_reference_uniqueness(self, info: ValidationInfo):

        # Kinds that must appear at most once per node
        unique_kinds = {"definition", "initialization", "implementation"}

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

        # 1) Bottom-up recursion: close all children first
        for child in self.nodes or []:
            child.propagate_required_keys_bottom_up(info)
        for tool in self.tool_list or []:
            tool.propagate_required_keys_bottom_up(info)

        # 2) Collect keys from self and children
        merged: Set[str] = set()

        # self keys
        if self.required_keys.enabled and self.required_keys.keys:
            merged.update(k for k in self.required_keys.keys if k)

        # children keys
        for child in self.nodes or []:
            rk = child.required_keys
            if rk.enabled and rk.keys:
                merged.update(k for k in rk.keys if k)

        # tool children keys
        for tool in self.tool_list or []:
            rk = tool.required_keys
            if rk.enabled and rk.keys:
                merged.update(k for k in rk.keys if k)

        # 3) Write back in a way that respects RequiredKeys.validate_keys()
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

    @model_validator(mode="after")
    def auto_ids_and_rewrite_refs(self, info: ValidationInfo) -> "NodeSpecTop":
        if not (info.context or {}).get("run_root_pass", False):
            return self

        seen: set[str] = set()

        def children(n: "NodeSpec") -> list["NodeSpec"]:
            out = []
            if n.nodes:
                out += n.nodes
            if n.tool_list:
                out += n.tool_list
            return out

        def qualify(parent_id: str | None, nm: str) -> str:
            return nm if not parent_id else f"{parent_id}_{nm}"

        def rewrite(refs, scope_map: dict[str, str]):
            def one(x: str) -> str:
                return scope_map.get(x, x)  # START/END or already-qualified stay

            if isinstance(refs, str):
                return one(refs)
            return [one(x) for x in refs]

        def walk(
            n: "NodeSpec", parent_id: str | None, scope_map: dict[str, str] | None
        ):
            # 1) set this node's id if missing/unqualified
            if n.id is None or n.id == n.name:
                n.id = qualify(parent_id, n.name)

            if n.id in seen:
                raise ValueError(f"duplicate qualified id '{n.id}'")
            seen.add(n.id)

            # 2) build THIS node's scope map for resolving references to its children
            kids = children(n)
            this_scope: dict[str, str] = {}
            for ch in kids:
                ch_id = (
                    ch.id
                    if (ch.id is not None and ch.id != ch.name)
                    else qualify(n.id, ch.name)
                )
                this_scope[ch.name] = ch_id

            # 3) rewrite THIS node's internal edges using THIS scope
            if n.internal_edges:
                for e in n.internal_edges:
                    if isinstance(getattr(e, "from_", None), str):
                        e.from_ = rewrite(e.from_, this_scope)
                    if isinstance(getattr(e, "to", None), str):
                        e.to = rewrite(e.to, this_scope)

            # 4) rewrite THIS node's external connections using PARENT scope (siblings live there)
            if n.external_connections and scope_map is not None:
                n.external_connections.in_ = rewrite(
                    n.external_connections.in_, scope_map
                )
                n.external_connections.out = rewrite(
                    n.external_connections.out, scope_map
                )

            # 5) recurse into children, passing THIS node's scope as their parent scope
            for ch in kids:
                walk(ch, n.id, this_scope)

        # root has no parent scope
        walk(self, parent_id=None, scope_map=None)

        # mark duplicates by exact name (now that IDs exist)
        all_nodes: list["NodeSpec"] = []
        stack: list["NodeSpec"] = [self]
        while stack:
            cur = stack.pop()
            all_nodes.append(cur)
            stack.extend(children(cur))

        # reset
        for n in all_nodes:
            n.duplicates = Duplication(exists=False, ids=None)

        # group by exact name
        from collections import defaultdict

        groups = defaultdict(list)
        for n in all_nodes:
            groups[n.name].append(n)

        # mark
        for group in groups.values():
            if len(group) <= 1:
                continue
            ids = sorted({g.id for g in group if g.id})
            if not ids:
                continue
            for g in group:
                g.duplicates = Duplication(exists=True, ids=ids)

        return self


# IMPORTANT: enable recursive NodeSpec type
NodeSpec.model_rebuild()
