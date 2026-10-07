from __future__ import annotations
# NOTE: This file is a commented copy of structure_schema.py.
# It is intended for human reading and prompt context, not as the canonical schema.
# The classes and fields are identical to the original; only extra comments are added.
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Literal, DefaultDict
from pathlib import Path
from pydantic import BaseModel, Field, model_validator, ValidationInfo
from collections import Counter, defaultdict
import json

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
    parent: Optional[str] = None
    in_: Union[str, List[str]]
    out: Union[str, List[str]]

# ============================================================
# LLM-SPECIFIC OBJECTS
# LLM configuration is optional and only applies to LLM nodes.
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

        # Rule 1: "other" must be selected alone
        if has_other and len(self.type) != 1:
            raise ValueError("'other' must be the only selected system type when present")

        # Rule 2: other_description iff "other"
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
    Indicates the degree of autonomy using a numeric scale from 0 to 4, with an
    optional sub-level used only when level == 3.
    """
    level: Literal[0, 1, 2, 3, 4]
    description: str
    sub_level: Optional[Literal["low", "medium", "high"]] = None

    @model_validator(mode="after")
    def validate_sub_level(self):
        level = self.level
        sub = self.sub_level

        # level == 3 → require sub_level
        if level == 3 and sub is None:
            raise ValueError("sub_level is required when level == 3")

        # level != 3 → always set sub_level = None silently
        if level != 3:
            self["sub_level"] = None

        return self

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
    content: Optional[str] = None
    tool_calls: Optional[List[FlowToolCallSpec]] = None


class FlowSpec(BaseModel):
    """
    Simple-flow artifact derived from runtime traces.
    """
    flow_id: int
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
    version: ClassVar[str] = "1.0"

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

    @staticmethod
    def hierarchy_summary_from_parsed(
        *,
        var_order: List[str],
        nodes_by_var: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build a hierarchical summary from parsed nodespec assignment maps
        (as returned by global_utils.load_nodes_by_var).

        Output schema:
        {
          "top_level": {
            "name": str,
            "type": str,
            "description": str,
            "ids": [str],
            "children": [{"name": str, "id": str}]
          },
          "all_graph_nodes": [
            {
              "name": str,
              "type": str,
              "description": str,
              "ids": [str],
              "children": [{"name": str, "id": str}]
            }
          ]
        }
        """
        if not var_order or not nodes_by_var:
            return {}

        def _node_name(var: str) -> str:
            node = nodes_by_var.get(var, {})
            return str(node.get("name") or "").strip() or var

        def _node_ids(var: str) -> List[str]:
            node = nodes_by_var.get(var, {})
            ids: Set[str] = set(inferred_ids_by_var.get(var, set()))
            nid = str(node.get("id") or "").strip()
            if nid:
                ids.add(nid)
            dup = node.get("duplicates") if isinstance(node.get("duplicates"), dict) else {}
            inst = dup.get("instances") if isinstance(dup, dict) and isinstance(dup.get("instances"), list) else []
            for x in inst:
                if isinstance(x, dict):
                    iid = str(x.get("id") or "").strip()
                    if iid:
                        ids.add(iid)
            if not ids:
                ids.add(_node_name(var))
            return sorted(ids)

        def _node_instances(node: Dict[str, Any]) -> List[Dict[str, str]]:
            dup = node.get("duplicates") if isinstance(node.get("duplicates"), dict) else {}
            inst = dup.get("instances") if isinstance(dup, dict) and isinstance(dup.get("instances"), list) else []
            out: List[Dict[str, str]] = []
            for x in inst:
                if not isinstance(x, dict):
                    continue
                iid = str(x.get("id") or "").strip()
                pid = str(x.get("parent_id") or "").strip()
                if iid:
                    out.append({"id": iid, "parent_id": pid})
            return out

        def _children_of(var: str) -> List[str]:
            node = nodes_by_var.get(var, {})
            out: List[str] = []
            for fld in ("nodes", "tool_list"):
                vals = node.get(fld)
                if not isinstance(vals, list):
                    continue
                for x in vals:
                    s = str(x or "").strip()
                    if s and s in nodes_by_var:
                        out.append(s)
            # stable + unique while preserving first occurrence order
            seen: Set[str] = set()
            uniq: List[str] = []
            for c in out:
                if c not in seen:
                    seen.add(c)
                    uniq.append(c)
            return uniq

        # parent map to locate roots
        parent_count: DefaultDict[str, int] = defaultdict(int)
        for pv in var_order:
            for cv in _children_of(pv):
                parent_count[cv] += 1

        root_var = None
        for v in var_order:
            if parent_count.get(v, 0) == 0 and v in nodes_by_var:
                root_var = v
                break
        if root_var is None:
            root_var = var_order[0]

        # Infer deterministic hierarchical IDs even if root-pass wasn't run.
        inferred_ids_by_var: DefaultDict[str, Set[str]] = defaultdict(set)
        roots = [v for v in var_order if parent_count.get(v, 0) == 0 and v in nodes_by_var]
        if not roots and root_var in nodes_by_var:
            roots = [root_var]
        for rv in roots:
            inferred_ids_by_var[rv].add(_node_name(rv))
        queue: List[str] = list(roots)
        queued: Set[str] = set(queue)
        while queue:
            pv = queue.pop(0)
            pids = inferred_ids_by_var.get(pv, set()) or {_node_name(pv)}
            for cv in _children_of(pv):
                cname = _node_name(cv)
                before = len(inferred_ids_by_var[cv])
                for pid in pids:
                    inferred_ids_by_var[cv].add(f"{pid}_{cname}")
                if len(inferred_ids_by_var[cv]) > before and cv not in queued:
                    queue.append(cv)
                    queued.add(cv)
        for v in var_order:
            if v in nodes_by_var and not inferred_ids_by_var.get(v):
                inferred_ids_by_var[v].add(_node_name(v))

        def _pick_child_id(child_var: str, parent_ids: List[str]) -> str:
            cnode = nodes_by_var.get(child_var, {})
            cids = _node_ids(child_var)
            if not cids:
                return ""
            instances = _node_instances(cnode)
            if instances and parent_ids:
                pset = {p for p in parent_ids if p}
                for inst in instances:
                    if inst.get("parent_id") in pset and inst.get("id"):
                        return str(inst["id"])
            if parent_ids:
                for pid in parent_ids:
                    pref = f"{pid}_"
                    for cid in cids:
                        if cid.startswith(pref):
                            return cid
            return cids[0]

        def _block(var: str) -> Dict[str, Any]:
            node = nodes_by_var.get(var, {})
            nt = node.get("node_type") if isinstance(node.get("node_type"), dict) else {}
            name = str(node.get("name") or "").strip() or var
            ntype = str((nt or {}).get("type") or "").strip()
            desc = str(node.get("description") or "").strip()
            ids = _node_ids(var)
            kids = []
            for cv in _children_of(var):
                cnode = nodes_by_var.get(cv, {})
                cname = str(cnode.get("name") or "").strip() or cv
                kids.append({"name": cname, "id": _pick_child_id(cv, ids)})
            return {
                "name": name,
                "type": ntype,
                "description": desc,
                "ids": ids,
                "children": kids,
            }

        # Parent-complete topological ordering:
        # a node appears only after ALL of its parents appeared.
        var_pos = {v: i for i, v in enumerate(var_order)}
        parents_by_child: DefaultDict[str, Set[str]] = defaultdict(set)
        for pv in var_order:
            if pv not in nodes_by_var:
                continue
            for cv in _children_of(pv):
                parents_by_child[cv].add(pv)

        emitted: Set[str] = {root_var}
        remaining: Set[str] = {v for v in var_order if v in nodes_by_var and v != root_var}
        ordered_vars: List[str] = []

        # Ready queue with FIFO semantics on time-of-readiness.
        ready_queue: List[str] = []
        in_queue: Set[str] = set()

        # Structural priority for ready-node selection:
        # 1) deeper descendant depth first
        # 2) more direct children first
        # 3) more parents first
        # 4) FIFO fallback (queue order)
        depth_cache: Dict[str, int] = {}

        def _max_desc_depth(v: str, stack: Optional[Set[str]] = None) -> int:
            if v in depth_cache:
                return depth_cache[v]
            st = set(stack or set())
            if v in st:
                return 0
            st.add(v)
            kids = _children_of(v)
            if not kids:
                depth_cache[v] = 0
                return 0
            d = 1 + max(_max_desc_depth(c, st) for c in kids if c in nodes_by_var)
            depth_cache[v] = d
            return d

        def _enqueue_new_ready() -> None:
            for v in sorted(remaining, key=lambda x: var_pos.get(x, 10**9)):
                if v in in_queue:
                    continue
                if parents_by_child.get(v, set()).issubset(emitted):
                    ready_queue.append(v)
                    in_queue.add(v)

        _enqueue_new_ready()
        while remaining:
            if not ready_queue:
                # Cycle/ambiguous ordering fallback: deterministic by var order.
                v = min(remaining, key=lambda x: var_pos.get(x, 10**9))
                ready_queue.append(v)
                in_queue.add(v)

            # Pick from ready set by structural priority; FIFO fallback.
            pick_idx = 0
            best_key = None
            for i, cand in enumerate(ready_queue):
                key = (
                    -_max_desc_depth(cand),
                    -len(_children_of(cand)),
                    -len(parents_by_child.get(cand, set())),
                    i,  # FIFO fallback
                )
                if best_key is None or key < best_key:
                    best_key = key
                    pick_idx = i

            v = ready_queue.pop(pick_idx)
            in_queue.discard(v)
            if v not in remaining:
                continue
            ordered_vars.append(v)
            emitted.add(v)
            remaining.remove(v)
            _enqueue_new_ready()

        all_graph_nodes = [_block(v) for v in ordered_vars]
        return {
            "top_level": _block(root_var),
            "all_graph_nodes": all_graph_nodes,
        }

    @classmethod
    def hierarchy_summary_from_nodespec_py(cls, path: str) -> Dict[str, Any]:
        """
        Convenience wrapper: parse a final_nodespec.py file without executing it
        and return hierarchical summary.
        """
        from modular_imp.global_utils import load_nodes_by_var  # lazy import

        var_order, nodes_by_var = load_nodes_by_var(Path(path))
        return cls.hierarchy_summary_from_parsed(var_order=var_order, nodes_by_var=nodes_by_var)


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

        agent_instances = flatten(agents)         # (agent_name, agent_id)
        tool_instances = flatten(tools)           # (tool_name, tool_id)
        llm_instances = flatten(llms)             # (llm_name, llm_id)
        mcp_local_instances = flatten(mcp_local)  # (mcp_name, mcp_id)
        mcp_ext_instances = flatten(mcp_external) # (mcp_name, mcp_id)

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
        lines.append(f"- MCP servers: {total_mcp} (local: {total_mcp_local}, external: {total_mcp_ext})")
        lines.append("")

        # ---- build fast lookup: agent_id -> agent_name ----
        agent_id_to_name: Dict[str, str] = {aid: aname for aname, aid in agent_instances}

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
                rest = instance_id[len(prefix):]
                agent_name = rest.split("_", 1)[0]
                return f"{TOP}_{agent_name}"
            # fallback: first two tokens
            return "_".join(parts[:2]) if len(parts) >= 2 else instance_id

        # ---- per-agent aggregation (by instance ids) ----
        tools_by_agent: Dict[str, Set[str]] = defaultdict(set)  # agent_id -> set(tool_name)
        llms_by_agent: Dict[str, Set[str]] = defaultdict(set)   # agent_id -> set(llm_name)
        mcps_by_agent: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)  # agent_id -> {(mcp_name, kind)}

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
            key=lambda x: x
        )

        # ---- per-system section (assume single top system id == TOP) ----
        lines.append(f"System: {TOP}")
        lines.append(f"- Agents ({len(agent_ids_in_system)}):")
        for aid in agent_ids_in_system:
            tnames = sorted(tools_by_agent.get(aid, set()))
            lnames = sorted(llms_by_agent.get(aid, set()))
            mcps = sorted(mcps_by_agent.get(aid, set()), key=lambda x: x[0])

            lines.append(f"  - {aid}")
            lines.append(f"    - Tools ({len(tnames)}): {', '.join(tnames) if tnames else ''}".rstrip())
            lines.append(f"    - LLMs ({len(lnames)}): {', '.join(lnames) if lnames else ''}".rstrip())
            if mcps:
                mcp_str = ", ".join([f"{n} [{k}]" for n, k in mcps])
                lines.append(f"    - MCP servers ({len(mcps)}): {mcp_str}")
            else:
                lines.append("    - MCP servers (0)")
        lines.append("")

        # ---- global summaries (already merged by name+type) ----
        def render_summary(title: str, entries: List[Dict[str, Any]], *, mcp_kind: str | None = None) -> None:
            lines.append(f"{title}:")
            if not entries:
                lines.append("- (none)")
                lines.append("")
                return
            for e in sorted(entries, key=lambda x: x["name"]):
                name = e["name"]
                ids = sorted(e.get("ids", []) or [])
                if mcp_kind:
                    lines.append(f"- {name} [{mcp_kind}]: {len(ids)} instance{'s' if len(ids) != 1 else ''}")
                else:
                    lines.append(f"- {name}: {len(ids)} instance{'s' if len(ids) != 1 else ''}")
                for _id in ids:
                    lines.append(f"  - {_id}")
            lines.append("")

        render_summary("Tools summary", tools)
        render_summary("LLMs summary", llms)
        render_summary("MCP servers summary (local)", mcp_local, mcp_kind="local")
        render_summary("MCP servers summary (external)", mcp_external, mcp_kind="external")

        return "\n".join(lines)


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

        # Accept both local names (pre-close-id) and qualified ids (post-close-id).
        valid_refs = {node.name for node in nodes if isinstance(node.name, str)}
        valid_refs.update({node.id for node in nodes if isinstance(node.id, str) and node.id})

        # Special non-node identifiers that are allowed as endpoints
        allowed_special = {"START", "END"}

        def _check_ref(ref: str, context: str):
            if ref not in valid_refs and ref not in allowed_special:
                raise ValueError(
                    f"Invalid reference '{ref}' in {context}: "
                    f"not a node id/name in this graph (valid: {sorted(valid_refs)})"
                )

        # 1) Validate each child node's connections (Connection.in_ / Connection.out)
        for node in nodes:
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
                for ref in outs:
                    _check_ref(ref, f"connections.out of node '{node.name}'")

        # 2) Validate graph-level edges (Edge.from_ / Edge.to)
        for edge in edges:
            _check_ref(edge.from_, f"edge.from_ in graph '{self.name}'")
            _check_ref(edge.to, f"edge.to in graph '{self.name}'")

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

        # 1) Recurse first so descendants are already closed.
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
    def validate_code_execution(self, info: ValidationInfo):
        """
        code_execution may be True for:
        - explicitly executable node types (Agent/Tool/System/other), or
        - any parent node that inherits code_execution=True from children.
        """
        
        if self.code_execution:
            allowed = {"Agent", "Tool", "System", "other"}
            nt = self.node_type.type
            inherited_from_child = any(bool(ch.code_execution) for ch in (self.nodes or [])) or any(
                bool(t.code_execution) for t in (self.tool_list or [])
            )
            if nt not in allowed and not inherited_from_child:
                raise ValueError(
                    f"Node '{self.name}' has code_execution=True, but node_type='{self.node_type.type}' is not allowed "
                    f"for direct execution and no executing child was found."
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
        for child in (self.nodes or []):
            child.propagate_required_keys_bottom_up(info)
        for tool in (self.tool_list or []):
            tool.propagate_required_keys_bottom_up(info)

        # 2) Collect keys from self and children
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
        
    # Root-only pass: auto-ids, edge rewrites, and duplicate marking.
    @model_validator(mode="after")
    def auto_ids_and_rewrite_refs(self, info: ValidationInfo) -> "NodeSpecTop":
        if not (info.context or {}).get("run_root_pass", False):
            return self

        seen: set[str] = set()
        shareable_alias_types = {"Tool", "LLM", "Local_MCP_server", "External_MCP_server", "Database"}

        def children(n: "NodeSpec") -> list["NodeSpec"]:
            out = []
            if n.nodes: out += n.nodes
            if n.tool_list: out += n.tool_list
            return out

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
                ch_id = ch.id if (ch.id is not None and ch.id != ch.name) else qualify(n.id, ch.name)
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

            # 5) recurse into children, passing THIS node's scope as their parent scope
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
            stack.extend(children(cur))

        # reset
        for n in all_nodes:
            n.duplicates = Duplication(exists=False, instances=None)

        # group by exact name only for shareable runtime components
        from collections import defaultdict
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

# IMPORTANT: enable recursive NodeSpec type
NodeSpec.model_rebuild()
