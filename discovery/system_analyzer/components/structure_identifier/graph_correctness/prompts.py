"""Prompt constants for graph_correctness component."""
try:
    from ..global_utils import (
        agentic_terminology_instructions,
        context_request_format_template,
        guidance_summary_instructions,
        retrieval_instructions,
        vars_fields_context_request_format_template,
    )
except ImportError:
    from global_utils import (
        agentic_terminology_instructions,
        context_request_format_template,
        guidance_summary_instructions,
        retrieval_instructions,
        vars_fields_context_request_format_template,
    )

SINGLE_CHILD_DECISION_PROMPT = """You are resolving one graph-correctness issue for an agentic system graph.

Goal:
Decide whether a parent node with exactly one direct child should keep that child as a distinct agentic component or merge it as the same component.

<<AGENTIC_TERMINOLOGY>>

Inputs:
- parent_var
- parent_node
- child_var
- child_node
- guidance_summary
- retrieved_context (optional)

Decision rules:
1) Do not invent agentic components, relationships, files, code references, or runtime behavior.
2) Search for evidence until the decision is evidence-backed; if evidence is missing, return context_request instead of guessing. RAG will retrieve information.
3) Return action=merge_components when the two nodes are not distinct agentic components and one is an internal/support implementation detail of the other.
4) Separate code objects can still represent one agentic component (for example wrapper + underlying function/class/config used as one runtime unit); in such cases prefer merge_components.
5) If parent Agent is a wrapper whose primary runtime is invoking exactly one inner Agent function and then returning that result, treat them as one agentic component and return action=merge_components.
6) Return action=keep_distinct when both nodes are independently executable or callable runtime components with separate runtime identity, lifecycle, or endpoint, even if one imports, starts, proxies to, or delegates to the other. Return action=merge_components when one node is only a thin in-process wrapper, adapter or alias for the same underlying component and does not introduce an independently callable runtime role or boundary.
7) <<GUIDANCE_SUMMARY_RULE>>

Output rules:
1) Use one of two response shapes only:
   - context_request object only, OR
   - final output JSON object.
2) Return JSON only.
3) Final output must contain exactly these keys: action, reason.
4) action must be one of: merge_components, keep_distinct.
5) When returning context_request, each needs[].query must target exactly one missing fact using concrete code literals.

Final output JSON:
{
  "action": "merge_components|keep_distinct",
  "reason": "<short evidence-based reason>"
}

<<CONTEXT_REQUEST_FORMAT>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_RULE>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT>>", context_request_format_template).replace("<<VARS_FIELDS_CONTEXT_REQUEST_FORMAT>>", vars_fields_context_request_format_template).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


AGENT_LLM_DECISION_PROMPT = """You are resolving one graph-correctness issue for an agentic system graph.

Goal:
Classify whether this node is a real LLM-based Agent runtime component.

<<AGENTIC_TERMINOLOGY>>
- Real LLM-based Agent: a runtime Agent component that performs agentic reasoning/orchestration and is concretely powered by at least one LLM call/config/instance.

Inputs:
- agent_var
- agent_node
- guidance_summary
- retrieved_context (optional)

Decision rules:
1) Do not invent components, relationships, files, or behavior.
2) Search for evidence until decision is evidence-backed; if evidence is missing, return context_request instead of guessing. RAG will retrieve information.
3) Use action=real_llm_agent when concrete evidence shows this node is an agent runtime and is LLM-powered, OR when it is instantiated via a known agent framework constructor (e.g., crewai.Agent / AssistantAgent / create_react_agent) and there is no concrete counter-evidence that it is non-LLM.
4) Use action=not_real_agent only when there is concrete positive evidence that this node is not an Agent runtime or is explicitly non-LLM. Lack of LLM evidence alone is not enough for not_real_agent; request context instead.
5) <<GUIDANCE_SUMMARY_RULE>>

Output rules:
1) Use one of two response shapes only:
   - context_request object only, OR
   - final output JSON object.
2) Return JSON only.
3) Final output must contain exactly these keys: action, reason.
4) action must be one of: real_llm_agent, not_real_agent.
5) When returning context_request, each needs[].query must target exactly one missing fact using concrete code literals.

Final output JSON:
{
  "action": "real_llm_agent|not_real_agent",
  "reason": "<short evidence-based reason>"
}

<<CONTEXT_REQUEST_FORMAT>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_RULE>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT>>", context_request_format_template).replace("<<VARS_FIELDS_CONTEXT_REQUEST_FORMAT>>", vars_fields_context_request_format_template).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


AGENT_LLM_RESOLUTION_PROMPT = """You are resolving one graph-correctness issue for an agentic system graph.

Goal:
For a confirmed real LLM-based Agent, produce exactly one LLM child proposal.

<<AGENTIC_TERMINOLOGY>>

Inputs:
- agent_var
- agent_node
- guidance_summary
- retrieved_context (optional)

Decision rules:
1) Do not invent components, relationships, files, code references, or runtime behavior.
2) Search for evidence until output is evidence-backed; if evidence is missing, return context_request instead of guessing. RAG will retrieve information.
3) child_proposal.node_type must be LLM or syn_llm. Do not return other types.
4) If LLM cannot be defensibly identified yet, return context_request (do not return fabricated child_proposal).
5) <<GUIDANCE_SUMMARY_RULE>>
6) Synthetic fallback allowed only for known agent frameworks:
   if code shows an Agent runtime created by a known framework constructor
   (e.g., crewai.Agent / AssistantAgent / create_react_agent) but no explicit
   LLM object/config is found, set child_proposal.node_type to syn_llm.
   For syn_llm, all other child_proposal fields may be empty.

Output rules:
1) Use one of two response shapes only:
   - context_request object only, OR
   - final output JSON object with child_proposal.
2) Return JSON only.
3) Final output must contain only one top-level key: child_proposal.
4) child_proposal must include only these fields: name, node_type, description, code_references_add_child.
For node_type=syn_llm, name/description/code_references_add_child may be empty.
5) When returning context_request, each needs[].query must target exactly one missing fact using concrete code literals.

Final output JSON:
{
  "child_proposal": {
    "name": "<child runtime name>",
    "node_type": "LLM|syn_llm",
    "description": "<short runtime role>",
    "code_references_add_child": [
      {
        "kind": "usage",
        "file": "<abs_file_path>",
        "line_start": <int>,
        "line_end": <int>,
        "note": "<short note describing exactly where child is attached/used by parent>"
      }
    ]
  }
}

<<CONTEXT_REQUEST_FORMAT>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_RULE>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT>>", context_request_format_template).replace("<<VARS_FIELDS_CONTEXT_REQUEST_FORMAT>>", vars_fields_context_request_format_template).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


AGENT_LLM_REUSE_PROMPT = """You are resolving one graph-correctness issue for an agentic system graph.

Goal:
Before creating a new LLM child for an Agent, decide whether one existing LLM node in the graph is already the related runtime LLM for this agent.

<<AGENTIC_TERMINOLOGY>>

Inputs:
- agent_var
- agent_node
- llm_catalog (existing LLM nodes only)
- guidance_summary
- retrieved_context (optional)

Decision rules:
1) Do not invent components, relationships, files, code references, or runtime behavior.
2) Search for evidence until output is evidence-backed; if evidence is missing, return context_request instead of guessing. RAG will retrieve information.
3) Choose action=reuse_existing only when one candidate in llm_catalog is clearly related to this agent's LLM usage/configuration.
4) Choose action=create_new when no candidate is clearly related.
5) If action=reuse_existing, selected_llm_var must be one of llm_catalog.var values.
6) <<GUIDANCE_SUMMARY_RULE>>

Output rules:
1) Use one of two response shapes only:
   - context_request object only, OR
   - final output JSON object.
2) Return JSON only.
3) Final output must contain exactly these keys: action, selected_llm_var, reason.
4) action must be one of: reuse_existing, create_new.
5) selected_llm_var must be null when action=create_new.

Final output JSON:
{
  "action": "reuse_existing|create_new",
  "selected_llm_var": "<llm_var_or_null>",
  "reason": "<short evidence-based reason>"
}

<<CONTEXT_REQUEST_FORMAT>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_RULE>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT>>", context_request_format_template).replace("<<VARS_FIELDS_CONTEXT_REQUEST_FORMAT>>", vars_fields_context_request_format_template).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


SYSTEM_SUMMARY_GENERATION_PROMPT = """You are generating a concise system summary from graph evidence.

Goal:
Produce a clear system-level summary of the current graph after rule-based corrections.

<<AGENTIC_TERMINOLOGY>>

Inputs:
- summary_node_catalog
- nodespec_class_fields
- additional_context (optional; contains retrieved_context + context_request_history)

Rules:
1) Treat NodeSpec nodes as the source of truth. Summarize from node identity + node relations (parent/children) first.
2) Use `summary_node_catalog` for structure (var, name, type, description, children (nodes/tool list fields)).
3) Summary scope is Agentic Components only (System, Agent, LLM, Tool, Local_MCP_server, External_MCP_server, Database, Deterministic_controller, other runtime component). Do not summarize code objects.
4) Do not invent components, relations, or runtime behavior.
5) Ensure completeness: no missing component and no missing direct relation from `summary_node_catalog`.
6) When referring to a set of components, list each component by its exact name and type explicitly (groups may be unified, e.g., tools: tool_A, tool_B, ...); do not replace named lists with grouped labels like "all sub-agents" or "all tools".
The summary is invalid if any mentioned component is missing an explicit type label.
7) Describe each direct parent -> child relation explicitly. Do not skip any direct relation present in `summary_node_catalog`.
8) Use `nodespec_class_fields` to decide what additional node fields can be requested when structure is insufficient.
9) If evidence is missing, return `context_request` using a simple var->fields map:
   `context_request.vars_fields = {"<var_name>": ["<field1>", "<field2>"]}`.
10) Request only missing facts. Do not request broad text search; request explicit NodeSpec fields per var.
11) Build incrementally across rounds: use `additional_context`; do not re-request the same var/fields.

Output rules:
1) Return JSON only.
2) Use one of two response shapes only:
   - context_request object only, OR
   - final output JSON.
3) Final output must contain exactly these keys: summary, reason.
4) `summary` must be plain text (not bullets/template-locked) and describe the system in natural form:
   "System structure: <system_name> includes <agent_1>, <agent_2>, <...>. <agent_1> has tools <tool_1>, <tool_2>, an MCP server <server_A> with tools <tool_A>, <tool_B>, <tool_C>, <tool_D>, <tool_E>, and database <DB_1>.",
   while still naming components explicitly and stating direct relations clearly.

Final output JSON:
{
  "summary": "<concise system summary>",
  "reason": "<short evidence-based reason>"
}

<<VARS_FIELDS_CONTEXT_REQUEST_FORMAT>>
When returning context_request, return only that object.
""".replace("<<AGENTIC_TERMINOLOGY>>", agentic_terminology_instructions).replace("<<VARS_FIELDS_CONTEXT_REQUEST_FORMAT>>", vars_fields_context_request_format_template)


SYSTEM_NO_AGENT_DECISION_PROMPT = """You are resolving one graph-correctness issue for an agentic system graph.

Goal:
Classify whether a System node with no direct Agent child is truly an agentic system component.

<<AGENTIC_TERMINOLOGY>>
- System: an agentic system runtime component that contains Agent nodes as children.

Inputs:
- system_var
- system_node
- guidance_summary
- retrieved_context (optional)

Decision rules:
1) Do not invent components, relationships, files, or behavior.
2) Search for evidence until decision is evidence-backed; if evidence is missing, return context_request instead of guessing. RAG will retrieve information.
3) Use action=agentic_system only when evidence supports this node as a System under the definition above (agentic runtime component that contains agents).
4) Use action=not_agentic_system when evidence supports this node is not an agentic system runtime component.
5) <<GUIDANCE_SUMMARY_RULE>>

Output rules:
1) Use one of two response shapes only:
   - context_request object only, OR
   - final output JSON object.
2) Return JSON only.
3) Final output must contain exactly these keys: action, reason.
4) action must be one of: agentic_system, not_agentic_system.
5) When returning context_request, each needs[].query must target exactly one missing fact using concrete code literals.

Final output JSON:
{
  "action": "agentic_system|not_agentic_system",
  "reason": "<short evidence-based reason>"
}

<<CONTEXT_REQUEST_FORMAT>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_RULE>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT>>", context_request_format_template).replace("<<VARS_FIELDS_CONTEXT_REQUEST_FORMAT>>", vars_fields_context_request_format_template).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


LLM_NO_CHILDREN_DECISION_PROMPT = """You are resolving one graph-correctness issue for an agentic system graph.

Goal:
Resolve an invalid LLM node that has direct children.

<<AGENTIC_TERMINOLOGY>>

Inputs:
- llm_var
- llm_node
- direct_children (nodes + tool_list)
- guidance_summary
- retrieved_context (optional)

Decision rules:
1) Do not invent components, relationships, files, or behavior.
2) Search for evidence until decision is evidence-backed; if evidence is missing, return context_request instead of guessing. RAG will retrieve information.
3) LLM nodes should not own direct children as a graph parent.
4) Use action=change_type_to_agent only when concrete evidence shows this node is actually an Agent runtime component mislabeled as LLM.
5) Otherwise use action=remove_children.
6) Return only one action.

Output rules:
1) Use one of two response shapes only:
   - context_request object only, OR
   - final output JSON object.
2) Return JSON only.
3) Final output must contain exactly these keys: action, reason.
4) action must be one of: change_type_to_agent, remove_children.
5) When returning context_request, each needs[].query must target exactly one missing fact using concrete code literals.

Final output JSON:
{
  "action": "change_type_to_agent|remove_children",
  "reason": "<short evidence-based reason>"
}

<<CONTEXT_REQUEST_FORMAT>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_RULE>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT>>", context_request_format_template).replace("<<VARS_FIELDS_CONTEXT_REQUEST_FORMAT>>", vars_fields_context_request_format_template).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


MULTI_PARENT_DECISION_PROMPT = """You are resolving one graph-correctness issue for an agentic system graph.

Goal:
Resolve a node with multiple parents by choosing the valid parent set.

<<AGENTIC_TERMINOLOGY>>

Inputs:
- node_var
- node
- parent_vars
- node_type
- guidance_summary
- retrieved_context (optional)

Decision rules:
1) Do not invent components, relationships, files, or behavior.
2) Search for evidence until decision is evidence-backed; if evidence is missing, return context_request instead of guessing. RAG will retrieve information.
3) Tool/Server/LLM/Database may keep multiple parents only when evidence supports true sharing.
4) Prefer nearest operational owners over wrapper/container parents only within the same parent hierarchy level: when one candidate is mainly orchestration (for example controller or top-level wrapper) and another candidate is the concrete runtime owner for that same hierarchy branch, keep the concrete owner. Do not use this rule to drop parents that represent different agentic components or branches where this node is also present.
5) A Tool parent is valid only when the tool contains/owns the child as part of its runtime component boundary (creation, lifecycle, or direct encapsulation evidence). Mere interaction/calls/reads/writes/routing is NOT containment.
6) For Tool/LLM/Database with both wrapper-level and concrete-worker parent candidates, drop the wrapper only if the concrete worker for that same component path is also in parent_vars; otherwise keep the wrapper as the component’s representative parent. Do not drop a wrapper in favor of a different component branch. Routing/edges alone are not ownership evidence.
7) keep_parents must be subset of parent_vars.
8) <<GUIDANCE_SUMMARY_RULE>>

Output rules:
1) Use one of two response shapes only:
   - context_request object only, OR
   - final output JSON object.
2) Return JSON only.
3) Final output must contain exactly these keys: keep_parents, reason.
4) keep_parents must include only vars from parent_vars.
5) When returning context_request, each needs[].query must target exactly one missing fact using concrete code literals.

Final output JSON:
{
  "keep_parents": ["<var>", "..."],
  "reason": "<short evidence-based reason>"
}

<<CONTEXT_REQUEST_FORMAT>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_RULE>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT>>", context_request_format_template).replace("<<VARS_FIELDS_CONTEXT_REQUEST_FORMAT>>", vars_fields_context_request_format_template).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


AGENT_SYSTEM_SINGLE_PARENT_DECISION_PROMPT = """You are resolving one graph-correctness issue for an agentic system graph.

Goal:
For an Agent or System node with multiple parents, choose exactly one valid parent.

<<AGENTIC_TERMINOLOGY>>

Inputs:
- node_var
- node
- parent_vars
- node_type
- guidance_summary
- retrieved_context (optional)

Decision rules:
1) Do not invent components, relationships, files, or behavior.
2) Search for evidence until decision is evidence-backed; if evidence is missing, return context_request instead of guessing. RAG will retrieve information.
3) You MUST select exactly one parent from parent_vars.
4) selected_parent must be a single var and must be in parent_vars.
5) <<GUIDANCE_SUMMARY_RULE>>

Output rules:
1) Use one of two response shapes only:
   - context_request object only, OR
   - final output JSON object.
2) Return JSON only.
3) Final output must contain exactly these keys: selected_parent, reason.
4) selected_parent must be one var from parent_vars.
5) When returning context_request, each needs[].query must target exactly one missing fact using concrete code literals.

Final output JSON:
{
  "selected_parent": "<var>",
  "reason": "<short evidence-based reason>"
}

<<CONTEXT_REQUEST_FORMAT>>

When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_RULE>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT>>", context_request_format_template).replace("<<VARS_FIELDS_CONTEXT_REQUEST_FORMAT>>", vars_fields_context_request_format_template).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)
