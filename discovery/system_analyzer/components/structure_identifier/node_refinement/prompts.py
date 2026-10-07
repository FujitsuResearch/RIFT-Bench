"""Prompt templates for node_refinement component."""

from __future__ import annotations
try:
    from ..global_utils import (
        agentic_terminology_instructions,
        context_request_format_template,
        guidance_summary_instructions,
        retrieval_instructions,
    )
except ImportError:
    from global_utils import (
        agentic_terminology_instructions,
        context_request_format_template,
        guidance_summary_instructions,
        retrieval_instructions,
    )
from typing import Dict

TYPE_CLASSIFICATION_PROMPT = f"""You are resolving node type validation for a single node.

Goal:
For the given node, verify existing node_type or change it to the correct final type.

<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>

Type rubric (evidence-backed):
- Agent: LLM-driven component with its own role.
- System: contains multiple agents (and possibly also other components) as a unit.
- LLM: model client/config used for inference.
- Tool: registered callable capability used by an agent/system; requires evidence of direct registration/usage under an agent or server (not just a helper, container, middleware, backend, or config artifact).
- Database: storage client or engine.
- Local_MCP_server: MCP server defined in the local codebase.
- External_MCP_server: MCP server accessed via remote endpoint/config.
- Deterministic_controller: deterministic router/control node that mediates interactions between other components. It does not contain components, but does interact with other nodes.
- other: helper/support artifact when no specific type above is evidence-backed.

Inputs:
- node_var
- node_block
- child_type_hints (optional; direct child vars with their current node types)
- guidance_summary
- retrieved_context (optional)

<<GUIDANCE_SUMMARY_INSTRUCTIONS>>

General rules:
1) Classify the node as the system component it represents. Code objects are only evidence. Do not classify a node as Support Artifact just because it looks like a wrapper/helper/adapter. A wrapper is a Support Artifact only when the wrapped component is clearly represented as a separate node. If that separate component is not clearly present in the evidence, classify the node based on its own observable behavior and role (not as other), and do not assume or invent a separate underlying component.
2) Base the decision on code evidence, not labels: inspect node_block.code_references first (definition + usage/call sites), and if that is insufficient, request additional code context before deciding. RAG will retrieve information.
3) If code evidence is conflicting or incomplete, return context_request instead of choosing by wording.
4) Return final_type from the known list only.
5) Choose the most specific type supported by evidence; do not upgrade to Agent/System without explicit LLM-driven behavior.
6) Only use other when retrieval is exhausted and no non-other type is evidence-backed.
7) Keep the reason short and tied to a concrete artifact (class/function/config/edge).
8) Use child_type_hints when available: if contained children include Agent nodes, prefer System over Deterministic_controller.
9) Hard constraint: if retrieved_context is empty, you MUST NOT output final_type="other". The ONLY valid response in that case is a context_request for missing evidence.
FORBIDDEN: any final_type="other" output when retrieved_context is empty. This is an invalid response and must be retried with context_request.

Type-specific rules:
10) Tool registration gate: choose Tool only when the node itself is directly exposed/registered as an invokable capability (for example bound in an agent/server tool registry or explicit callable tool interface). If direct registration evidence is missing, do not classify as Tool.
11) Containers/wrappers are not Tool by default: if a node only assembles/holds/routes/wraps Tool or Server connections, classify as other unless evidence shows it implements/hosts an MCP server runtime; only then use Local_MCP_server or External_MCP_server.
12) Agent vs System: “orchestrates/routes” alone is weak evidence. Prefer Agent for a single LLM actor; use System when evidence supports a composed multi-agent structure, including deterministic orchestration of those components. If unclear, request context.
13) Select Deterministic_controller only when the node acts purely as a deterministic router or middleware that forwards, dispatches, or intercepts interactions between components, and does not contain or own them as subcomponents.
If the node contains, aggregates, or manages multiple Agent nodes (or other components) as part of its structure, it must be classified as a System, not a Deterministic_controller.

14) If final_type is "other", include other_description with a concrete support-artifact label (for example: helper wrapper, config artifact, template, adapter, utility class).


Output JSON:
{{
  "final_type": "Agent|System|LLM|Tool|Database|Local_MCP_server|External_MCP_server|Deterministic_controller|other",
  "reason": "<short evidence-based reason>",
  "other_description": "<required if final_type='other'; else null>"
}}

<<CONTEXT_REQUEST_FORMAT_TYPE_CLASSIFICATION>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_INSTRUCTIONS>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT_TYPE_CLASSIFICATION>>", context_request_format_template.replace("<<REQUEST_ID>>", "node_refine_type_classification_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


COMPONENT_EXISTENCE_CHECK_PROMPT = f"""You are resolving whether a node is an actual agentic component.

Goal:
Decide wether the given node is a real agentic component in the agentic system.

<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>

Inputs:
- node_var
- node_block
- guidance_summary
- retrieved_context (optional)

<<GUIDANCE_SUMMARY_INSTRUCTIONS>>

Rules:
1) Use concrete evidence only.
2) Search for evidence until decision is evidence-backed; if evidence is missing, return context_request instead of guessing. Rag will retreive information.
3) Choose action=keep_component only when evidence supports this node as a real runtime agentic component (not just a helper/support artifact).
4) If node is not an actual agentic runtime component, choose action=remove_node.
5) Do not invent components, files, or runtime behavior.
6) Return only one of the allowed actions.

Output JSON:
{{
  "action": "keep_component|remove_node",
  "reason": "<short evidence-based reason>"
}}

<<CONTEXT_REQUEST_FORMAT_COMPONENT_EXISTENCE>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_INSTRUCTIONS>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT_COMPONENT_EXISTENCE>>", context_request_format_template.replace("<<REQUEST_ID>>", "node_refine_component_existence_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


GROUP_PROMPT_TEMPLATE = """You are filling in NodeSpec fields group that charecterize an agentic component.

Goal:
Update only the requested fields for this node.

Inputs:
- node_var
- node_block
- guidance_summary
- retrieved_context (optional)

<<GUIDANCE_SUMMARY_INSTRUCTIONS>>

Target fields (only these can be updated):
<<TARGET_FIELDS>>

General rules:
1) Use concrete evidence only.
2) Search for evidence until decision is evidence-backed; if evidence is missing, return context_request instead of guessing. Rag will retreive information.
3) Do not stop early: issue focused context_request hops for each missing fact and continue until relevant retrievable evidence is exhausted.
4) Return updates only for target fields; do not include any other field.

Output rules:
1) Use one of two response shapes only:
   - context_request object only, OR
   - final output JSON object.
2) Return JSON only.
3) Final output must contain exactly the requested fields.

Field-specific guidance (rules + schema):
<<TARGET_RULES_AND_SCHEMA>>

<<CONTEXT_REQUEST_FORMAT_GROUP>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<GUIDANCE_SUMMARY_INSTRUCTIONS>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT_GROUP>>", context_request_format_template.replace("<<REQUEST_ID>>", "node_refine_group_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


GROUP_TARGET_FIELDS: Dict[str, str] = {
    "group_1_identity": "name, description, framework",
    "group_2_io": "inputs, outputs",
    "group_3_execution_keys": "code_execution, required_keys",
    "group_5_metadata": "metadata",
    "group_6_llm": "llm_config, system_prompt, user_prompt_template",
    "group_7_mcp": "tool_list",
    "group_8_tool_flags": "read_internal, read_external, write_internal, write_external, is_rag_tool",
    "group_9_agent": "agent_type, llm_config, system_prompt, user_prompt_template",
    "group_10_system": "system_type, llm_config, system_prompt, user_prompt_template",
    "group_11_database": "data_path",
}


GROUP_TARGET_RULES: Dict[str, str] = {
    "group_1_identity": """For field `name`:
1) Set the component name in the system, based on the code object name that represents this component.
2) Rename only with concrete evidence.
For field `description`:
1) Prefer explicit description text from code/docs when available.
2) Otherwise write a concise functional description; do not describe implementation internals.
For field `framework`:
1) Set only with explicit evidence.
2) Use a single framework name (for example LangGraph, CrewAI, AutoGen); if unresolved, omit `framework` from updates.

Final output JSON:
{
  "updates": {
    "name": "<component_name>",
    "description": "<runtime_functional_description>",
    "framework": {"framework":"LangGraph","other_description":null}
  },
  "Explanations": {
    "name": "<short evidence-based reason>",
    "description": "<short evidence-based reason>",
    "framework": "<short evidence-based reason>"
  }
}
""",

    "group_2_io": """For fields `inputs` and `outputs`:
1) Focus on the runtime interface of this agentic component.
2) Include only evidence-backed entries; do not invent ports.
3) Each item must follow the exact InputPort/OutputPort schema.
4) If unresolved, keep [] for that side.
5) For each output item, set `output_kind` exactly to one of:
   - `data`: output contains information/data only (read-only behavior, no external state change).
   - `action`: tool effect is an external side effect/state change, with no data payload returned.
   - `data_and_action`: both a data payload is returned and an external side effect/state change occurred.
6) Treat `action` as TRUE side-effect semantics only: something changed in an external system or durable state
   (for example: email sent, buy/sell order executed, record created/updated/deleted, file written/deleted).
7) Do NOT label as `action` for read-only retrieval/inspection operations
   (for example: fetch/get/list/read/search/query/describe/check/status), even though a function call happened.
8) If evidence is ambiguous between read-only and side-effect behavior, prefer `data`.

Final output JSON:
{
  "updates": {
    "inputs": [
      {"name":"<input_name>","dtype":"<input_dtype>","description":"<optional_input_description_or_null>","required":true,"default":"<default_or_null>"}
    ],
    "outputs": [
      {"name":"<output_name>","dtype":"<output_dtype>","description":"<optional_output_description_or_null>","output_kind":"data|action|data_and_action","other_output_kind_description":null}
    ]
  },
  "Explanations": {
    "inputs": "<short evidence-based reason>",
    "outputs": "<short evidence-based reason>"
  }
}""",

    "group_3_execution_keys": """For field `code_execution`:
1) Must be boolean.
2) Set `code_execution=true` ONLY when this component executes runtime-provided code/commands as data.
   Valid positive evidence includes interpreter/sandbox/REPL behavior, code-runner tools, `exec`/`eval`, notebook execution,
   shell-command execution, or equivalent "run arbitrary code/command text" semantics.
3) Set `code_execution=false` for components that run only predefined implementation logic, even when that logic is local Python code.
   This includes normal business tools, API wrappers, DB/file operations, routing/orchestration, state updates, and text generation.
4) Calling an LLM/tool/API is NOT code execution by itself.
5) If dynamic code/command execution is not explicit in evidence, set `code_execution=false`.

For field `required_keys`:
- env_var_names: optional list of env variable names from .env

1) Must follow RequiredKeys shape exactly.
2) Set `enabled=true` only when this node needs runtime credentials; otherwise set `enabled=false` and `keys=[]`.
3) Use explicit OR implicit evidence. Implicit evidence is valid when node/provider identity indicates authenticated external API usage, even without direct env/header/api_key code in the snippet.
Valid evidence includes:
   - direct key usage (for example env/header/api_key/auth references),
   - provider/client wrapper identity with clear credential requirements,
   - constructor/config/call bindings that tie this node to a keyed service.
4) Choose service-specific keys only. Do not guess generic keys.
5) If a matched key exists in `env_var_names`, use that exact spelling.
6) For agent or LLM nodes, return the complete evidence-backed credential bundle, not API key alone. Include companion fields when used by this node at runtime (for example key + endpoint/base_url + api_version + deployment/model selector).
7) For Tool and External_MCP_server nodes, include auth/resource keys only when tied to this node’s runtime calls (explicitly or via a clearly identified wrapper/provider).
8) Set enabled=false only when evidence indicates no credentials are needed (local-only operations). Do not disable just because evidence is implicit.

Final output JSON:
{
  "updates": {
    "code_execution": true,
    "required_keys": {"enabled":true,"keys":["OPENAI_API_KEY"]}
  },
  "Explanations": {
    "code_execution": "<short evidence-based reason>",
    "required_keys": "<short evidence-based reason>"
  }
}""",

    "group_5_metadata": """For field `metadata`:
1) Clean metadata from cross-component/logging noise and keep only this component's important metadata.
2) Keep only evidence-backed, component-specific details useful for downstream passes.
3) Remove duplicated information already represented in core fields; do not add speculative notes.

Final output JSON:
{
  "updates": {
    "metadata": {<string_key>: <json_value>}
  },
  "Explanations": {
    "metadata": "<short evidence-based reason>"
  }
}""",

    "group_6_llm": """For fields `llm_config`, `system_prompt`, `user_prompt_template`:
1) Use only direct code evidence from this component; no invention, no paraphrasing, no inferred values.
2) Keep values exactly as represented in code (object/text/params), following the exact schema shape.
3) If a field is not concretely present in code for this node, omit that field from updates.

Final output JSON:
{
  "updates": {
    "llm_config": {"provider":"<provider_or_null>","class_name":"<class_name_or_null>","model_name":"<model_name_or_null>","temperature":0.0},
    "system_prompt": "<system_prompt_text>",
    "user_prompt_template": "<user_prompt_template_text>"
  },
  "Explanations": {
    "llm_config": "<short evidence-based reason>",
    "system_prompt": "<short evidence-based reason>",
    "user_prompt_template": "<short evidence-based reason>"
  }
}""",

    "group_7_mcp": """For field `tool_list`:
1) Validate existing MCP tool children and keep only evidence-backed Tool nodes.
2) Add missing children only when they are concrete Tool runtime capabilities.
3) Never add MCP servers to `tool_list` (no Local_MCP_server / External_MCP_server here).
4) If an MCP server is discovered, do not place it in this field; keep `tool_list` tool-only.

Final output JSON:
{
  "updates": {
    "tool_list": ["<tool_node_var>"]
  },
  "Explanations": {
    "tool_list": "<short evidence-based reason>"
  }
}""",

    "group_8_tool_flags": """For fields `read_internal`, `read_external`, `write_internal`, `write_external`, `is_rag_tool`:
Definitions:
- Internal = local resources in this environment (local files, local DB/storage, local indexes, local services).
- External = remote resources outside this environment (web pages, external APIs, remote DB/storage/services).
- RAG tool = a retrieval-oriented tool that queries/searches a knowledge base, vector store, documents, or other indexed corpus for context.

Rules:
1) Each field must be boolean.
2) Set each flag independently, using explicit evidence for that exact capability; do not infer one flag from another.
3) `read_internal` = True only if this node can read local/internal resources.
4) `read_external` = True only if this node can read/fetch from remote/external resources.
5) `write_internal` = True only if this node can create/update/delete local/internal resources.
6) `write_external` = True only if this node can create/update/delete remote/external resources.
7) `is_rag_tool` = True only if this tool is retrieval-focused (for example vector/keyword retrieval over docs/KB/index).
8) If any flag is unresolved, return `context_request` for the missing evidence (do not guess).

Final output JSON:
{
  "updates": {
    "read_internal": false,
    "read_external": false,
    "write_internal": false,
    "write_external": false,
    "is_rag_tool": false
  },
  "Explanations": {
    "read_internal": "<short evidence-based reason>",
    "read_external": "<short evidence-based reason>",
    "write_internal": "<short evidence-based reason>",
    "write_external": "<short evidence-based reason>",
    "is_rag_tool": "<short evidence-based reason>"
  }
}""",

    "group_9_agent": """For fields `agent_type`, `llm_config`, `system_prompt`, `user_prompt_template`:
1) Use only direct code evidence from this component; no invention, no paraphrasing, no inferred values.
2) For `agent_type`, choose labels only from explicit behavior evidence. Allowed examples: ReAct, CodeAct, Orchestrator, Planner, Executor, other.
3) Keep exact schema shape; if unresolved, omit field from updates (except `agent_type` may use `other` when explicitly justified).

Final output JSON:
{
  "updates": {
    "agent_type": {"type":["ReAct"],"other_description":null},
    "llm_config": {"provider":"<provider_or_null>","class_name":"<class_name_or_null>","model_name":"<model_name_or_null>","temperature":0.0},
    "system_prompt": "<system_prompt_text>",
    "user_prompt_template": "<user_prompt_template_text>"
  },
  "Explanations": {
    "agent_type": "<short evidence-based reason>",
    "llm_config": "<short evidence-based reason>",
    "system_prompt": "<short evidence-based reason>",
    "user_prompt_template": "<short evidence-based reason>"
  }
}""",

    "group_10_system": """For fields `system_type`, `llm_config`, `system_prompt`, `user_prompt_template`:
1) Use only direct code evidence from this component; no invention, no paraphrasing, no inferred values.
2) For `system_type`, choose labels only from explicit orchestration evidence. Allowed examples: Sequential, Orchestrator, Router, Loop, Hierarchical, other.
3) Keep exact schema shape; if unresolved, omit field from updates (except `system_type` may use `other` when explicitly justified).

Final output JSON:
{
  "updates": {
    "system_type": {"type":["Orchestrator"],"other_description":null},
    "llm_config": {"provider":"<provider_or_null>","class_name":"<class_name_or_null>","model_name":"<model_name_or_null>","temperature":0.0},
    "system_prompt": "<system_prompt_text>",
    "user_prompt_template": "<user_prompt_template_text>"
  },
  "Explanations": {
    "system_type": "<short evidence-based reason>",
    "llm_config": "<short evidence-based reason>",
    "system_prompt": "<short evidence-based reason>",
    "user_prompt_template": "<short evidence-based reason>"
  }
}""",

    "group_11_database": """For field `data_path`:
1) Must be a concrete runtime DB path/URI/identifier string.
2) Do not invent paths.
3) If unresolved, omit `data_path` from updates.

Final output JSON:
{
  "updates": {
    "data_path": "<db_path_or_uri>"
  },
  "Explanations": {
    "data_path": "<short evidence-based reason>"
  }
}""",
}


def _build_group_prompt(group_key: str) -> str:
    rules_and_schema = f"Rules:\n{GROUP_TARGET_RULES[group_key]}"
    return GROUP_PROMPT_TEMPLATE.replace(
        "<<TARGET_FIELDS>>", GROUP_TARGET_FIELDS[group_key]
    ).replace(
        "<<TARGET_RULES_AND_SCHEMA>>", rules_and_schema
    )


GROUP_REFINEMENT_PROMPTS: Dict[str, str] = {k: _build_group_prompt(k) for k in GROUP_TARGET_RULES.keys()}
