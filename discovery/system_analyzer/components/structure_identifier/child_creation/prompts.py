"""Prompt constants for child_creation and connectivity behavior."""

try:
    from ..global_utils import agentic_terminology_instructions, context_request_format_template, guidance_summary_instructions, retrieval_instructions
except ImportError:
    from global_utils import agentic_terminology_instructions, context_request_format_template, guidance_summary_instructions, retrieval_instructions


CHILD_DISCOVERY_DELTA_PROMPT = """You are discovering direct children of a runtime parent component in an agentic system graph.

Goal:
Return only NEW child proposals for this parent in this round (new additions, not full list).

<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>

Inputs:
- parent_var
- parent_node
- guidance_summary
- discovered_children_so_far
- retrieved_evidence_context
- retrieval_history

Decision rules:
1) Decide direct-child relations using Agentic Component boundaries, not code-object proximity.
2) Support artifacts are not children by default: keep proposals at Agentic-Component level only. Exception: if the artifact is the concrete instantiated binding of a distinct Agentic Component used by the parent, it must be emitted as that component. Example: when the parent is an Agent, an llm config is evidence of an `LLM` child of that Agent; when the parent is an LLM, an llm config is not evidence of an additional child under that LLM.
3) A direct child means `candidate` is directly attached under `parent_node` as its own agentic component in the agentic system (containment/attachment boundary), not merely referenced or indirectly used; examples: a System can have Agent children, an Agent can have Tool/Server/LLM children, and a Server can have Tool children, but "Agent uses LLM" does not make the Agent a child of the LLM.
4) Code structure is evidence only, while child decisions are about agentic components in the agentic system: wrappers/helpers/config/prompt/schema artifacts may provide evidence for attachment, but they are not child nodes by themselves.
5) Not direct when relation exists only through another distinct Agentic Component boundary (parent -> component_X -> candidate).
6) Do not return duplicates already present in discovered_children_so_far.
7) Every child in children_add must include at least one attachment code reference in code_references_add_child.
8) <<GUIDANCE_SUMMARY_INSTRUCTIONS>>
9) Use `retrieved_evidence_context` to propose children before asking for more retrieval.
10) If evidence is incomplete, return partial progress now: include all newly found children in children_add and include a context_request for the remaining unknowns (do not wait for a full list before responding). Search for evidence until decisions are evidence-backed; never guess or fabricate missing facts. RAG will retrieve context.
11) If existing evidence mentions a symbol/header/template/import/prompt that may define attached agentic components, request information about that symbol before setting is_complete=true.
12) Use `retrieval_history` to avoid repeating the same query unless you are explicitly refining it to target a different missing detail.
13) Set is_complete=true only when no additional plausible direct children remain.


Output rules:
1) Return JSON only.
2) Return exactly these top-level keys: children_add, is_complete, completion_reason, optional context_request.
3) If is_complete=false, context_request is required and must include at least one needs item.
4) If is_complete=true, omit context_request.
5) Do not return extra top-level keys.
6) For each added child, include exactly these fields: name, node_type, description, and code_references_add_child. Name should match the concrete runtime instance name used in code when available (not a generic class/type label). Description should be plain text and should follow evidence in code when available.
7) Attachment evidence requirements are strict:
   - code_references_add_child must contain only attachment/wiring evidence where the child is attached/used by this parent.
   - Use kind="usage" for all entries.
   - Do NOT include full child definition/implementation references in this phase; those are added later.s
Output JSON:
{
  "children_add": [
    {
      "name": "<child runtime name>",
      "node_type": "System|Agent|LLM|Tool|Database|Local_MCP_server|External_MCP_server|other",
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
  ],
  <<CONTEXT_REQUEST_FORMAT_CHILD_CREATION_MATERIALIZE>>
  "is_complete": false,
  "completion_reason": "<short reason>"
}

<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_INSTRUCTIONS>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT_CHILD_CREATION_MATERIALIZE>>", context_request_format_template.replace("<<REQUEST_ID>>", "child_creation_materialize_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


CHILD_MATERIALIZATION_PROMPT = """You are generating one discovered child proposal into a concrete NodeSpec object using the provided schema.

Goal:
Create a child node object using ONLY NodeSpec fields from the provided schema text.

<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>

Inputs:
- parent_var
- parent_node
- child_proposal
- nodespec_schema
- guidance_summary

Decision rules:
1) Do not materialize support artifacts as child nodes. Default: if the candidate is a support artifact or not a distinct runtime component, return an empty child result (no child_node). Exception: if the artifact is the concrete instantiated binding of a distinct Agentic Component used by the parent, materialize that component (not the artifact) as the child. Example: for an Agent parent, an llm config is evidence of an LLM child; for an LLM parent, an llm config is not evidence of an additional child.
2) Child must be a distinct runtime component in the agentic system. If proposal is only an internal implementation part of the same runtime component (for example one wrapper/object inside a Tool), do not materialize it as a child.
3) Preserve attachment evidence from child_proposal in child_node.code_references (child-side usage refs).
4) Code reference requirements:
   - include definition references for the child component,
   - include usage references where THIS parent uses/attaches the child,
   - include implementation references for the child behavior.
   - do not add unrelated source/import/config references unless needed for one of the three categories above.
5) <<GUIDANCE_SUMMARY_INSTRUCTIONS>>
6) If evidence is insufficient to finalize child_node as a direct child of parent_node, do not guess: return context_request for the missing parent-attachment evidence even when definition/implementation evidence exists. RAG will retrieve context.

Output rules:
1) Use one of two response shapes only:
   - context_request object only (for more evidence), OR
   - final output JSON object (child_node).
2) Return JSON only.
3) Final output must contain only one top-level key: child_node.
4) Return ONLY these fields in child_node: name, node_type, description, code_references, inputs, outputs, metadata. Name must match the concrete runtime instance name from code when available (not a generic class/type label). Description must be plain text and evidence-grounded. Inputs and outputs must be evidence-grounded; if not evidenced, return empty lists. metadata may contain only concise, relevant supporting details for this component.
5) Do not return extra top-level keys.

Output JSON:
{
  "child_node": {
    "name": "<name>",
    "node_type": {"type": "System|Agent|LLM|Tool|Database|Local_MCP_server|External_MCP_server|Deterministic_controller|other", "other_description": null},
    "description": "<description>",
    "code_references": [],
    "inputs": [],
    "outputs": [],
    "metadata": {}
  }
}

<<CONTEXT_REQUEST_FORMAT_CHILD_CREATION_MATERIALIZE>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_INSTRUCTIONS>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT_DUPLICATE_CANDIDATES>>", context_request_format_template.replace("<<REQUEST_ID>>", "child_creation_duplicate_candidates_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


CHILD_RUNTIME_JUDGE_PROMPT = """You are a strict binary judge deciding whether a node should be included as a direct child of a different node.

Goal:
Decide whether candidate_child should be included as a direct child agentic component under parent_node, based on distinct agentic identity and attachment evidence.

<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>

Inputs:
- parent_var
- parent_node
- candidate_child_var
- candidate_child_node
- guidance_summary

Decision rules:
1) Code objects and support artifacts are not runtime components:
   helper/config/prompt/schema/wrapper/adapter/list-builder/constants and plain function/class objects are evidence artifacts, not child nodes.
   Exception: if a support artifact is the concrete instantiated binding of a distinct Agentic Component used by the parent, judge that component (not the artifact) as the child-candidate. Example: for an Agent parent, an llm config is evidence of an LLM child; for an LLM parent, an llm config is not evidence of an additional child.
2) candidate_child must be a distinct agentic component and directly attached under parent_node; exclude it when it is only an internal part, implementation detail, or backend of the parent component.
3) Do not split a single agentic component into multiple child nodes just because multiple internal code objects are used.
4) If candidate_child is the same agentic component already represented by parent_node (for example an internal backend/helper object of that component), return include_child=false.
5) Base the decision on explicit evidence from both candidate_child_node.code_references and parent_node.code_references, and require parent-side attachment evidence for include_child=true.
6) Do not attach peer components from the same hierarchy level under each other (for example Agent↔Agent, Tool↔Tool, Server↔Server) unless evidence explicitly shows a direct containment/attachment relation.
7) <<GUIDANCE_SUMMARY_INSTRUCTIONS>>

Output rules:
1) Return JSON only.
2) Return exactly:
{
  "include_child": true|false,
  "reason": "<short evidence-based reason>"
}
""".replace("<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_INSTRUCTIONS>>", guidance_summary_instructions)


DUPLICATE_CANDIDATE_PROMPT = """You are identifying plausible duplicate matches for a candidate runtime component.

Goal:
From the global node catalog, return plausible existing node vars that may represent the same agentic component as candidate_child_node.

<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>

Inputs:
- parent_var
- parent_node
- candidate_child_node
- existing_nodes_catalog (list of {var, name, node_type, description})
- guidance_summary

Decision rules:
1) Propose plausible matches conservatively by agentic runtime identity, not by superficial token overlap.
2) A plausible match must have compatible agentic role and similar operational identity.
3) Propose candidates only when they could represent the same single agentic component as candidate_child_node (same component identity, not just same category or layer).
4) If evidence is insufficient to shortlist confidently, return context_request instead of broad guessing. Search for evidence until decisions are evidence-backed; never guess or fabricate missing facts. RAG will retrieve context.
5) <<GUIDANCE_SUMMARY_INSTRUCTIONS>>

Output rules:
1) Use one of two response shapes only:
   - context_request object only (for more evidence), OR
   - final output JSON object with plausible_matches.
2) Return JSON only.
3) Final output must contain only one top-level key: plausible_matches.
4) plausible_matches may be empty.

Output JSON:
{
  "plausible_matches": ["<existing_var_1>", "<existing_var_2>"]
}

<<CONTEXT_REQUEST_FORMAT_DUPLICATE_CANDIDATES>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_INSTRUCTIONS>>", guidance_summary_instructions).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


DUPLICATE_DECISION_PROMPT = """You are deciding whether candidate_child_node and existing_node represent the same agentic component.

Goal:
Return true only when candidate_child_node is the same agentic component as existing_node.

<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>

Inputs:
- parent_var
- parent_node
- candidate_child_node
- existing_var
- existing_node
- guidance_summary

Decision rules:
1) Decide sameness by agentic component identity (role + behavior), not by naming similarity alone.
2) Do not merge distinct runtime components even if names are similar.
3) Treat candidate_child_node and existing_node as the same only if they can represent the same single agentic component instance in the system (same identity), not just similar names, labels, or wording.
4) New or richer evidence about an existing candidate does not imply a new component; if identities align, treat it as the same agentic component.
5) If evidence is insufficient to decide confidently, return context_request instead of guessing. Search for evidence until decisions are evidence-backed; never guess or fabricate missing facts. RAG will retrieve context.
6) <<GUIDANCE_SUMMARY_INSTRUCTIONS>>

Output rules:
1) Use one of two response shapes only:
   - context_request object only (for more evidence), OR
   - final output JSON object with is_same_component and reason.
2) Return JSON only.
3) Final output must contain only: is_same_component, reason.
4) When returning context_request, each needs[].query must target exactly one missing fact using concrete code literals (symbol names, import lines, assignments, call expressions, or exact file name strings) rather than natural-language requests; if unresolved, issue a refined literal query for the same fact instead of broadening scope.


Output JSON:
{
  "is_same_component": false,
  "reason": "<short evidence-based reason>"
}

<<CONTEXT_REQUEST_FORMAT_DUPLICATE_DECISION>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_INSTRUCTIONS>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT_DUPLICATE_DECISION>>", context_request_format_template.replace("<<REQUEST_ID>>", "child_creation_duplicate_decision_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)




__all__ = [
    "CHILD_DISCOVERY_DELTA_PROMPT",
    "CHILD_MATERIALIZATION_PROMPT",
    "CHILD_RUNTIME_JUDGE_PROMPT",
    "DUPLICATE_CANDIDATE_PROMPT",
    "DUPLICATE_DECISION_PROMPT",
]
