"""Prompt constants for static_validation."""

try:
    from ..global_utils import agentic_terminology_instructions, context_request_format_template, guidance_summary_instructions, retrieval_instructions
except ImportError:
    from global_utils import agentic_terminology_instructions, context_request_format_template, guidance_summary_instructions, retrieval_instructions

REPRESENTATION_SHORTLIST_PROMPT = """You are validating whether a candidate node is already represented in an existing graph.

Goal:
Return plausible graph node vars that may already represent candidate_node.

<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>

Inputs:
- candidate_var
- candidate_node
- graph_catalog: list of {var, name, node_type, description}
- guidance_summary
- retrieved_context (optional)

Priority rules (apply in this order):
1) Representation boundary first:
   - Include graph vars that plausibly represent the same Agentic Component as candidate_node.
   - Also include graph vars that plausibly own candidate_node when candidate_node is a Support Artifact internal to that component.
2) Distinguish component vs implementation detail:
   - If candidate looks like a function/helper/wrapper/config that implements another runtime component, shortlist likely owner components (do not treat the artifact as an independent component).

Operational rules:
1) Search for evidence until shortlist is evidence-backed; if evidence is missing, return context_request rather than guessing (RAG will retrieve context).
2) Do not invent agentic components, relationships, code references, files, or behavior.
3) This is shortlist-only with high recall: include all evidence-plausible owners/candidates at this stage rather than over-restricting early.
4) <<GUIDANCE_SUMMARY_INSTRUCTIONS>>
5) Return JSON only.

Return one of:
A) final response
{
  "plausible_matches": ["<graph_var>", "..."],
  "reason": "<short evidence-based reason>"
}

B) context_request only
<<CONTEXT_REQUEST_FORMAT_REPRESENTATION_SHORTLIST>>

When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_INSTRUCTIONS>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT_REPRESENTATION_SHORTLIST>>", context_request_format_template.replace("<<REQUEST_ID>>", "static_validation_rep_shortlist_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


REPRESENTATION_PAIR_PROMPT = """You are validating if a candidate node and an existing graph node represent the same agentic component.

Goal:
Decide whether graph_node already represents candidate_node.

<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>

Inputs:
- candidate_var
- candidate_node
- graph_var
- graph_node
- guidance_summary
- retrieved_context (optional)

Priority rules (apply in this order):
1) Representation boundary first:
   - is_represented=true when graph_node covers (bi-directional) candidate_node as the same runtime Agentic Component.
   - is_represented=true also when candidate_node is clearly a Support Artifact contained within graph_node.
2) Component vs implementation detail:
   - If candidate_node is implementation/support internals of graph_node (for example helper/function/wrapper/config used by that component), treat it as represented by graph_node.
3) Evidence quality:
   - Require semantic/runtime evidence (definition, usage, attachment, or call-flow).
   - Name overlap alone is not sufficient.
4) Treat representation as true not only for one-to-one identity, but also when candidate_node is a collection/factory placeholder and graph_node is a concretely instantiated member produced by that candidate’s definition.

Operational rules:
1) Search for evidence until the decision is evidence-backed; if evidence is incomplete, return context_request rather than guessing (RAG will retrieve context).
2) Do not invent agentic components, relationships, code references, files, or behavior.
3) <<GUIDANCE_SUMMARY_INSTRUCTIONS>>
4) Return JSON only.

Return one of:
A) final response
{
  "is_represented": true|false,
  "reason": "<short evidence-based reason>"
}

B) context_request
<<CONTEXT_REQUEST_FORMAT_REPRESENTATION_PAIR>>

When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_INSTRUCTIONS>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT_REPRESENTATION_PAIR>>", context_request_format_template.replace("<<REQUEST_ID>>", "static_validation_rep_pair_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


MISSING_NODE_DECISION_PROMPT = """You are deciding whether a candidate node must be attached to the graph.

Goal:
For candidate_node, decide if it should be attached to graph or remain isolated.

<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>

Inputs:
- candidate_var
- candidate_node
- root_var
- graph_catalog: list of {var, name, node_type, description}
- guidance_summary
- retrieved_context (optional)

Rules:
1) Do not invent agentic components, relationships, code references, files, or behavior.
2) Search for evidence until the decision is evidence-backed; if evidence is missing or conflicting, return context_request instead of guessing. If evidence shows dynamic or indirect context injection/composition (for example imported templates/headers/preambles/config/prompt bundles combined at runtime), include that injected context in scope and inspect it before deciding. Do not return a negative decision while such injected context is unresolved; return context_request for the missing injected context evidence first.
3) Return need_to_attach=true only when there is concrete evidence that candidate_node is part of runtime execution (direct participation, orchestration role, tool/server usage, or explicit attachment path to the graph).
4) Return need_to_attach=false when there is concrete evidence candidate_node should not be added as a new attached node: either (a) it is truly non-main/isolated (support-only artifact or out-of-graph utility), or (b) it should remain unattached because there is no strong evidence of direct attachment into the graph.
5) attach_parent_var, when provided, must be a var from graph_catalog.
6) If evidence supports both possibilities weakly, do not choose; request targeted context to resolve the conflict. If evidence is missing, return context_request instead of guessing.
7) <<GUIDANCE_SUMMARY_INSTRUCTIONS>>
8) If evidence is incomplete, return a context_request for the remaining unknowns. Search for evidence until decisions are evidence-backed; never guess or fabricate missing facts. RAG will retrieve context.
9) Return JSON only.

Return one of:
A) final response
{
  "need_to_attach": true|false,
  "attach_parent_var": "<var_or_null>",
  "reason": "<short evidence-based reason>"
}

B) context_request only
<<CONTEXT_REQUEST_FORMAT_MISSING_NODE_DECISION>>

When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_INSTRUCTIONS>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT_MISSING_NODE_DECISION>>", context_request_format_template.replace("<<REQUEST_ID>>", "static_validation_missing_decision_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


ADDING_NODE_PARENT_DECISION_PROMPT = """You are deciding if a newly validated node should be attached under a specific existing parent in the graph.

Goal:
Return a binary decision for one parent-candidate pair: attach under this parent or not.

<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>

Inputs:
- candidate_var
- candidate_node
- root_var
- parent_var
- parent_node
- missing_decision_prior (optional): prior result from missing-node step, including need_to_attach / attach_parent_var / reason
- guidance_summary
- retrieved_context (optional)

Rules:
1) Do not invent agentic components, relationships, code references, files, or behavior.
2) Search for evidence until decisions are evidence-backed; never guess or fabricate missing facts.
3) parent_var must be treated as fixed input for this call. Decide attach_to_parent only for this parent.
4) attach_to_parent=true only when there is concrete runtime attachment/ownership evidence that candidate_node is directly under parent_var.
5) attach_to_parent=false when evidence indicates no direct ownership, only indirect relation, or support-only relation to another component.
6) <<GUIDANCE_SUMMARY_INSTRUCTIONS>>
7) If evidence supports both outcomes weakly, return context_request instead of guessing. RAG will retrieve context. If attachment may be established through dynamic or indirect context injection/composition (for example imported templates/headers/preambles/config/prompt bundles combined at runtime), include that injected context in scope before deciding attach_to_parent. Do not return attach_to_parent=false while such injected context is unresolved; return context_request for the missing injected context evidence first.
8) Use missing_decision_prior as prior context when it is available, and consider it during the decision, but do not rely on it alone; attach_to_parent must still be supported by runtime evidence for this specific parent.
9) Return JSON only.

Return one of:
A) final response
{
  "attach_to_parent": true|false,
  "reason": "<short evidence-based reason>"
}

B) context_request only
<<CONTEXT_REQUEST_FORMAT_ADDING_PARENT_DECISION>>

When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>", agentic_terminology_instructions).replace("<<GUIDANCE_SUMMARY_INSTRUCTIONS>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT_ADDING_PARENT_DECISION>>", context_request_format_template.replace("<<REQUEST_ID>>", "static_validation_adding_parent_decision_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)

__all__ = [
    "REPRESENTATION_SHORTLIST_PROMPT",
    "REPRESENTATION_PAIR_PROMPT",
    "MISSING_NODE_DECISION_PROMPT",
    "ADDING_NODE_PARENT_DECISION_PROMPT"
]
