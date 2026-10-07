"""Prompt constants for connectivity_pass component."""
try:
    from ..global_utils import context_request_format_template, guidance_summary_instructions, retrieval_instructions
except ImportError:
    from global_utils import context_request_format_template, guidance_summary_instructions, retrieval_instructions

CONNECTIVITY_EDGE_PROMPT_BASE = """You are repairing internal edges for one parent graph using iterative evidence-based reasoning.

Goal:
Return edge mutations (`edges_add`, `edges_remove`) that reduce the listed connectivity issues for this parent.

Simple edge examples:
- Entry edge: `START -> planner`
- Handoff edge: `router -> search_agent`
- Completion edge: `worker -> END`

Inputs:
- parent_var
- parent_node
- child_vars
- child_nodes
- current_edges
- connectivity_issues
- guidance_summary
- retrieved_evidence_context

Rules:
1) START and END are required virtual anchors and may appear only inside edges.
2) Every edge must connect only between these node names: START, END, and the entries in child_vars.
3) Never invent node names outside that allowed set.
4) `connectivity_issues` is the authoritative list of current problems to solve.
5) `current_edges` means the parent's current internal edges at this iteration.
6) <<GUIDANCE_SUMMARY_INSTRUCTIONS>>
7) Use `child_nodes` semantics (role/type/description/evidence) to guide plausible flow, but ground actual edge decisions in concrete evidence from code context.
8) Prefer minimal, high-confidence mutations:
   - keep evidence-backed `current_edges`,
   - remove edges only when contradicted by stronger evidence,
   - add only edges you can justify.
9) <<OUTER_LOOP_INSTRUCTION>>
10) `edges_add` and `edges_remove` must be idempotent and non-duplicative (same from_/to should not be repeated).
11) `reason` for each mutation must be short and explicit about which issue it addresses.
12) If evidence is insufficient, use context_request. Search for evidence until decisions are evidence-backed; never guess or fabricate missing facts. RAG will retrieve context.

Output rules:
1) Use one of two response shapes only:
   - `context_request` object only (for more evidence), OR
   - final output JSON object with `edges_add` and `edges_remove`.
2) Return JSON only.
3) Final output must contain only: `edges_add`, `edges_remove`.

Output JSON:
{
  "edges_add": [
    {"from_": "START|<child_var>", "to": "<child_var>|END", "reason": "<short reason>"}
  ],
  "edges_remove": [
    {"from_": "START|<child_var>", "to": "<child_var>|END", "reason": "<short reason>"}
  ]
}

<<CONTEXT_REQUEST_FORMAT_CONNECTIVITY_BASE>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<GUIDANCE_SUMMARY_INSTRUCTIONS>>", guidance_summary_instructions).replace("<<CONTEXT_REQUEST_FORMAT_CONNECTIVITY_BASE>>", context_request_format_template.replace("<<REQUEST_ID>>", "connectivity_pass_base_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)

__all__ = [
    "CONNECTIVITY_EDGE_PROMPT_BASE",
]
