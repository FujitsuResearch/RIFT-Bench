"""Prompt constants for root_finder."""
try:
    from ..global_utils import agentic_terminology_instructions, context_request_format_template, retrieval_instructions
except ImportError:
    from global_utils import agentic_terminology_instructions, context_request_format_template, retrieval_instructions

ROOT_LIST_CANDIDATE_PROMPT = """You are selecting the best root candidate from ONE file-level node list.

Goal:
From the given nodes list (one file's catalog output), choose the single node most likely to be the system root candidate.

<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>

Inputs:
- current_filepath: absolute path of the file this nodes list came from
- nodes: list[NodeSpec] for this file only

Graph framing:
1) The repository is modeled as a runtime graph:
   - Nodes are runtime components (System, Agent, LLM, Tool, Database, controllers, etc.).
2) A root node is the highest-level runtime source component in this graph:
   - the component from which the system's runtime orchestration originates,
   - not merely a CLI/bootstrap helper that loads config and delegates.
3) In this call, you only choose the best local candidate from this file list.

Output JSON:
{
  "selected_node": NodeSpec or null,
  "reason": "<short evidence-based reason>"
}

Rules:
1) selected_node must be exactly one node from the provided nodes list, or null when the list is empty.
2) Prefer top-level runtime orchestrators (System/Agent/controller) over helper/config/bootstrap artifacts.
3) Do not invent fields or modify node content.
4) Return JSON only.
5) If evidence is insufficient, request repository context via context_request (RAG will retreive it).

<<CONTEXT_REQUEST_FORMAT_ROOT_LIST_BEST>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>", agentic_terminology_instructions).replace("<<CONTEXT_REQUEST_FORMAT_ROOT_LIST_BEST>>", context_request_format_template.replace("<<REQUEST_ID>>", "root_finder_list_best_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


ROOT_CANDIDATE_COMPARE_PROMPT = """You are comparing TWO root candidates and picking the stronger one.

Goal:
Choose which candidate is more likely to be the true system root.

<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>

Inputs:
- stored_candidate: NodeSpec (current best candidate)
- contender_candidate: NodeSpec (new candidate from current file list)

Graph framing:
1) The repository is modeled as a runtime graph where nodes are runtime components.
2) The root node is the highest-level runtime source component for overall orchestration.
3) Compare candidates by global graph role, not local code convenience:
   - prefer true runtime orchestrators over wrappers/bootstrap/config helpers,
   - prefer broader system-level ownership/composition semantics.

Output JSON:
{
  "winner": "stored|contender",
  "reason": "<short evidence-based reason>"
}

Rules:
1) winner must be exactly one of: stored, contender.
2) Prefer candidates representing the top-level runtime source of the full agentic system.
3) Return JSON only.
4) If evidence is insufficient, request repository context via context_request (RAG will retreive it).

<<CONTEXT_REQUEST_FORMAT_ROOT_COMPARE>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>", agentic_terminology_instructions).replace("<<CONTEXT_REQUEST_FORMAT_ROOT_COMPARE>>", context_request_format_template.replace("<<REQUEST_ID>>", "root_finder_compare_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


__all__ = [
    "ROOT_LIST_CANDIDATE_PROMPT",
    "ROOT_CANDIDATE_COMPARE_PROMPT",
]
