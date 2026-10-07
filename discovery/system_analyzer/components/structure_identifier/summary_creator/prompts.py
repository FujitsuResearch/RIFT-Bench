"""Prompt constants for summary_creator."""
try:
    from ..global_utils import agentic_terminology_instructions, context_request_format_template, retrieval_instructions
except ImportError:
    from global_utils import agentic_terminology_instructions, context_request_format_template, retrieval_instructions

SYSTEM_GUIDANCE_SUMMARY_PROMPT = """You are summarizing an agentic system from evidence.

Goal:
Produce a clear plain-text guidance note that downstream relation-check stages can use directly.

<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>

Input:
- code_references: initial evidence list.

Rules:
1) `summary` must be plain text and must start with exactly:
   Agentic System Guidance:
2) Use only Agentic Component concepts in summary text:
   - System, Agent, LLM instance, Server/MCP server, Database/Store, Tool, major runtime controller components.
3) Use concrete runtime instance names from evidence (variable/component names), not generic class-only wording.
4) Focus on what exists in the active runtime path and direct attachment relations.
5) Be explicit about direct attachment boundaries (for example: agents in a system, tools/servers under an agent, tools under a server, LLMs used by agents).
6) If an Agent/Server/System is identified but its direct attached components are still unclear, request context instead of finalizing.
7) Keep summary text at Agentic Component level only. Do not include implementation internals (imports, class internals, helper function details).
8) Prefer short declarative sentences and unambiguous relation verbs ("uses", "has", "exposes").
9) Add only newly identified evidence as `code_references_add` (additions from this round).
10) If uncertainty remains about active runtime attachments (agents, tools, servers, LLMs, middleware/controllers, stores/databases), return a context_request and continue retrieval.
11) Use one of two response shapes only:
    - `context_request` object only (for more evidence), OR
    - final JSON response object (`summary`, `code_references_add`).
12) Do not mix both shapes in the same response.

Style:
- Keep the summary compact and relation-focused.
- Prefer explicit component-name lists (agents, LLMs, tools, servers, stores) over narrative prose.
- Mention file paths or variable names only when needed to disambiguate components.

Final output JSON:
{
  "summary": "<plain-text Agentic System Guidance block>",
  "code_references_add": [
    {
      "kind": "<definition|initialization|invocation|wiring|config|usage|other>",
      "file": "<abs_file_path>",
      "line_start": "<line_start_int>",
      "line_end": "<line_end_int>",
      "note": "<short evidence note>"
    }
  ]
}

<<CONTEXT_REQUEST_FORMAT_SUMMARY_GUIDANCE>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<AGENTIC_TERMINOLOGY_INSTRUCTIONS>>", agentic_terminology_instructions).replace("<<CONTEXT_REQUEST_FORMAT_SUMMARY_GUIDANCE>>", context_request_format_template.replace("<<REQUEST_ID>>", "summary_creator_guidance_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


__all__ = [
    "SYSTEM_GUIDANCE_SUMMARY_PROMPT",
]
