"""Prompt builders for the MCP live-tool-discovery stage."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_mcp_connection_resolver_prompt(
    *,
    node_name: str,
    node_description: str,
    component_handle: str,
    constructor_snippet: str,
    constructor_file: str,
    assignment_snippet: str,
    assignment_file: str,
    file_contents: Dict[str, str],
    project_root: str,
    prior_attempts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the LLM payload asking for only the body of `_resolve_connection()`."""
    file_context = "\n\n".join(
        f"### FILE: {path}\n```python\n{text}\n```" for path, text in file_contents.items()
    )

    prior_feedback = ""
    if prior_attempts:
        blocks = []
        for i, attempt in enumerate(prior_attempts, start=1):
            blocks.append(
                f"Attempt {i} failed.\n"
                f"error_type: {attempt.get('error_type')}\n"
                f"error: {attempt.get('error')}\n"
                f"stderr_tail: {attempt.get('stderr_tail')}"
            )
        prior_feedback = "\n\nPREVIOUS FAILED ATTEMPTS (fix these specific issues):\n" + "\n---\n".join(blocks)

    prompt = f"""
You are writing ONE Python function body for a fixed probe-script template that will run INSIDE
the target repository's own environment (working directory = project root: {project_root}), with
the target repo's own `.env` already sourced into the process environment.

Target MCP server node: {node_name}
Description: {node_description or "<unknown>"}
Component handle in source: {component_handle or "<unknown>"}

The constructor call that builds this MCP client (verbatim source, file {constructor_file}):
```python
{constructor_snippet}
```

The enclosing assignment/definition this constructor appears inside (verbatim source, file {assignment_file}):
```python
{assignment_snippet}
```

Full contents of the relevant source file(s), for import/context purposes:
{file_context}

YOUR TASK
Write only the BODY of this exact function (do not change its name or signature, do not add any
other top-level code, do not import `mcp`/`asyncio` yourself -- the surrounding template already
handles the MCP SDK transport clients and protocol handshake):

def _resolve_connection():
    ...

It must return EXACTLY ONE of these shapes:
  {{"transport": "http", "url": <str>, "headers": <dict|None>}}
  {{"transport": "sse", "url": <str>, "headers": <dict|None>}}
  {{"transport": "stdio", "command": <str>, "args": <list[str]>, "env": <dict|None>}}

Rules:
- Resolve the REAL connection: import the enclosing module and/or call the real factory
  function/class/method exactly as the target code does, so any environment-variable-based
  secrets or config resolve for real. The working directory is already the project root, so
  ordinary imports of the target codebase's own modules will work.
- Never fabricate a fallback URL, header, or command. If the real connection cannot be resolved,
  raise an exception (any exception type/message) instead of guessing.
- If the target code constructs an object that already exposes attributes like `url`/`headers` or
  `command`/`args`/`env`, read them directly from that object rather than re-deriving them by hand.
- Keep it self-contained: only reference names you import or define within this function body.
- Return ONLY the function body (the lines that go inside the function, indented by exactly 4
  spaces relative to the `def` line), wrapped in <resolver>...</resolver> tags. Do not include the
  `def _resolve_connection():` line itself, and do not wrap the answer in markdown code fences.
{prior_feedback}
""".strip()

    return {"prompt": prompt, "temperature": 0}
