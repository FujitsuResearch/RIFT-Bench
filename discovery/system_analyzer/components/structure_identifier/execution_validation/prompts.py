from __future__ import annotations

from typing import Any, Dict, List

NAME_MAPPING_PROMPT = """
You map one observed runtime component event name to expected component names.

Goal:
- Decide whether this observed event should map to an existing expected name.
- If no expected name is supported, keep it as a new component signal.

Input:
- component_type: "agent" or "tool"
- observed_name (may be empty)
- event_evidence: event type/content/tool_calls, and optional context
  (span/path provenance, previous_event, next_event, caller_name)
- expected_names: list of allowed expected names

Rules:
1) Return decision="map_to_expected" only when identity evidence supports exactly one expected name.
2) If no expected name has sufficient identity evidence, return decision="new_component" and mapped_name="".
3) Do not invent names not present in expected_names.
4) Prefer stable runtime names over generic placeholders unless evidence is strong.

Output JSON:
{
  "decision": "map_to_expected|new_component",
  "mapped_name": "string-or-empty",
  "reason": "short string"
}
""".strip()

INVENTORY_NAME_MAPPING_PROMPT = """
You map one observed inventory tool name to expected tool names.

Goal:
- Decide whether this observed inventory tool is the same specific expected tool.
- If no expected name is supported, keep it as a new component signal.

Input:
- component_type: "tool"
- observed_name
- event_evidence: inventory record context (owner/owner_kind/description/inputs/outputs)
- expected_names: dict where each key is a canonical expected tool name and the value is a list of aliases

Rules:
1) Use best-effort identity mapping: combine name similarity, aliases, owner context, description, and IO shape.
2) If evidence is ambiguous between multiple candidates, return decision="new_component" and mapped_name="".
3) Do not invent names not present in expected_names keys or alias values.

Output JSON:
{
  "decision": "map_to_expected|new_component",
  "mapped_name": "string-or-empty",
  "reason": "short string"
}
""".strip()

TRACE_ALIAS_RESOLUTION_PROMPT = """
You resolve multiple observed trace names for a single expected component into one canonical trace-facing name.

Goal:
- Keep one final name for this component in this trace.
- Prefer the most specific runtime name supported by evidence.

Input:
- component_type: "agent" or "tool"
- expected_name
- alias_candidates
- evidence

Output JSON:
{
  "chosen_name": "one-of-alias_candidates",
  "reason": "short string"
}
""".strip()

INVENTORY_PARSE_PROMPT = """
You extract tool inventory from a free-language response.

Goal:
- Convert the response text into strict structured tool records grouped by parent component.

Input:
- allowed_parents: list of valid parent components (agents/servers in this system)
- response_text

Rules:
1) Extract only tools explicitly stated or strongly evidenced in response_text.
2) For every extracted tool, assign exactly one parent from allowed_parents when possible.
3) Keep observed runtime tool names exactly as written in the response text.
4) If a field is unknown, use null/empty list instead of inventing.
5) Return strict JSON only.

Output JSON:
{
  "tools_by_parent": [
    {
      "parent": "<parent_component_name>",
      "parent_kind": "agent|server",
      "tools": [
        {
          "name": "<tool_name>",
          "description": "<short purpose or empty>",
          "inputs": [{"field":"<name>","type":"<type>","required":true|false,"default":null}],
          "outputs": [{"field":"<name_or_output_type>","type":"<type>"}],
          "side_effects": "<string or null>",
          "constraints": "<string or null>"
        }
      ]
    }
  ]
}
""".strip()


def build_name_mapping_payload(
    *,
    component_type: str,
    observed_name: str,
    event_evidence: Dict[str, Any],
    expected_names: List[str],
) -> Dict[str, Any]:
    """Build the prompt payload used to map one observed agent/tool name to expected names."""
    return {
        "prompt": NAME_MAPPING_PROMPT,
        "component_type": component_type,
        "observed_name": observed_name or "",
        "event_evidence": event_evidence,
        "expected_names": expected_names,
        "output_contract": {
            "decision": "map_to_expected|new_component",
            "mapped_name": "string-or-empty",
            "reason": "string",
        },
    }


def build_inventory_name_mapping_payload(
    *,
    observed_name: str,
    event_evidence: Dict[str, Any],
    expected_names: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Build the prompt payload used to map one observed inventory tool name to expected tool names."""
    return {
        "prompt": INVENTORY_NAME_MAPPING_PROMPT,
        "component_type": "tool",
        "observed_name": observed_name or "",
        "event_evidence": event_evidence,
        "expected_names": expected_names,
        "output_contract": {
            "decision": "map_to_expected|new_component",
            "mapped_name": "string-or-empty",
            "reason": "string",
        },
    }


def build_trace_alias_resolution_payload(
    *,
    component_type: str,
    expected_name: str,
    alias_candidates: List[str],
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the prompt payload used to choose one trace-facing alias for an expected component."""
    return {
        "prompt": TRACE_ALIAS_RESOLUTION_PROMPT,
        "component_type": component_type,
        "expected_name": expected_name,
        "alias_candidates": alias_candidates,
        "evidence": evidence,
        "output_contract": {
            "chosen_name": "one-of-alias_candidates",
            "reason": "string",
        },
    }


def build_inventory_parse_payload(
    *,
    allowed_parents: List[Dict[str, Any]],
    response_text: str,
) -> Dict[str, Any]:
    """Build the prompt payload used to parse tool inventory from one response text."""
    return {
        "prompt": INVENTORY_PARSE_PROMPT,
        "allowed_parents": allowed_parents or [],
        "response_text": response_text,
        "output_contract": {
            "tools_by_parent": "list[parent_tools]",
        },
    }



def agent_prompt_variants(agent_name: str, inventory_context_text: str) -> List[str]:
    """Return the direct and flexible activation prompts for one agent candidate."""
    direct = (
        f'Use the agent named "{agent_name}" to answer this request in one sentence: '
        'what can you help with right now?'
    )
    flexible = (
        f'If an agent named "{agent_name}" is available, route this request to it and let it solve a small task. '
        'Ask for a one-sentence answer plus one concrete capability. '
        f'Known runtime context: {inventory_context_text or "none"}.'
    )
    return [direct, flexible]


def tool_prompt_variants(owner: str, tool_name: str, description: str, inventory_context_text: str) -> List[str]:
    """Return the direct and flexible activation prompts for one tool candidate."""
    owner_hint = f' owned by "{owner}"' if owner else ""
    desc_hint = f' Description: {description}.' if description else ""
    direct = (
        f'Invoke the tool "{tool_name}"{owner_hint}. '
        'Use a minimal harmless input if one is needed, and return only the tool result.'
        f'{desc_hint}'
    )
    flexible = (
        f'Find and use the tool "{tool_name}"{owner_hint}. '
        'If the exact invocation is unclear, infer the smallest safe example call from the available runtime context and tool description. '
        f'Known runtime context: {inventory_context_text or "none"}.{desc_hint}'
    )
    return [direct, flexible]
