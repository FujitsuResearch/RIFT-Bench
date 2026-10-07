"""Prompt constants for post_processing component."""

import json
from typing import Any, Dict, List

try:
    from ..global_utils import context_request_format_template, retrieval_instructions
except ImportError:
    from global_utils import context_request_format_template, retrieval_instructions

NAME_PROMPT = """You are a code analysis expert specializing in Python variable tracing and runtime component identification.

Goal:
Find the code identifier that represents the given Agentic Component.

Important distinction:
- Prefer the variable name that holds the component instance.
- If no variable assignment exists, fall back to the function or class name when that definition itself is the component representation in code.
- Do NOT return a display name, label, package name, or guessed identifier.

Definition of "component_handle":
The code identifier that most directly represents the component in Python code. Prefer:
1) assignment target / attribute target holding the component
2) explicit alias variable
3) function name, if the component is represented directly by a function definition
4) class name, if the component is represented directly by a class definition

Examples:
- `pikachu_instance = Pokemon(pikachu, 4048)` -> component_handle is `pikachu_instance`
- `agent = build_agent(...)` -> component_handle is `agent`
- `self.router = Router(...)` -> component_handle is `self.router`
- `@tool
def best_tool(urls: List[str]) -> str:` -> component_handle is `best_tool`
- `class PlannerAgent(BaseAgent): ...` -> component_handle is `PlannerAgent`

Inputs:
- node_var
- node
- guidance_summary
- retrieved_context (optional)

Task:
Identify the single best code identifier that represents the component and return both:
1) the identifier name itself as component_handle
2) the supporting code_reference

Reasoning procedure:
1. Search for direct evidence that the component is bound to a variable or attribute.
2. If found, prefer that binding target.
3. If no such binding exists, search for a function definition or class definition that directly represents the component.
4. Choose the most canonical runtime identifier for the node.
5. Never return null just because there is no assignment if a function/class definition clearly represents the component.
6. Use only code evidence.

Selection policy:
Prefer, in order:
1) direct assignment target of the component instance
2) stable attribute assignment target holding the component
3) explicit alias variable of the component
4) function name if the component is directly represented by a function definition
5) class name if the component is directly represented by a class definition


Avoid:
- import-only names unless they are also the component representation in local code
- string labels
- config keys
- dict keys
- package/module names
- guessed identifiers
- component display names

Strict rules:
1. Do NOT invent files, symbols, or identifiers.
2. Use only evidence-backed reasoning.
3. Return exactly one result.
4. code_reference.kind must be "component_handle".
5. component_handle must be the exact identifier text as it appears in code.
6. The returned snippet must allow recovery of the identifier directly from code.
7. If multiple candidates exist, choose the most canonical runtime identifier for the node.
8. Only return context_request if evidence is insufficient to determine even a function/class fallback.

Output rules:
1. Return JSON only.
2. Use exactly one of:
   - context_request object only, OR
   - final output JSON object only
3. Final output must contain exactly these two top-level keys:
   - component_handle
   - code_reference
4. code_reference must contain exactly these keys:
   - kind
   - file
   - line
   - snippet
   - other_kind_description

Final output:
{
  "component_handle": "<exact identifier text>",
  "code_reference": {
    "kind": "component_handle",
    "file": "<path>",
    "line": 123,
    "snippet": "<short exact or near-exact code snippet containing the identifier>",
    "other_kind_description": null
  }
}

Examples of acceptable outputs:

- assignment case:
  {
    "component_handle": "planner",
    "code_reference": {
      "kind": "component_handle",
      "file": "app/agents.py",
      "line": 42,
      "snippet": "planner = PlannerAgent(model=model, tools=tools)",
      "other_kind_description": null
    }
  }

- function fallback case:
  {
    "component_handle": "best_tool",
    "code_reference": {
      "kind": "component_handle",
      "file": "tools.py",
      "line": 10,
      "snippet": "def best_tool(urls: List[str]) -> str:",
      "other_kind_description": null
    }
  }


<<CONTEXT_REQUEST_FORMAT_COMPONENT_HANDLE>>

<<RETRIEVAL_INSTRUCTIONS>>

Quality bar:
A correct answer returns the single best code identifier representing the component as component_handle, together with the supporting code_reference:
prefer a variable/attribute binding when present, otherwise fall back to a function name or class name that directly represents the component.
""".replace("<<CONTEXT_REQUEST_FORMAT_COMPONENT_HANDLE>>", context_request_format_template.replace("<<REQUEST_ID>>", "post_processing_component_handle_ref_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)

ASSIGNMENT_REFERENCE_PROMPT ="""You are a coding expert with specisilsaztion in LLM-based agents components assignment statements.

Goal:
You task is to find where the assignment statement of the given 'Agentic Component' and create a code-refernace object for it (detailed bellow).
Return exactly one code reference object that points to the most suting assignment statement location for this node's represented runtime component.

Terminology (mandatory):
- Agentic Component: a runtime unit with its own operational role in the agentic system (for example System, Agent, LLM instance, Tool, MCP server/server, Database, major custom runtime component).
- Code Object: function/class/variable/module in code. Code objects are evidence only; decisions are about Agentic Components.
- Node's assignment statement - The place where the component is assigned

Inputs:
- node_var
- node
- guidance_summary
- retrieved_context (optional)

Decision rules:
1) Do not invent files, snippets, symbols, components, or behavior.
2) Search for evidence until output is evidence-backed; if evidence is missing, return context_request instead of guessing.
3) Return exactly one code reference object describing the component assignment statement location.
4) kind must be "assignment".
5) in most cases, the first line of the snippet should start with '<node_var> = ...' or with '... as <node_var>'.
6) <node_var> can also be the Name of the component and not the name of the varible, so be flexible.
6) snippet must be a short exact or near-exact code fragment from the cited location.
7) line must be either integer or [line_start, line_end].
8) If multiple assignment statement sites exist, choose the most canonical runtime assignment statement used by the graph node.
9) guidance_summary is high-level direction only and cannot override concrete code evidence.

Output rules:
1) Use one of two response shapes only:
   - context_request object only, OR
   - final output JSON object.
2) Return JSON only.
3) Final output must contain exactly one top-level key: code_reference.
4) code_reference must contain exactly these keys:
   - kind
   - file
   - line
   - snippet

Final output JSON:
{
  "code_reference": {
    "kind": "assignment",
    "file": "<path>",
    "line": 123,
    "snippet": "<short snippet>",
    "other_kind_description": null
  }
}

<<CONTEXT_REQUEST_FORMAT_ASSIGNMENT>>
When returning context_request, return only that object.
<<RETRIEVAL_INSTRUCTIONS>>
""".replace("<<CONTEXT_REQUEST_FORMAT_ASSIGNMENT>>", context_request_format_template.replace("<<REQUEST_ID>>", "post_processing_assignment_ref_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)

IMPLEMENTATION_REFERENCE_PROMPT = """You are a code analysis expert specializing in LLM-based agent systems and runtime component tracing.

Objective:
Given a target Agentic Component, identify the single best code location that defines its runtime behavior, and return exactly one code reference object for that definition.

Primary task:
Find the most canonical functional definition for the runtime component represented by the given node.

Definitions:
- Agentic Component: a runtime unit with a distinct operational role in an agentic system. Examples include: System, Agent, LLM instance, Tool, MCP server, database adapter, orchestrator, router, memory manager, or any major custom runtime component.
- Code Object: a function, class, variable, module, or config-backed symbol found in code. Code objects are evidence only. The task is to locate the functional definition of the Agentic Component, not merely any related code object.
- Functional definition: the code location where the core runtime behavior of the component is actually implemented or concretely bound. Prefer executable logic over metadata, registration, wiring, imports, schemas, or high-level configuration.
- Canonical functional definition: the single location that best explains how this component behaves at runtime for the represented graph node.

Inputs:
- node_var
- node
- guidance_summary
- retrieved_context (optional)

Reasoning policy:
- Use code evidence only.
- Treat guidance_summary as a hint, not as authority.
- Do not infer nonexistent files, snippets, symbols, relationships, or behavior.
- If evidence is insufficient, request more context instead of guessing.

Selection policy:
1. Return exactly one code reference object.
2. The returned reference must point to the best available functional definition for the component represented by the node.
3. Always prefer:
   a) concrete runtime implementation,
   b) then the function/class that directly implements the behavior,
   c) then the binding site that attaches external package behavior to this node,
   d) then nulls if no functional definition exists in repository evidence.
4. Prefer implementation over:
   - imports
   - registrations
   - graph assembly
   - dependency injection
   - config files
   - schema/type declarations
   - tests
   - documentation
5. If multiple candidate definitions exist, choose the one most directly responsible for runtime behavior of this node.
6. If the node refers to behavior provided entirely by an external package or remote system and no repository-local functional definition exists, return a code_reference object with null file/line/snippet, not a guess.
7. Some components are only instantiated, configured, or referenced, but not functionally defined in the repository. In those cases, return nulls.
8. kind must always be "definition".
9. snippet must be a short exact or near-exact fragment from the cited location.
10. line must be either:
   - a single integer, or
   - a two-element array: [line_start, line_end]

Important disambiguation rules:
- For classes, prefer the class definition if it is the main runtime behavior container.
- For functions, prefer the function definition if it directly implements the component behavior.
- For callable instances, wrappers, factories, or builders, prefer the underlying implementation actually executed at runtime, not the factory, unless the factory itself contains the core behavior.
- For externally provided tools/models/clients, prefer the local wrapper/adapter if that wrapper defines the runtime behavior relevant to this node.
- For config-defined components, only cite config if the config itself is the closest available binding of behavior and no repository-local implementation exists; otherwise prefer the implementation target.
- Do not return a call site unless that call site is also the most concrete binding of runtime behavior available.
- Do not return a symbol merely because its name matches the node.

Failure policy:
If you cannot identify the functional definition with evidence, return a context_request object only.

Output rules:
1. Return JSON only.
2. Use exactly one of these two shapes:
   - a context_request object only, OR
   - a final output object only.
3. Final output must contain exactly one top-level key:
   - code_reference
4. code_reference must contain exactly these keys:
   - kind
   - file
   - line
   - snippet
   - other_kind_description
5. other_kind_description must always be null.

Final output shape:
{
  "code_reference": {
    "kind": "definition",
    "file": "<path or null>",
    "line": 123,
    "snippet": "<short exact or near-exact code snippet, or null>",
    "other_kind_description": null
  }
}

Allowed null-output shape when no repository-local functional definition is found:
{
  "code_reference": {
    "kind": "definition",
    "file": null,
    "line": null,
    "snippet": null,
    "other_kind_description": null
  }
}

<<CONTEXT_REQUEST_FORMAT_IMPLEMENTATION>>

<<RETRIEVAL_INSTRUCTIONS>>

Quality bar:
A valid answer identifies the single strongest evidence-backed location where the node's represented component is functionally defined at runtime, or explicitly returns nulls / a context_request when that cannot be established from evidence.
""".replace("<<CONTEXT_REQUEST_FORMAT_IMPLEMENTATION>>", context_request_format_template.replace("<<REQUEST_ID>>", "post_processing_definition_ref_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)

def build_definition_prompt(assignment_code_reference: Dict[str, Any]) -> str:
    """
    Build a prompt for finding the functional definition of an agentic component,
    using a resolved assignment code reference as the primary tracing anchor.

    Parameters
    ----------
    assignment_code_reference : dict
        Expected shape:
        {
            "kind": "assignment",
            "file": "path/to/file.py",
            "line": 123,               # or [123, 126]
            "snippet": "x = Foo(...)",
            "other_kind_description": None
        }

    Returns
    -------
    str
        A ready-to-use prompt string with the assignment reference embedded as JSON.

    Raises
    ------
    TypeError
        If assignment_code_reference is not a dict.
    ValueError
        If required keys are missing or the payload is malformed.
    """
    if not isinstance(assignment_code_reference, dict):
        raise TypeError("assignment_code_reference must be a dict")

    required_keys = {"kind", "file", "line", "snippet"}
    missing = required_keys - set(assignment_code_reference.keys())
    if missing:
        raise ValueError(
            f"assignment_code_reference is missing required keys: {sorted(missing)}"
        )

    kind = assignment_code_reference.get("kind")
    file_ = assignment_code_reference.get("file")
    line = assignment_code_reference.get("line")
    snippet = assignment_code_reference.get("snippet")

    if kind not in {"assignment", "definition"}:
        raise ValueError(
            "assignment_code_reference['kind'] must be "
            f"'assignment' or 'definition', got {kind!r}"
        )

    if not isinstance(file_, str) or not file_.strip():
        raise ValueError("assignment_code_reference['file'] must be a non-empty string")

    valid_line = isinstance(line, int) or (
        isinstance(line, list)
        and len(line) == 2
        and all(isinstance(x, int) for x in line)
    )
    if not valid_line:
        raise ValueError(
            "assignment_code_reference['line'] must be an int or a two-element list of ints"
        )

    if not isinstance(snippet, str) or not snippet.strip():
        raise ValueError("assignment_code_reference['snippet'] must be a non-empty string")

    assignment_ref_json = json.dumps(
        {
            "kind": kind,
            "file": file_,
            "line": line,
            "snippet": snippet,
            "other_kind_description": assignment_code_reference.get(
                "other_kind_description", None
            ),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    return f"""You are a code analysis expert specializing in LLM-based agent systems and runtime component tracing.

Objective:
Given an Agentic Component and an existing assignment code reference, trace from that assignment to identify the single most accurate functional definition of the component.

Core rule:
Do not return a code_reference unless you can point to a concrete file location with a valid line number and supporting snippet.
If you cannot ground the answer in exact code evidence, return context_request instead.

Definitions:
- Agentic Component: a runtime unit with a distinct operational role (Agent, Tool, LLM, MCP server, DB adapter, orchestrator, router, memory manager, or similar).
- Code Object: any function, class, variable, module, config-backed symbol, or alias. Code objects are evidence only.
- Functional definition: the location where the component's runtime logic is implemented or concretely bound.
- Assignment code reference: the already-resolved assignment site for this component; this is the primary tracing anchor.

Inputs:
- node_var
- node
- guidance_summary
- assignment_code_reference
- retrieved_context (optional)

Provided assignment_code_reference:
{assignment_ref_json}

Mandatory reasoning procedure:
1. Start from assignment_code_reference.
2. Identify the assigned expression at that location:
   - constructor call
   - function call
   - factory/builder call
   - imported alias
   - attribute access
   - wrapper object
3. Resolve the runtime target step by step:
   - follow imports
   - follow local aliases
   - follow wrapper/factory calls
   - follow attribute bindings
4. Prefer the underlying runtime implementation over registration, wiring, or graph assembly code.
5. Stop only when you have the strongest evidence-backed functional definition.
6. If exact grounding is missing at any step, return context_request instead of guessing.

Selection policy:
- Return exactly one code reference: the best functional definition.
- Prefer, in order:
  1) the concrete implementation that contains runtime logic
  2) the directly invoked function/class definition
  3) the local wrapper/adapter around external behavior
- Do not prefer:
  - the assignment site itself
  - imports
  - registrations
  - dependency injection containers
  - graph wiring
  - config-only declarations
  - tests
  - docs

Special cases:
- If the assignment instantiates an external library component:
  - return the local wrapper/adapter if it exists and defines the relevant runtime behavior
  - otherwise return the all-null code_reference object
- If the assignment points to a factory/builder but you cannot resolve what it returns:
  - return context_request
- If multiple definitions exist:
  - choose the one most directly responsible for the runtime behavior of this node
- If no repository-local functional definition exists:
  - return the all-null code_reference object

Strict evidence rules:
1. Do NOT invent files, symbols, line numbers, snippets, components, or behavior.
2. Use only evidence-backed reasoning.
3. guidance_summary is a hint only and cannot override concrete code evidence.
4. kind must always be "definition".
5. other_kind_description must always be null.
6. snippet must be a short exact or near-exact fragment from the cited location.
7. line must be either:
   - an integer, or
   - [line_start, line_end]

Critical validity constraints:
- A code_reference is valid only if all of the following are true:
  - file is non-null
  - line is non-null
  - snippet is non-null
- The following are INVALID and must never be returned:
  - file != null with line == null
  - file != null with snippet == null
  - line != null with file == null
  - snippet != null with file == null
- If you cannot provide exact file+line+snippet grounding, do NOT return a partial code_reference.
- Either:
  A) return a fully grounded code_reference, or
  B) return the all-null code_reference for confirmed no-local-definition cases, or
  C) return context_request.

Decision boundary:
- Return a fully grounded code_reference only when exact evidence exists.
- Return the all-null code_reference only when evidence shows there is no repository-local functional definition for this component.
- Return context_request when evidence is insufficient to distinguish between those two cases.

Output rules:
1. Return JSON only.
2. Use exactly one of these shapes:
   - context_request object only
   - final output object only
3. Final output must contain exactly one top-level key:
   - code_reference
4. code_reference must contain exactly these keys:
   - kind
   - file
   - line
   - snippet
   - other_kind_description

Fully grounded final output:
{{
  "code_reference": {{
    "kind": "definition",
    "file": "<path>",
    "line": 123,
    "snippet": "<short exact or near-exact code snippet>",
    "other_kind_description": null
  }}
}}

Confirmed no-local-definition output:
{{
  "code_reference": {{
    "kind": "definition",
    "file": null,
    "line": null,
    "snippet": null,
    "other_kind_description": null
  }}
}}

<<CONTEXT_REQUEST_FORMAT_BUILD_DEFINITION>>

<<RETRIEVAL_INSTRUCTIONS>>

Mandatory self-check before final answer:
1. If code_reference.file is not null, then code_reference.line and code_reference.snippet must also be not null.
2. If exact line grounding is missing, output context_request instead.
3. Never output a guessed line number.
4. Never output a placeholder null line for a non-null file.

Quality bar:
A correct answer must never fabricate a location. It must return exactly one of:
- an exact grounded code_reference,
- a confirmed all-null code_reference for no-local-definition cases, or
- a context_request when exact grounding is not yet possible.
""".replace("<<CONTEXT_REQUEST_FORMAT_BUILD_DEFINITION>>", context_request_format_template.replace("<<REQUEST_ID>>", "post_processing_definition_ref_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


def build_system_prompt_prompt(assignment_code_reference: Dict[str, Any]) -> str:
    """
    Build a prompt for finding the canonical system prompt definition associated
    with an agentic component, using a resolved assignment code reference as the
    primary tracing anchor.

    Parameters
    ----------
    assignment_code_reference : dict
        Expected shape:
        {
            "kind": "assignment",
            "file": "path/to/file.py",
            "line": 123,               # or [123, 126]
            "snippet": "agent = MyAgent(...)",
            "other_kind_description": None
        }

    Returns
    -------
    str
        A ready-to-use prompt string with the assignment reference embedded as JSON.

    Raises
    ------
    TypeError
        If assignment_code_reference is not a dict.
    ValueError
        If required keys are missing or the payload is malformed.
    """
    if not isinstance(assignment_code_reference, dict):
        raise TypeError("assignment_code_reference must be a dict")

    required_keys = {"kind", "file", "line", "snippet"}
    missing = required_keys - set(assignment_code_reference.keys())
    if missing:
        raise ValueError(
            f"assignment_code_reference is missing required keys: {sorted(missing)}"
        )

    kind = assignment_code_reference.get("kind")
    file_ = assignment_code_reference.get("file")
    line = assignment_code_reference.get("line")
    snippet = assignment_code_reference.get("snippet")

    if kind not in {"assignment", "definition"}:
        raise ValueError(
            "assignment_code_reference['kind'] must be "
            f"'assignment' or 'definition', got {kind!r}"
        )

    if not isinstance(file_, str) or not file_.strip():
        raise ValueError("assignment_code_reference['file'] must be a non-empty string")

    valid_line = isinstance(line, int) or (
        isinstance(line, list)
        and len(line) == 2
        and all(isinstance(x, int) for x in line)
    )
    if not valid_line:
        raise ValueError(
            "assignment_code_reference['line'] must be an int or a two-element list of ints"
        )

    if not isinstance(snippet, str) or not snippet.strip():
        raise ValueError("assignment_code_reference['snippet'] must be a non-empty string")

    assignment_ref_json = json.dumps(
        {
            "kind": kind,
            "file": file_,
            "line": line,
            "snippet": snippet,
            "other_kind_description": assignment_code_reference.get(
                "other_kind_description", None
            ),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    return f"""You are a code analysis expert specializing in LLM-based agent systems and prompt tracing.

Objective:
Given an Agentic Component and an existing assignment code reference, trace from that assignment to identify the single most accurate system prompt definition associated with the component.

Core rule:
Do not return a code_reference unless you can point to a concrete file location with a valid line number and supporting snippet.
If you cannot ground the answer in exact code evidence, return context_request instead.

Definitions:
- Agentic Component: a runtime unit with a distinct operational role (Agent, Tool, LLM, MCP server, DB adapter, orchestrator, router, memory manager, or similar).
- System prompt: the instruction text used to define the high-level behavior, role, policy, or operating constraints of the runtime component.
- System prompt definition: the most canonical repository-local code location where that system prompt text is defined, assembled, or bound for the represented runtime node.
- Code Object: any function, class, variable, module, config-backed symbol, template, or alias. Code objects are evidence only.
- Assignment code reference: the already-resolved assignment site for this component; this is the primary tracing anchor.

Inputs:
- node_var
- node
- guidance_summary
- assignment_code_reference
- retrieved_context (optional)

Provided assignment_code_reference:
{assignment_ref_json}

Mandatory reasoning procedure:
1. Start from assignment_code_reference.
2. Identify the assigned expression at that location:
   - constructor call
   - function call
   - factory/builder call
   - imported alias
   - attribute access
   - wrapper object
3. Resolve the component step by step to find where its system prompt comes from:
   - follow imports
   - follow local aliases
   - follow wrapper/factory calls
   - inspect constructor arguments
   - inspect keyword arguments such as system_prompt, prompt, instructions, prefix, message, system_message, template
   - inspect class attributes, constants, templates, and helper functions that build prompt text
   - inspect inherited defaults or local overrides when clearly evidenced
4. Prefer the location where the actual system prompt text is defined or assembled, not merely where it is passed through.
5. Stop only when you have the strongest evidence-backed system prompt definition.
6. If exact grounding is missing at any step, return context_request instead of guessing.

Selection policy:
- Return exactly one code reference: the best system prompt definition.
- Prefer, in order:
  1) the literal string / template / constant that defines the system prompt text
  2) the helper function or method that assembles the final system prompt
  3) the binding site where the final system prompt is passed to the runtime component, if and only if the actual definition is not available but the binding clearly contains the full text
- Do not prefer:
  - the assignment site itself, unless it contains the actual full system prompt text
  - imports
  - registrations
  - generic config wiring
  - graph wiring
  - tests
  - docs
  - unrelated user prompts or task prompts

Special cases:
- If the component uses a prompt loaded from an external package, remote source, or hidden runtime and no repository-local prompt definition exists:
  - return the all-null code_reference object
- If the component references a prompt-building helper but you cannot resolve the helper implementation:
  - return context_request
- If multiple prompt fragments exist:
  - choose the one that most directly defines the canonical system-level behavior of this node
- If a final prompt is composed from multiple local pieces:
  - return the single most canonical definition site, usually the assembly function or the primary base prompt constant
- If no repository-local system prompt definition exists:
  - return the all-null code_reference object

Strict evidence rules:
1. Do NOT invent files, symbols, line numbers, snippets, prompts, or behavior.
2. Use only evidence-backed reasoning.
3. guidance_summary is a hint only and cannot override concrete code evidence.
4. kind must always be "system_prompt".
5. other_kind_description must always be null.
6. snippet must be a short exact or near-exact fragment from the cited location.
7. line must be either:
   - an integer, or
   - [line_start, line_end]
8. The returned snippet must support that this location defines or assembles the system prompt, not merely references the component.

Critical validity constraints:
- A code_reference is valid only if all of the following are true:
  - file is non-null
  - line is non-null
  - snippet is non-null
- The following are INVALID and must never be returned:
  - file != null with line == null
  - file != null with snippet == null
  - line != null with file == null
  - snippet != null with file == null
- If you cannot provide exact file+line+snippet grounding, do NOT return a partial code_reference.
- Either:
  A) return a fully grounded code_reference, or
  B) return the all-null code_reference for confirmed no-local-system-prompt cases, or
  C) return context_request.

Decision boundary:
- Return a fully grounded code_reference only when exact evidence exists.
- Return the all-null code_reference only when evidence shows there is no repository-local system prompt definition for this component.
- Return context_request when evidence is insufficient to distinguish between those two cases.

Output rules:
1. Return JSON only.
2. Use exactly one of these shapes:
   - context_request object only
   - final output object only
3. Final output must contain exactly one top-level key:
   - code_reference
4. code_reference must contain exactly these keys:
   - kind
   - file
   - line
   - snippet
   - other_kind_description

Fully grounded final output:
{{
  "code_reference": {{
    "kind": "system_prompt",
    "file": "<path>",
    "line": 123,
    "snippet": "<short exact or near-exact code snippet>",
    "other_kind_description": null
  }}
}}

Confirmed no-local-system-prompt output:
{{
  "code_reference": {{
    "kind": "system_prompt",
    "file": null,
    "line": null,
    "snippet": null,
    "other_kind_description": null
  }}
}}

<<CONTEXT_REQUEST_FORMAT_SYSTEM_PROMPT>>

<<RETRIEVAL_INSTRUCTIONS>>

Mandatory self-check before final answer:
1. If code_reference.file is not null, then code_reference.line and code_reference.snippet must also be not null.
2. If exact line grounding is missing, output context_request instead.
3. Never output a guessed line number.
4. Never output a placeholder null line for a non-null file.
5. Ensure the returned location is about system prompt definition, not general component implementation.

Quality bar:
A correct answer must never fabricate a location. It must return exactly one of:
- an exact grounded code_reference for the canonical system prompt definition,
- a confirmed all-null code_reference for no-local-system-prompt cases, or
- a context_request when exact grounding is not yet possible.
""".replace("<<CONTEXT_REQUEST_FORMAT_SYSTEM_PROMPT>>", context_request_format_template.replace("<<REQUEST_ID>>", "post_processing_system_prompt_ref_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


def build_source_prompt(assignment_code_reference: Dict[str, Any]) -> str:
    """
    Build a prompt that traces from an assignment code reference to the import
    statement that provides the main builder symbol of the assigned object.

    "Source" here means:
    the import line from which the main builder of the object came.

    Example:
        assignment snippet: pikachu = Pokemon(pikachu, 4048)
        desired source:     from Pokemon_mastery import Pokemon

    Parameters
    ----------
    assignment_code_reference : dict
        Expected shape:
        {
            "kind": "assignment",
            "file": "path/to/file.py",
            "line": 123,               # or [123, 126]
            "snippet": "x = Foo(...)",
            "other_kind_description": None
        }

    Returns
    -------
    str
        A ready-to-use prompt string with the assignment reference embedded as JSON.
    """
    if not isinstance(assignment_code_reference, dict):
        raise TypeError("assignment_code_reference must be a dict")

    required_keys = {"kind", "file", "line", "snippet"}
    missing = required_keys - set(assignment_code_reference.keys())
    if missing:
        raise ValueError(
            f"assignment_code_reference is missing required keys: {sorted(missing)}"
        )

    assignment_ref_json = json.dumps(
        assignment_code_reference,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    return f"""You are a code analysis expert specializing in symbol tracing and import resolution in Python codebases.

Objective:
Given an Agentic Component and an existing assignment code reference, identify the import source of the component's main builder.

Core task:
Trace from the provided assignment_code_reference to find the import line that introduces the main builder symbol used to construct or obtain the assigned object.

Definition of "source":
- The source is the import statement that brings the main builder symbol into scope.
- The main builder is the primary callable or symbol used on the right-hand side of the assignment to create, construct, or obtain the component.
- In most cases this is the outermost constructor/function/factory symbol in the assignment expression.

Example:
- Assignment snippet: `pikachu = Pokemon(pikachu, 4048)`
- Source import: `from Pokemon_mastery import Pokemon`

Inputs:
- node_var
- node
- guidance_summary
- assignment_code_reference
- retrieved_context (optional)

Provided assignment_code_reference:
{assignment_ref_json}

Mandatory reasoning procedure:
1. Start from assignment_code_reference.
2. Inspect the right-hand side of the assignment.
3. Identify the main builder symbol.
   Examples:
   - `x = Foo(...)` -> main builder is `Foo`
   - `x = make_agent(...)` -> main builder is `make_agent`
   - `x = pkg.Builder(...)` -> main builder is `pkg.Builder`, and you must resolve how `pkg` entered scope
   - `x = wrapper(factory(...))` -> prefer the outermost builder actually assigned unless evidence shows the node's represented component is built by the inner callable
4. Find the import statement that brings that builder into scope.
5. Return exactly one code reference object pointing to that import line.

Important distinctions:
- You are locating the source import, not the functional definition.
- Do not return the builder definition unless it is also literally the import line required by the task.
- Do not return the assignment itself.
- Do not return a usage site.
- Do not return a module path guess without import evidence.

Selection policy:
- Prefer the exact import line in the same file as the assignment.
- If the builder is re-exported or aliased, return the import line actually used by the assignment file.
- If the symbol is accessed through a module alias, return the import line that introduced that module alias.
- If multiple imports could apply, choose the one that directly explains the builder symbol used in the assignment snippet.
- If the builder is locally defined and not imported, return nulls.
- If the source cannot be established from evidence, return context_request.

Strict rules:
1. Do NOT invent files, imports, modules, or symbols.
2. Use only evidence-backed reasoning.
3. kind must be "source".
4. snippet must be a short exact or near-exact fragment of the import line.
5. line must be integer or [start, end].
6. If no import source exists in repo evidence, return nulls.
7. guidance_summary is only a hint and cannot override concrete code evidence.

Output rules:
- Return JSON only.
- Use exactly one of:
  (A) context_request
  (B) final output

Final output:
{{
  "code_reference": {{
    "kind": "source",
    "file": "<path or null>",
    "line": 123,
    "snippet": "<short import snippet or null>",
    "other_kind_description": null
  }}
}}

Null case:
{{
  "code_reference": {{
    "kind": "source",
    "file": null,
    "line": null,
    "snippet": null,
    "other_kind_description": null
  }}
}}

<<CONTEXT_REQUEST_FORMAT_BUILD_SOURCE>>

<<RETRIEVAL_INSTRUCTIONS>>

Quality bar:
A correct answer identifies the exact import line that introduced the main builder symbol used by the assignment.
""".replace("<<CONTEXT_REQUEST_FORMAT_BUILD_SOURCE>>", context_request_format_template.replace("<<REQUEST_ID>>", "post_processing_source_ref_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


def build_children_guidance(child_component_names):
    if not child_component_names:
        return ASSIGNMENT_REFERENCE_PROMPT
    children_json = json.dumps(child_component_names, ensure_ascii=False)
    return f"""You are a code analysis expert specializing in LLM-based agent systems and assignment-site detection for runtime components.

Objective:
Find the single best assignment statement for the given Agentic Component and return exactly one code reference object for that assignment.

Primary task:
Locate the canonical binding site where the runtime component represented by the node is assigned to a variable, alias, or named handle in code.

Definitions:
- Agentic Component: a runtime unit with its own operational role in the agentic system (for example: System, Agent, LLM instance, Tool, MCP server, database adapter, orchestrator, router, memory manager, or major custom runtime component).
- Child component: a sub-component that is part of, injected into, composed into, configured into, or otherwise used to construct the parent component represented by the node. Example: an `llm` can be a child of an `llm_based_agent`.
- Code Object: a function, class, variable, module, alias, or other code-level symbol. Code objects are evidence only. The task is to identify the assignment site of the Agentic Component.
- Assignment statement: the code location where the represented component is bound to a symbol, usually in forms such as:
  - <node_var> = ...
  - <node_var>: Type = ...
  - <node_var> := ...
  - with ... as <node_var>
  - except ... as <node_var>
- Assignment reference: the concrete code statement that binds the runtime component to the symbol used to represent that component.

Inputs:
- node_var
- node
- child_component_names
- guidance_summary
- retrieved_context (optional)

Core distinction (MANDATORY):
You must return the assignment/binding site of the component, NOT a later usage site.
A function, method, or node implementation that merely calls, invokes, reads, forwards, or wraps the component is NOT the assignment unless that same code fragment also performs the binding.

Child-component-guided reasoning (MANDATORY):
Use child_component_names as a structural signal to identify the most suitable parent-component assignment.

Provided child_component_names:
{children_json}""" +"""

How to use child_component_names:
1. Prefer candidate assignments whose right-hand side (RHS) directly contains, passes, injects, configures, or composes one or more child component names.
2. Treat presence of child component names in constructor/factory arguments as strong positive evidence that the assignment is constructing the parent component.
3. If multiple assignment candidates exist for the same node_var, prefer the one whose RHS best explains how the parent is assembled from its children.
4. If a candidate is only a later usage site of the parent and does not compose or bind the child components, it is weaker evidence and should not be selected over a true assignment site.
5. Child names are supporting evidence only:
   - they help rank assignment candidates
   - they do NOT justify inventing an assignment
   - they do NOT override concrete code evidence

Examples of child evidence:
- `agent_runnable = create_react_agent(llm, tools=[scrape_webpages])`
  If child_component_names include `llm` and/or `scrape_webpages`, this is strong evidence that this assignment is the parent-agent construction site.
- `result = agent_runnable.invoke(state)`
  This is a usage/invocation site, not the assignment site for the parent component, even if it appears in the node implementation.

Selection policy:
1. Return exactly one code reference object.
2. kind must be "assignment".
3. Prefer the most canonical binding site for the runtime component represented by the node.
4. Prefer direct assignment/binding of the represented component over code that merely uses that component later.
5. Use child_component_names to rank candidates:
   - strongest: assignment whose RHS composes the parent from its children
   - next: assignment whose RHS clearly constructs/configures the parent even without explicit child names
   - weaker: alias-only rebinding
   - invalid: later invocation/usage site
6. If multiple candidate assignments exist, choose the one that most directly binds the actual runtime component used by the node.
7. Prefer:
   a) direct variable assignment of the component
   b) annotated assignment of the component
   c) aliasing/binding that clearly establishes the runtime handle
   d) context-manager binding only if that is the actual runtime binding form
8. Do NOT return:
   - function definitions that use the component
   - method bodies that call `.invoke()`, `.run()`, `.call()`, `.execute()`, `.ainvoke()`, etc.
   - return statements
   - registration sites unless they also contain the actual binding
   - import statements unless the imported alias is the only binding actually representing the node
   - graph edges, routing logic, or node wrappers that merely reference the component
   - a larger enclosing block when the exact assignment statement is available

Snippet requirements:
1. The snippet must be a short exact or near-exact code fragment from the cited location.
2. The snippet must be centered on the assignment statement itself.
3. If the assignment is a single line, return only that line.
4. If the assignment spans multiple lines, return only the minimal contiguous assignment block required to capture the full assignment.
5. Do NOT include unrelated enclosing function definitions, surrounding calls, or downstream usage lines.
6. The first line of the snippet must begin with the assignment/binding statement itself, such as:
   - <node_var> = ...
   - <node_var>: ... = ...
   - with ... as <node_var>
   - except ... as <node_var>
7. If the candidate snippet begins with `def`, `class`, `return`, or another non-assignment header while the actual assignment exists inside or outside that block, that candidate is invalid.

Decision rules:
1. Do not invent files, snippets, symbols, components, or behavior.
2. Search for evidence until the output is evidence-backed; if evidence is missing, return context_request instead of guessing.
3. Return exactly one code reference object describing the component assignment statement location.
4. guidance_summary is high-level direction only and cannot override concrete code evidence.
5. If the node_var appears only in usage sites but no binding site is found, return context_request.
6. If the component is represented through aliasing, factories, or wrappers, still return the binding statement for the symbol representing the node, not the later call site.
7. If multiple mentions of node_var exist, distinguish:
   - binding/definition site
   - use/reference site
   Always choose the binding/definition site.
8. If child_component_names are available, use them to prefer the assignment site that most plausibly assembles the parent component from those children.

Validity rules:
1. A valid assignment code_reference must identify a concrete file location with:
   - non-null file
   - non-null line
   - non-null snippet
2. line must be either:
   - an integer, or
   - [line_start, line_end]
3. snippet must be short and assignment-centered.
4. Never return a function body or invocation block when the actual assignment statement is available.
5. Never return a snippet whose primary purpose is using the component rather than binding it.

Output rules:
1. Use one of two response shapes only:
   - context_request object only, OR
   - final output JSON object
2. Return JSON only.
3. Final output must contain exactly one top-level key: code_reference.
4. code_reference must contain exactly these keys:
   - kind
   - file
   - line
   - snippet
   - other_kind_description
5. other_kind_description must always be null.

Final output JSON:
{
  "code_reference": {
    "kind": "assignment",
    "file": "<path>",
    "line": 123,
    "snippet": "<short snippet>",
    "other_kind_description": null
  }
}

<<CONTEXT_REQUEST_FORMAT_CHILD_GUIDANCE>>

<<RETRIEVAL_INSTRUCTIONS>>

Mandatory self-check before final answer:
1. Is the returned snippet the binding site of the component, rather than a usage site?
2. Does the snippet begin with the assignment/binding statement itself?
3. If a smaller exact assignment statement exists, have you avoided returning a larger enclosing block?
4. Does the chosen assignment RHS contain or structurally compose any child component names? If yes, that is strong positive evidence.
5. If the snippet begins with `def` or contains `.invoke(...)` as the main evidence, it is probably a usage-site false positive and should not be returned.
6. If two assignment candidates compete, prefer the one that best assembles the parent from child_component_names.

Quality bar:
A correct answer returns the single strongest evidence-backed assignment statement that binds the runtime component represented by the node, and uses child_component_names to prefer the parent-construction site over later usage or invocation sites.
""".replace("<<CONTEXT_REQUEST_FORMAT_CHILD_GUIDANCE>>", context_request_format_template.replace("<<REQUEST_ID>>", "post_processing_assignment_ref_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


####################        UNUSED PROMPT TEMPLATES        ####################
# def build_input_schema_prompt(assignment_code_reference: Dict[str, Any]) -> str:
#     """
#     Build a prompt that traces from an assignment code reference to the input
#     schema of the component's main builder.

#     "input_schema" here means:
#     the code location that defines the expected input structure for the main
#     builder used in the assignment. This can be, for example:
#     - a Pydantic model
#     - a TypedDict
#     - a dataclass used as input
#     - a function parameter annotation
#     - an args_schema / input_model binding
#     - an explicit schema dict / JSON schema
#     - a tool input schema declaration

#     Parameters
#     ----------
#     assignment_code_reference : dict
#         Expected shape:
#         {
#             "kind": "assignment",
#             "file": "path/to/file.py",
#             "line": 123,               # or [123, 126]
#             "snippet": "x = Foo(...)",
#             "other_kind_description": None
#         }

#     Returns
#     -------
#     str
#         A ready-to-use prompt string with the assignment reference embedded as JSON.
#     """
#     if not isinstance(assignment_code_reference, dict):
#         raise TypeError("assignment_code_reference must be a dict")

#     required_keys = {"kind", "file", "line", "snippet"}
#     missing = required_keys - set(assignment_code_reference.keys())
#     if missing:
#         raise ValueError(
#             f"assignment_code_reference is missing required keys: {sorted(missing)}"
#         )

#     assignment_ref_json = json.dumps(
#         assignment_code_reference,
#         ensure_ascii=False,
#         indent=2,
#         sort_keys=True,
#     )

#     return f"""You are a code analysis expert specializing in Python symbol tracing and runtime interface discovery.

# Objective:
# Given an assignment_code_reference, locate the input schema of the component's main builder.

# Definition of "input_schema":
# The input schema is the code location that defines the expected structure, fields, types, or validation rules for inputs consumed by the main builder or the runtime component it constructs.

# Possible forms of input_schema:
# - Pydantic model
# - TypedDict
# - dataclass
# - explicit schema dict / JSON schema
# - function signature with meaningful typed parameters
# - args_schema / input_model / request_model binding
# - tool input schema declaration
# - parser schema object
# - validator-backed input class

# Inputs:
# - node_var
# - node
# - guidance_summary
# - assignment_code_reference
# - retrieved_context (optional)

# Provided assignment_code_reference:
# {assignment_ref_json}

# Mandatory reasoning procedure:
# 1. Start from assignment_code_reference.
# 2. Inspect the right-hand side of the assignment.
# 3. Identify the main builder symbol.
#    Examples:
#    - `x = Foo(...)` -> main builder is `Foo`
#    - `x = make_agent(...)` -> main builder is `make_agent`
#    - `x = pkg.Builder(...)` -> main builder is `pkg.Builder`
# 4. Resolve the builder to its implementation or local binding as needed.
# 5. Locate the code object that defines the input shape expected by that builder or component.
# 6. Return exactly one code reference object pointing to the best input schema location.

# Selection policy:
# - Return exactly one code reference: the single best input schema location.
# - Prefer, in order:
#   1) explicit schema classes or objects bound to the builder/component
#   2) args_schema / input_model / schema fields or constructor bindings
#   3) dedicated request/input model classes
#   4) typed function signature of the canonical builder
#   5) nulls if no meaningful input schema exists in repository evidence
# - Avoid:
#   - the assignment itself
#   - imports, unless the import line is the only repository-local evidence of the schema binding
#   - unrelated config
#   - output schemas
#   - tests
#   - docs

# Important distinctions:
# - You are locating the input schema, not the functional definition.
# - You are locating the schema for the main builder/component represented by the assignment, not just any nearby model.
# - Prefer the schema actually used at runtime.
# - If multiple candidate schemas exist, choose the one most directly governing the runtime inputs of the represented component.

# Special cases:
# - If the builder uses an external package but a local wrapper binds an args_schema/input_model, return the local binding or local schema definition.
# - If the builder accepts only primitive parameters and has no dedicated schema object, you may return the canonical function/class definition line only if its typed signature is the strongest available input-schema evidence.
# - If no meaningful input schema exists in the repo, return nulls.
# - If evidence is insufficient to identify the schema confidently, return context_request.

# Strict rules:
# 1. Do NOT invent files, symbols, schemas, or behavior.
# 2. Use only evidence-backed reasoning.
# 3. kind must be "input_schema".
# 4. snippet must be a short exact or near-exact fragment from the cited location.
# 5. line must be integer or [start, end].
# 6. guidance_summary is only a hint and cannot override concrete code evidence.

# Output rules:
# - Return JSON only.
# - Use exactly one of:
#   (A) context_request
#   (B) final output

# Final output:
# {{
#   "code_reference": {{
#     "kind": "input_schema",
#     "file": "<path or null>",
#     "line": 123,
#     "snippet": "<short schema snippet or null>",
#     "other_kind_description": null
#   }}
# }}

# Null case:
# {{
#   "code_reference": {{
#     "kind": "input_schema",
#     "file": null,
#     "line": null,
#     "snippet": null,
#     "other_kind_description": null
#   }}
# }}

# <<CONTEXT_REQUEST_FORMAT_INPUT_SCHEMA>>

# <<RETRIEVAL_INSTRUCTIONS>>

# Quality bar:
# A correct answer identifies the single strongest evidence-backed code location that defines the runtime input shape of the component built by the assignment.
# """.replace("<<CONTEXT_REQUEST_FORMAT_INPUT_SCHEMA>>", context_request_format_template.replace("<<REQUEST_ID>>", "post_processing_input_schema_ref_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)


# def build_output_schema_prompt(assignment_code_reference: Dict[str, Any]) -> str:
#     """
#     Build a prompt that traces from an assignment code reference to the output
#     schema of the component's main builder.

#     "output_schema" here means:
#     the code location that defines the expected returned/output structure of the
#     main builder or of the runtime component it constructs.

#     This can be, for example:
#     - a Pydantic response model
#     - a TypedDict
#     - a dataclass used as output
#     - a return type annotation
#     - an output_model / response_model binding
#     - an explicit schema dict / JSON schema
#     - a structured output schema declaration
#     - a parser or formatter schema object

#     Parameters
#     ----------
#     assignment_code_reference : dict
#         Expected shape:
#         {
#             "kind": "assignment",
#             "file": "path/to/file.py",
#             "line": 123,               # or [123, 126]
#             "snippet": "x = Foo(...)",
#             "other_kind_description": None
#         }

#     Returns
#     -------
#     str
#         A ready-to-use prompt string with the assignment reference embedded as JSON.
#     """
#     if not isinstance(assignment_code_reference, dict):
#         raise TypeError("assignment_code_reference must be a dict")

#     required_keys = {"kind", "file", "line", "snippet"}
#     missing = required_keys - set(assignment_code_reference.keys())
#     if missing:
#         raise ValueError(
#             f"assignment_code_reference is missing required keys: {sorted(missing)}"
#         )

#     assignment_ref_json = json.dumps(
#         assignment_code_reference,
#         ensure_ascii=False,
#         indent=2,
#         sort_keys=True,
#     )

#     return f"""You are a code analysis expert specializing in Python symbol tracing and runtime interface discovery.

# Objective:
# Given an assignment_code_reference, locate the output schema of the component's main builder.

# Definition of "output_schema":
# The output schema is the code location that defines the expected returned structure, fields, types, or validation rules for the main builder or for the runtime component it constructs.

# Possible forms of output_schema:
# - Pydantic response model
# - TypedDict
# - dataclass
# - explicit schema dict / JSON schema
# - return type annotation
# - output_model / response_model / result_model binding
# - structured output schema declaration
# - parser / formatter / serializer schema object
# - validator-backed output class

# Inputs:
# - node_var
# - node
# - guidance_summary
# - assignment_code_reference
# - retrieved_context (optional)

# Provided assignment_code_reference:
# {assignment_ref_json}

# Mandatory reasoning procedure:
# 1. Start from assignment_code_reference.
# 2. Inspect the right-hand side of the assignment.
# 3. Identify the main builder symbol.
#    Examples:
#    - `x = Foo(...)` -> main builder is `Foo`
#    - `x = make_agent(...)` -> main builder is `make_agent`
#    - `x = pkg.Builder(...)` -> main builder is `pkg.Builder`
# 4. Resolve the builder to its implementation or local binding as needed.
# 5. Locate the code object that defines the output shape produced by that builder or by the runtime component it constructs.
# 6. Return exactly one code reference object pointing to the best output schema location.

# Selection policy:
# - Return exactly one code reference: the single best output schema location.
# - Prefer, in order:
#   1) explicit output schema classes or objects bound to the builder/component
#   2) output_model / response_model / result_model / structured-output bindings
#   3) dedicated response/result model classes
#   4) typed return annotation of the canonical builder or callable component
#   5) nulls if no meaningful output schema exists in repository evidence
# - Avoid:
#   - the assignment itself
#   - imports, unless the import line is the only repository-local evidence of the schema binding
#   - unrelated config
#   - input schemas
#   - tests
#   - docs

# Important distinctions:
# - You are locating the output schema, not the functional definition.
# - You are locating the schema for the main builder/component represented by the assignment, not just any nearby model.
# - Prefer the schema actually governing runtime outputs.
# - If multiple candidate schemas exist, choose the one most directly governing the runtime outputs of the represented component.

# Special cases:
# - If the builder uses an external package but a local wrapper binds an output_model/response_model/structured output parser, return the local binding or local schema definition.
# - If the builder has no dedicated output schema object, you may return the canonical function/class definition line only if its typed return signature is the strongest available output-schema evidence.
# - If the component returns primitive or unstructured values and no meaningful output schema exists, return nulls.
# - If evidence is insufficient to identify the schema confidently, return context_request.

# Strict rules:
# 1. Do NOT invent files, symbols, schemas, or behavior.
# 2. Use only evidence-backed reasoning.
# 3. kind must be "output_schema".
# 4. snippet must be a short exact or near-exact fragment from the cited location.
# 5. line must be integer or [start, end].
# 6. guidance_summary is only a hint and cannot override concrete code evidence.

# Output rules:
# - Return JSON only.
# - Use exactly one of:
#   (A) context_request
#   (B) final output

# Final output:
# {{
#   "code_reference": {{
#     "kind": "output_schema",
#     "file": "<path or null>",
#     "line": 123,
#     "snippet": "<short schema snippet or null>",
#     "other_kind_description": null
#   }}
# }}

# Null case:
# {{
#   "code_reference": {{
#     "kind": "output_schema",
#     "file": null,
#     "line": null,
#     "snippet": null,
#     "other_kind_description": null
#   }}
# }}

# <<CONTEXT_REQUEST_FORMAT_OUTPUT_SCHEMA>>

# <<RETRIEVAL_INSTRUCTIONS>>

# Quality bar:
# A correct answer identifies the single strongest evidence-backed code location that defines the runtime output shape of the component built by the assignment.
# """.replace("<<CONTEXT_REQUEST_FORMAT_OUTPUT_SCHEMA>>", context_request_format_template.replace("<<REQUEST_ID>>", "post_processing_output_schema_ref_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)

# def build_usage_prompt(
#     assignment_code_reference: Dict[str, Any],
#     found_usage_code_references: List[Dict[str, Any]],
# ) -> str:
#     """
#     Build a prompt that traces from an assignment code reference and an existing
#     list of found usage references to identify one NEW usage location of the
#     assigned component.

#     "usage" here means:
#     any code location where the assigned object is used after its assignment.

#     Because usage can have multiple valid locations, this prompt is designed for
#     iterative discovery:
#     - each run returns exactly one NEW usage code reference, or
#     - returns a null code_reference when no additional usage is found.

#     Parameters
#     ----------
#     assignment_code_reference : dict
#         Expected shape:
#         {
#             "kind": "assignment",
#             "file": "path/to/file.py",
#             "line": 123,               # or [123, 126]
#             "snippet": "x = Foo(...)",
#             "other_kind_description": None
#         }

#     found_usage_code_references : list[dict]
#         Existing usage references already found in prior iterations.
#         These should be excluded from the next result.

#     Returns
#     -------
#     str
#         A ready-to-use prompt string with the assignment reference and prior
#         usage references embedded as JSON.
#     """
#     if not isinstance(assignment_code_reference, dict):
#         raise TypeError("assignment_code_reference must be a dict")

#     if not isinstance(found_usage_code_references, list):
#         raise TypeError("found_usage_code_references must be a list")

#     required_keys = {"kind", "file", "line", "snippet", "other_kind_description"}
#     missing = required_keys - set(assignment_code_reference.keys())
#     if missing:
#         raise ValueError(
#             f"assignment_code_reference is missing required keys: {sorted(missing)}"
#         )

#     for i, ref in enumerate(found_usage_code_references):
#         if not isinstance(ref, dict):
#             raise TypeError(
#                 f"found_usage_code_references[{i}] must be a dict, got {type(ref).__name__}"
#             )
#         ref_missing = required_keys - set(ref.keys())
#         if ref_missing:
#             raise ValueError(
#                 f"found_usage_code_references[{i}] is missing required keys: {sorted(ref_missing)}"
#             )

#     assignment_ref_json = json.dumps(
#         assignment_code_reference,
#         ensure_ascii=False,
#         indent=2,
#         sort_keys=True,
#     )
#     found_refs_json = json.dumps(
#         found_usage_code_references,
#         ensure_ascii=False,
#         indent=2,
#         sort_keys=True,
#     )

#     return f"""You are a code analysis expert specializing in Python symbol tracing, data-flow tracking, and object usage discovery.

# Objective:
# Given an assignment_code_reference and a list of already found usage code references, find exactly one NEW usage location of the assigned object.

# Definition of "usage":
# A usage is any code location where the assigned object is referenced, invoked, passed, accessed, returned, stored, mutated, or otherwise participates in runtime behavior after its assignment.

# Examples of valid usage:
# - method call on the object
# - attribute access
# - passing the object as an argument
# - returning the object
# - storing it in a container
# - reassigning it to another variable
# - conditional checks involving it
# - serialization / logging / registration involving it
# - invocation if the object is callable

# Examples:
# - assignment: `pikachu = Pokemon(pikachu, 4048)`
# - valid usage:
#   - `battle(pikachu)`
#   - `pikachu.attack()`
#   - `team.append(pikachu)`
#   - `return pikachu`

# Inputs:
# - node_var
# - node
# - guidance_summary
# - assignment_code_reference
# - found_usage_code_references
# - retrieved_context (optional)

# Provided assignment_code_reference:
# {assignment_ref_json}

# Already found usage code references:
# {found_refs_json}

# Mandatory reasoning procedure:
# 1. Start from assignment_code_reference.
# 2. Identify the assigned variable/object symbol.
# 3. Search for places where that object is used after assignment.
# 4. Exclude all usages already present in found_usage_code_references.
# 5. Return exactly one NEW usage code reference if one exists.
# 6. If no additional usage exists, return a null code_reference object.

# Selection policy:
# - Return exactly one code reference on each run.
# - It must be a NEW usage not already listed in found_usage_code_references.
# - Prefer, in order:
#   1) direct runtime usages of the exact assigned symbol
#   2) semantically meaningful usages (method call, argument passing, return, container insertion)
#   3) nearer/local usages before distant/indirect ones when equally strong
# - Avoid:
#   - the original assignment itself
#   - imports
#   - definitions
#   - comments
#   - docs
#   - tests, unless test code is the only usage evidence available and repository policy permits it

# Important distinctions:
# - You are locating a usage, not the functional definition.
# - You are locating a usage of the assigned object, not merely a usage of the builder symbol.
# - Prefer exact symbol usage over speculative aliasing.
# - Only follow aliases if the aliasing step is explicit in code evidence.
# - Do not return a usage already included in found_usage_code_references.

# Deduplication rules:
# - A candidate usage is NOT new if it matches an existing found reference by the same file and overlapping line and substantially the same snippet.
# - Do not return semantically duplicate references to the same usage site.
# - Return the next best distinct usage site.

# Special cases:
# - If the object is reassigned to another name, that reassignment can count as a usage.
# - If the object is passed through several aliases, only count later alias usages when the alias relationship is explicit in code evidence.
# - If the object is created but never used, return nulls.
# - If evidence is insufficient to identify a new usage confidently, return context_request.

# Strict rules:
# 1. Do NOT invent files, symbols, aliases, or behavior.
# 2. Use only evidence-backed reasoning.
# 3. kind must be "usage".
# 4. snippet must be a short exact or near-exact fragment from the cited location.
# 5. line must be integer or [start, end].
# 6. guidance_summary is only a hint and cannot override concrete code evidence.
# 7. Return only one new usage per run.

# Output rules:
# - Return JSON only.
# - Use exactly one of:
#   (A) context_request
#   (B) final output

# Final output:
# {{
#   "code_reference": {{
#     "kind": "usage",
#     "file": "<path or null>",
#     "line": 123,
#     "snippet": "<short usage snippet or null>",
#     "other_kind_description": null
#   }}
# }}

# Null case (when no more usages exist):
# {{
#   "code_reference": {{
#     "kind": "usage",
#     "file": null,
#     "line": null,
#     "snippet": null,
#     "other_kind_description": null
#   }}
# }}

# <<CONTEXT_REQUEST_FORMAT_USAGE>>

# <<RETRIEVAL_INSTRUCTIONS>>

# Quality bar:
# A correct answer identifies exactly one new, evidence-backed usage site of the assigned object that is not already present in found_usage_code_references, or returns nulls when no more usage sites exist.
# """.replace("<<CONTEXT_REQUEST_FORMAT_USAGE>>", context_request_format_template.replace("<<REQUEST_ID>>", "post_processing_usage_ref_hop1")).replace("<<RETRIEVAL_INSTRUCTIONS>>", retrieval_instructions)
