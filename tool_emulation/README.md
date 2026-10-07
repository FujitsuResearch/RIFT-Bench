# Tool emulation

This package creates a sandboxed twin of an agentic system, emulates its tools, and exports the artifacts needed by scanning.

The entry point is [`run_tool_emulation.py`](run_tool_emulation.py). Apart from this runner, this directory keeps only the original `components/ToolEmulation/` implementation and a small runner-specific wrapper (`runner_runtime.py`). None of them imports `discovery`.

## What the runner does

1. Resolves the target system below `agentic_systems/`.
2. Loads a source NodeSpec and adds its runtime-mapping metadata.
3. Builds a temporary Docker sandbox image from an original source ZIP.
4. Emulates tools inside the sandbox.
5. Saves the updated NodeSpec and exports `/sandbox` as a ZIP.
6. Deletes both Docker image tags created for emulation.

The runner returns `(node_spec_path, post_emulation_zip_path)` when called as Python code, and prints both paths when used from the CLI.

## Requirements

- Run commands from the repository root.
- Use the repository virtual environment (`.venv`).
- Docker must be available to build and run the sandbox.
- `rift.env` must contain the emulation model configuration.
- The system's environment file, runtime mapping, and NodeSpec source must be present, unless explicitly overridden.

## Quick start

```bash
.venv/bin/python tool_emulation/run_tool_emulation.py \
  --user-id user \
  --system-name domain_systems/finance/autogen_router
```

`--system-name` is relative to `agentic_systems/`; the `domain_systems/` prefix is optional when the system is unique.

## Inputs and defaults

| CLI option | Default |
| --- | --- |
| `--user-id` | Required identifier used in sandbox image names. |
| `--system-name` | Required path below `agentic_systems/`. |
| `--node-spec-path` | `final_spec_gt_spec.json` for domain systems, otherwise `<system>_spec.json`. |
| `--runtime-mapping-path` | `<repository>/runtime_mapping.json`. |
| `--node-spec-py-path` | `<repository>/<system>_spec.py`, falling back to `final_spec_gt_spec.py`. |
| `--env-file-path` | `<system-root>/<system>.env`, then the repository-level equivalent. |
| `--codebase-zip-path` | A temporary ZIP made from the system repository. |
| `--output-node-spec-path` | `<system-root>/<system>_after_discovery.json`. |
| `--output-codebase-zip-path` | `experiments_v2/<system-key>_post_emulation.zip`. |

Use an existing original-source ZIP when needed:

```bash
.venv/bin/python tool_emulation/run_tool_emulation.py \
  --user-id user \
  --system-name domain_systems/finance/autogen_router \
  --codebase-zip-path /path/to/original-code.zip \
  --output-node-spec-path /path/to/emulated_nodespec.json \
  --output-codebase-zip-path /path/to/post_emulation_code.zip
```

The output code ZIP path must end in `.zip`.

Both hyphenated and underscore forms are accepted for `--user-id` and `--system-name`.

## Outputs and scanning handoff

The output ZIP contains the sandbox filesystem rooted at `sandbox/`. It includes the code changes made during tool emulation and is directly accepted by the scanning runner.

```bash
.venv/bin/python scanning/run_scanning.py \
  --user-id user \
  --system-name domain_systems/finance/autogen_router \
  --node-spec-path agentic_systems/domain_systems/finance/autogen_router/autogen_router_after_discovery.json \
  --codebase-zip-path experiments_v2/domain_systems__finance__autogen_router_post_emulation.zip
```

`scanning/run_scanning.py` also accepts an original code ZIP instead of the post-emulation ZIP. It builds a temporary scan image from whichever ZIP is supplied and removes that image when scanning finishes.

## Docker image lifecycle

The emulation runner creates a `:v1` image during environment preparation and a `:latest` image when it commits the emulated sandbox. It removes both tags in cleanup, including when emulation fails after image creation. The persistent artifacts are the updated NodeSpec and the post-emulation ZIP.
