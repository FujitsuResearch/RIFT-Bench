# Scanning runner

This directory contains the standalone scanning runner: [run_scanning.py](run_scanning.py).

The runner receives a NodeSpec plus a code ZIP, builds an isolated Docker image, runs the scanning pipeline, saves results, and removes the temporary Docker image tags when it finishes.

Scanning is file-based: it does not require PostgreSQL or the removed scanning/databases package.

## Requirements

- Run commands from the repository root.
- Use the repository virtual environment.
- Docker must be available.
- rift.env must contain the required runtime configuration.
- The target system environment file and NodeSpec must exist, unless overridden.

## Run scanning

Use the NodeSpec and post-emulation ZIP produced by the tool-emulation runner:

```bash
.venv/bin/python scanning/run_scanning.py \
  --user-id user \
  --system-name domain_systems/finance/autogen_router \
  --node-spec-path agentic_systems/domain_systems/finance/autogen_router/autogen_router_after_discovery.json \
  --codebase-zip-path experiments_v2/domain_systems__finance__autogen_router_post_emulation.zip
```

The runner also accepts an original source ZIP instead of a post-emulation ZIP.

## CLI options

| Option | Default |
| --- | --- |
| --user-id | Required identifier used for sandbox image names. |
| --system-name | Required path below agentic_systems/. The domain_systems/ prefix is optional when unambiguous. |
| --node-spec-path | <system-root>/<system>_after_discovery.json. |
| --codebase-zip-path | Required original or post-emulation source ZIP. |
| --env-file-path | <system-root>/<system>.env, then the repository-level equivalent. |
| --scan-results-dir | experiments_v2/<system-key>_scan_results. |
| --scan-timeout-sec | 600 seconds. |
| --flows-per-probe | 3. |
| --apply-defenses | Enables scanning defenses. |

Both hyphenated and underscore forms are accepted for user-id, system-name, scan-timeout-sec, and flows-per-probe.

## Output

Results are written to the scan-results directory. The runner removes the scan image and the environment image even if scanning fails.
