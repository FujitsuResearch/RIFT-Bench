# Use Case 4: Hierarchical Research App

This app wires together LangGraph-based teams (search, scraping, writing) and runs them for a single query or a batch of queries with optional local MLflow logging.

**DATE OF CREATION:** 03/12/2025

## Setup
- Python 3.10+ recommended.
- Install deps (ideally in a venv): `pip install -r requirements.txt` (includes MLflow for logging)
- Provide the required environment variables (via `.env` or shell):
  - `AZURE_API_KEY`
  - `AZURE_API_BASE`
  - `AZURE_API_VERSION`
  - `model` (Azure deployment name)
  - `TAVILY_API_KEY`
  - Optional: set `USER_AGENT` to a sensible identifier for web requests (a default is set if missing).
- Dependencies include `beautifulsoup4` for web scraping via `WebBaseLoader`.

## Running
- Single query:  
  `python3 run_hierarchical_research.py --query "Research AI agents and write a brief report about them."`
- Batch via JSON (example provided):  
  `python3 run_hierarchical_research.py --queries-file queries_example.json`
- Outputs:
  - Single run prints the answer string.
  - Batch run prints a JSON list of answers (same order as input).
  - The agents run non-interactively; they auto-use `outline.txt`/`report.txt` and the final output is the report text only (no preamble/meta/offers).

## MLflow logging (local, optional)
- Enabled automatically (MLflow is in requirements).
- Defaults: experiment `hierarchical_research`, tracking URI `MLFLOW_TRACKING_URI` env var or `file://<repo>/Research_Orchestrator/mlruns`.
- Override as needed:
  - `--mlflow-experiment <name>`
  - `--mlflow-tracking-uri <uri>`
- Artifacts: answer text stored as `answer.txt` per run; any files produced in the temp working directory (e.g., `outline.txt`, `report.txt`) are logged as artifacts; LangGraph autologging is enabled.

## Agents, tools, memory
- **Research Team**
  - Supervisor routes between: `search` (Tavily web search) and `web_scraper` (WebBaseLoader for page scraping).
- **Document Writing Team**
  - Supervisor routes between: `note_taker` (creates `outline.txt`), `doc_writer` (writes/edits `report.txt`, uses private memory), `chart_generator` (Python REPL + document reader).
  - Doc writer memory tools (independent): `doc_writer_remember`, `doc_writer_recall` (simple in-memory store for that agent).
- **Top-Level Supervisor**
  - Routes between research and writing teams until completion.

## Files
- `hierarchical_research_app_script.py`: graph construction, helper runners, MLflow setup.
- `run_hierarchical_research.py`: CLI entry point (single or batch).
- `queries_example.json`: sample batch input structure.
- `requirements.txt`: Python dependencies for this use case.
