# Overview

This use case implements a single ReAct agent with the following configuration:

- **Framework**: LangGraph
- **Agent Type**: ReAct
- **Tools**:
  1. Short-term memory (checkpointer) - conversation state
  2. Long-term memory (store) - persistent facts via remember/recall tools
  3. DALL-E API tool

The agent uses async SQLite databases to persist memories across runs:
- `checkpoints.db` - stores conversation history per thread
- `memory_store.db` - stores user facts across all conversations

**DATE OF CREATION:** 04/12/2025 
## Memory Usage

**To save information:**
- Explicitly ask the agent to remember: "Remember my name is <user_name>"

**To retrieve information:**
- Ask the agent to recall: "Recall my name"

Additionally, inside the `resources/` directory you will find an `tasks.json` file containing a list of example tasks you can run with the agent.

# Getting Started

Navigate to the `memory_agent` directory and set up the virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

**Linux / macOS**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

You are now ready to use `memory_agent`!


# Running MLflow

To start the MLflow server for tracing and logging:

```bash
mlflow server --host 0.0.0.0 --port 5000
```
# Running `main.py`

The `main.py` script runs the CodeAct agent using command-line arguments.

## Must-Haves

To successfully run the code, make sure your `.env` file includes the following keys:


- `AZURE_DEPLOYMENT_NAME`
- `AZURE_MODEL_NAME`
- `AZURE_API_VERSION`
- `AZURE_OPENAI_ENDPOINT`
- `DALLE_API_KEY`
- `DALLE_API_BASE`
- `DALLE_API_VERSION`
- `DALLE_DEPLOYMENT_NAME`


## Basic Command

```bash
python main.py --task "YOUR INSTRUCTION HERE"
```

## Arguments

- **`--task`** (required): User instruction or task for the agent.
- **`--exp_name`** (default: `codeact_experiment`): MLflow experiment name.
- **`--tool_file`** (default: `./resources/langraph_react_tools_list.json`): list of tools that can be used by the agent.

## Example Usage

```bash
python main.py \
    --task "create an image of a cat wearing a red hat" \
    --exp_name "react_dalle_api_tool" \
```
