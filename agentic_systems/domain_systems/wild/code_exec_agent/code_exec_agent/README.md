# Overview

This use case implements a single CodeAct agent with the following configuration:

- **Framework**: CrewAI  
- **Agent Type**: CodeAct  
- **Tools**:  
  1. External MCP  
  2. Local MCP  
  3. CodeInterpreter  

Additionally, inside the `resources/` directory you will find an `tasks.json` file containing a list of example tasks you can run with the agent.

**DATE OF CREATION:** 02/12/2025

# Getting Started

Navigate to the `code_exec_agent` directory and set up the virtual environment:

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

You are now ready to use `code_exec_agent`!

# Running MLflow

To start the MLflow server for tracing and logging:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

If you change the port, provide the new MLflow URI when running `main.py`.

# Running `main.py`

The `main.py` script runs the CodeAct agent using command-line arguments.

## Must-Haves

To successfully run the code, make sure your `.env` file includes the following keys:


- `AZURE_DEPLOYMENT_NAME`
- `AZURE_MODEL_NAME`
- `AZURE_API_VERSION`
- `AZURE_OPENAI_ENDPOINT`
- `GITHUB_PERSONAL_ACCESS_TOKEN`: for external MCP.
- `SERPER_API_KEY`: for local MCP.

In case you're having issues running the code due to telemetry errors you can set the following to `true` in your `.env` file:

- `OTEL_SDK_DISABLED` 
- `CREWAI_DISABLE_TELEMETRY` 
- `CREWAI_DISABLE_TRACKING` 

## Basic Command

```bash
python main.py --task "YOUR INSTRUCTION HERE"
```

## Arguments

- **`--task`** (required): User instruction or task for the agent.
- **`--exp_name`** (default: `codeact_experiment`): MLflow experiment name.
- **`--max_iter`** (default: `8`): Maximum number of reasoning/tool steps.
- **`--expected_output`** (optional): Expected output description.
- **`--verbose`**: Enables verbose logging.
- **`--mlflow_uri`** (default: `http://localhost:5000`): MLflow tracking server URI.

## Example Usage

```bash
python main.py \
    --task "Write a Python function that computes factorial" \
    --exp_name "codeact_factorial_test" \
    --max_iter 10 \
    --verbose \
    --mlflow_uri http://localhost:5000
```
