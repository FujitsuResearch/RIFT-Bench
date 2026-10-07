# Overview

This use case runs a medical domain multi-agent flow.

- **Framework**: crewai
- **Mode**: agent
- **Entrypoint**: `main.py`

# Getting Started

Navigate to this use case directory and set up a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Running MLflow

```bash
mlflow server --host 0.0.0.0 --port 5000
```

# Running `main.py`

## Must-Haves

Ensure your `.env` includes the model/provider credentials required by this project.

## Basic Command

```bash
python main.py --query "YOUR INSTRUCTION HERE"
```

## Arguments

- **`--query`** (required): User instruction/task.
- **`--exp_name`** (default: `medical_crewai_agent`): MLflow experiment name.
- **`--port`** (default: `5000`): MLflow tracking port.
- **`--verbose`** (optional): Print intermediate agent messages.

## Example Usage

```bash
python main.py \
  --query "Example task for medical" \
  --exp_name "medical_crewai_agent" \
  --port 5000
```
