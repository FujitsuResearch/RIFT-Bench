import argparse
import os, sys

import mlflow

from crewai import Task, Crew, Process
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(override=True)

from agents.code_act import CodeAct


def setup_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)


setup_paths()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CodeAct Use Case")
    parser.add_argument(
        "--task", type=str, required=True, help="User task / Instruction for the agent"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="codeact_experiment",
        help="Experiment name for logging and tracking in mlflow",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=8,
        help="Max tool / reasoning iterations for the agent to perform",
    )

    parser.add_argument(
        "--expected_output",
        type=str,
        default=None,
        help="Expected output for the task (optional)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking server URI",
    )
    return parser.parse_args()


def mlflow_setup(exp_name: str, mlflow_ur: str) -> None:
    mlflow.set_tracking_uri(mlflow_ur)
    mlflow.set_experiment(exp_name)
    mlflow.crewai.autolog()


def create_agent(max_iter, verbose) -> CodeAct:
    code_act = CodeAct(
        max_iter=max_iter,
        verbose=verbose,
    )
    return code_act


def create_task(description: str, expected_output: str, agent: CodeAct) -> Task:

    task = Task(
        description=description,
        expected_output=(
            expected_output
            if expected_output is not None
            else "A clear, concise answer beginning with 'FINAL:' and no extra commentary afterwards."
        ),
        agent=agent,
    )
    return task


def crew_setup(code_act: CodeAct, task: Task, verbose: bool) -> Crew:
    crew = Crew(
        tasks=[task],
        agents=[code_act],
        process=Process.sequential,
        verbose=verbose,
    )
    return crew


def main():
    
    args = parse_args()
    mlflow_setup(args.exp_name, args.mlflow_uri)


    code_act = create_agent(args.max_iter, args.verbose)
    tools_list = code_act.get_all_tools()
    task = create_task(args.task, args.expected_output, code_act)
    crew = crew_setup(code_act, task, args.verbose)

    with mlflow.start_run(
        run_name=args.exp_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    ):
        mlflow.log_params(
            {
                "model": code_act.llm,
                "tools_list": tools_list,
                "mcp_servers": [type(mcp).__name__ for mcp in code_act.mcps],
            }
        )

        mlflow.log_text(args.task, "task.txt")
        result = crew.kickoff()
        try:
            final_text = str(result) if result is not None else ""
            mlflow.log_text(final_text, "final.txt")
            if hasattr(task, "output") and task.output is not None:
                mlflow.log_text(task.output.raw or "", "task_output_raw.txt")
        except Exception as e:
            print(f"[mlflow] logging error: {e}")

    print("Crew run completed.")
    if hasattr(task, "output") and task.output is not None:
        print("Final Output:")
        print("Summary:", task.output.raw[:1000])


if __name__ == "__main__":
    main()
