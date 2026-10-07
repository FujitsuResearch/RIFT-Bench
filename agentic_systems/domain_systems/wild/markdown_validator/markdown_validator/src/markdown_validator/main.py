#!/usr/bin/env python
import argparse
import sys
import os
from pathlib import Path
import mlflow
from dotenv import load_dotenv

if __package__ in (None, ""):
    this_file = Path(__file__).resolve()
    src_root = this_file.parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from markdown_validator.crew import MarkDownValidatorCrew
else:
    from markdown_validator.crew import MarkDownValidatorCrew

# Load environment variables from .env file
load_dotenv()


def configure_model_env() -> None:
    deployment_name = os.environ.get("AZURE_DEPLOYMENT_NAME")
    if deployment_name:
        os.environ["OPENAI_MODEL_NAME"] = deployment_name
        os.environ["MODEL_NAME"] = deployment_name
        os.environ["MODEL"] = deployment_name


configure_model_env()


def configure_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.autolog()


configure_mlflow()


def run(filename: str | None = None):
    """
    Run the markdown validation crew to analyze the markdown file.
    """
    # Get the input markdown file from command line arguments
    inputs = {
        'query': 'Please provide the markdown file to analyze:',
        'filename': filename if filename else (sys.argv[1] if len(sys.argv) > 1 else None),
    }

    # Check if the markdown file path is provided
    if inputs['filename']:
        print(f"Starting markdown validation for file: {inputs['filename']}")
        crewResult = MarkDownValidatorCrew().crew().kickoff(inputs=inputs)
        print("Markdown validation completed")
        return crewResult
    else:
        raise ValueError("Error: No markdown file provided. Please provide a file path as a command-line argument.")


def train(n_iterations: int | None = None, filename: str | None = None):
    """
    Train the markdown validator crew for a given number of iterations.
    """
    # Get the number of iterations and markdown file path from command line arguments
    inputs = {
        'query': 'Training the markdown validation model.',
        'filename': filename if filename else (sys.argv[2] if len(sys.argv) > 2 else None),
    }

    # Check if the markdown file path is provided
    if inputs['filename']:
        try:
            print(f"Starting training for file: {inputs['filename']}")
            iterations = n_iterations if n_iterations is not None else int(sys.argv[1])
            MarkDownValidatorCrew().crew().train(n_iterations=iterations, filename=inputs['filename'])
            print("Training completed successfully.")
        except Exception as e1:
            raise Exception(f"An error occurred while training the crew: {e1}")
    else:
        raise ValueError(
            "Error: No markdown file provided for training. Please provide the number of iterations and a file path.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run markdown validator commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--filename", required=True, help="Markdown file path to validate.")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--n-iterations", type=int, required=True)
    train_parser.add_argument("--filename", required=True, help="Markdown file path for training.")

    return parser.parse_args()


if __name__ == "__main__":
    print("## Welcome to Markdown Validator Crew")
    print('-------------------------------------')

    args = parse_args()
    try:
        if args.command == "run":
            result = run(filename=args.filename)
            print("\n\n########################")
            print("## Validation Report")
            print("########################\n")
            print(f"Final Recommendations: {result}")
        elif args.command == "train":
            train(n_iterations=args.n_iterations, filename=args.filename)
    except Exception as e:
        print(f"An error occurred: {e}")
