from dataclasses import dataclass, field

import argparse
import contextlib
import os
import datasets
import duckdb
import mlflow
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.azure import AzureProvider

DEFAULT_EXPERIMENT_NAME = "pydanticai_analyst"
DEFAULT_MLFLOW_URI = "http://localhost:5000"


@dataclass
class AnalystAgentDeps:
    output: dict[str, pd.DataFrame] = field(default_factory=dict[str, pd.DataFrame])

    def store(self, value: pd.DataFrame) -> str:
        """Store the output in deps and return the reference such as Out[1] to be used by the LLM."""
        ref = f'Out[{len(self.output) + 1}]'
        self.output[ref] = value
        return ref

    def get(self, ref: str) -> pd.DataFrame:
        if ref not in self.output:
            raise ModelRetry(
                f'Error: {ref} is not a valid variable reference. Check the previous messages and try again.'
            )
        return self.output[ref]


model = OpenAIChatModel(
    'gpt-4o',
    provider=AzureProvider(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version='2025-01-01-preview',
    ),
)

analyst_agent = Agent(
    model,
    deps_type=AnalystAgentDeps,
    instructions='You are a data analyst and your job is to analyze the data according to the user request.',
)


def _mlflow_span(name: str, attributes: dict[str, str] | None = None):
    """Create an MLflow trace span when tracing is available, else no-op."""
    start_span = getattr(mlflow, "start_span", None)
    if callable(start_span):
        return start_span(name=name, attributes=attributes or {})
    return contextlib.nullcontext()


def mlflow_setup(exp_name: str, mlflow_uri: str) -> None:
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(exp_name)
    if hasattr(mlflow, "openai") and hasattr(mlflow.openai, "autolog"):
        mlflow.openai.autolog()


@analyst_agent.tool
def load_dataset(
    ctx: RunContext[AnalystAgentDeps],
    path: str,
    split: str = 'train',
) -> str:
    """Load the `split` of dataset `dataset_name` from huggingface.

    Args:
        ctx: Pydantic AI agent RunContext
        path: name of the dataset in the form of `<user_name>/<dataset_name>`
        split: load the split of the dataset (default: "train")
    """
    # begin load data from hf
    builder = datasets.load_dataset_builder(path)  # pyright: ignore[reportUnknownMemberType]
    splits: dict[str, datasets.SplitInfo] = builder.info.splits or {}
    if split not in splits:
        raise ModelRetry(
            f'{split} is not valid for dataset {path}. Valid splits are {",".join(splits.keys())}'
        )

    builder.download_and_prepare()  # pyright: ignore[reportUnknownMemberType]
    dataset = builder.as_dataset(split=split)
    assert isinstance(dataset, datasets.Dataset)
    dataframe = dataset.to_pandas()
    assert isinstance(dataframe, pd.DataFrame)
    # end load data from hf

    # store the dataframe in the deps and get a ref like "Out[1]"
    ref = ctx.deps.store(dataframe)
    # construct a summary of the loaded dataset
    output = [
        f'Loaded the dataset as `{ref}`.',
        f'Description: {dataset.info.description}'
        if dataset.info.description
        else None,
        f'Features: {dataset.info.features!r}' if dataset.info.features else None,
    ]
    return '\n'.join(filter(None, output))


@analyst_agent.tool
def run_duckdb(ctx: RunContext[AnalystAgentDeps], dataset: str, sql: str) -> str:
    """Run DuckDB SQL query on the DataFrame.

    Note that the virtual table name used in DuckDB SQL must be `dataset`.

    Args:
        ctx: Pydantic AI agent RunContext
        dataset: reference string to the DataFrame
        sql: the query to be executed using DuckDB
    """
    data = ctx.deps.get(dataset)
    result = duckdb.query_df(df=data, virtual_table_name='dataset', sql_query=sql)
    # pass the result as ref (because DuckDB SQL can select many rows, creating another huge dataframe)
    ref = ctx.deps.store(result.df())
    return f'Executed SQL, result is `{ref}`'


@analyst_agent.tool
def display(ctx: RunContext[AnalystAgentDeps], name: str) -> str:
    """Display at most 5 rows of the dataframe."""
    dataset = ctx.deps.get(name)
    return dataset.head().to_string()  # pyright: ignore[reportUnknownMemberType]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default=(
            "In dataset `cornell-movie-review-data/rotten_tomatoes`, "
            "count rows where label='neg'."
        ),
        help="Task for the analyst agent to answer.",
    )
    parser.add_argument("--exp_name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--mlflow_uri", default=DEFAULT_MLFLOW_URI)
    args = parser.parse_args()

    exp_name = args.exp_name
    mlflow_uri = args.mlflow_uri
    mlflow_setup(exp_name, mlflow_uri)

    task = args.task
    deps = AnalystAgentDeps()
    with mlflow.start_run(run_name="pydanticai-analyst-session"):
        mlflow.log_params(
            {
                "task": task,
                "experiment_name": exp_name,
                "mlflow_uri": mlflow_uri,
                "model": "gpt-4o",
            }
        )
        with _mlflow_span("pydanticai_run_sync", {"agent": "analyst_agent"}):
            result = analyst_agent.run_sync(
                user_prompt=task,
                deps=deps,
            )

        mlflow.log_text(task, "task.txt")
        mlflow.log_text(str(result.output), "result_output.txt")
        mlflow.log_metric("stored_dataframes", len(deps.output))

    print(result.output)
