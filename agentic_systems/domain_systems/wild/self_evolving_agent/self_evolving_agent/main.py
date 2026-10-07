import argparse
import os, sys

import mlflow


from datetime import datetime
from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from UserQueryAgent import createAgent, get_all_tools, get_model_used


def setup_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)


setup_paths()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use case 6")
    parser.add_argument(
        "--task", type=str, required=True, help="User task / Instruction for the agent"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="usecase6_experiment",
        help="Experiment name for logging and tracking in mlflow",
    )

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
    mlflow.langchain.autolog()


def main():
    
    args = parse_args()

    mlflow_setup(args.exp_name, args.mlflow_uri)

    userQueryAgent = createAgent()
   
    with mlflow.start_run(
        run_name=args.exp_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    ):
        
        tools_list = get_all_tools()
        model_used = get_model_used()
        user_task = args.task

        task_response_messages = userQueryAgent.invoke({"messages": [HumanMessage(user_task)]})
        task_response = task_response_messages["messages"][-1].content

        final_response = str(task_response) if task_response is not None else ""
        mlflow.log_params(
            {
                "model": model_used,
                "tools_list": tools_list,
                "user_query": user_task,  
                "final_response": final_response              
            }
        )        

    print("Agent invoke completed.")
    print(f"""\n\nFINAL OUTPUT:\n {final_response}""")


if __name__ == "__main__":
    main()
