import datetime
import os

import mlflow
import pandas as pd
from langchain_openai import ChatOpenAI

from src.evaluation import SQLQueryComparison, test_agent_on_dataset
from src.helpers import load_test_dataset
from src.prompts import (plan_and_solve_cot_prompt,
                         plan_and_solve_cot_prompt_v2, zero_shot_cot_prompt)


def run_benchmarks():
    open_ai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=open_ai_api_key)
    evaluation_agent = SQLQueryComparison(llm)
    test_datasets = load_test_dataset("data/")

    agents_definitons = [
        ("", "no_cot"),
        (zero_shot_cot_prompt, "zero_shot_cot"),
        (plan_and_solve_cot_prompt, "plan_and_solve_cot"),
        (plan_and_solve_cot_prompt_v2, "plan_and_solve_cot_v2"),
    ]

    mlflow.set_experiment(
        f"sql_agent_interview_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    for cot_prompt, agent_name in agents_definitons:
        results = []
        with mlflow.start_run(run_name=agent_name):
            for dataset in test_datasets:
                evaluations = test_agent_on_dataset(
                    dataset,
                    llm,
                    cot_prompt,
                    agent_name,
                    evaluation_agent,
                    max_questions=50,
                )
                results += evaluations

            df = pd.DataFrame([i.dict() for i in results])
            df[["is_runnable", "returns_correct_result"]] = df[
                ["is_runnable", "returns_correct_result"]
            ].astype(int)
            mean_resutls = (
                df[
                    [
                        "is_runnable",
                        "returns_correct_result",
                        "evaluation_time",
                        "tokens_used",
                    ]
                ]
                .mean()
                .to_dict()
            )
            mlflow.log_params({"cot_prompt": cot_prompt})
            mlflow.log_metric("is_runnable_ratio", mean_resutls["is_runnable"])
            mlflow.log_metric(
                "correct_results_ratio", mean_resutls["returns_correct_result"]
            )
            mlflow.log_metric(
                "average_evaluation_time", mean_resutls["evaluation_time"]
            )
            mlflow.log_metric("average_tokens_used", mean_resutls["tokens_used"])
            df.to_csv(f"{agent_name}_detailed_metrics.csv", index=False)
            mlflow.log_artifact(f"{agent_name}_detailed_metrics.csv")
            os.remove(f"{agent_name}_detailed_metrics.csv")


if __name__ == "__main__":
    run_benchmarks()
