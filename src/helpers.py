import json
import os

from .data_models import Question, TestDataset


def get_json(path: str) -> str:
    with open(path, "r") as f:
        return json.load(f)


def load_test_dataset(path: str) -> list[TestDataset]:
    table_metadata = get_json(os.path.join(path, "dev_tables.json"))
    questions = get_json(os.path.join(path, "dev.json"))

    test_datasets = []
    for table in table_metadata:
        database_path = os.path.join(
            path, "dev_databases", table["db_id"], f"{table['db_id']}.sqlite"
        )
        approperiate_questions = [
            question for question in questions if question["db_id"] == table["db_id"]
        ]
        approperiate_questions = [
            Question(**question) for question in approperiate_questions
        ]
        test_dataset = TestDataset(
            db_id=table["db_id"],
            database_url=f"sqlite:///{database_path}",
            questions=approperiate_questions,
        )
        test_datasets.append(test_dataset)
    return test_datasets
