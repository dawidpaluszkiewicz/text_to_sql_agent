import pandas as pd
import sqlalchemy
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.sql_database import SQLDatabase
from langchain_core.language_models.base import BaseLanguageModel
from sqlalchemy import create_engine

from src.agent import TextToSQLConverter

from .data_models import Evaluation, TestDataset, TrueOrFalse


def get_query_result(sql_query: str, db_url: str) -> dict:
    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
            df = pd.read_sql_query(sql_query, connection)
        result_dict = df.to_dict(orient="list")
        for key, value in result_dict.items():
            result_dict[key] = [
                (
                    str(item)
                    if not isinstance(item, (int, float, str, bool, type(None)))
                    else item
                )
                for item in value
            ]

        return result_dict
    except sqlalchemy.exc.SQLAlchemyError as e:
        return {"error": str(e)}


def is_runnable(sql_query: str, database_url: str) -> bool:
    try:
        db = SQLDatabase.from_uri(database_url)
        db.run(sql_query)
        return True
    except Exception as e:
        return False


class SQLQueryComparison:
    def __init__(self, llm: str):
        self.parser = PydanticOutputParser(pydantic_object=TrueOrFalse)
        self.prompt_template = """
        You are an AI assistant tasked with comparing the results of two SQL queries to determine if they are equivalent, ignoring potential differences in column names.

        Question: {question}

        Ground Truth Query:
        {ground_truth_query}

        Evaluated Query:
        {evaluated_query}

        Ground Truth Result:
        {ground_truth_result}

        Evaluated Result:
        {evaluated_result}

        Please analyze the results and determine if they are equivalent. Consider the following:
        1. The number of rows in both results should be the same.
        2. The data in each row should match, even if the column names or order are different.
        3. Numerical values should be considered equal if they're within a small margin of error due to floating-point representation.
        4. The results should be identical in content, ignoring any differences in column order or names.
        5. If evaluated query contains a correct result but it also contains some additional columns that are not present in the ground truth, it's still a correct result.

        {format_instructions}
        """
        self.prompt = PromptTemplate(
            input_variables=[
                "question",
                "ground_truth_query",
                "evaluated_query",
                "ground_truth_result",
                "evaluated_result",
                "format_instructions",
            ],
            template=self.prompt_template,
        )
        self.llm = llm
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def compare_queries(
        self,
        question: str,
        ground_truth_query: str,
        evaluated_query: str,
        ground_truth_result: str,
        evaluated_result: str,
    ) -> bool:
        result = self.chain.run(
            {
                "question": question,
                "ground_truth_query": ground_truth_query,
                "evaluated_query": evaluated_query,
                "ground_truth_result": ground_truth_result,
                "evaluated_result": evaluated_result,
                "format_instructions": self.parser.get_format_instructions(),
            }
        )
        parsed_result: TrueOrFalse = self.parser.parse(result)
        return parsed_result.result


def test_agent_on_dataset(
    dataset: TestDataset,
    llm: BaseLanguageModel,
    cot_prompt: str,
    agent_name: str,
    evaluation_agent: SQLQueryComparison,
    max_questions: int | None = None,
) -> list[Evaluation]:
    agent = TextToSQLConverter(
        dataset.database_url, llm, chain_of_thoughts_prompt=cot_prompt
    )
    evaluation_results = []

    max_questions = max_questions or len(dataset.questions)
    if max_questions > len(dataset.questions):
        max_questions = len(dataset.questions)

    for question in dataset.questions[:max_questions]:
        try:
            sql, evaluation_time, tokens_used = agent._get_sql_query_and_metrics(
                question.question
            )

            expected_results = (get_query_result(question.SQL, dataset.database_url),)
            generated_results = (get_query_result(sql, dataset.database_url),)
            returns_correct_result = evaluation_agent.compare_queries(
                question.question,
                question.SQL,
                sql,
                expected_results,
                generated_results,
            )

            results = Evaluation(
                agent=agent_name,
                db_id=dataset.db_id,
                question_id=question.question_id,
                question_difficulty=question.difficulty,
                question=question.question,
                generated_SQL=sql,
                expected_SQL=question.SQL,
                is_runnable=is_runnable(sql, dataset.database_url),
                returns_correct_result=returns_correct_result,
                evaluation_time=evaluation_time,
                tokens_used=tokens_used,
                expected_result=str(expected_results),
                generated_result=str(generated_results),
            )

            evaluation_results.append(results)
        except Exception as e:
            print(e)
    return evaluation_results
