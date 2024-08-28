import time

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.sql_database import SQLDatabase
from langchain_core.language_models.base import BaseLanguageModel
from langchain_openai.chat_models.base import ChatOpenAI


class TextToSQLConverter:
    def __init__(
        self, db_uri: str, llm: BaseLanguageModel, chain_of_thoughts_prompt: str = ""
    ):
        self.db = SQLDatabase.from_uri(db_uri)
        self.llm = llm

        self.system_prompt = """
        __INSTRUCTIONS__
        You are an AI assistant that translates natural language queries into SQL.
        Given a question create a syntactically correct sqlite3 query.
        Your task is to generate SQL queries based on the user's questions about the database.
        Whenever you need to filter by text data make sure to use appropriate LIKE operator.
        Try to avoid errors caused by case sensitivity.
        {chain_of_thoughts_prompt}

        DON'T OVERCOMPLICATE IT. DON'T CREATE UNNECESSARY COLUMNS. 

        __OUTPUT_FORMAT__
        At the end of the output, please include the SQL query itself surrounded by &&& without any other text.
        For example: &&&SELECT * FROM users WHERE age > 20&&&""".format(
            chain_of_thoughts_prompt=chain_of_thoughts_prompt
        )

        self.template = (
            self.system_prompt
            + """

        __DATABASE_SCHEMA__
        {db_schema}

        User question: {question}

        SQL query:
        """
        )

        self.prompt = PromptTemplate(
            input_variables=["db_schema", "question"], template=self.template
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def _get_db_schema(self):
        return self.db.get_table_info()

    def run(self, question: str) -> str:
        db_schema = self._get_db_schema()
        return self.chain.run(db_schema=db_schema, question=question)

    def convert_to_sql(self, question: str) -> str:
        response = self.run(question)
        sql_query = response.split("&&&")[1].strip()  # TODO: make it more robust
        return sql_query

    def _get_sql_query_and_metrics(self, question: str) -> tuple[str, float, int]:
        start_time = time.time()
        response = self.run(question)
        elapsed_time = time.time() - start_time
        try:
            sql_query = response.split("&&&")[1].strip()
        except IndexError:
            sql_query = ""
        tokens_used = self.llm.get_num_tokens(response)

        return sql_query, elapsed_time, tokens_used
