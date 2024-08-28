from pydantic import BaseModel, Field


class Question(BaseModel):
    question_id: int
    db_id: str
    question: str
    evidence: str
    SQL: str
    difficulty: str


class TestDataset(BaseModel):
    db_id: str
    database_url: str
    questions: list[Question] = Field(default_factory=list)


class Evaluation(BaseModel):
    agent: str
    db_id: str
    question_id: int
    question_difficulty: str
    question: str
    generated_SQL: str
    expected_SQL: str
    expected_result: str
    generated_result: str
    is_runnable: bool
    returns_correct_result: bool
    evaluation_time: float
    tokens_used: int


class TrueOrFalse(BaseModel):
    result: bool
