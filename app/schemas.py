"""Pydantic models: what the LLM must return, and what the API accepts and returns."""
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class RAGResponse(BaseModel):
    """Structured answer produced by the model."""
    answer: str = Field(description="The conversational answer to the user's question.")
    citations: List[str] = Field(
        default_factory=list,
        description="Short exact quotes copied from the context that support the answer.",
    )
    source_ids: List[int] = Field(
        default_factory=list,
        description="Numbers of the context chunks (the [n] labels) the answer is based on.",
    )
    confidence_score: float = Field(
        description="From 0.0 to 1.0: how well the context answers the question.",
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="1 to 2 relevant follow-up questions the user might ask next.",
    )

    @field_validator("confidence_score")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return min(1.0, max(0.0, v))


class GroundedResponse(RAGResponse):
    """RAGResponse plus fields computed by our code, not by the model."""
    standalone_question: str = ""
    sources: List[str] = Field(default_factory=list)              # file / page of the chunks used
    citations_verified: List[bool] = Field(default_factory=list)  # is each quote really in the context?


class StandaloneQuestion(BaseModel):
    """The user's latest message rewritten as a self-contained search question."""
    question: str = Field(description="The rewritten question. A question, never an answer.")


# --- API ---

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = Field(
        default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$",
        description="Send the value returned by the previous answer to continue that conversation.",
    )


class ChatResponse(GroundedResponse):
    session_id: str


class Health(BaseModel):
    stage: str          # starting | pulling model | indexing | ready | error
    detail: str = ""
    provider: str
    llm_model: str
    embed_model: str
    files: int = 0
    chunks: int = 0


class DocumentInfo(BaseModel):
    name: str
    pages: int          # 0: no text could be read from the file, so it is not searchable
    characters: int
    low_text: bool      # very little text per page: scanned or image-only?


class RetrievedChunk(BaseModel):
    n: int
    source: str
    text: str
