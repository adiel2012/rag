"""Fakes so the tests run without Ollama, AWS or a GPU."""
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.config import Settings
from app.schemas import RAGResponse, StandaloneQuestion

MANUAL = (
    "Falcon drone user manual.\n\n"
    "The Falcon drone has a warranty period of 24 months from the date of purchase.\n\n"
    + "General safety advice: keep away from people and animals while flying. " * 12
    + "\n\nTo calibrate the compass, open the Sensors tab and press the ZX-CAL button.\n\n"
    + "Then rotate the drone slowly three times around each axis until the light turns green. " * 4
    + "\n\nBattery care: store the battery at half charge in a cool and dry place. " * 6
)
NOTES = "# Shipping notes\n\nOrders are shipped from the Rotterdam warehouse every Tuesday.\n"


class FakeLLM:
    """Answers with a real quote from the first chunk plus an invented one."""

    def __init__(self):
        self.rewrite = "What is the warranty period of the Falcon drone?"
        self.prompts = []

    def _answer(self, prompt):
        self.prompts.append(prompt)
        context = prompt.to_messages()[0].content.split("Context:\n", 1)[1]
        first_chunk = context.split("\n", 1)[1]
        return RAGResponse(
            answer="It is in the manual.",
            citations=[first_chunk[:40], "this sentence appears in no document at all"],
            source_ids=[1, 99],
            confidence_score=1.7,
        )

    def with_structured_output(self, schema, **kwargs):
        if schema is RAGResponse:
            return RunnableLambda(self._answer)
        return RunnableLambda(lambda prompt: StandaloneQuestion(question=self.rewrite))

    def invoke(self, prompt, **kwargs):
        return AIMessage("plain text answer")


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "manual.txt").write_text(MANUAL, encoding="utf-8")
    (d / "notes.md").write_text(NOTES, encoding="utf-8")
    (d / "ignored.xyz").write_text("not a supported type", encoding="utf-8")
    return d


@pytest.fixture
def settings(tmp_path, data_dir):
    return Settings(data_dir=data_dir, db_dir=tmp_path / "db", chunk_size=300, chunk_overlap=40, top_k=3)


@pytest.fixture
def embeddings():
    return DeterministicFakeEmbedding(size=64)


@pytest.fixture
def llm():
    return FakeLLM()
