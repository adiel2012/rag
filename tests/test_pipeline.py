import pytest

from app.bot import ConversationalRAGBot, Engine
from app.config import Settings
from app.index import Index, collection_name
from app.ingest import file_report, load_documents, split_documents
from app.schemas import RAGResponse, StandaloneQuestion


def make_index(settings, embeddings):
    docs, _ = load_documents(settings.data_dir)
    splits = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    return Index(splits, embeddings, settings.db_dir, "test", top_k=settings.top_k)


def make_bot(settings, embeddings, llm):
    index = make_index(settings, embeddings)
    engine = Engine(index, llm, llm.with_structured_output(RAGResponse), llm.with_structured_output(StandaloneQuestion))
    return ConversationalRAGBot(engine)


def test_load_documents_and_report(settings):
    (settings.data_dir / "scan.txt").write_text("   ", encoding="utf-8")
    docs, unreadable = load_documents(settings.data_dir)
    assert unreadable == ["scan.txt"]
    assert file_report(docs, unreadable)[-1].pages == 0
    assert sorted(d.metadata["source"].replace("\\", "/").rsplit("/", 1)[1] for d in docs) == ["manual.txt", "notes.md"]
    report = {r.name: r for r in file_report(docs)}
    assert report["notes.md"].low_text and not report["manual.txt"].low_text


def test_index_sync_is_incremental(settings, embeddings):
    first = make_index(settings, embeddings)
    assert first.added == len(first) > 3 and first.removed == 0

    again = make_index(settings, embeddings)
    assert (again.added, again.removed, len(again)) == (0, 0, len(first))

    (settings.data_dir / "notes.md").unlink()
    after = make_index(settings, embeddings)
    assert after.added == 0 and after.removed == 1 and len(after) == len(first) - 1


def test_retrieve_finds_exact_term_and_adds_next_chunk(settings, embeddings):
    index = make_index(settings, embeddings)
    # Fake embeddings carry no meaning, so only the keyword side can find this code.
    found = index.retrieve("ZX-CAL", k=1, neighbors=1)
    assert "ZX-CAL" in found[0].page_content
    assert len(found) == 2                                   # the hit and the chunk that follows it
    positions = [index.position[d.id] if getattr(d, "id", None) in index.position
                 else index.chunk_docs.index(d) for d in found]
    assert positions[1] == positions[0] + 1                  # in document order


def test_retrieve_weights(settings, embeddings):
    index = make_index(settings, embeddings)
    # With no weight the second query cannot contribute anything.
    only_first = index.retrieve(["Rotterdam warehouse", "ZX-CAL"], k=1, neighbors=0, weights=[1.0, 0.0])
    assert "Rotterdam" in only_first[0].page_content


def test_answer_is_checked_in_code(settings, embeddings, llm):
    r = make_bot(settings, embeddings, llm).ask("How long is the warranty?")
    assert r.citations_verified == [True, False]             # the invented quote is caught
    assert len(r.sources) == 1                               # chunk 99 does not exist and is dropped
    assert r.confidence_score == 1.0                         # clamped
    assert r.standalone_question == "How long is the warranty?"   # first turn: no rewrite


def test_follow_up_is_rewritten_and_history_is_sent(settings, embeddings, llm):
    bot = make_bot(settings, embeddings, llm)
    bot.ask("Tell me about the Falcon drone.")
    r = bot.ask("How long is it?")
    assert r.standalone_question == llm.rewrite
    assert len(llm.prompts[-1].to_messages()) == 4           # system, previous question and answer, question


def test_rewrite_that_looks_like_an_answer_is_discarded(settings, embeddings, llm):
    bot = make_bot(settings, embeddings, llm)
    bot.ask("Tell me about the Falcon drone.")
    llm.rewrite = "The warranty lasts 24 months."            # lost the question mark
    assert bot.ask("How long is it?").standalone_question == "How long is it?"
    llm.rewrite = "x" * 300 + "?"                            # far too long
    assert bot.ask("And the battery?").standalone_question == "And the battery?"


def test_plain_text_fallback(settings, embeddings, llm):
    bot = make_bot(settings, embeddings, llm)
    bot.engine.chain = bot.engine.chain | (lambda _: (_ for _ in ()).throw(ValueError("bad json")))
    r = bot.ask("How long is the warranty?")
    assert r.answer == "plain text answer" and r.confidence_score == 0.0 and r.citations == []


def test_settings_defaults_and_bedrock_requires_a_model(monkeypatch):
    assert Settings().llm_model == "qwen3.5:9b" and Settings().temperature == 0
    assert Settings(thinking="").thinking is None
    with pytest.raises(ValueError, match="LLM_MODEL is required"):
        Settings(provider="bedrock")
    s = Settings(provider="bedrock", llm_model="some.model-v1:0")
    assert s.embed_model == "amazon.titan-embed-text-v2:0" and s.temperature is None
    assert collection_name("bedrock", s.embed_model) != collection_name("ollama", "bge-m3")
