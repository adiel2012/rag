"""FastAPI app: a chat page and a JSON API over the RAG pipeline of the notebook."""
import logging
import threading
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import providers
from .bot import ConversationalRAGBot, Engine, _label
from .config import Settings
from .index import Index, collection_name
from .ingest import file_report, load_documents, split_documents
from .schemas import (
    ChatRequest, ChatResponse, DocumentInfo, Health, RAGResponse, RetrievedChunk,
    StandaloneQuestion,
)

log = logging.getLogger("rag")
STATIC = Path(__file__).parent / "static"


class State:
    """Everything built at startup, plus the conversations."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.stage = "starting"     # starting | pulling model | indexing | ready | error
        self.detail = ""
        self.engine: Optional[Engine] = None
        self.documents: List[DocumentInfo] = []
        self.sessions: "OrderedDict[str, tuple]" = OrderedDict()   # id -> (bot, lock)
        self.lock = threading.Lock()

    def report(self, stage: str, detail: str = "") -> None:
        self.stage, self.detail = stage, detail

    def session(self, session_id: Optional[str]):
        """The bot and lock of a conversation, created on first use. The oldest are dropped."""
        with self.lock:
            session_id = session_id or uuid.uuid4().hex
            if session_id not in self.sessions:
                self.sessions[session_id] = (ConversationalRAGBot(self.engine), threading.Lock())
                while len(self.sessions) > self.settings.max_sessions:
                    self.sessions.popitem(last=False)
            self.sessions.move_to_end(session_id)
            return session_id, *self.sessions[session_id]


def build(state: State, llm=None, embeddings=None) -> None:
    """Get the models, read the documents and build the index. Runs once, in a background thread."""
    s = state.settings
    try:
        if llm is None:
            providers.ensure_models(s, state.report)
            llm = providers.build_llm(s)
        if embeddings is None:
            embeddings = providers.build_embeddings(s)

        state.report("indexing", "reading documents")
        docs, unreadable = load_documents(s.data_dir)
        if not docs:
            raise RuntimeError(
                f"No text could be extracted from the files in '{s.data_dir}'. "
                "Put PDF, DOCX, CSV, TXT or MD files there and restart."
            )
        state.documents = file_report(docs, unreadable)
        splits = split_documents(docs, s.chunk_size, s.chunk_overlap)
        index = Index(
            splits, embeddings, s.db_dir, collection_name(s.provider, s.embed_model),
            top_k=s.top_k, report=lambda detail: state.report("indexing", detail),
        )
        log.info("Index synced: %d added, %d removed, %d total.", index.added, index.removed, len(index))

        state.engine = Engine(
            index, llm,
            providers.structured(llm, RAGResponse, s),
            providers.structured(llm, StandaloneQuestion, s),
        )
        state.report("ready")
    except Exception as e:
        log.exception("Startup failed")
        state.report("error", str(e))


def create_app(settings: Optional[Settings] = None, llm=None, embeddings=None) -> FastAPI:
    """llm and embeddings can be passed in (the tests do); otherwise they come from the provider."""
    state = State(settings or Settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # In the background, so the server answers /api/health while models download.
        threading.Thread(target=build, args=(state, llm, embeddings), daemon=True).start()
        yield

    app = FastAPI(title="RAG chatbot", lifespan=lifespan)
    app.state.rag = state

    def engine() -> Engine:
        if state.stage != "ready":
            raise HTTPException(503, f"Not ready ({state.stage}): {state.detail}".rstrip(": "))
        return state.engine

    @app.get("/api/health", response_model=Health)
    def health():
        s = state.settings
        return Health(
            stage=state.stage, detail=state.detail, provider=s.provider,
            llm_model=s.llm_model, embed_model=s.embed_model,
            files=sum(d.pages > 0 for d in state.documents), chunks=len(state.engine.index) if state.engine else 0,
        )

    @app.get("/api/documents", response_model=List[DocumentInfo])
    def documents():
        return state.documents

    # Plain "def" endpoints run in a thread pool: the model calls block for seconds.
    @app.post("/api/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        engine()
        session_id, bot, lock = state.session(req.session_id)
        with lock:      # one question at a time per conversation, so the history stays in order
            try:
                answer = bot.ask(req.question.strip())
            except Exception as e:
                log.exception("Question failed")
                raise HTTPException(502, f"The model call failed: {e}")
        return ChatResponse(**answer.model_dump(), session_id=session_id)

    @app.delete("/api/sessions/{session_id}", status_code=204)
    def forget(session_id: str):
        with state.lock:
            state.sessions.pop(session_id, None)

    @app.get("/api/inspect", response_model=List[RetrievedChunk])
    def inspect(q: str = Query(min_length=1, max_length=4000), k: Optional[int] = Query(None, ge=1, le=50)):
        """The chunks retrieved for a question: use it to see why an answer was wrong."""
        return [
            RetrievedChunk(n=n, source=_label(d), text=d.page_content)
            for n, d in enumerate(engine().index.retrieve(q, k=k), 1)
        ]

    @app.get("/", include_in_schema=False)
    def page():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
