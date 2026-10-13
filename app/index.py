"""Vector store (Chroma, persisted) + keyword index (BM25) and hybrid retrieval (notebook section 7)."""
import hashlib
import re
from pathlib import Path
from typing import Callable, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

BATCH = 100


def chunk_id(d: Document) -> str:
    key = f"{d.metadata.get('source')}|{d.metadata.get('page')}|{d.page_content}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def collection_name(provider: str, embed_model: str) -> str:
    # One collection per embedding model: vectors from different models are not comparable.
    return ("rag_" + re.sub(r"[^a-z0-9]+", "_", f"{provider}_{embed_model}".lower()).strip("_"))[:60].strip("_")


def _tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class Index:
    def __init__(
        self,
        splits: List[Document],
        embeddings: Embeddings,
        db_dir: Path,
        collection: str,
        top_k: int = 6,
        report: Optional[Callable[[str], None]] = None,
    ):
        self.top_k = top_k
        # Cosine distance: unlike the notebook's embeddings, the providers' vectors are not normalised.
        self.vectorstore = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=str(db_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

        # Incremental sync: only new chunks are embedded, and chunks whose file
        # was removed or changed are deleted.
        wanted = {chunk_id(d): d for d in splits}
        existing = set(self.vectorstore.get(include=[])["ids"])
        stale = [i for i in existing if i not in wanted]
        new = [i for i in wanted if i not in existing]
        for i in range(0, len(stale), BATCH):
            self.vectorstore.delete(ids=stale[i:i + BATCH])
        for i in range(0, len(new), BATCH):
            if report:
                report(f"embedding chunks {i}/{len(new)}")
            batch = new[i:i + BATCH]
            self.vectorstore.add_documents([wanted[j] for j in batch], ids=batch)
        self.added, self.removed = len(new), len(stale)

        # Keyword index (BM25) over the same chunks, for exact terms that embeddings can miss
        # (field names, codes, product-specific words).
        self.chunk_ids = list(wanted)                 # in document order
        self.chunk_docs = [wanted[i] for i in self.chunk_ids]
        self.position = {cid: n for n, cid in enumerate(self.chunk_ids)}
        self.bm25 = BM25Okapi([_tokens(d.page_content) for d in self.chunk_docs])

    def __len__(self) -> int:
        return len(self.chunk_docs)

    def retrieve(self, queries, k: Optional[int] = None, fetch: int = 20, neighbors: int = 3, weights=None) -> List[Document]:
        """Hybrid search: merge semantic (vector) and keyword (BM25) rankings.
        Accepts one query or a list of queries (e.g. the rewritten and the original question).
        weights, one per query, says how much each query counts in the merge (default 1.0 each)."""
        k = k or self.top_k
        chunk_docs, position = self.chunk_docs, self.position
        if isinstance(queries, str):
            queries = [queries]
        if weights is None:
            weights = [1.0] * len(queries)
        weight_of = {}  # query -> weight, duplicates merged
        for query, w in zip(queries, weights):
            weight_of[query] = max(w, weight_of.get(query, 0.0))

        rankings = []  # (weight, ranked chunks)
        for query, w in weight_of.items():
            rankings.append((w, self.vectorstore.similarity_search(query, k=min(fetch, len(chunk_docs)))))
            scores = self.bm25.get_scores(_tokens(query))
            order = sorted(range(len(chunk_docs)), key=lambda i: scores[i], reverse=True)[:fetch]
            rankings.append((w, [chunk_docs[i] for i in order if scores[i] > 0]))

        fused = {}  # reciprocal rank fusion: chunk id -> score
        for w, ranking in rankings:
            for rank, d in enumerate(ranking):
                cid = getattr(d, "id", None) or chunk_id(d)
                if cid in position:
                    fused[cid] = fused.get(cid, 0.0) + w / (60 + rank)
        best = sorted(fused, key=fused.get, reverse=True)[:k]

        # A procedure often continues in the next chunk (e.g. across a page break):
        # also take the chunk that follows each of the top hits.
        picked = set(best)
        for cid in best[:neighbors]:
            nxt = position[cid] + 1
            if nxt < len(chunk_docs) and chunk_docs[nxt].metadata.get("source") == chunk_docs[position[cid]].metadata.get("source"):
                picked.add(self.chunk_ids[nxt])

        # Return in document order so multi-step procedures read coherently.
        return [chunk_docs[i] for i in sorted(position[cid] for cid in picked)]
