# RAG explained, plus a local conversational RAG chatbot

This folder contains:

- `README.md`: the theory of Retrieval-Augmented Generation (this file)
- [`rag_chatbot.ipynb`](rag_chatbot.ipynb): a working notebook for Google Colab (T4) or your own machine: local Qwen 3.5 (Ollama) + local embeddings + Chroma + Pydantic structured answers + chat memory + document upload

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/rag/blob/main/rag_chatbot.ipynb)

Click the button to open the notebook in Google Colab, then choose Runtime -> Change runtime type -> **T4 GPU** and run the cells top to bottom.

---

## 1. What is RAG?

A language model only knows what was in its training data. It does not know your private documents, it has a knowledge cutoff, and when it lacks a fact it may invent one (hallucination).

**Retrieval-Augmented Generation (RAG)** fixes this without retraining the model. Before the model answers, the system *retrieves* the most relevant passages from your own knowledge base and puts them in the prompt. The model then answers *from that text*.

> Think of an open-book exam: the model does not need to memorize everything, it only needs to read the right page and answer from it.

Why use it instead of fine-tuning:

| | RAG | Fine-tuning |
|---|---|---|
| Add or update knowledge | Re-index a file | Retrain |
| Cost | Low | High |
| Can cite sources | Yes | No |
| Reduces hallucination | Yes, when retrieval finds the right text | Not reliably |
| Good for | Facts, documents, changing data | Style, format, behavior |

RAG is only as good as its retrieval: if the right passage is not retrieved, the model cannot answer correctly, however capable it is. Most RAG tuning is retrieval tuning.

---

## 2. The pipeline

```
          INDEXING (done once per document)
 Documents -> Load -> Chunk -> Embed -> Vector store

          QUERYING (every question)
 Question -> Embed -> Search vector store -> Top-k chunks
          -> Prompt = instructions + chunks + question -> LLM -> Answer
```

### 2.1 Load
Read files (PDF, DOCX, CSV, text, web pages) into plain text plus metadata (source file, page number). Parsing quality matters a lot: garbage text in means garbage retrieval.

### 2.2 Chunk
Documents are split into pieces (typically 300-1000 characters) because:
- the model's context window is limited,
- one embedding per whole book would be too vague to match a specific question,
- smaller, focused chunks retrieve more precisely.

**Overlap** (e.g. 10-15%) between neighboring chunks keeps a sentence that straddles a boundary from being cut off from its context. `RecursiveCharacterTextSplitter` splits on paragraphs first, then lines, then sentences, so chunks follow natural boundaries.

Trade-off: chunks too small lose context; chunks too large dilute the match and waste prompt space.

### 2.3 Embed
An **embedding model** turns text into a vector (a list of a few hundred numbers) so that *texts with similar meaning have vectors that point in similar directions*. "How do I cancel my subscription?" lands near "Steps to terminate your plan" even though they share no words. This is **semantic search**, as opposed to keyword search.

The same embedding model must be used for indexing and querying, and changing the model means re-embedding everything (the notebook keeps one Chroma collection per embedding model for this reason).

Two properties of the embedding model matter in practice:
- **Language**: `all-MiniLM-L6-v2` is trained on English. For Spanish or mixed-language documents use a multilingual model (e.g. `BAAI/bge-m3`).
- **Maximum input length**: text beyond the model's limit is silently cut off, so chunks must fit (MiniLM reads about 256 tokens, roughly 1000 characters of English).

### 2.4 Vector store
A database (Chroma, FAISS, Qdrant, Pinecone, pgvector) that stores each chunk's vector and finds the nearest ones to a query vector quickly, usually using cosine similarity. It also stores metadata so you can cite sources and filter results.

### 2.5 Retrieve
Embed the question, fetch the **top-k** closest chunks (k = 3-6 is common). Raising k improves recall but adds noise and cost.

### 2.6 Generate
Build a prompt:

```
System: Answer ONLY from the context. If it's not there, say you don't know.
Context: <retrieved chunks>
Question: <user question>
```

The model answers using the context. The "say you don't know" instruction is what turns retrieval into a reduction in hallucinations.

Everything (instructions, chunks, chat history, question) must fit in the model's **context window**. Ollama's default window is small and it truncates silently when the prompt is too long, so the notebook sets `num_ctx` explicitly.

### 2.7 What top-k RAG cannot do
Retrieval returns a handful of chunks, so questions that need the *whole* corpus ("summarize all the documents", "how many invoices are there?") get an answer based on only those few chunks. Those tasks need a different approach (summarize each document and then combine, or query structured data directly).

---

## 3. Making it conversational

Plain RAG treats each question alone. In a chat, users say things like *"and how does it connect to Pydantic?"*. Two additions are needed:

1. **Memory**: keep the previous turns and pass them to the model.
2. **Question rewriting (history-aware retrieval)**: the vector search cannot understand "it". Before retrieving, the model rewrites the question into a standalone one ("How does Qwen connect to Pydantic?") using the chat history. Retrieval then uses the rewritten question, while the answer prompt gets the history.

The notebook implements both. It also trims history to the last few turns so the prompt does not grow forever.

---

## 4. Structured output with Pydantic

Free-text answers are hard for programs to use. With **Pydantic** you define a schema:

```python
class RAGResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence_score: float   # 0.0-1.0
    follow_up_questions: list[str]
```

and `llm.with_structured_output(RAGResponse)` makes the model return JSON matching it (Ollama constrains generation to the schema). Pydantic validates the result, so your code receives a real Python object (`res.answer`, `res.confidence_score`) or a clear error, never malformed text. If structured output still fails (for example the JSON is cut off), the notebook falls back to a plain-text answer.

**Model-reported is not verified.** A schema guarantees the *shape* of the answer, not its truth: the model can invent a quote or report 0.9 confidence on a wrong answer. So the notebook separates what the model says from what the code checks:

| Field | Filled by | Trust |
|---|---|---|
| `answer`, `follow_up_questions` | Model | Read critically |
| `confidence_score` | Model | A hint only, not a probability |
| `citations` (quotes) | Model | Checked: `citations_verified` says whether each quote really appears in the retrieved text |
| `source_ids` -> `sources` (file, page) | Model picks chunk numbers, code maps them to files | Tells you where to look to confirm |
| `standalone_question` | Code | Shows what was actually searched |

---

## 5. Running it (Qwen + Ollama, on Colab or your own machine)

The model runs inside the notebook's own runtime through Ollama, with no API keys and no calls to an external AI service:

| Component | Choice | Role |
|---|---|---|
| LLM | Qwen 3.5 9B via **Ollama** | Generates the answer |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) | Text -> vectors |
| Vector DB | **Chroma** (persisted in `chroma_db/`) | Stores and searches chunks |
| Orchestration | **LangChain** | Glues the steps together |
| Validation | **Pydantic** | Typed, validated answers |

LangChain is the glue, not the whole system: you still need the model, the embedding model, the vector DB and the document parsers. For a production system, many teams use the underlying SDKs directly for more control and easier debugging; LangChain is excellent for getting a working pipeline quickly.

### Why Qwen 3.5 9B

Qwen 3.5's small series (0.8B, 2B, 4B, 9B) is the newest Qwen line made for edge hardware; the newer 3.6 and 3.8 releases are only published at 27B and up. The 9B is the largest of the small series. It is about a 7 GB download, so it fits the 16 GB of a Colab T4 GPU with room for the context window, while the next size up (27B) does not fit.

Qwen 3.5 has a "thinking" mode that reasons before answering. The notebook turns it off (`THINKING = False`): it adds latency and is not needed to answer from retrieved text.

### Setup on Google Colab (T4)

1. Open the notebook in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/rag/blob/main/rag_chatbot.ipynb)
2. Runtime -> Change runtime type -> **T4 GPU**.
3. Run the cells top to bottom. Cell 3 installs Ollama, starts it and downloads the model (about 7 GB, a few minutes).
4. Cell 5 asks for your PDF files: an upload button appears and the notebook waits until you choose them. The following cells index them.

Colab sessions are temporary: after a runtime reset the model, your uploaded files and the index are gone and the cells must be run again. Note also that on Colab your documents are uploaded to Google's servers, so "local" means "no AI API", not "never leaves your computer".

### Setup on your own machine

1. Install [Python](https://www.python.org/downloads/) 3.10 or newer (tick "Add python.exe to PATH" on Windows) and the Jupyter extension for VS Code.
2. Install [Ollama](https://ollama.com) (version 0.17.4 or newer for Qwen 3.5).
3. Open `rag_chatbot.ipynb` and run the cells top to bottom. Cell 3 starts Ollama if needed and pulls the model.
4. Cell 5 opens a file dialog asking for your PDF files and copies them into `data/`. (DOCX, CSV, TXT and MD files placed in `data/` by hand are indexed too.)

On a machine with 8 GB of RAM use `qwen3.5:4b` instead; without a GPU the 9B will be slow.

### Settings

All settings (model, thinking, context window, embedding model, chunk size, top-k) are in the config cell. Re-run the "Load and chunk" and "Embeddings" cells whenever the documents change: the index is synced incrementally, so only new chunks are embedded and chunks from removed files are deleted.

Smaller models are faster but follow the "answer only from context" rule less reliably. For a model without thinking mode (such as `qwen2.5:7b`) set `THINKING = None`.

### What the notebook does

| Cells | Step |
|---|---|
| 1-3 | Install, configure, install/start Ollama and download the model |
| 4 | Pydantic schemas: `RAGResponse` (from the model) and `GroundedResponse` (plus code-checked fields) |
| 5-6 | Ask for the PDF files, load them and split them into chunks |
| 7 | Embed and sync the persistent Chroma index |
| 8-9 | Prompts, Qwen chain, and the conversational bot (rewrite -> retrieve -> generate -> check) |
| 10-11 | Example questions (edit them to match your PDFs) and an interactive chat loop |

---

## 6. Common problems and tuning

| Symptom | Likely cause | Fix |
|---|---|---|
| Answers ignore the documents | Retrieval missed the right chunk | Raise `k`, adjust chunk size/overlap, check parsing |
| "I don't know" but the answer is in the file | Chunk split the answer apart, or poor PDF text extraction (scanned PDFs need OCR) | More overlap, inspect `splits` |
| Invented details | Model not constrained | Stronger system prompt, lower temperature, bigger model |
| Follow-up questions retrieve junk | No question rewriting | Use the rewrite step (included) |
| Citations marked "NOT found in context" | Model paraphrased or invented the quote | Check the listed source yourself; try a larger model |
| Poor results on non-English documents | English-only embedding model | Switch `EMBED_MODEL` to a multilingual one |
| Instructions ignored on long chats or many chunks | Prompt exceeds the context window | Raise `NUM_CTX`, lower `TOP_K` or chunk size |
| "Summarize everything" gives a partial answer | Only top-k chunks are seen (section 2.7) | Ask specific questions, or summarize per document |
| Structured output falls back to plain text | Small model, or output cut off | Larger model, simpler schema |
| Slow | Large model on CPU | Smaller model, GPU, fewer chunks |

Documents are also a way in for **prompt injection**: text inside a file ("ignore previous instructions...") may be followed by the model. The prompt tells the model to treat context as reference material only, which helps but is not a guarantee, so only index documents you trust.

### Beyond the basics
- **Hybrid search** (vector + keyword/BM25) for exact terms like IDs and names
- **Re-ranking** the top 20 results with a cross-encoder, then keeping the best 4
- **Metadata filters** (by file, date, author)
- **Evaluation**: build a set of questions with known answers and measure retrieval hit-rate and answer correctness before changing parameters
