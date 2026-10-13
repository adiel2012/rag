# Chat with your documents: RAG explained, with a notebook and a Docker app

Ask questions in plain language and get answers taken from your own PDFs, with the file and page each answer came from. This technique is called **Retrieval-Augmented Generation (RAG)**. The repository explains how it works and gives you two working versions of the same pipeline.

| | Notebook | Web app |
|---|---|---|
| What it is | [`rag_chatbot.ipynb`](rag_chatbot.ipynb), run cell by cell | A chat page and a JSON API, in Docker ([`app/`](app)) |
| Best for | Learning how RAG works, step by step | Using it day to day, or calling it from other programs |
| You need | A Google account (it runs on Google Colab) | [Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| Models | Qwen 3.5, running inside the notebook | Qwen or any other Ollama model on your machine, or models from AWS (Amazon Bedrock) |
| Documents | You upload PDFs when asked | A folder on your machine |
| Start here | [Quick start: the notebook](#quick-start-the-notebook) | [Quick start: the web app](#quick-start-the-web-app) |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/rag/blob/main/rag_chatbot.ipynb)

## Contents

- [Quick start: the notebook](#quick-start-the-notebook)
- [Quick start: the web app](#quick-start-the-web-app)
- [Words you will meet](#words-you-will-meet)
- How it works
  - [1. What is RAG?](#1-what-is-rag)
  - [2. The pipeline](#2-the-pipeline)
  - [3. Making it conversational](#3-making-it-conversational)
  - [4. Structured output with Pydantic](#4-structured-output-with-pydantic)
- Using it
  - [5. Running the notebook](#5-running-it-qwen--ollama-on-colab-or-your-own-machine)
  - [6. Common problems and tuning](#6-common-problems-and-tuning)
  - [7. Web app and API (Docker)](#7-web-app-and-api-docker)
- [What is in this repository](#what-is-in-this-repository)

## Quick start: the notebook

You need a Google account and one or two PDF files. Nothing is installed on your computer and nothing costs money.

1. Click the **Open in Colab** button above. The notebook opens in your browser.
2. In the menu choose **Runtime -> Change runtime type -> T4 GPU -> Save**. Do this first: changing it later restarts everything.
3. Click the first grey box (a "code cell") and press **Shift+Enter** to run it. Wait until it finishes, then do the same with the next one, in order, down the page. The first time Colab asks whether to run a notebook that is not from Google, choose "Run anyway".
4. Section 3 downloads the AI model (about 7 GB). It takes several minutes; let it finish.
5. In section 5 a **Choose Files** button appears. Pick your PDFs.
6. Keep running the cells. In section 10, replace the example questions with your own; in section 11 you can chat freely.

Some output looks like an error and is not: see [Messages that look alarming but are normal](#messages-that-look-alarming-but-are-normal).

## Quick start: the web app

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running, and about 16 GB of free disk.

1. Download this repository and put your documents (PDF, DOCX, CSV, TXT, MD) in its `data/` folder.
2. Open a terminal in the repository folder and run:

   ```bash
   docker compose -f docker-compose.local.yml up --build
   ```

3. Open <http://localhost:8000>. The first start downloads the models, which takes several minutes; the page shows the progress and enables the question box when it is ready.

Everything else (a GPU, other models, AWS, a different documents folder, the API) is in [section 7](#7-web-app-and-api-docker).

## Words you will meet

| Word | Meaning |
|---|---|
| **LLM** (large language model) | The AI model that reads text and writes the answer. Here: Qwen. |
| **Prompt** | The text given to the model: instructions, passages from your documents, and your question. |
| **Token** | The unit models count text in. One token is roughly three quarters of an English word. |
| **Context window** | The maximum number of tokens the model can read at once. Whatever does not fit is cut off. |
| **Hallucination** | An answer that sounds right but that the model made up. |
| **Chunk** | A piece of a document, a few paragraphs long. Documents are cut into chunks so that the relevant passage can be found. |
| **Embedding** | A list of numbers that represents the meaning of a text. Texts with similar meaning have similar numbers. |
| **Vector store** | A database of embeddings that can find the chunks closest in meaning to a question. Here: Chroma. |
| **Retrieval** | Finding the chunks relevant to a question. The "R" in RAG. |
| **Top-k** | How many chunks are retrieved and given to the model (k = 6 in the notebook). |
| **Ollama** | The program that runs the model on the machine, instead of calling a paid online service. |
| **GPU** | A graphics processor. Models run many times faster on it than on a normal processor (CPU). "T4" is the GPU Colab offers for free. |
| **Schema** | A fixed list of fields an answer must have, like a form. Defined here with Pydantic. |

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
- **Language**: small models such as `all-MiniLM-L6-v2` are trained on English. The notebook uses `BAAI/bge-m3`, a larger multilingual model that also handles Spanish and mixed-language documents.
- **Maximum input length**: text beyond the model's limit is silently cut off, so chunks must fit (MiniLM reads about 256 tokens, roughly 1000 characters of English).

### 2.4 Vector store
A database (Chroma, FAISS, Qdrant, Pinecone, pgvector) that stores each chunk's vector and finds the nearest ones to a query vector quickly, usually using cosine similarity. It also stores metadata so you can cite sources and filter results.

### 2.5 Retrieve
Embed the question, fetch the **top-k** closest chunks (k = 3-6 is common). Raising k improves recall but adds noise and cost.

**Hybrid search.** Embeddings match meaning but can miss exact terms that carry little meaning on their own: field names, codes, product-specific words ("Shipper tab", "PO-1234"). Keyword search (BM25) is the opposite: it matches exact words and ignores meaning. The notebook runs both and merges the two rankings (reciprocal rank fusion), which is more robust than either alone.

**Neighbor expansion.** PDFs are loaded one page at a time, so a numbered procedure that runs over a page break ends up in separate chunks, and the best-matching chunk may hold only the first half of the answer. For the top hits the notebook also adds the chunk that follows, and passes the chunks to the model in document order.

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

   This step has its own failure mode: a small model may *answer* the question from its general knowledge instead of rewriting it, and the search then runs on an invented answer. The notebook guards against this in three ways: the history is given as quoted text rather than as chat turns, a rewrite that looks like an answer (much longer, or no longer a question) is discarded, and the search always includes the user's original words too.

The notebook implements both. It also trims history to the last few turns so the prompt does not grow forever.

---

## 4. Structured output with Pydantic

Free-text answers are hard for programs to use. With **Pydantic** you define a schema:

```python
class RAGResponse(BaseModel):
    answer: str
    citations: list[str]
    source_ids: list[int]     # numbers of the context chunks used
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

This section is about the notebook; the Docker app is in [section 7](#7-web-app-and-api-docker). The model runs inside the notebook's own runtime through Ollama, with no API keys and no calls to an external AI service:

| Component | Choice | Role |
|---|---|---|
| LLM | Qwen 3.5 9B via **Ollama** | Generates the answer |
| Embeddings | `BAAI/bge-m3` (HuggingFace, multilingual) | Text -> vectors |
| Keyword search | BM25 (`rank_bm25`) | Finds exact words; merged with the vector results |
| Vector DB | **Chroma** (persisted in `chroma_db/`) | Stores and searches chunks |
| Orchestration | **LangChain** | Glues the steps together |
| Validation | **Pydantic** | Typed, validated answers |

LangChain is the glue, not the whole system: you still need the model, the embedding model, the vector DB and the document parsers. For a production system, many teams use the underlying SDKs directly for more control and easier debugging; LangChain is excellent for getting a working pipeline quickly.

### Why Qwen 3.5 9B

Qwen 3.5's small series (0.8B, 2B, 4B, 9B) is the newest Qwen line made for edge hardware; the newer 3.6 and 3.8 releases are only published at 27B and up. The 9B is the largest of the small series. It is about a 7 GB download, so it fits the 16 GB of a Colab T4 GPU with room for the context window, while the next size up (27B) does not fit.

Qwen 3.5 has a "thinking" mode that reasons before answering. The notebook turns it off (`THINKING = False`): it adds latency and is not needed to answer from retrieved text.

### Setup on Google Colab (T4)

Follow [Quick start: the notebook](#quick-start-the-notebook). Two details it leaves out: section 3 installs and starts Ollama for you, and DOCX, CSV, TXT and MD files put in `data/` through Colab's Files panel are indexed along with the PDFs.

Colab sessions are temporary: after a runtime reset the model, your uploaded files and the index are gone and the cells must be run again. Note also that on Colab your documents are uploaded to Google's servers, so "local" means "no AI API", not "never leaves your computer".

### Messages that look alarming but are normal

| You see | Where | What it means |
|---|---|---|
| Red lines starting with `ERROR: pip's dependency resolver...` that mention `google-colab` or `google-adk` | Section 1 | Two packages that come with Colab wanted older versions of some libraries. The notebook does not depend on that. The install worked: continue. |
| `WARNING: systemd is not running` | Section 3 | Colab cannot run Ollama as a background service, so the notebook starts it itself. Ignore it. |
| A cell that seems stuck for minutes | Section 3 | The 7 GB model is downloading. Percentages are printed as it goes. |
| A cell that waits and does nothing | Section 5 | It is waiting for you to click **Choose Files**. |
| `RuntimeError: This Colab runtime has no GPU` | Section 3 | A real error, with a simple fix: Runtime -> Change runtime type -> T4 GPU, then run again from the first cell. If T4 GPU was already selected, your free Colab GPU time is used up for now: try again in a few hours. |
| The first answer is slow | Section 10 | The model is being loaded onto the GPU. Later answers are faster. |

### Setup on your own machine

1. Install [Python](https://www.python.org/downloads/) 3.10 or newer (tick "Add python.exe to PATH" on Windows) and the Jupyter extension for VS Code.
2. Install [Ollama](https://ollama.com) (version 0.17.4 or newer for Qwen 3.5).
3. Open `rag_chatbot.ipynb` and run the cells top to bottom. Section 3 starts Ollama if needed and pulls the model.
4. Section 5 opens a file dialog asking for your PDF files and copies them into `data/`. (DOCX, CSV, TXT and MD files placed in `data/` by hand are indexed too.)

On a machine with 8 GB of RAM use `qwen3.5:4b` instead; without a GPU the 9B will be slow.

### Settings

All settings (model, thinking, context window, embedding model, chunk size, top-k) are in the config cell. Re-run the "Load and chunk" and "Embeddings" cells whenever the documents change: the index is synced incrementally, so only new chunks are embedded and chunks from removed files are deleted.

Smaller models are faster but follow the "answer only from context" rule less reliably. For a model without thinking mode (such as `qwen2.5:7b`) set `THINKING = None`.

### What the notebook does

| Sections | Step |
|---|---|
| 1-3 | Install, configure, install/start Ollama and download the model |
| 4 | Pydantic schemas: `RAGResponse` (from the model) and `GroundedResponse` (plus code-checked fields) |
| 5-6 | Ask for the PDF files, load them and split them into chunks |
| 7 | Embed and sync the persistent Chroma index |
| 8-9 | Prompts, Qwen chain, and the conversational bot (rewrite -> retrieve -> generate -> check) |
| 10-11 | Example questions (edit them to match your PDFs) and an interactive chat loop |
| 12 | `inspect()`: show the chunks retrieved for a question, to diagnose wrong answers |

Each section of the notebook says what its cell does, how long it takes and what you should see when it finishes.

---

## 6. Common problems and tuning

| Symptom | Likely cause | Fix |
|---|---|---|
| Not sure why an answer failed | You cannot see what was retrieved | Run `inspect("your question")` (notebook section 12) to see the chunks the model was given |
| Answers ignore the documents | Retrieval missed the right chunk | Raise `k`, adjust chunk size/overlap, check parsing |
| "I don't know" but the answer is in the file | Chunk split the answer apart, or poor PDF text extraction (scanned PDFs need OCR) | More overlap, inspect `splits` |
| Invented details | Model not constrained | Stronger system prompt, lower temperature, bigger model |
| Follow-up questions retrieve junk | No question rewriting | Use the rewrite step (included) |
| Citations marked "NOT found in context" | Model paraphrased or invented the quote, or the quote is under 15 characters or spans two chunks | Check the listed source yourself; try a larger model |
| Poor results on non-English documents | English-only embedding model | Switch `EMBED_MODEL` to a multilingual one |
| Instructions ignored on long chats or many chunks | Prompt exceeds the context window | Raise `NUM_CTX`, lower `TOP_K` or chunk size |
| "Summarize everything" gives a partial answer | Only top-k chunks are seen (section 2.7) | Ask specific questions, or summarize per document |
| Structured output falls back to plain text | Small model, or output cut off | Larger model, simpler schema |
| Slow | Large model on CPU | Smaller model, GPU, fewer chunks |

Documents are also a way in for **prompt injection**: text inside a file ("ignore previous instructions...") may be followed by the model. The prompt tells the model to treat context as reference material only, which helps but is not a guarantee, so only index documents you trust.

### Beyond the basics
- **Re-ranking** the top 20 results with a cross-encoder, then keeping the best 4
- **Metadata filters** (by file, date, author)
- **Evaluation**: build a set of questions with known answers and measure retrieval hit-rate and answer correctness before changing parameters

---

## 7. Web app and API (Docker)

The `app/` folder is the notebook's pipeline as a service: a chat page in the browser and a JSON API. It needs only [Docker Desktop](https://www.docker.com/products/docker-desktop/). The documents come from a folder that is copied into the Docker image when it is built; there is no upload.

There are two ways to run it. They differ only in where the models run:

| | Local | AWS |
|---|---|---|
| Compose file | `docker-compose.local.yml` | `docker-compose.aws.yml` |
| Answering model | Ollama container, default `qwen3.5:4b` | Amazon Bedrock, the model you choose |
| Embeddings | Ollama, `bge-m3` | Bedrock, `amazon.titan-embed-text-v2:0` |
| Your documents and questions | Stay on your machine | Are sent to AWS |
| Cost | Free | Pay per use |
| Needs | About 16 GB of disk (Ollama image 9.4 GB, models 4.5 GB, app 1 GB); a GPU helps a lot | An AWS account with Bedrock access |

### Run it locally

1. Put your documents (PDF, DOCX, CSV, TXT, MD) in the `data/` folder, or point `DOCS_DIR` at the folder that has them (see [The documents folder](#the-documents-folder)).
2. Start it:

   ```bash
   docker compose -f docker-compose.local.yml up --build
   ```

   With an NVIDIA GPU (uses the larger `qwen3.5:9b`):

   ```bash
   docker compose -f docker-compose.local.yml -f docker-compose.gpu.yml up --build
   ```

3. Open <http://localhost:8000>.

**What to expect.** The first start downloads the Ollama image and the models, then reads the documents and builds the index. The page shows each stage (`pulling model`, `indexing`) and enables the question box at `ready`. As a reference, on a 16-core machine without a GPU, two PDF manuals (337 pages, 516 chunks) took about 4 minutes to index, and each answer took 70 to 90 seconds with `qwen3.5:4b`. Later starts skip the downloads and re-embed nothing if the documents are unchanged. A GPU makes answers much faster.

**What the status line means.** While the app starts, the page shows what it is doing next to the title:

| Status | What is happening |
|---|---|
| `starting` | The app is waiting for the Ollama container to answer. |
| `pulling model: qwen3.5:4b: 40%` | The Ollama container on your machine is downloading the model that **writes the answers** (3.3 GB) from the public registry at ollama.com. Nothing is uploaded and no account is used. |
| `pulling model: bge-m3: 40%` | The same, for the second model (1.2 GB). It is the **embedding model**: it turns text into numbers that represent its meaning, which is how the passages relevant to a question are found. The percentage starts again from 0 for this model. |
| `indexing: embedding chunks 200/516` | `bge-m3` is reading your documents, chunk by chunk, to build the search index. |
| `qwen3.5:4b (ollama) · 2 file(s), 516 chunks` | Ready. The question box is enabled. |
| `error: ...` | Startup failed; see [If something goes wrong](#if-something-goes-wrong). |

The percentages come from Ollama, which reports its progress while it downloads; the app passes them on in steps of 10%, and the page asks the app for the status every two seconds. The same lines appear in the app's log. The models are stored in a Docker volume, so they are downloaded only once: they are fetched again only after `down -v` or after deleting Docker's volumes.

**Always start it with `docker compose`.** Starting the image by itself (for example with the Run button in Docker Desktop) gives a container with no Ollama next to it and no published port, so the page cannot be opened.

**Another model.** Copy `.env.example` to `.env` and set `LLM_MODEL` to any model from the [Ollama library](https://ollama.com/library), for example `LLM_MODEL=llama3.2`. It is downloaded on the next start. For a model without a "thinking" mode also set `THINKING=` (empty).

### Run it with AWS models

1. In the AWS console, check that your account can use the model you want in Amazon Bedrock, and that your user or role may call `bedrock:InvokeModel`.
2. Copy `.env.example` to `.env` and fill in the AWS part: `AWS_REGION`, `BEDROCK_LLM_MODEL` (for example `us.anthropic.claude-haiku-5-5`; more examples are in `.env.example`) and your credentials.
3. Put your documents in `data/` (or set `DOCS_DIR`) and start it:

   ```bash
   docker compose -f docker-compose.aws.yml up --build
   ```

4. Open <http://localhost:8000>.

`.env` holds your credentials: it is listed in `.gitignore` and is not copied into the image. Indexing sends the text of your documents to Bedrock once, to compute the embeddings; each question then sends the question and the retrieved passages.

### The documents folder

The documents are copied into the Docker image when it is built, so the image always carries its own documents. The folder is set in one of two ways:

- **In the `Dockerfile`:** by default it is `data/` (the line `COPY data/ /` at the top).
- **As a parameter:** set `DOCS_DIR` to any folder on your machine, in `.env` or on the command line:

  ```bash
  DOCS_DIR=/path/to/my/pdfs docker compose -f docker-compose.local.yml up --build
  ```

  In PowerShell, run `$env:DOCS_DIR = "C:\Users\me\pdfs"` first and then the `docker compose` command.

After adding, changing or removing files, build and start again with the same command (`up --build`). Only new or changed chunks are embedded again. Each mode keeps its own index, so switching between local and AWS does not mix them.

The page lists the files it read under the title. A file marked "no text could be read" is usually a scanned PDF: it holds pictures of pages, not text, and cannot be searched until it has been through OCR.

Because the documents are inside the image, do not push the image to a public registry if they are private.

### Everyday commands

Run them in the repository folder. For AWS mode, replace `docker-compose.local.yml` with `docker-compose.aws.yml`.

| To | Command |
|---|---|
| Start in the background | `docker compose -f docker-compose.local.yml up -d --build` |
| See what the app is doing | `docker compose -f docker-compose.local.yml logs -f app` |
| Stop (keeps the models and the index) | `docker compose -f docker-compose.local.yml down` |
| Stop and delete the downloaded models and the index | `docker compose -f docker-compose.local.yml down -v` |

### The API

The page uses the same API you can call from your own programs. Interactive documentation is at <http://localhost:8000/docs>.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Startup stage (`starting`, `pulling model`, `indexing`, `ready`, `error`), models, number of files and chunks |
| GET | `/api/documents` | The files that were read; `pages: 0` means no text could be read from the file |
| POST | `/api/chat` | Ask a question. Body: `{"question": "...", "session_id": "..."}` |
| DELETE | `/api/sessions/{id}` | Forget a conversation |
| GET | `/api/inspect?q=...` | The chunks retrieved for a question, to see why an answer was wrong |

```bash
curl -s -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" -d '{"question": "How many paid holiday days do employees get?"}'
```

```json
{
  "answer": "Employees get 23 days of paid holiday per year.",
  "citations": [],
  "source_ids": [2],
  "confidence_score": 1.0,
  "follow_up_questions": [],
  "standalone_question": "How many paid holiday days do employees get?",
  "sources": ["handbook.md"],
  "citations_verified": [],
  "session_id": "febe58d2c8b94b9c90618339c9e3aca6"
}
```

The fields are described in section 4. Send the `session_id` with the next question to continue the same conversation; leave it out to start a new one. While the app is still starting, `/api/chat` answers with status 503.

### If something goes wrong

| You see | Cause | Fix |
|---|---|---|
| The page does not open | The app was not started with `docker compose`, or it is still building | Start it with the compose command; check the logs (see Everyday commands) |
| The page stays on `pulling model` | The models are downloading (about 4.5 GB the first time) | Wait; the percentage is shown |
| `error: No Ollama server answered` | The app container is running without the Ollama container | Start both with the compose command |
| `error: No text could be extracted from the files` | The documents folder is empty, or holds only scanned PDFs | Put documents with real text in `data/` (or set `DOCS_DIR`) and run `up --build` again |
| `error: Unable to locate credentials` | AWS mode without credentials | Fill in the AWS keys (or `AWS_PROFILE`) in `.env` |
| An access or validation error from Bedrock | The model is not enabled for your account or Region, or the ID is wrong | Check the model ID and Region in `.env` against the Bedrock console |
| `port is already allocated` | Another program uses port 8000 | Set `PORT=8080` in `.env` and open <http://localhost:8080> |
| Answers take a minute or more | The model is running on the CPU | Use the GPU file, a smaller model (`LLM_MODEL=qwen3.5:2b`), or AWS mode |
| A wrong answer, or "I don't know" | The right passage was not retrieved, or the file has no text | Open `/api/inspect?q=your question` to see what the model was given; see section 6 |
| Answers come without quotes | Small models often leave the optional `citations` field empty | Use a larger model |

### Good to know

- **No login.** The compose files publish the page on this machine only (`127.0.0.1`). Do not expose it to a network as it is.
- **Conversations are kept in memory** and are lost when the app restarts.
- **The server logs every question and every API response** (status, time taken and the response body, cut to 4,000 characters), except the health check. Read them with the logs command above. The logs therefore contain text from your documents; set `LOG_RESPONSE_CHARS=0` in `.env` to turn response logging off.
- **Settings** (`TOP_K`, `CHUNK_SIZE`, `NUM_CTX`, `PORT`, `DOCS_DIR`...) are explained in `.env.example`.
- **Differences from the notebook:** embeddings come from the model provider instead of running inside the app (so the image needs no PyTorch), and the app keeps one conversation per browser session instead of a single one.
- **Tested:** the local mode has been run end to end on CPU. The GPU file and the AWS mode have not yet been run against a real GPU or a real AWS account.
- **Tests:** `pip install -r requirements-dev.txt` and then `pytest`. They use fake models, so they need neither Ollama nor AWS.

---

## What is in this repository

| Path | What it is |
|---|---|
| `README.md` | This file |
| `rag_chatbot.ipynb` | The notebook |
| `app/` | The web app: `config.py` (settings), `providers.py` (Ollama and Bedrock), `ingest.py` (load and chunk), `index.py` (Chroma, BM25, retrieval), `bot.py` (prompts and the conversational bot), `main.py` (the API), `static/index.html` (the chat page) |
| `Dockerfile` | Builds the app image, with the documents inside |
| `docker-compose.local.yml` | Runs the app with Ollama on this machine |
| `docker-compose.gpu.yml` | Addition to the local file for an NVIDIA GPU |
| `docker-compose.aws.yml` | Runs the app with Amazon Bedrock |
| `.env.example` | Every setting, with explanations. Copy it to `.env` to change them |
| `data/` | Default documents folder (its contents are not committed) |
| `tests/` | Tests for the app |
| `requirements.txt`, `requirements-dev.txt` | Python dependencies of the app and of the tests |
