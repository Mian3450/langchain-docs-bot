# 🦜 LangChain Documentation Assistant Bot

> **AI assistant that answers questions about LangChain documentation. Built with RAG, Python, and ChromaDB.**

A Telegram bot that answers questions about the LangChain framework using a
Retrieval-Augmented Generation pipeline over the official LangChain docs. Every
answer cites its sources with clickable links back to GitHub.

This is a portfolio/demo project showcasing a clean, typed, tested, dockerized
RAG implementation.

## Demo

<!-- Replace with a real screenshot/GIF: docs/screenshots/demo.gif -->
![Bot demo](docs/screenshots/demo.gif)

```
You: What is LCEL and why use it?

Bot: LangChain Expression Language (LCEL) is a declarative way to compose
chains using the pipe operator [1]. Chains built with LCEL get streaming,
async, batch, and parallel execution support for free [2].

📚 Sources:
[1] LCEL concepts — github.com/langchain-ai/langchain/blob/master/docs/docs/concepts/lcel.mdx
[2] LCEL how-to    — github.com/langchain-ai/langchain/blob/master/docs/docs/how_to/lcel.mdx
```

## Architecture

```
Telegram ─► aiogram (rate-limit + logging middleware)
                │  answer_question(text)
                ▼
         ┌──────────────────────────────────────────┐
         │  RAG pipeline                              │
         │  retrieve (Chroma, similarity/MMR, top_k)  │
         │      │                                     │
         │      ▼                                     │
         │  generate (Groq Llama 3.1, [N] citations)  │
         └──────────────────────────────────────────┘
                │
                ▼
   answer + 📚 numbered source links

Offline ingestion (run once):
  GitHub docs ─► strip MDX ─► chunk ─► embed (bge-small, local ONNX) ─► ChromaDB
```

> **Note on the doc source:** LangChain moved its documentation out of the
> `langchain-ai/langchain` monorepo into the unified `langchain-ai/docs` repo.
> The bot ingests the open-source docs from `src/oss` on the `main` branch
> (configurable via `DOCS_REPO` / `DOCS_REF` / `DOCS_SUBDIR`).

See [docs/architecture.md](docs/architecture.md) for the full diagram and module map.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ (tested on 3.12) |
| Bot framework | aiogram 3.x |
| RAG framework | LangChain 0.3 |
| Vector store | ChromaDB 1.x (local, persistent) |
| Embeddings | FastEmbed `BAAI/bge-small-en-v1.5` (local ONNX, free) — swappable to OpenAI |
| LLM | Groq `llama-3.1-8b-instant` (free tier) — swappable to OpenAI `gpt-4o-mini` |
| Doc source | `langchain-ai/docs` GitHub repo (`src/oss`) |
| Quality metrics | DeepEval (Answer Relevancy, Faithfulness, Contextual Precision) |
| Testing | pytest, pytest-asyncio |
| Deployment | Docker + docker-compose |
| Config | pydantic-settings + `.env` |
| Logging | structlog |
| Lint / types | ruff, mypy `--strict` |

## Quick start

### 1. Configure

```bash
cp .env.example .env
# Edit .env: set TELEGRAM_BOT_TOKEN (from @BotFather) and OPENAI_API_KEY
```

### 2. Run with Docker (recommended)

```bash
# Populate the vector store once (downloads docs, embeds — costs a few cents):
docker compose run --rm bot python -m scripts.ingest_docs

# Start the bot:
docker compose up -d

# Then open Telegram and message your bot.
```

The `./data` directory is volume-mounted, so the ChromaDB collection persists
across restarts.

### 3. Run locally (without Docker)

```bash
# Using uv (fast); falls back to pip if you prefer.
uv venv && uv pip install -e ".[dev]"

# Ingest docs (once), then start the bot:
python -m scripts.ingest_docs
python -m src.main
```

> Tip: `python -m scripts.ingest_docs --limit 10` does a quick partial ingest
> for a fast end-to-end smoke test.

## Configuration

All settings come from environment variables / `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | — | **Required.** Bot token from @BotFather |
| `GROQ_API_KEY` | — | **Required when `LLM_PROVIDER=groq` (default).** Free tier at https://console.groq.com |
| `OPENAI_API_KEY` | — | Required only when a provider is set to `openai`, or for `scripts/eval_rag.py` |
| `LLM_PROVIDER` | `groq` | `groq` or `openai` |
| `EMBEDDING_PROVIDER` | `fastembed` | `fastembed` (local ONNX) or `openai` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Must match the selected provider |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Must match the selected provider |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `200` | Chunking parameters |
| `TOP_K` | `5` | Chunks retrieved per query |
| `RETRIEVAL_STRATEGY` | `similarity` | `similarity` or `mmr` |
| `RATE_LIMIT_PER_MINUTE` | `10` | Per-user message limit |
| `LOG_FORMAT` | `console` | `console` (dev) or `json` (prod) |

## Testing

Unit tests run **fully offline** — no API keys, no network. The LLM is injected
as a fake and retrieval uses a deterministic local embedding.

```bash
pytest            # all tests
ruff check .      # lint
mypy --strict src # types
```

## Evaluation (DeepEval)

> ⚠️ **Evaluation is not a unit test.** DeepEval uses an LLM as a judge, so it
> **requires `OPENAI_API_KEY`**, **consumes tokens** (~$0.10–0.50 for a full
> run), is **slow** (~1–2 min/question), and produces **non-deterministic**
> scores that vary a few percent between runs.

```bash
uv pip install -e ".[eval]"      # install the eval extra
python -m scripts.eval_rag --smoke   # quick sanity check (3 questions)
python -m scripts.eval_rag           # full dataset -> eval_report.md
```

Metrics measured against `tests/eval_dataset.json` (18 hand-crafted Q/A pairs):

- **Answer Relevancy** — does the answer address the question?
- **Faithfulness** — does the answer stick to retrieved context (no hallucinations)?
- **Contextual Precision** — are retrieved chunks actually relevant?

### Results

_Run `python -m scripts.eval_rag` to populate `eval_report.md`. Example shape:_

| Metric | Mean |
|---|---|
| Answer Relevancy | _e.g. 0.9x_ |
| Faithfulness | _e.g. 0.8x_ |
| Contextual Precision | _e.g. 0.8x_ |

Targets: Faithfulness > 0.8, Answer Relevancy > 0.8.

## Project layout

```
src/
  config.py              # typed settings
  main.py                # entry point
  bot/                   # aiogram handlers, middleware, keyboards
  rag/                   # ingestion, embeddings, vectorstore, retriever,
                         #   generator, pipeline
  utils/                 # logging, source formatting
scripts/
  ingest_docs.py         # one-off: scrape + embed docs
  eval_rag.py            # DeepEval metrics -> eval_report.md
tests/                   # offline unit tests + eval_dataset.json
docs/                    # architecture.md, screenshots
```

## License

MIT — see [LICENSE](LICENSE).
