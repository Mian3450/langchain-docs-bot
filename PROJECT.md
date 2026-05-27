# LangChain Documentation Assistant Bot

A Telegram bot that answers user questions about LangChain documentation using a RAG (Retrieval-Augmented Generation) pipeline.

This is a portfolio/demo project showcasing production-grade RAG implementation.

## Goals

- Demonstrate a complete RAG pipeline (ingestion → chunking → embeddings → retrieval → generation)
- Show clean Python project structure (typed, tested, dockerized)
- Provide answer quality metrics via DeepEval
- Be reproducible: any user with API keys can clone and run

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Bot framework | aiogram 3.x |
| RAG framework | LangChain |
| Vector store | ChromaDB (local, persistent) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` (configurable) |
| Document loader | `langchain-community` document loaders + custom GitHub scraper |
| Quality metrics | DeepEval (Answer Relevancy, Faithfulness, Context Precision) |
| Testing | pytest, pytest-asyncio |
| Deployment | Docker + docker-compose |
| Config | pydantic-settings + .env |
| Logging | structlog |

## Project Structure

```
langchain-docs-bot/
├── README.md
├── PROJECT.md
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Settings via pydantic-settings
│   ├── main.py                # Entry point: starts the bot
│   │
│   ├── bot/
│   │   ├── __init__.py
│   │   ├── handlers.py        # aiogram message handlers
│   │   ├── keyboards.py       # Inline keyboards (optional)
│   │   └── middleware.py      # Rate limiting, logging
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingestion.py       # Document loading + chunking
│   │   ├── embeddings.py      # Embedding model wrapper
│   │   ├── vectorstore.py     # ChromaDB wrapper
│   │   ├── retriever.py       # Retrieval logic + reranking
│   │   ├── generator.py       # Answer generation with citations
│   │   └── pipeline.py        # End-to-end RAG orchestration
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py         # structlog config
│       └── source_formatter.py # Format source citations
│
├── scripts/
│   ├── ingest_docs.py         # Run once: scrape + embed LangChain docs
│   └── eval_rag.py            # Run DeepEval metrics on test set
│
├── data/
│   ├── raw/                   # Downloaded markdown files
│   └── chroma/                # ChromaDB persistent storage
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── eval_dataset.json      # Hand-crafted Q/A pairs for DeepEval
│
└── docs/
    ├── architecture.md        # Architecture overview
    └── screenshots/           # Bot screenshots for README
```

## Detailed Requirements

### 1. Configuration (`src/config.py`)

Use `pydantic-settings` to load from `.env`:

```python
class Settings(BaseSettings):
    telegram_bot_token: SecretStr
    openai_api_key: SecretStr
    
    # Model configuration
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    
    # RAG parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    
    # Paths
    chroma_persist_dir: str = "./data/chroma"
    raw_docs_dir: str = "./data/raw"
    
    # Bot behavior
    max_message_length: int = 4000
    rate_limit_per_minute: int = 10
```

### 2. Document Ingestion (`src/rag/ingestion.py`)

Approach: scrape LangChain documentation from the **official GitHub repository** (`langchain-ai/langchain` → `/docs/docs/` folder) rather than scraping the website. This avoids JavaScript rendering issues and respects the project's open-source nature.

Steps:
1. Clone the `langchain-ai/langchain` repo (shallow, depth=1) into a temp dir, OR use GitHub Contents API for selected paths
2. Walk `/docs/docs/` recursively and collect both `.md` and `.mdx` files
3. **Preprocess MDX files** (critical — see below)
4. Parse YAML frontmatter (if present) for `title` and `description`
5. Build metadata for each document:
   - `source_path` — relative path inside the repo (e.g. `docs/concepts/rag.mdx`)
   - `source_url` — full GitHub URL: `https://github.com/langchain-ai/langchain/blob/master/{source_path}`
   - `title` — from frontmatter, else first `# heading`, else filename
   - `section` — top-level docs folder (`concepts`, `how_to`, `tutorials`, etc.)
6. Chunk using `RecursiveCharacterTextSplitter` with markdown-aware separators (`["\n## ", "\n### ", "\n\n", "\n", ". ", " "]`)
7. Filter out chunks shorter than 100 characters (nav fragments, empty sections)

#### MDX Preprocessing

LangChain docs are mostly `.mdx` (Markdown + JSX), which contains JSX imports, component tags, and other syntax that `UnstructuredMarkdownLoader` will pass through as garbage into chunks. Implement a dedicated preprocessor before chunking:

```python
def preprocess_mdx(content: str) -> str:
    """Strip MDX-specific syntax to leave clean Markdown."""
    # 1. Remove import statements: `import X from 'Y';`
    # 2. Remove export statements: `export const X = ...;`
    # 3. Remove JSX component tags: `<Tabs>`, `</Tabs>`, `<TabItem value="..." label="...">`, etc.
    #    Self-closing: `<ComponentName ... />`
    #    Paired: `<ComponentName ...>...</ComponentName>` — keep inner text, drop tags
    # 4. Remove JSX expressions: `{someVariable}` inside text (but NOT inside code blocks)
    # 5. Collapse 3+ blank lines into 2
    # 6. Preserve fenced code blocks (```...```) and inline code (`...`) AS-IS
```

Implementation hint: use regex with care around fenced code blocks — split content by code fences first, apply stripping only to non-code segments, then rejoin. A naive global regex will mutilate code samples.

Add `tests/test_mdx_preprocess.py` with 5–6 representative fixtures (Tabs example, import-heavy file, file with JSX inside prose, file that's pure markdown).

Target: ~500–2000 chunks total after filtering.

### 3. Embeddings & Vector Store (`src/rag/embeddings.py`, `vectorstore.py`)

- Use `OpenAIEmbeddings` from `langchain-openai`
- Use `Chroma` from `langchain-chroma`
- Persistent storage in `./data/chroma`
- Add a singleton-style factory so vector store is loaded once per process

**Version pinning is mandatory** for LangChain ecosystem — the API breaks between minor versions. Pin these in `requirements.txt`:

```
langchain==0.3.*
langchain-openai==0.2.*
langchain-chroma==0.1.*
langchain-community==0.3.*
chromadb==0.5.*
```

Use compatible release operator (`~=`) or exact pins. Avoid open-ended `langchain>=0.3`.

### 4. Retriever (`src/rag/retriever.py`)

- Default: similarity search with `top_k=5`
- Bonus: implement an MMR (Maximum Marginal Relevance) variant for diversity
- Return both content and metadata (for source citations)

### 5. Generator (`src/rag/generator.py`)

**Source ID mapping**: each retrieved chunk gets a short numeric ID (`[1]`, `[2]`, etc.) before being passed to the LLM. The LLM cites by ID. The bot layer then maps IDs back to clickable GitHub URLs when formatting the reply.

Prompt template:

```
You are a helpful assistant specialized in the LangChain framework.
Answer the user's question using ONLY the provided context.
If the context doesn't contain the answer, say so honestly — do not invent facts.
Cite the sources you used by their numeric ID, like [1] or [2, 3].
Place citations at the end of sentences they support.

Context:
[1] (from {title_1})
{chunk_1}

[2] (from {title_2})
{chunk_2}

...

Question: {question}

Answer:
```

Generator returns:

```python
@dataclass
class AnswerResult:
    text: str                # Raw LLM output with [N] citations
    sources: list[Source]    # Ordered list, indices match citation IDs
    latency_ms: float

@dataclass
class Source:
    id: int                  # The [N] used in the prompt
    title: str
    source_path: str
    source_url: str          # GitHub blob URL
```

Use `ChatOpenAI` with `gpt-4o-mini` and temperature 0.2 (low for factual accuracy).

### 6. Pipeline (`src/rag/pipeline.py`)

A single async function:

```python
async def answer_question(question: str) -> AnswerResult:
    """
    Returns:
        AnswerResult with .text (str), .sources (list[Source]), .latency_ms (float)
    """
```

This is the only thing the bot handler calls.

### 7. Bot Handlers (`src/bot/handlers.py`)

Commands:
- `/start` — welcome message with usage hint
- `/help` — explain what the bot does + examples
- `/about` — info about the demo project (link to GitHub repo)

Default behavior: any text message → `answer_question(text)` → formatted reply with sources.

**Citation rendering**: the LLM output contains `[N]` markers. The handler keeps those `[N]` inline in the answer text (don't strip them) and appends a numbered source list mapping each `[N]` to a clickable GitHub URL. Use Telegram MarkdownV2 or HTML parse mode for links.

Example reply:
```
LangChain Expression Language (LCEL) is a declarative way to compose chains [1].
It supports streaming, async, and parallel execution out of the box [2].

📚 Sources:
[1] LCEL concepts — github.com/langchain-ai/langchain/blob/master/docs/concepts/lcel.mdx
[2] LCEL how-to — github.com/langchain-ai/langchain/blob/master/docs/how_to/lcel_cheatsheet.mdx
```

Handle long answers (>4000 chars) by splitting into multiple messages, keeping the sources block in the final message.

### 8. Middleware (`src/bot/middleware.py`)

- Rate limiting: max 10 questions per user per minute (in-memory dict, no Redis needed for demo)
- Logging: log every question + response time + user ID

### 9. Testing (`tests/`)

- `test_chunking.py` — chunks have correct size and overlap
- `test_retrieval.py` — known question retrieves expected document
- `test_generation.py` — mock LLM call, verify prompt structure and source extraction
- All tests should be runnable without API keys (use mocks for OpenAI)

### 10. Evaluation (`scripts/eval_rag.py`)

Use DeepEval to measure:
- **Answer Relevancy** — does the answer address the question?
- **Faithfulness** — does the answer stick to retrieved context (no hallucinations)?
- **Contextual Precision** — are retrieved chunks actually relevant?

Test dataset (`tests/eval_dataset.json`): 15–20 hand-crafted Q/A pairs covering different LangChain topics (chains, agents, retrievers, callbacks, etc.).

Output: a markdown report with scores per metric, saved to `eval_report.md`.

**Important — DeepEval is not a unit test suite.** DeepEval uses an LLM (by default GPT-4) as a judge to score each metric, which means:
- Running eval **requires API keys** and consumes tokens (a 20-question run costs roughly $0.10–0.50 depending on chunk sizes)
- Eval is **non-deterministic** — scores will vary between runs by a few percent
- Eval is **slow** (1–2 min per question)

Therefore:
- `pytest tests/` must run fully offline with mocked LLM/embeddings (existing requirement from section 9)
- `python scripts/eval_rag.py` is a separate command, run manually before releases or after major prompt changes
- README must clearly explain the difference between unit tests and eval, and document expected cost per eval run

Add a sanity-check mode to the eval script: `--smoke` flag that runs only 3 questions for a quick "did I break the pipeline" check.

### 11. Docker

- `Dockerfile`: Python 3.11-slim base, install requirements, copy src, run `python -m src.main`
- `docker-compose.yml`: one service for the bot, volume mount for `./data` (so Chroma persists across restarts)

### 12. README.md

Must include:
- Hero section: "AI assistant that answers questions about LangChain documentation. Built with RAG, Python, and ChromaDB."
- Demo: animated GIF or 2–3 screenshots
- Quick start: clone, set up `.env`, `docker-compose up`, talk to bot
- Architecture diagram (use ASCII art or link to docs/architecture.md)
- Tech stack table
- Evaluation results section (DeepEval scores)
- License: MIT

## Implementation Order

Suggest building in this order to test early:

1. Project scaffolding + `.env.example` + `pyproject.toml` + pinned `requirements.txt`
2. `config.py` + basic structlog setup
3. **MDX preprocessor** (`src/rag/ingestion.py::preprocess_mdx`) + its tests first — single contained unit, easy to verify
4. Full ingestion script using the preprocessor (test with 10 docs, then full scrape)
5. Vector store + embeddings (verify persistence)
6. Retriever (test against known queries)
7. Generator with `[N]` citation IDs (test the full RAG chain)
8. Bot handlers with `[N]` → URL mapping (test in real Telegram)
9. Middleware (rate limiting + logging)
10. Unit tests + DeepEval evaluation
11. Docker + README polish

## Non-Goals

- No multi-language support in v1 (English only)
- No conversation history / multi-turn (each question is independent)
- No fine-tuning, no custom embeddings
- No authentication (it's a public demo)
- No payment / monetization logic

## Acceptance Criteria

The project is complete when:
- [ ] User can clone repo, fill `.env`, run `docker-compose up`, and chat with the bot
- [ ] Ingestion script processes all LangChain docs in under 30 minutes
- [ ] Bot responds within 5 seconds for typical questions
- [ ] All tests pass (`pytest`)
- [ ] DeepEval scores are documented in README (Faithfulness > 0.8, Answer Relevancy > 0.8)
- [ ] README has architecture diagram and screenshots
- [ ] Code passes `ruff check` and `mypy --strict`

## Notes for Claude Code

- Use `uv` if available for faster dependency management; otherwise standard `pip`
- Pin all dependencies in `requirements.txt` (no loose versions)
- Add type hints everywhere (the user's portfolio emphasizes QA quality)
- Use `async/await` consistently in bot and pipeline code
- Add docstrings for every public function/class
- Do NOT commit `.env`, `data/raw/`, `data/chroma/`, `__pycache__/` (add to `.gitignore`)