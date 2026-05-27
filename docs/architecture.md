# Architecture

## Overview

The bot is a thin Telegram front-end over a Retrieval-Augmented Generation
(RAG) pipeline. Ingestion is a one-off offline step; query answering happens at
request time.

## Ingestion (offline, run once)

```
GitHub (langchain-ai/docs, src/oss)
        │  Trees API + raw.githubusercontent.com
        ▼
   data/raw/*.mdx ──► preprocess_mdx() ──► RecursiveCharacterTextSplitter
   (strip JSX/imports/:::dirs)      (markdown-aware chunks, drop <100 chars)
                                              │
                                              ▼
                                  OpenAIEmbeddings (text-embedding-3-small)
                                              │
                                              ▼
                                      ChromaDB (persistent, ./data/chroma)
```

Driven by `scripts/ingest_docs.py`. Metadata attached to every chunk:
`source_path`, `source_url` (GitHub blob URL), `title`, `section`.

## Query time (online)

```
Telegram user
     │  message
     ▼
aiogram Dispatcher
     │   ├── LoggingMiddleware      (structlog: user id, latency)
     │   └── RateLimitMiddleware    (in-memory sliding window, N/min)
     ▼
handlers.handle_question
     │  answer_question(text)            ← the single public RAG entry point
     ▼
rag.pipeline
     ├── retriever.retrieve()        → ChromaDB similarity / MMR search (top_k)
     └── generator.generate_answer() → ChatOpenAI (gpt-4o-mini), [N] citations
     ▼
AnswerResult(text, sources, latency_ms)
     │  format_reply(): HTML-escape answer + numbered source links
     │  split_for_telegram(): chunk long replies under max_message_length
     ▼
Telegram reply (answer with [N] markers + 📚 Sources block)
```

## Citation flow

1. `generator._build_context()` assigns each retrieved chunk a numeric ID
   (`[1]`, `[2]`, ...) and builds a parallel `list[Source]`.
2. The LLM is instructed to cite by those IDs.
3. `handlers` keeps the `[N]` markers inline in the answer and appends a
   `📚 Sources` block mapping each ID to a clickable GitHub URL.

This keeps the LLM's job simple (cite a number) while the application owns the
ID → URL mapping, so citations are always real links, never hallucinated.

## Module map

| Module | Responsibility |
|---|---|
| `src/config.py` | Typed settings (pydantic-settings) |
| `src/utils/logging.py` | structlog + stdlib logging setup |
| `src/utils/tls.py` | Enable OS trust store for TLS (corporate proxies) |
| `src/utils/source_formatter.py` | HTML rendering + message splitting |
| `src/rag/ingestion.py` | MDX preprocessing, fetch, load, chunk |
| `src/rag/embeddings.py` | Cached embeddings client |
| `src/rag/vectorstore.py` | Cached persistent Chroma + write helpers |
| `src/rag/retriever.py` | similarity / MMR retrieval |
| `src/rag/generator.py` | Prompt, LLM call, `AnswerResult`/`Source` |
| `src/rag/pipeline.py` | `answer_question()` orchestration |
| `src/bot/handlers.py` | Commands + default question handler |
| `src/bot/middleware.py` | Rate limiting + request logging |
| `src/bot/keyboards.py` | Inline keyboard (repo link) |
| `src/main.py` | Wire-up + long polling |

## Notable design decisions

- **chromadb 1.x** (not 0.x): ships prebuilt wheels with Rust HNSW bindings, so
  no C++ toolchain is needed to install on Windows/slim images. 0.x pulled
  `chroma-hnswlib`, which must compile from source.
- **GitHub source over web scraping**: avoids JS rendering, and the Trees API
  needs a single request while raw file downloads don't count against the API
  rate limit.
- **Source repo moved**: LangChain's docs left the `langchain-ai/langchain`
  monorepo for the unified `langchain-ai/docs` (Mintlify) repo. We ingest
  `src/oss` (the open-source docs) on `main`; repo/ref/subdir are all settings.
- **MDX preprocessing splits on code fences first**: stripping regexes never
  touch fenced/inline code, so example snippets survive intact. In addition to
  imports/exports/JSX tags, it strips Mintlify `:::` container directives.
- **OS trust store (`truststore`)**: enabled at startup so TLS works behind
  corporate proxies with a private root CA; a no-op elsewhere.
- **Single `answer_question()` surface**: the bot layer is decoupled from all
  retrieval/generation internals, which also makes the pipeline trivial to unit
  test and to reuse from the eval script.
