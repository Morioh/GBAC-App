# TPP Labs — GBAC Policy Intelligence

A Streamlit application for exploring and querying meeting notes from DC DOEE's Green Building Advisory Council (GBAC), built by the Georgetown McCourt School's Tech & Public Policy (TPP) Lab.

The app combines hybrid semantic/keyword search over a ChromaDB vector store with optional LLM-powered chat, summarization, and cross-meeting comparison (via Anthropic Claude or OpenAI). It's aimed at policy researchers and lab staff who need to quickly find, summarize, and compare what was discussed across dozens of GBAC meetings without manually re-reading PDFs.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-FF4B4B)
![License](https://img.shields.io/badge/license-MIT-green)

## Key Features

- **Hybrid retrieval** — combines ChromaDB semantic search with a BM25 keyword index, merged via reciprocal rank fusion ([gbac_rag_app.py:113-179](gbac_rag_app.py#L113-L179))
- **AI chat with citations** — ask natural-language questions and get answers grounded in retrieved meeting excerpts, with source passages shown alongside (Claude or GPT, whichever key is configured)
- **Graceful no-key fallback** — without an API key configured, the Chat tab still returns the most relevant raw excerpts instead of failing
- **Meeting summarization** — generate a structured summary (topics, decisions, action items, concerns) for any single meeting date
- **Cross-meeting comparison** — track how a topic evolved across 2–5 selected meetings
- **Document browsing** — list all meeting dates/section types, filter chunks by section name, or search within a custom date range
- **Persistent vector store** — ChromaDB data lives on disk in `chroma_db/` and is loaded once per session via `st.cache_resource`

## Tech Stack / Dependencies

| Component | Library | Purpose |
| --- | --- | --- |
| UI framework | `streamlit>=1.28.0` | Web app / interactive UI |
| Vector database | `chromadb>=0.4.0` | Persistent semantic search index |
| Embeddings | `sentence-transformers>=2.2.0` | `all-MiniLM-L6-v2` model for query/document embeddings |
| Keyword search | `rank-bm25>=0.2.2` | BM25 scoring, fused with semantic scores |
| LLM (optional) | `anthropic>=0.7.0` | Claude-powered chat/summarization/comparison |
| LLM (optional) | `openai>=1.0.0` | GPT-powered chat/summarization/comparison (fallback provider) |
| Numerics | `numpy>=1.24.0` | Score fusion/normalization |
| Env loading (optional, unpinned) | `python-dotenv` | Loads `.env` for API keys, imported with a try/except so it's not a hard dependency — see [gbac_rag_app.py:9-14](gbac_rag_app.py#L9-L14) |

## Architecture / Project Structure

The app is a single Streamlit script with two layers:

- **Backend** (`RAGDatabase`, `HybridRetriever`, `rag_answer`, `summarize_meeting`, `compare_meetings`) — wraps ChromaDB + BM25 retrieval and, when an API key is available, sends retrieved context to Claude/OpenAI to produce grounded answers.
- **UI** (`render_chat_tab`, `render_analysis_tab`, `render_documents_tab`, `main`) — a three-tab interface (Chat / Analysis / Documents) styled with injected custom CSS, with tab and chip state driven by URL query params.

Important: **the app only reads an existing `chroma_db/` vector store** — it does not ingest `gbac_chunks.json` at runtime ([`load_system`](gbac_rag_app.py#L368-L381) just opens `./chroma_db` if present and non-empty). There is no ingestion script in this repo; see [Known Limitations](#roadmap--known-limitations).

```
.
├── gbac_rag_app.py           # Main Streamlit app: RAG backend + Chat/Analysis/Documents UI (entry point)
├── requirements.txt          # Python dependencies
├── .env                      # API keys (ANTHROPIC_API_KEY / OPENAI_API_KEY), not committed
├── config_template.py        # Sample config values (persist dir, defaults) — NOT imported by the app; defaults are hardcoded inline instead
├── setup.sh / setup.bat      # Create venv + install dependencies
├── run.sh / run.bat          # Activate venv (if present) and launch Streamlit
├── chroma_db/                # Persisted ChromaDB store (auto-loaded at startup; must already exist and be non-empty)
├── gbac_chunks.json          # Flattened chunk records (text/file/date/section) — the format the app's readers expect
├── processed_chunks.json     # Intermediate chunk export (chunk_id/metadata/text/token_count) from the notebook pipeline
├── RAG_Setup.ipynb           # Google Colab prototype notebook for the search/RAG pipeline (reference only; not wired to the app)
├── tpp_lab_interface.html    # Standalone static HTML mockup of the interface (design reference, not served by the app)
├── Sample Questions*.rtf     # Example questions used for manual testing
├── QUICKSTART.md             # Condensed quick-start guide
└── venv/                     # Local virtual environment (not committed)
```

## Getting Started

### Prerequisites

- Python 3.9+ (the working `venv/` in this project was created with 3.12.2)
- pip

### Installation

```bash
# From the project root
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Or use the bundled setup script, which does the same thing:

```bash
./setup.sh          # macOS/Linux
setup.bat           # Windows
```

### Environment Setup

AI-powered features (Chat answers, Meeting Analysis) require an API key from **either** Anthropic or OpenAI. Without one, Chat still works in excerpt-only mode. Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here
# or
OPENAI_API_KEY=sk-your-key-here
```

Keys are loaded automatically via `python-dotenv` if it's installed ([gbac_rag_app.py:9-14](gbac_rag_app.py#L9-L14)); otherwise export them as real environment variables before launching. If both keys are set, Anthropic is preferred ([`get_api_config`](gbac_rag_app.py#L384-L392)).

> **TODO:** `python-dotenv` is used but not listed in `requirements.txt` — add it there if you want `.env` loading guaranteed on a fresh install.

You also need a populated `chroma_db/` directory in the project root (see [Known Limitations](#roadmap--known-limitations) for how this repo currently expects that to be built).

## Usage

Run the app:

```bash
streamlit run gbac_rag_app.py
```

or:

```bash
./run.sh     # macOS/Linux — activates venv if present, then runs Streamlit
run.bat      # Windows
```

Streamlit will print a local URL, typically:

```
Local URL: http://localhost:8501
```

Once open, the app has three tabs:

- **💬 Chat** — ask a question (e.g. *"What key decisions were made in recent GBAC meetings?"*); with an API key configured you get a cited, synthesized answer plus the underlying source passages, otherwise you get the top matching passages directly.
- **📊 Analysis** *(requires an API key)* — summarize a single meeting by date, or compare how a topic evolved across 2–5 selected meetings.
- **📂 Documents** — list all meeting dates/section types, search chunks by section name, or search within a custom date range (dates must be entered as `Month DD, YYYY`, e.g. `February 5, 2025`).

To share the app on your local network instead of just `localhost`:

```bash
streamlit run gbac_rag_app.py --server.address 0.0.0.0
```

## Configuration

Runtime configuration is read from environment variables; there is no config file actually wired into the app (`config_template.py` is a reference of hardcoded defaults, not a loaded settings file).

| Variable | Required | Description |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | No | Claude API key. Preferred provider if both keys are set. Used for chat answers, summaries, and comparisons via `claude-sonnet-4-20250514` ([gbac_rag_app.py:234](gbac_rag_app.py#L234)). |
| `OPENAI_API_KEY` | No | OpenAI API key, used as fallback provider. Chat tries `gpt-4o` → `gpt-4` → `gpt-3.5-turbo` in order ([gbac_rag_app.py:243](gbac_rag_app.py#L243)); Analysis tab uses `gpt-4o` directly. |

If neither key is set, `render_analysis_tab` disables itself with a prompt to set one, and Chat falls back to excerpt-only mode ([gbac_rag_app.py:1374-1376](gbac_rag_app.py#L1374-L1376)).

## Roadmap / Known Limitations

- **No ingestion path in this repo.** The app only reads an already-built `chroma_db/` ([`load_system`](gbac_rag_app.py#L368-L381)); no script here populates it from `gbac_chunks.json`. The only ChromaDB-writing logic that exists is illustrative code inside `RAG_Setup.ipynb`, written for a Google Colab session (`google.colab.files` uploads, a differently-named collection `gbac_meetings`, etc.) rather than this app's `gbac_chunks` collection. TODO: add a standalone ingestion script if the database needs to be rebuilt from scratch.
- **`config_template.py` is unused.** It documents defaults (embedding model, top-k bounds, example queries) that are actually hardcoded inline in `gbac_rag_app.py` — it isn't imported anywhere.
- **`QueryAnalyzer` is defined but unused** ([gbac_rag_app.py:39-76](gbac_rag_app.py#L39-L76)) — it detects attendance/aggregation/metrics query intents but nothing in the current app calls it.
- No automated test suite exists in this repo.

## Contributing

1. Add new features directly in `gbac_rag_app.py` (the app is a single-file Streamlit script).
2. Update `requirements.txt` if you introduce a new dependency.
3. Test manually against your data (`Sample Questions.rtf` / `Sample Questions - V2.rtf` have example queries used previously).
4. Update this README if behavior, structure, or configuration changes.

## License

Released under the [MIT License](LICENSE).

## Acknowledgments

Built for the Georgetown McCourt School of Public Policy's [Tech & Public Policy Lab](https://mccourt.georgetown.edu/tech-and-public-policy/), for research use over DC DOEE's Green Building Advisory Council meeting notes.
