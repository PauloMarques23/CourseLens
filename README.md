# CourseLens — RAG Assistant for Information Retrieval and Text Mining

CourseLens is a Retrieval-Augmented Generation prototype for the MINTRI assignment. It answers
questions over a corpus of IR / RAG / Text-Mining papers and slides, shows grounded answers with
their retrieved sources and citations, and adds text-mining views (keywords, summary, topic
clusters) over the retrieved evidence.

## Features

- Ingestion of PDF, PPTX, TXT and Markdown from `data/raw/`
- Cleaning, chunking and metadata creation
- **Hybrid retrieval**: BM25 (`rank-bm25`) + dense embeddings (BGE + ChromaDB via LlamaIndex), fused with Reciprocal Rank Fusion
- Retrieval-mode selector in the UI (`hybrid` / `dense` / `lexical`)
- Grounded answer generation with Gemini, with source citations
- Local extractive fallback when no Gemini key is configured
- Text mining over the retrieved results (keywords, summary, topic clusters)
- Streamlit demo UI

## Requirements

- Python 3.10 or newer
- ~2 GB free disk space (embedding model + index)
- Internet access on first run (to download the embedding model)
- A Gemini API key

## Installation

From the project folder:

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows (cmd)**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Gemini setup

```text
copy .env.example .env     (Windows)   |   cp .env.example .env   (macOS/Linux)
```

Then edit `.env` and set `GEMINI_API_KEY`. Without a key the app uses the local extractive fallback.

## Build the index

Run once before the first launch (and again whenever the corpus changes):

```powershell
python scripts/ingest.py
```

## Run

```powershell
streamlit run app.py
```

## Usage

1. Type a question.
2. Choose the retrieval mode (`hybrid` recommended) and the number of chunks.
3. Read the grounded answer with its `[S1] [S2]` citations.
4. Open the Sources / Keywords / Summary / Topics tabs to inspect the retrieved evidence.

## Project layout

```text
app.py                  Streamlit application
requirements.txt        dependencies
.env.example            Gemini configuration template
src/mintri_rag/         core package (loaders, chunking, retrieval, generation, text mining)
scripts/ingest.py       builds the retrieval index from data/raw/
data/raw/               source documents (the corpus)
```
