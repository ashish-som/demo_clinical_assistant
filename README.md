# Clinical Research Assistant

An Agentic RAG solution for clinical research papers. The assistant can index PDF papers, answer research questions with citations, and generate structured summaries from the indexed evidence.

## Capabilities

1. Ingest clinical research PDFs
2. Extract page-level text and metadata
3. Create section-aware chunks
4. Store embeddings in ChromaDB
5. Retrieve evidence using hybrid retrieval
6. Generate query variations for better recall
7. Check whether retrieved evidence is sufficient
8. Answer questions with source citations
9. Summarize indexed papers
10. Run through CLI or Streamlit UI

## Project Flow

```text
PDFs
-> Extract page text
-> Convert pages to Documents
-> Split Documents into chunks
-> Tag chunks with likely paper sections
-> Add chunk IDs
-> Create embeddings
-> Store in ChromaDB
-> Ask question or request summary
-> Generate query variations
-> Retrieve evidence with hybrid search
-> Check evidence quality
-> Retry retrieval if needed
-> Generate cited answer or summary
```

## Folder Structure

```text
clinical_assistant/
  app.py
  cli_app.py
  src/
    agents.py
    chunking.py
    config.py
    interfaces.py
    pdf_loader.py
    retrieval.py
    services.py
    vector_store.py
  pdf/
  uploaded_pdfs/
  research_db/
  requirements.txt
  requirements-core.txt
  .env.example
```

## Main Files

`app.py`

Streamlit UI for uploading papers, building the index, asking questions, and summarizing papers.

`cli_app.py`

Command-line version of the same assistant. Useful for running the core RAG workflow without a UI.

Supported commands:

```text
index
ask
summarize
list-papers
```

`src/pdf_loader.py`

Reads PDFs using PyMuPDF. Each non-empty page becomes a LangChain `Document` with metadata such as source file, study id, author/year, and page number.

`src/chunking.py`

Splits page documents into smaller chunks. Each chunk receives a `chunk_id` and a likely paper section such as `abstract`, `methods`, `results`, `discussion`, `limitations`, or `conclusion`.

`src/vector_store.py`

Creates embeddings and stores chunks in a persistent ChromaDB database.

`src/retrieval.py`

Contains retrieval logic:

1. semantic vector search
2. query variation
3. keyword search
4. hybrid retrieval
5. paper-specific filtering
6. section-aware filtering
7. deduplication
8. source-diverse retrieval for broad summaries

`src/agents.py`

Builds the LangGraph workflows.

Question answering:

```text
retrieve -> grade -> retrieve_again OR generate -> END
```

Summarization:

```text
retrieve_for_summary -> summarize -> END
```

`src/interfaces.py`

Defines lightweight interfaces using Python `Protocol`:

```python
DocumentLoader
ChunkingStrategy
RetrievalStrategy
```

`src/services.py`

Contains application-level services:

```python
PaperIndexingService
HybridResearchRetriever
ClinicalResearchAssistant
```

These services keep the RAG logic separate from the UI.

## Architecture

```text
CLI or Streamlit UI
-> ClinicalResearchAssistant service
-> LangGraph workflow
-> Hybrid retrieval strategy
-> ChromaDB vector store
-> PDF chunks with metadata
```

Indexing uses:

```text
PaperIndexingService
-> PdfDocumentLoader
-> SectionAwareChunker
-> Chroma vector store
```

## Retrieval Design

### Semantic Vector Search

The query is embedded and matched against PDF chunks stored in ChromaDB.

Code:

```python
create_retriever(vectorstore, top_k)
```

### Query Variation

The LLM generates alternate search queries for the same user intent. This improves recall when research papers use different wording.

Example:

```text
Original:
What are the cardiovascular outcomes?

Variations:
cardiovascular endpoints reported in the study
clinical outcomes related to heart disease
mortality and morbidity findings for cardiovascular disease
```

Code:

```python
create_query_variations(query, llm)
```

### Hybrid Search

The system combines:

1. vector search for semantic similarity
2. keyword search for exact term matching

This is useful for clinical terms, drug names, biomarkers, abbreviations, and numeric outcomes.

Code:

```python
hybrid_retrieve_with_query_variations(vectorstore, query, llm, top_k)
```

### Section-Aware Retrieval

Chunks are tagged with likely paper sections. Focused questions can prefer the most relevant section.

Examples:

```text
"What was the sample size?" -> methods
"What were the outcomes?" -> results
"What are the limitations?" -> limitations
```

Code:

```python
detect_section(text)
add_section_labels(chunks)
infer_section_filter(query)
```

### Paper-Specific Retrieval

If the user names a specific paper, such as `Honghao 2025`, the retriever detects that source from metadata and keeps only chunks from that paper.

Code:

```python
detect_requested_sources(query, docs)
filter_by_requested_sources(docs, requested_sources)
```

### Source-Diverse Retrieval

For broad summaries, the system avoids pulling all context from only one paper. This helps represent multiple indexed papers in the summary context.

If the user asks for one specific paper, source diversity is skipped.

Code:

```python
diversify_by_source(docs)
```

### Evidence Quality Check

The agent checks whether retrieved context is sufficient before generating the final answer. If evidence is weak, it rewrites the query and retrieves again.

### Citation-Aware Generation

Final answers and summaries are grounded in retrieved chunks and cite sources like `[Source 1]`.

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install core dependencies for the CLI version:

```bash
pip install -r requirements-core.txt
```

Or install all dependencies, including Streamlit:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

## Run Without Streamlit

Build or rebuild the vector index:

```bash
python cli_app.py index
```

Ask a research question:

```bash
python cli_app.py ask "What are the cardiovascular outcomes reported across the studies?"
```

Summarize indexed papers:

```bash
python cli_app.py summarize
```

Summarize one specific paper:

```bash
python cli_app.py summarize --focus "Summarize Honghao 2025"
```

List indexed papers:

```bash
python cli_app.py list-papers
```

## Run With Streamlit

Run the Streamlit app:

```bash
streamlit run app.py
```

The Streamlit app supports:

1. building or rebuilding the index
2. asking research questions
3. summarizing indexed papers

## Persistence

ChromaDB stores the processed chunks and embeddings in:

```text
research_db/
```

PDFs do not need to be uploaded again after the index is created. Rebuild the index only when the source PDFs change or when a fresh database is needed.

## Environment Variables

`.env` should contain:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

Do not hardcode API keys in Python files.

## Possible Enhancements

1. Add BM25 for stronger lexical retrieval
2. Add reranking before generation
3. Add DeepEval metrics for faithfulness and relevance
4. Add authentication for deployed usage
5. Deploy to Streamlit Cloud, Hugging Face Spaces, or a cloud VM
