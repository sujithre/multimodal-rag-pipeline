# Multimodal RAG Pipeline

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline that processes PDFs with text and images, indexes them into Azure AI Search, and answers questions with citations — optionally powered by AI agents.

## Pipeline Overview

```
PDF → Extract Text + Images → Describe Images (GPT-4o Vision) → Chunk & Embed
    → Index (Azure AI Search) → Query (Hybrid + Semantic Reranking) → Answer with Citations
```

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `01_pdf_parsing.ipynb` | Extract text and images from PDF using PyMuPDF, preserving page-level metadata |
| 2 | `02_image_descriptions.ipynb` | Generate text descriptions of extracted images using GPT-4o Vision |
| 3 | `03_chunking_and_indexing.ipynb` | Split pages into overlapping chunks, generate embeddings, and upload to Azure AI Search |
| 4 | `04_rag_query.ipynb` | Hybrid search + semantic reranking, then GPT-4 answer generation with page citations |
| 5 | `05_rag_agent.ipynb` | Wrap the RAG pipeline using Microsoft Agent Framework for autonomous search |
| 6 | `06_rag_foundry_agent.ipynb` | Agent connected via FoundryChatClient to Azure AI Foundry for managed model access |
| 7 | `07_rag_foundry_service_agent.ipynb` | Service-managed Prompt Agent in Azure AI Foundry (visible in the Foundry portal) |

## Key Technologies

- **PDF Parsing:** PyMuPDF
- **AI Models:** Azure OpenAI (GPT-4o, GPT-4o Vision, text-embedding-3-small)
- **Search:** Azure AI Search (hybrid keyword + vector + semantic reranking)
- **Chunking:** LangChain `RecursiveCharacterTextSplitter`
- **Agents:** Microsoft Agent Framework, Azure AI Foundry
- **Auth:** Azure Identity (`DefaultAzureCredential`)

## Prerequisites

### Azure Resources

- **Azure OpenAI** with deployments for chat (GPT-4o), vision (GPT-4o), and embeddings (text-embedding-3-small)
- **Azure AI Search** service
- **Azure AI Foundry** project (for notebooks 6–7)

### Environment Variables

Create a `.env` file in the project root:

```env
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=<chat-deployment-name>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding-deployment-name>
AZURE_SEARCH_ENDPOINT=https://<search-service>.search.windows.net/
AZURE_SEARCH_INDEX_NAME=rag-index
FOUNDRY_PROJECT_ENDPOINT=<foundry-endpoint>   # For notebooks 6-7
FOUNDRY_MODEL=<foundry-model-name>            # For notebooks 6-7
```

### Python Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Design Decisions

- **Embedding dimensions: 256** — reduced from 3,072 to fit Azure AI Search free tier
- **Chunk size: 1,000 chars** with 200-char overlap for semantic coherence
- **Hybrid search** — BM25 keyword + HNSW vector + semantic reranking for precision
- **Image enrichment** — images sent to GPT-4o Vision with surrounding text context for better descriptions

## Getting Started

1. Set up Azure resources and configure `.env`
2. Place a PDF in the project root (default: `harrypotter.pdf`)
3. Run notebooks 1–4 sequentially to build the pipeline
4. Optionally run notebooks 5–7 to explore agent-based architectures
