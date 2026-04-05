# Retrieval-Augmented Generation (RAG) — How It Works

## What is RAG?

RAG (Retrieval-Augmented Generation) is a pattern that improves AI-generated answers by **grounding** them in your own data. Instead of relying solely on what a language model was trained on, RAG retrieves relevant information from a knowledge base at query time and feeds it to the model as context.

**The core idea:** *Don't make the model memorize everything — let it look things up.*

### Why RAG?

| Problem without RAG | How RAG solves it |
|---|---|
| LLMs can hallucinate facts | Answers are grounded in retrieved documents |
| Training data has a knowledge cutoff | Your index can contain the latest data |
| No access to private/proprietary data | You index your own documents |
| No source attribution | Retrieved chunks carry metadata (page numbers, filenames) for citations |

### RAG Pipeline at a Glance

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. Parse    │ ──▶ │  2. Enrich   │ ──▶ │  3. Chunk &  │ ──▶ │  4. Query &  │
│  Documents   │     │  (images,    │     │  Index       │     │  Generate    │
│              │     │   metadata)  │     │              │     │              │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
     PDF/docs       Image descriptions     Embeddings +          Search +
     → raw text       → richer text       vector store         LLM answer
```

---

## Step 1: Document Parsing (PDF → Text + Images)

> **Notebook:** `01_pdf_parsing.ipynb`

The first step is turning raw documents into machine-readable text. In this project, we parse a PDF using **PyMuPDF** (`fitz`).

### What happens

1. **Open the PDF** and iterate over every page.
2. **Extract text** from each page using `page.get_text()`.
3. **Extract images** embedded in each page — skip tiny decorative images (< 5 KB) to avoid noise.
4. **Save everything** — text goes into a JSON structure keyed by page number; images are saved as files on disk.

### Key concepts

- **Page-level granularity:** We keep track of which page each piece of text and each image came from. This metadata flows through the entire pipeline and enables page citations in the final answer.
- **Image extraction:** PDFs often contain images (diagrams, illustrations, charts) that carry important information. Extracting them now allows us to process them in the next step.

### Output

- `extracted_data/parsed_pages.json` — Array of page objects with `page_number`, `text`, `has_images`, and `images` metadata.
- `extracted_data/images/` — Extracted image files named by page and index (e.g., `page_5_img_1.png`).

---

## Step 2: Image Enrichment (Images → Text Descriptions)

> **Notebook:** `02_image_descriptions.ipynb`

Images contain information that text search can't find. This step uses **GPT-4o Vision** to generate text descriptions of every extracted image, making image content searchable.

### What happens

1. **Load parsed pages** from Step 1.
2. For each image, **encode it as base64** and send it to GPT-4o along with surrounding page text for context.
3. The model returns a **text description** of the image (what it shows, any text in it, diagram structure, etc.).
4. **Merge descriptions back** into the page text as `text_with_images` — the original text plus an `[Image descriptions]` section appended at the end.

### Key concepts

- **Multi-modal enrichment:** By converting images to text, the entire document — text and visuals — becomes searchable with the same text-based retrieval methods.
- **Context-aware descriptions:** Sending the surrounding page text helps the vision model produce more relevant descriptions (e.g., "This is an illustration of Hogwarts castle" vs. "A drawing of a building").
- **Resumable processing:** Progress is saved every 10 images to `image_descriptions_progress.json`, so the process can resume after interruptions or rate limits.

### Output

- `extracted_data/parsed_pages_with_images.json` — Same structure as Step 1, but with `text_with_images` and `description` fields added.

---

## Step 3: Chunking, Embedding & Indexing

> **Notebook:** `03_chunking_and_indexing.ipynb`

This is the heart of the RAG pipeline — turning enriched text into a searchable vector index.

### 3a. Chunking

Full pages are often too long to use as search results. We split them into smaller, overlapping **chunks**.

- **Tool used:** `RecursiveCharacterTextSplitter` from LangChain.
- **Chunk size:** 1,000 characters with 200-character overlap.
- **How it works:** The splitter recursively tries separators in order — `\n\n` → `\n` → ` ` → `""` — to split text at natural boundaries (paragraphs first, then lines, then words). This produces chunks that are semantically coherent rather than cutting mid-sentence.
- **Overlap:** The 200-character overlap ensures that information near chunk boundaries isn't lost — a fact that spans two chunks will appear in both.

Each chunk retains metadata: `page_number`, `chunk_index`, and `has_images`.

### 3b. Embedding

Each chunk is converted into a **vector embedding** — a list of numbers that represents the chunk's meaning in a high-dimensional space.

- **Model:** Azure OpenAI embedding model (e.g., `text-embedding-3-small`).
- **Dimensions:** 256 (reduced from 3,072 to save storage on the free tier).
- **Batching:** Chunks are embedded in batches of 100 for efficiency.

Chunks with similar meaning end up close together in vector space, enabling **semantic search** (finding relevant results even when the exact words don't match the query).

### 3c. Indexing in Azure AI Search

The chunks and their embeddings are uploaded to **Azure AI Search**, which provides:

| Feature | What it does |
|---|---|
| **Keyword search** | Traditional full-text search (BM25) on the `content` field |
| **Vector search** | HNSW-based approximate nearest neighbor search on the `embedding` field |
| **Hybrid search** | Combines keyword + vector results using Reciprocal Rank Fusion (RRF) |
| **Semantic reranking** | A Microsoft-hosted model re-scores the top results for relevance |

The index schema includes:
- `id` — Unique chunk identifier (e.g., `page-5-chunk-2`)
- `content` — The chunk text (searchable)
- `embedding` — The vector (searchable via HNSW)
- `page_number`, `chunk_index`, `has_images` — Filterable metadata

### Output

- An Azure AI Search index (e.g., `rag-index`) populated with all chunks, ready for querying.

---

## Step 4: Query & Answer Generation

> **Notebook:** `04_rag_query.ipynb`

This is where the user asks a question and gets a grounded, cited answer.

### What happens (the RAG loop)

```
User Question
     │
     ▼
┌──────────────────┐
│ 1. Embed query   │  Convert question to a vector
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 2. Hybrid search │  Keyword + vector search in Azure AI Search
│    + reranking   │  Semantic reranker re-scores top results
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 3. Build context │  Assemble top-k chunks into a prompt
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 4. Generate      │  Send context + question to GPT-4.1
│    answer        │  Model answers using ONLY the context
└────────┬─────────┘
         ▼
   Answer with page citations
```

### Step-by-step

1. **Embed the query** — The user's question is converted to a vector using the same embedding model from Step 3.
2. **Hybrid search + semantic reranking** — Azure AI Search runs both keyword and vector search, merges results with RRF, then the semantic reranker re-scores the top candidates for precision.
3. **Build context** — The top-k chunks (default: 5) are assembled into a context string, each labeled with its page number.
4. **Generate answer** — The context and question are sent to GPT-4.1 with a system prompt that instructs it to:
   - Answer **only** from the provided context
   - Cite **page numbers** in the answer
   - Admit when the context doesn't contain enough information

### Example

```
Question: "Who is Harry's godfather and how is he related to his parents?"

Answer: Harry's godfather is Sirius Black. He was James Potter's best friend
at Hogwarts... (Page 112, Page 245)

Sources: Page 112, Page 245, Page 301, Page 89, Page 156
```

---

## Summary

| Step | Input | Process | Output |
|---|---|---|---|
| **1. Parse** | PDF file | Extract text + images per page | `parsed_pages.json` + image files |
| **2. Enrich** | Pages + images | GPT-4o describes images → merge into text | `parsed_pages_with_images.json` |
| **3. Index** | Enriched pages | Chunk → embed → upload to search index | Azure AI Search index |
| **4. Query** | User question | Search → retrieve → generate with LLM | Grounded answer with citations |

The power of RAG is that **your data stays current and private** — you control what goes into the index, and the LLM only sees what the retriever surfaces. No fine-tuning needed.
