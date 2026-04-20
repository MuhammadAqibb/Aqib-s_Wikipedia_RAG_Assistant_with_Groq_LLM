# WikiRAG — Real-Time Wikipedia Assistant

A production-ready conversational AI app that combines Wikipedia knowledge, vector search, and Groq's LLaMA 3.3 70B to answer any question with rich, structured responses, with full conversation memory.

---

## What It Does

WikiRAG is a **Retrieval Augmented Generation (RAG)** application. Instead of relying solely on an LLM's training data, it:

1. Takes your question
2. Searches a **Pinecone vector index** of Wikipedia articles for relevant content
3. Feeds that Wikipedia context to **Groq LLaMA 3.3 70B**
4. Returns a rich, structured answer combining both Wikipedia facts and the LLM's broad knowledge
5. Remembers your conversation so follow-up questions work naturally

## Architecture

User Question
      │
      ▼
 Follow-up? ──Yes──► Reuse cached context
      │
      No
      │
      ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
      │
      ▼
Pinecone Vector Search (Top 5 Wikipedia chunks)
      │
      ▼
Groq LLaMA 3.3 70B (Wikipedia context + training knowledge + conversation history)
      │
      ▼
Structured Answer → FastAPI → Chat UI

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq LLaMA 3.3 70B Versatile |
| Vector DB | Pinecone Serverless |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (384 dims) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (dark theme) |
| Data | Wikipedia (wikimedia/wikipedia HuggingFace dataset) |

## Setup & Installation

### Prerequisites
You need free API keys from three services:
| Service | Purpose | Get Key At |
|---------|---------|------------|
| **Groq** | LLM inference (fast & free) | [console.groq.com](https://console.groq.com) |
| **Pinecone** | Vector database | [app.pinecone.io](https://app.pinecone.io) |
| **HuggingFace** | Embeddings model (no key needed, optional for higher limits) | [huggingface.co](https://huggingface.co) |

### Step 1 — Clone or download the project files

### Step 2 — Install dependencies

Using pip (recommended):
bash
---bash
pip install fastapi uvicorn langchain langchain-groq langchain-huggingface langchain-pinecone langchain-community python-dotenv requests pydantic sentence-transformers datasets "pinecone-client==5.0.1"
---
Or using Poetry:
pip install poetry
poetry install

### Step 3 — Configure environment variables

Copy `.env.example` to `.env` and fill in your keys:
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
PINECONE_API_KEY=pcsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
PINECONE_INDEX_NAME=

### Step 4 — Create a Pinecone index

1. Go to [app.pinecone.io](https://app.pinecone.io)
2. Click **Create Index**
3. Set:
   - **Name:** whatever you put in `.env
   - **Dimensions:** `384`
   - **Metric:** `cosine`
   - **Cloud:** AWS, Region: us-east-1 (free tier)
4. Click **Create Index**

### Step 5 — Load Wikipedia data into Pinecone

This is a **one-time step**. It fetches Wikipedia articles from HuggingFace and loads them into your Pinecone index:

```bash
python load_wikipedia.py
```

This will load 50,000 Wikipedia articles. You'll see progress every 100 articles:

### Step 6 — Start the server

uvicorn server:app --host 0.0.0.0 --port 8000

### Step 7 — Open the app

Visit **[http://localhost:8000](http://localhost:8000)** in your browser.


## How It Works — Deep Dive
### RAG (Retrieval Augmented Generation)
Traditional LLMs answer from training data alone, which can be outdated or hallucinated. RAG fixes this by:
1. **Retrieving** relevant real documents (Wikipedia) before answering
2. **Augmenting** the LLM prompt with those documents
3. **Generating** an answer grounded in real facts
### Vector Search
Text is converted into numerical vectors (embeddings) using `all-MiniLM-L6-v2`. Similar text has similar vectors. Pinecone finds the most semantically similar Wikipedia chunks to your question — even if the exact words don't match.
### Conversation Memory
- The frontend stores conversation history in a JavaScript array
- Every request sends the full history to the backend
- The server caches the last retrieved Wikipedia context per session
- Follow-up questions reuse the cached context instead of triggering a new search
- Meta instructions like "too short" or "elaborate" are detected and never sent to Pinecone
### Smart Follow-up Detection
The app distinguishes between:
- **New topics** → searches Pinecone fresh (e.g. "What is CRISPR?")
- **Follow-ups** → reuses cached context (e.g. "why is it important?")
- **Meta instructions** → reuses cached context (e.g. "make it shorter")

## Real-World Applications
This exact pattern — RAG + vector search + LLM — is used in production for:

- **Customer support bots** that answer only from company documentation
- **Legal assistants** grounded in specific case files or legislation
- **Medical assistants** using approved clinical guidelines
- **Internal knowledge bases** for companies and teams
- **Research assistants** over scientific papers
- **Educational tools** that explain topics with cited sources

## Extending the Project

**Add persistent chat history:**
Replace the in-memory `history` array with a SQLite database to survive page refreshes.

**Deploy to the web:**

pip install gunicorn
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
