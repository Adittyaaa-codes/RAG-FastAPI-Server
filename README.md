# StudyBot — AI-Powered Study Assistant

StudyBot lets students upload their notes and documents, then chat with an AI that answers questions **directly from their own content** — not from the internet.

---

## 🔗 Repositories

- **Frontend** (React + Vite + TypeScript) → [github.com/Adittyaaa-codes/Frontend](https://github.com/Adittyaaa-codes/Frontend)
- **Backend** (Node.js + Express + MongoDB) → [github.com/Adittyaaa-codes/Backend](https://github.com/Adittyaaa-codes/Backend)
- **AI Server** (FastAPI + LangChain + Qdrant + OpenAI) → This repo

---

## How It Works

1. **Sign up / Login** — handled by the Express backend with JWT authentication.
2. **Upload documents** — send PDFs or text files; the AI server splits them into chunks, converts them to vectors using OpenAI embeddings, and stores them in Qdrant under your own isolated collection.
3. **Chat** — ask a question; the server finds the most relevant chunks from your documents and feeds them to GPT-4o-mini, which answers based only on your content.
4. **YouTube Summarizer** — paste any YouTube link to get an AI-generated summary with key points and keywords.

Each user's data is fully isolated — no one can access another user's documents.

---

## What Each Service Does

**FastAPI AI Server**
The core brain of the application. It exposes the entire RAG pipeline as an API — handling document indexing, vector storage, semantic retrieval from Qdrant, and LLM calling via LangChain. It also manages multi-agent routing (QA vs Explain modes) using LangGraph and handles multi-user isolation by giving each user their own Qdrant collection.

**MERN Backend**
Handles everything related to users — signup, login, JWT token generation, and managing user chat history. Acts as the authentication layer between the frontend and the AI server.

**React Frontend**
Provides a clean, responsive UI for uploading documents, managing subjects and chapters, chatting with the AI, and viewing YouTube summaries. Built with Vite, TypeScript, and Tailwind CSS for a fast and modern experience.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vite, TypeScript, Tailwind CSS |
| Backend | Node.js, Express, MongoDB, JWT |
| AI Server | FastAPI, LangChain, LangGraph |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | Qdrant |

---

## Local Setup

```bash
git clone <this-repo>
cd RAG
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=sk-...
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
JWT_SECRET=your-jwt-secret
```

Run the server:

```bash
uvicorn app:app --host localhost --port 8000 --reload
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload_docs` | Upload and index documents |
| `GET` | `/list_docs` | List uploaded documents |
| `DELETE` | `/delete_docs/{filename}` | Delete a document |
| `POST` | `/chat/qa` | Ask a question (streamed) |
| `POST` | `/chat/explain` | Get a detailed explanation (streamed) |
| `POST` | `/summarize` | Summarize a YouTube video |

All endpoints require `Authorization: Bearer <token>`.

---

## Author

Built by [Aditya](https://github.com/Adittyaaa-codes)
