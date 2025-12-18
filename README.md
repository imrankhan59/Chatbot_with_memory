# AI Assistant with Tools, Memory, and RAG

This project is a **production-style AI assistant** built using **LangChain** and **LangGraph**. It supports persistent conversation memory, tool-based reasoning for up-to-date information, and document-based question answering using **Retrieval-Augmented Generation (RAG)**.

The goal of this project is to demonstrate how real-world LLM applications are architected beyond simple chatbot demos.

---

## 🚀 Key Features

* **Stateful AI Agent** using LangGraph for controlled conversation flow
* **Persistent Chat Memory** stored in PostgreSQL using LangGraph checkpointers
* **Tool Calling** for real-time web search (DuckDuckGo)
* **RAG Pipeline** for querying uploaded PDF documents
* **Thread-based Context Management** to scope conversations and documents
* **Streamlit UI** for interactive user experience

---

## 🏗️ High-Level Architecture

```
User (Streamlit UI)
        ↓
LangGraph Agent (LLM + Tools)
        ↓
---------------------------------
| PostgreSQL | Vector Store | Web |
---------------------------------
```

### Components

* **LLM**: Groq-hosted model (`openai/gpt-oss-120b`)
* **Agent Orchestration**: LangGraph
* **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
* **Vector Store**: FAISS (in-memory)
* **Database**: PostgreSQL (conversation persistence)
* **UI**: Streamlit

---

## 🧠 How the System Works

### 1️⃣ Conversation Flow

* User messages are handled by a LangGraph agent
* The agent decides whether to respond directly or call a tool
* Conversation state is persisted in PostgreSQL using a thread ID

### 2️⃣ Tool-Based Reasoning

* **Web Search Tool**: Used for up-to-date or external information
* **RAG Tool**: Used when questions relate to uploaded documents

### 3️⃣ Retrieval-Augmented Generation (RAG)

* Users upload a PDF file
* The document is:

  * Loaded and split into chunks
  * Converted into embeddings
  * Stored in a FAISS vector store
* When a document-related question is asked, relevant chunks are retrieved and provided to the LLM

---

## 📄 PDF Ingestion and RAG

* Each uploaded PDF is associated with a specific chat thread
* Document retrieval is scoped to that thread
* If no document is uploaded, the agent prompts the user to upload one

---

## 💾 Persistent Memory

* Conversation history is stored in PostgreSQL
* Users can leave and return to continue previous chats
* Connection pooling is used for efficient database access

---

## ⚠️ Current Limitations (MVP Stage)

This project is intentionally designed as a **production-style MVP**, not a full enterprise system.

* No user authentication (thread-based only)
* Vector store is in-memory (not persistent across restarts)
* No observability or monitoring (e.g., LangSmith / Langfuse)

---

## 🔮 Future Improvements

* Add user authentication and user-level data isolation
* Persist vector embeddings using a production-ready vector database
* Introduce FastAPI backend for better scalability
* Add observability for tracing, latency, and cost monitoring

---

## 🧪 Use Cases

* Internal document Q&A systems
* Customer support knowledge assistants
* Policy, compliance, or HR document assistants
* Research and technical documentation chatbots

---

## 🛠️ Tech Stack

* Python
* LangChain
* LangGraph
* Groq LLM API
* PostgreSQL
* FAISS
* HuggingFace Transformers
* Streamlit

---

## Summary

This project demonstrates how to build a **real-world AI assistant** with memory, tools, and document understanding. It focuses on correct architectural patterns used in modern GenAI systems rather than simple prompt-based chatbots.
