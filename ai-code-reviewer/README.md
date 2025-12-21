# 🧠 AI Code Reviewer (In Progress)

An AI-powered code review tool that analyzes GitHub repositories and provides insights such as bug detection, security issues, fix suggestions, test generation, and code summaries using LLM-based agents.

⚠️ Project Status: Work in progress (active development)

---

## 🚀 Features (Current)

- Clone and analyze public GitHub repositories
- Multi-agent architecture:
  - Bug detection
  - Security issue detection
  - Fix suggestions
  - Test case generation
  - Code summarization
- Modular agent design
- Centralized LLM abstraction
- REST API built with FastAPI
- Swagger UI for testing

---

## 🏗️ Project Structure

backend/
├── agents/              # AI agents
├── llm/                 # Centralized LLM logic
├── models/              # Response schemas
├── services/            # Orchestration & repo reading
├── utils/               # Repo cloning utilities
├── main.py              # FastAPI entry point
├── .env                 # Environment variables (ignored)

---

## 🧪 How It Works

1. User submits a GitHub repository URL
2. Repository is cloned temporarily
3. Supported source files are detected
4. Each file is analyzed by multiple AI agents
5. Results are returned as structured JSON

---

## ⚠️ Known Limitations / Failure Cases

- Long response times for repositories  
  Reason: Multiple LLM calls are executed synchronously in a single request

- Requests may appear to load forever  
  Reason: CPU-based local models are slow and no background job system exists yet

- No frontend UI (API-only)
- Limited file-type support
- No caching or persistence
- Minimal progress reporting

These issues are known and will be addressed in future updates.

---

## 🛠️ Planned Improvements

- Background job processing
- Progress tracking
- Frontend dashboard
- Performance optimizations
- Expanded language support
- Cloud deployment

---

## 🧩 Tech Stack

- Python
- FastAPI
- LangChain
- HuggingFace Transformers
- Pydantic
- Git

---

## 📌 Note

This project is built as a learning-focused, resume-oriented system to explore AI agents, LLM orchestration, and backend system design.
EOF