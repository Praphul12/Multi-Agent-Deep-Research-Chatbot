# 🤖 Multi-Agent Assistant Platform

A production-ready AI application integrating conversational chat, document Q&A, and autonomous research capabilities. Built with LangGraph, LangChain, and OpenAI GPT-4.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/langgraph-0.0.20+-green.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A unified platform with two operational modes:

**💬 Chat Mode** - Conversational AI with multi-tool capabilities:
- PDF document Q&A using RAG with FAISS vector store
- Real-time web search via DuckDuckGo
- Stock price queries through Alpha Vantage API
- Built-in calculator for arithmetic operations
- Persistent conversation threads with SQLite
- Human in the loop for **Explicit approval requirement** before tool execution

**📝 Research Mode** - Autonomous research agent:
- Intelligent routing (closed-book, hybrid, open-book strategies)
- Automated web research via Tavily API
- Parallel content generation with LangGraph
- AI-generated technical diagrams using Gemini 2.0
- Citation-backed blog posts with multi-format output

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│                  (integrated_app.py)                         │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │          Mode Selector (Session State)              │    │
│  │                                                      │    │
│  │    💬 Chat Mode        📝 Research Mode             │    │
│  └───────┬────────────────────────────┬─────────────────┘    │
└──────────┼────────────────────────────┼──────────────────────┘
           │                            │
           ▼                            ▼
    ┌──────────────────┐       ┌────────────────────────┐
    │  Chat Backend    │       │  Research Backend      │
    │  (LangGraph)     │       │  (LangGraph)           │
    │                  │       │                        │
    │  • chat_node     │       │  • router              │
    │  • tool_node     │       │  • research            │
    │    - RAG         │       │  • orchestrator        │
    │    - Search      │       │  • worker (parallel)   │
    │    - Stocks      │       │  • reducer             │
    │    - Calculator  │       │    - merge             │
    │                  │       │    - decide_images     │
    │  SQLite          │       │    - generate          │
    │  Checkpointer    │       │                        │
    └──────────────────┘       └────────────────────────┘
```

### Chat Mode Flow
```
User Input → LLM → Tool Decision → Tool Execution → Response
               ↑                           │
               └──────── Loop ────────────┘
```
![Architecture Diagram](utils/tools.png)

### Research Mode Flow
![Architecture Diagram](utils/workflow.png)


## Features

### Advanced AI Capabilities
- **Multi-Agent Architecture** - LangGraph-based orchestration with conditional routing
- **RAG Pipeline** - Document chunking, embedding, and semantic search
- **Parallel Processing** - Concurrent section generation for research blogs
- **Tool Use** - Dynamic function calling with LangChain tool binding
- **State Management** - Persistent conversation threads and research outputs
- **Human-in-the-Loop Guardrails** — Approval-based execution for high-risk actions


### Production-Ready
- **Scalable Backend** - Independent graph compilation for chat and research
- **Error Handling** - Graceful degradation and comprehensive error messages
- **Session Persistence** - SQLite checkpointing for conversation continuity
- **Modular Design** - Clean separation of concerns
- **Streaming Responses** - Real-time token streaming

## Tech Stack

### Core Framework
- **Python 3.8+** - Primary language
- **Streamlit 1.28+** - Interactive web interface
- **LangGraph 0.0.20+** - Agent workflow orchestration

### LLM & AI
- **OpenAI GPT-4o-mini** - Conversational AI and content generation
- **OpenAI Embeddings** - text-embedding-3-small for semantic search
- **Google Gemini 2.0 Flash** - AI image generation

### Data & Storage
- **FAISS** - Vector similarity search for RAG
- **SQLite** - Conversation checkpointing
- **Pandas** - Data manipulation and display

### APIs & Tools
- **Tavily API** - Web research and evidence gathering
- **DuckDuckGo API** - Web search functionality
- **Alpha Vantage API** - Stock market data
- **PyPDF** - PDF parsing and text extraction

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (required)
- Tavily API key (optional, for research mode)
- Google AI API key (optional, for image generation)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/yourusername/multi-agent-assistant.git
cd multi-agent-assistant

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Verify setup
python verify_setup.py

# Run application
streamlit run integrated_app.py
```

### Environment Configuration

Create a `.env` file:

```env
# Required
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx

# Optional
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx
GOOGLE_API_KEY=AIxxxxxxxxxxxxx
```

## Usage

### Chat Mode
1. Select "💬 Chat Mode" in sidebar
2. Upload a PDF (optional)
3. Ask questions or request actions

Example queries:
- "What is this document about?"
- "Search for the latest AI developments"
- "What's the current price of TSLA stock?"
- "Calculate 1234 * 5678"

### Research Mode
1. Select "📝 Research Mode" in sidebar
2. Enter research topic
3. Set as-of date
4. Click "🚀 Generate Research Blog"
5. Download markdown or bundled ZIP

Example topics:
- "Latest developments in quantum computing"
- "Comparison of vector databases for RAG systems"
- "Tutorial on building LangGraph agents"

## Project Structure

```
multi-agent-assistant/
├── integrated_app.py              # Main Streamlit application
├── chatbot_backend.py             # Chat agent LangGraph
├── bwa_backend.py                 # Research agent LangGraph
├── requirements.txt               # Dependencies
├── verify_setup.py                # Setup verification
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## License

This project is licensed under the MIT License.

## Contact

**GitHub**: [@yourusername](https://github.com/yourusername)

---

Built with LangGraph to demonstrate production-grade agent orchestration, safety guardrails, and deterministic workflows.
