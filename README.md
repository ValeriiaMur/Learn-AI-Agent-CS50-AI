# LearnAI Agent

An open-source **agentic AI tutor** for ML/AI concepts. Unlike typical "chat with your PDF" demos, LearnAI Agent uses multi-agent orchestration to actively **teach** — assessing knowledge gaps, guiding learners with Socratic questioning, and adapting explanations to your level.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Streamlit   │────▶│ Router Agent │────▶│ Retrieval Agent   │
│  Frontend    │     │ (classifies  │     │ (vector search +  │
│              │◀────│  intent)     │     │  hybrid ranking)  │
└─────────────┘     └──────┬───────┘     └────────┬─────────┘
                           │                       │
                    ┌──────▼───────┐     ┌────────▼─────────┐
                    │ Tutor Agent  │◀────│ Progress Tracker  │
                    │ (teach/quiz/ │     │ (knowledge model) │
                    │  explain)    │     │                   │
                    └──────────────┘     └──────────────────┘
```

**Agent Roles:**

- **Router Agent** — Classifies user intent (concept question, quiz request, deeper explanation, off-topic)
- **Retrieval Agent** — Hybrid search (semantic + keyword) over ingested educational materials
- **Tutor Agent** — Decides how to teach: explain, ask Socratic questions, use analogies, or quiz
- **Progress Tracker** — Maintains a simple knowledge model of topics covered and learner confidence

**Tiered Model Routing:**
- Simple definitions/glossary → Llama 3 via Ollama (free, local)
- Standard explanations → GPT-4o-mini (fast, cheap)
- Complex reasoning/analogies → GPT-4o (premium fallback)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangGraph + LangChain |
| Primary LLM | OpenAI GPT-4o / GPT-4o-mini |
| Local LLM | Llama 3 via Ollama |
| Vector Database | ChromaDB |
| Embeddings | OpenAI `text-embedding-3-small` |
| Frontend | Streamlit |
| Observability | LangSmith |
| Containerization | Docker + Docker Compose |

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) (optional, for local Llama model)
- OpenAI API key

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/learnai-agent.git
cd learnai-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Ingest sample educational data
python -m src.rag.ingest

# Run the app
streamlit run app.py
```

### Docker (one-command setup)

```bash
docker-compose up --build
```

Then open http://localhost:8501

## Project Structure

```
learnai-agent/
├── app.py                      # Streamlit entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── src/
│   ├── __init__.py
│   ├── config.py               # Settings, model routing config
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── router.py           # Intent classification agent
│   │   ├── retrieval.py        # RAG retrieval agent
│   │   ├── tutor.py            # Teaching/Socratic agent
│   │   ├── progress.py         # Knowledge tracking agent
│   │   └── graph.py            # LangGraph state machine
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingest.py           # Document ingestion pipeline
│   │   ├── embeddings.py       # Embedding utilities
│   │   └── retriever.py        # Hybrid search (semantic + keyword)
│   └── utils/
│       ├── __init__.py
│       ├── model_router.py     # Tiered LLM routing logic
│       └── prompts.py          # All prompt templates
├── data/
│   └── sample_docs/            # Sample AI/ML educational content
│       ├── ml_glossary.md
│       └── neural_networks_intro.md
└── tests/
    ├── __init__.py
    ├── test_router.py
    ├── test_retriever.py
    └── test_model_router.py
```

## Ingesting Your Own Content

Drop `.md`, `.txt`, or `.pdf` files into `data/sample_docs/` and re-run:

```bash
python -m src.rag.ingest
```

The system chunks documents, generates embeddings, and stores them in ChromaDB. Fork this repo and fill it with your own course materials.

## Observability

Set `LANGCHAIN_TRACING_V2=true` in your `.env` to enable LangSmith tracing. Every agent decision, retrieval, and model call is logged with full trace data.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT
