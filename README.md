# LearnAI Agent

An open-source **agentic AI tutor** built around Harvard's [CS50 Introduction to Artificial Intelligence with Python](https://cs50.harvard.edu/ai/2024/). Unlike typical "chat with your PDF" demos, LearnAI Agent uses multi-agent orchestration to actively **teach** — assessing knowledge gaps, guiding learners with Socratic questioning, and adapting explanations to your level.

<img width="1677" height="816" alt="Screenshot 2026-03-04 at 4 35 18 PM" src="https://github.com/user-attachments/assets/7d543f76-332b-455b-98ca-c2f128474777" />

## What's Inside

The knowledge base contains comprehensive, original study notes covering all 7 modules of the CS50 AI curriculum:

| Module | Topics Covered | Source File |
|--------|---------------|-------------|
| **Search** | BFS, DFS, A\*, greedy best-first, minimax, alpha-beta pruning | `cs50ai_search.md` |
| **Knowledge** | Propositional logic, first-order logic, inference, resolution, model checking | `cs50ai_knowledge.md` |
| **Uncertainty** | Bayes' Rule, Bayesian networks, Markov models, HMMs, sampling | `cs50ai_uncertainty.md` |
| **Optimization** | Hill climbing, simulated annealing, linear programming, CSPs, arc consistency | `cs50ai_optimization.md` |
| **Learning** | kNN, SVMs, regression, regularization, k-means clustering, Q-learning | `cs50ai_machine_learning.md` |
| **Neural Networks** | Perceptrons, backpropagation, CNNs, RNNs, LSTMs, dropout | `cs50ai_neural_networks.md` |
| **Language** | Bag of words, TF-IDF, word embeddings, attention, transformers, GPT, BERT | `cs50ai_nlp.md` |

Additional reference docs (`neural_networks_intro.md`, `ml_glossary.md`, `transformers_and_llms.md`) provide supplementary depth.

All notes are original educational content inspired by the CS50 AI syllabus — not copied from the course. For the official lectures, problem sets, and videos, visit the [CS50 AI course page](https://cs50.harvard.edu/ai/2024/).

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
- **Progress Tracker** — Maintains a knowledge model of topics covered and learner confidence, with curriculum-aligned topic suggestions

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
| Frontend | Streamlit (Material icons, Harvard theme) |
| Observability | LangSmith |
| Containerization | Docker + Docker Compose |

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- [Ollama](https://ollama.ai) (optional, for local Llama model)

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
# Edit .env with your OpenAI API key

# Ingest the CS50 AI study notes into ChromaDB
python -m src.rag.ingest

# Run the app
streamlit run app.py
```

### Docker (one-command setup)

```bash
docker-compose up --build
```

Then open http://localhost:8501

## Deployment

### Streamlit Community Cloud (free, easiest)

1. Push your repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo, set `app.py` as the main file
4. Add your `OPENAI_API_KEY` under **Advanced settings > Secrets**
5. Deploy — you'll get a public URL in under a minute

### Railway / Render

Both platforms support Docker. Point them at your repo and set the `OPENAI_API_KEY` environment variable. The included `Dockerfile` handles the rest.

### Self-hosted (VPS)

```bash
# On your server
git clone https://github.com/yourusername/learnai-agent.git
cd learnai-agent
cp .env.example .env
# Edit .env with your keys

docker-compose up -d --build
```

The app serves on port 8501. Put it behind nginx or Caddy for HTTPS.

## Project Structure

```
learnai-agent/
├── app.py                      # Streamlit entry point
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .streamlit/
│   └── config.toml             # Theme (Harvard crimson) and server config
├── src/
│   ├── config.py               # Settings, model routing config
│   ├── agents/
│   │   ├── router.py           # Intent classification agent
│   │   ├── retrieval.py        # RAG retrieval agent
│   │   ├── tutor.py            # Teaching / Socratic agent
│   │   ├── progress.py         # Knowledge tracking + CS50 AI topic graph
│   │   └── graph.py            # LangGraph state machine
│   ├── rag/
│   │   ├── ingest.py           # Document ingestion pipeline
│   │   ├── embeddings.py       # Embedding utilities
│   │   └── retriever.py        # Hybrid search (semantic + keyword)
│   └── utils/
│       ├── model_router.py     # Tiered LLM routing logic
│       └── prompts.py          # All prompt templates
├── data/
│   └── sample_docs/            # CS50 AI educational content (10 files)
│       ├── cs50ai_search.md
│       ├── cs50ai_knowledge.md
│       ├── cs50ai_uncertainty.md
│       ├── cs50ai_optimization.md
│       ├── cs50ai_machine_learning.md
│       ├── cs50ai_neural_networks.md
│       ├── cs50ai_nlp.md
│       ├── neural_networks_intro.md
│       ├── ml_glossary.md
│       └── transformers_and_llms.md
└── tests/
    ├── test_router.py
    ├── test_retriever.py
    └── test_model_router.py
```

## Ingesting Your Own Content

Drop `.md` or `.txt` files into `data/sample_docs/` and re-run:

```bash
python -m src.rag.ingest
```

The system chunks documents with markdown-aware splitting, generates embeddings, and stores them in ChromaDB. The UI topic graph in `progress.py` can also be extended to match your custom curriculum.

## Observability

Set `LANGCHAIN_TRACING_V2=true` in your `.env` to enable LangSmith tracing. Every agent decision, retrieval, and model call is logged with full trace data.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT

