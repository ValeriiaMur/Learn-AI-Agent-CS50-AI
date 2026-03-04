"""LearnAI Agent — Streamlit Frontend.

An agentic AI tutor for Harvard's CS50 AI with Python course,
with Socratic teaching, multi-agent orchestration, and tiered model routing.
"""

import streamlit as st
from src.agents.graph import LearnAIAgent

CS50_AI_URL = "https://cs50.harvard.edu/ai/2024/"

# --- Course curriculum ---
COURSE_MODULES = {
    "Search": {
        "icon": ":material/search:",
        "topics": ["BFS & DFS", "A* search", "minimax", "alpha-beta pruning"],
        "description": "Finding solutions by exploring state spaces",
    },
    "Knowledge": {
        "icon": ":material/menu_book:",
        "topics": ["propositional logic", "inference rules", "first-order logic", "resolution"],
        "description": "Representing and reasoning with logical knowledge",
    },
    "Uncertainty": {
        "icon": ":material/casino:",
        "topics": ["Bayesian networks", "Markov models", "HMMs", "sampling methods"],
        "description": "Reasoning under probabilistic uncertainty",
    },
    "Optimization": {
        "icon": ":material/trending_up:",
        "topics": ["hill climbing", "simulated annealing", "constraint satisfaction", "linear programming"],
        "description": "Finding the best solution among many candidates",
    },
    "Learning": {
        "icon": ":material/model_training:",
        "topics": ["k-nearest neighbors", "SVMs", "regression", "reinforcement learning"],
        "description": "Algorithms that improve from data and experience",
    },
    "Neural Networks": {
        "icon": ":material/hub:",
        "topics": ["backpropagation", "CNNs", "RNNs", "dropout & regularization"],
        "description": "Deep learning architectures and training",
    },
    "Language": {
        "icon": ":material/translate:",
        "topics": ["word embeddings", "attention mechanism", "transformers", "large language models"],
        "description": "Understanding and generating human language",
    },
}


# --- Page Config ---
st.set_page_config(
    page_title="LearnAI — CS50 AI Tutor",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    /* Tighten sidebar spacing */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.4rem; }

    /* Course link badge */
    .course-link {
        display: inline-block;
        background: #a51c30;
        color: #fff !important;
        padding: 6px 16px;
        border-radius: 6px;
        font-size: 0.82em;
        font-weight: 500;
        text-decoration: none;
        letter-spacing: 0.02em;
        transition: background 0.2s;
    }
    .course-link:hover { background: #8c1726; color: #fff !important; }

    /* Module context pill */
    .module-pill {
        display: inline-block;
        background: var(--secondary-background-color, #f0f2f6);
        border-left: 3px solid #a51c30;
        padding: 8px 14px;
        border-radius: 0 6px 6px 0;
        font-size: 0.88em;
        margin-bottom: 4px;
        color: var(--text-color);
    }

    /* Starter cards */
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        border-radius: 8px;
    }

    /* Cleaner chat area */
    .stChatMessage { border-radius: 8px; }

    /* Subheader tweaks */
    [data-testid="stSidebar"] h3 {
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.7;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State ---
if "agent" not in st.session_state:
    st.session_state.agent = LearnAIAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_module" not in st.session_state:
    st.session_state.current_module = None


# ===================== SIDEBAR =====================
with st.sidebar:
    st.title("LearnAI Agent")
    st.caption("Your AI tutor for CS50's Introduction to AI with Python")

    st.markdown(
        f'<a class="course-link" href="{CS50_AI_URL}" target="_blank">'
        "Harvard CS50 AI &mdash; Free Course &rarr;</a>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- Course Modules ----
    st.subheader("Course Modules")

    for module_name, module in COURSE_MODULES.items():
        with st.expander(f"{module['icon']} {module_name}", expanded=False):
            st.caption(module["description"])
            for topic in module["topics"]:
                if st.button(
                    topic,
                    key=f"topic_{module_name}_{topic}",
                    use_container_width=True,
                    icon=":material/arrow_forward:",
                ):
                    st.session_state.suggested_message = f"Explain {topic} and how it works"
                    st.session_state.current_module = module_name
                    st.rerun()

    st.divider()

    # ---- Progress ----
    st.subheader("Progress")
    progress = st.session_state.agent.get_progress()
    st.text(progress)

    st.divider()

    # ---- Knowledge Base Stats ----
    st.subheader("Knowledge Base")
    try:
        kb = st.session_state.agent.get_kb_stats()
        if kb["total_chunks"] > 0:
            m1, m2 = st.columns(2)
            m1.metric("Files indexed", kb["total_files"])
            m2.metric("Chunks stored", kb["total_chunks"])

            with st.expander(":material/folder: Indexed files", expanded=False):
                for fname in kb["files"]:
                    hits = kb["source_hits"].get(fname, 0)
                    if hits > 0:
                        st.markdown(
                            f"`{fname}` — referenced **{hits}** time{'s' if hits != 1 else ''}"
                        )
                    else:
                        st.markdown(f"`{fname}`")

            if kb["last_sources"]:
                st.caption("Last response drew from:")
                for src in kb["last_sources"]:
                    st.markdown(f"  `{src}`")
        else:
            st.caption(
                "No data ingested yet. Run `python -m src.rag.ingest` to index your study notes."
            )
    except Exception:
        st.caption("Vector DB not initialized. Run ingestion first.")

    st.divider()

    # ---- Quick Actions ----
    st.subheader("Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Quiz me", use_container_width=True, icon=":material/quiz:"):
            st.session_state.suggested_message = "Quiz me on what I've learned so far"
            st.rerun()
    with col2:
        if st.button("Progress", use_container_width=True, icon=":material/bar_chart:"):
            st.session_state.suggested_message = "Show me my learning progress"
            st.rerun()

    col3, col4 = st.columns(2)
    with col3:
        if st.button("Roadmap", use_container_width=True, icon=":material/route:"):
            st.session_state.suggested_message = (
                "What should I study next based on the CS50 AI curriculum?"
            )
            st.rerun()
    with col4:
        if st.button("Compare", use_container_width=True, icon=":material/compare_arrows:"):
            st.session_state.suggested_message = (
                "Compare and contrast the last two topics I asked about"
            )
            st.rerun()

    st.divider()

    # ---- Architecture ----
    with st.expander(":material/settings: How it works", expanded=False):
        st.markdown(
            """
            **Agents**
            - Router — intent classification
            - Retrieval — hybrid RAG search
            - Tutor — Socratic response generation
            - Progress — knowledge tracking

            **Knowledge Base**
            Comprehensive notes covering all 7 modules
            of CS50's Introduction to AI with Python.

            **Model Routing**
            - Simple queries → local model
            - Standard → GPT-4o-mini
            - Complex → GPT-4o
            """
        )

    if st.button("Reset Session", use_container_width=True, icon=":material/restart_alt:"):
        st.session_state.agent = LearnAIAgent()
        st.session_state.messages = []
        st.session_state.current_module = None
        st.rerun()


# ===================== MAIN AREA =====================
st.title("LearnAI Agent")

st.markdown(
    f"Comprehensive learning notes from **Harvard CS50's Introduction to Artificial "
    f"Intelligence with Python**. [View the original course]({CS50_AI_URL})"
)

# Module context bar
if st.session_state.current_module:
    mod = COURSE_MODULES[st.session_state.current_module]
    st.markdown(
        f'<div class="module-pill">'
        f"Currently studying: <strong>{st.session_state.current_module}</strong>"
        f" &mdash; {mod['description']}</div>",
        unsafe_allow_html=True,
    )

st.caption("Ask me anything about AI & ML — I use the Socratic method, so expect questions back.")

st.divider()

# ---- Starter cards (empty state) ----
if not st.session_state.messages:
    st.markdown("##### Pick a starting point")
    cols = st.columns(3)
    starters = [
        (":material/search:", "Search & Problem Solving",
         "Explain how A* search works and why the heuristic matters"),
        (":material/model_training:", "Machine Learning Basics",
         "What is the difference between supervised and unsupervised learning?"),
        (":material/hub:", "Neural Networks",
         "How does backpropagation train a neural network?"),
        (":material/menu_book:", "Knowledge & Logic",
         "How does an AI use propositional logic to reason?"),
        (":material/casino:", "Probability & Uncertainty",
         "Explain Bayes' Rule with a real-world example"),
        (":material/translate:", "NLP & Transformers",
         "What is the attention mechanism and why was it a breakthrough?"),
    ]
    for i, (icon, label, prompt) in enumerate(starters):
        with cols[i % 3]:
            if st.button(label, key=f"starter_{i}", use_container_width=True, icon=icon):
                st.session_state.suggested_message = prompt
                st.rerun()


# ---- Chat history ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle suggested message
suggested = st.session_state.pop("suggested_message", None)

# Chat input
user_input = suggested or st.chat_input("Ask about any AI/ML concept...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.chat(user_input)
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.info(
                    "Make sure your `.env` file has a valid `OPENAI_API_KEY`."
                )

    st.rerun()
