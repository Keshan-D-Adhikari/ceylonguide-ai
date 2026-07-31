# CeylonGuide AI Technical Architecture Guide

This document describes the technical architecture, design patterns, RAG data ingestion flow, and orchestration mechanics of the CeylonGuide AI travel planner.

---

## Component Architecture Overview
```
                   +----------------------------------+
                   |          Streamlit UI            |
                   |            (app.py)              |
                   +----------------+-----------------+
                                    |
                                    v
                   +----------------+-----------------+
                   |       LangGraph Workflow         |
                   |      (src/graph/workflow.py)     |
                   +----------------+-----------------+
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
            v                       v                       v
  +------------------+    +------------------+    +------------------+
  |   Router Agent   |    |  Retrieval Agent |    |  Planner Agent   |
  | (agents/router.py)    | (agents/retrieval)    | (agents/planner.py)
  +--------+---------+    +--------+---------+    +--------+---------+
           |                       |                       |
           v                       v                       v
  +--------+---------+    +--------+---------+    +--------+---------+
  |    Groq Client   |    |    Chroma DB     |    |   OpenRouter     |
  | (factory.py client)|  | (vector_store.py)|   | (factory.py client)|
  +------------------+    +------------------+    +------------------+
```

---

## 1. Multi-Agent Orchestration & Shared State
The agent workflow is structured as a state machine using **LangGraph**. The state is defined in [src/graph/state.py](file:///C:/Users/kesha/Desktop/ceylonguide-ai/src/graph/state.py) using `TypedDict` containing the shared execution keys.

### Nodes and Responsibilities
- **Router Node** (`router_node`): Runs the user query through the Router Agent, updating `intent`, `duration_days`, `starting_location`, `interests`, `budget`, and deciding if RAG/planning is needed.
- **Retrieval Node** (`retrieval_node`): Invokes semantic queries against ChromaDB if `requires_rag` is True, setting the `retrieved_documents` state.
- **Planner Node** (`planner_node`): Runs the Trip Planner Agent to build a structured `draft_itinerary`. If a critique exists, it reads feedback to perform a revision.
- **Critic Node** (`critic_node`): Reviews the `draft_itinerary` and records scorecard feedback in `critique`, incrementing the `revision_count`.
- **Synthesizer Node** (`synthesizer_node`): Generates the final Markdown response (`final_answer`) listing RAG sources.

### Conditional Edges
The workflow uses a conditional edge `route_critic_decision` after the Critic evaluation:
- If `critique["approved"]` is True or `revision_count >= max_revisions`, transition to `synthesizer`.
- Else, transition to `planner` (performing a revision cycle).

---

## 2. RAG Architecture & Ingestion Pipeline
The local RAG system uses a persistent ChromaDB database to ground generated schedules in verified guides.

```
+------------------+      +---------------+      +-------------+
| 22 Markdown Docs | ---> | loaders.py    | ---> | cleaning    |
| (data/raw/)      |      | frontmatter   |      | whitespaces |
+------------------+      +---------------+      +-------------+
                                                        |
                                                        v
+------------------+      +---------------+      +-------------+
| vector_store.py  | <--- | embeddings.py | <--- | chunking.py |
| ChromaDB HNWS    |      | local MiniLM  |      | 800 chars   |
+------------------+      +---------------+      +-------------+
```

- **Metadata Parsing**: YAML frontmatter tags are parsed to identify titles, provinces, categories, and source filenames.
- **Chunking**: Text is split into 800-character segments with a 120-character sliding overlap, making clean splits on spaces/newlines where possible.
- **Embeddings**: Text chunks are embedded locally using `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional cosine similarity vectors.

---

## 3. Configuration & Model Factory
- **Configuration** ([src/models/config.py](file:///C:/Users/kesha/Desktop/ceylonguide-ai/src/models/config.py)): Detects and imports variables from environmental variables, `.env` file, or Streamlit secrets in production.
- **Factory** ([src/models/factory.py](file:///C:/Users/kesha/Desktop/ceylonguide-ai/src/models/factory.py)): Centralizes API calls to prevent duplicated completion code. OpenRouter calls are executed using `requests` with standard timeouts, while Groq calls use the official client.

---

## 4. Security & Error Boundaries
- **Secrets Management**: Secrets are never hardcoded or printed. Error handlers catch API failure messages and format them into readable user notices instead of displaying raw headers or stack traces.
- **Infinite Loop Safeguards**: The revision loop has a hard limit (`max_revisions = 1`) that ensures the graph exits revision cycles and finishes synthesis safely.
- **Git Boundaries**: Environment configurations (`.env`), database persistent files (`data/chroma/`), and cached build folders are excluded from Git using `.gitignore`.
