# CeylonGuide AI – A Multi-Agent RAG-Based Sri Lanka Travel Planner

CeylonGuide AI is an intelligent Sri Lanka travel planning application built with Python, LangGraph, Streamlit, and ChromaDB. It helps tourists create highly personalized itineraries using a multi-agent orchestration approach, local RAG knowledge, and multiple configured AI models.

## Architecture Overview

### Agentic Design Patterns
- **Routing**: The Router Agent analyzes user intent and requests to decide if planning/retrieval is needed.
- **Planning / Task Decomposition**: The Trip Planner decomposes complex itineraries into structured sub-steps.
- **Reflection / Critique**: The Critic Agent reviews the plan against constraints and offers a maximum of one revision cycle.
- **Tool Use**: Deterministic local tools for Day Allocation, Budget Estimation, Interest Coverage, and Validation.

### LangGraph Workflow
```mermaid
graph TD
    START([Start]) --> Router[Router Agent]
    Router --> Decision{Is Planning/RAG Required?}
    Decision -- Yes --> Retrieval[Destination Retrieval Agent]
    Decision -- No --> FinalSynth[Final Synthesizer]
    Retrieval --> Planner[Trip Planner Agent]
    Planner --> Critic[Critic Agent]
    Critic --> Approved{Approved?}
    Approved -- No (Revision < 1) --> RevisionPlanner[Trip Planner Agent (Revision)]
    RevisionPlanner --> Critic
    Approved -- Yes / Max Revision Reached --> FinalSynth
    FinalSynth --> END([End])
```

## Folder Structure
- `src/`: Main source code package containing agents, workflow, RAG, models, tools, and utilities.
- `data/raw/`: Knowledge base raw tourism documents (20+ destination guides).
- `scripts/`: Ingestion and RAG evaluation tools.
- `tests/`: Automated unit tests.
- `evaluation/`: RAG evaluation queries and markdown results.
- `docs/`: In-depth documentation.

## Setup & Running
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure `.env` based on `.env.example`.
3. Ingest documents:
   ```bash
   python scripts/ingest.py
   ```
4. Run application:
   ```bash
   streamlit run app.py
   ```
