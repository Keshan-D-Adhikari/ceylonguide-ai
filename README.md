# CeylonGuide AI – A Multi-Agent RAG-Based Sri Lanka Travel Planner

CeylonGuide AI is an intelligent travel planning application that helps tourists create personalized Sri Lanka itineraries based on trip days, starting point, travel style, and specific interests. 

The application implements a multi-agent orchestration architecture utilizing **LangGraph** to coordinate agents, **Retrieval-Augmented Generation (RAG)** grounded in a local vector database for factual guides, and a **Streamlit** user interface.

## Key Features
- **Flexible Input Methods**: Structured parameters (days, starting location, interests, budget style) or natural-language prompt input.
- **Robust Orchestrated Planning**: Converts travel requirements into logical daily destinations and schedules.
- **Academic Data Integrity**: Explicitly lists reference RAG source documents, with disclosures showing they are template-based academic drafts.
- **Orchestration Log Trace**: Visualizes agent flow steps, decision factors, and Critic evaluation reports in the UI.

---

## Agentic Design Patterns
CeylonGuide AI explicitly demonstrates four core agentic design patterns:

1. **Routing**
   - *Implementation*: The **Router Agent** ([src/agents/router.py](file:///C:/Users/kesha/Desktop/ceylonguide-ai/src/agents/router.py)) analyzes requests using the fast Groq model to extract intent, duration constraints, starting point, budget, and decide if RAG/planning is required.
   
2. **Tool Use**
   - *Implementation*: Agents utilize deterministic helper calculations ([src/tools/travel_tools.py](file:///C:/Users/kesha/Desktop/ceylonguide-ai/src/tools/travel_tools.py)) to estimate budget styles, validate day allocations, check interest coverage, and identify plan errors.
   
3. **Planning / Task Decomposition**
   - *Implementation*: The **Trip Planner Agent** ([src/agents/planner.py](file:///C:/Users/kesha/Desktop/ceylonguide-ai/src/agents/planner.py)) decomposes complex itineraries day-by-day based on RAG evidence using the OpenRouter reasoning model.
   
4. **Reflection / Self-Critique**
   - *Implementation*: The **Critic Agent** ([src/agents/critic.py](file:///C:/Users/kesha/Desktop/ceylonguide-ai/src/agents/critic.py)) reviews drafts against requested guidelines, scores them (0-10), and details revision requirements, directing the planner to execute exactly one revision loop if needed.

---

## Multi-Agent Architecture

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

### Agent Communication
Agents communicate using a shared LangGraph state structure ([src/graph/state.py](file:///C:/Users/kesha/Desktop/ceylonguide-ai/src/graph/state.py)) tracking extracted parameters (`intent`, `duration_days`, `starting_location`, `interests`, `budget`), retrieved documentation chunks (`retrieved_documents`), draft itineraries (`draft_itinerary`), critic reports (`critique`, `revision_count`), execution step logs, and internal errors.

---

## Technology Stack
- **Core Logic**: Python 3.11
- **UI Framework**: Streamlit
- **Agent Orchestration**: LangGraph
- **Validation**: Pydantic v2
- **Vector Database**: ChromaDB (Local persistent)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Local execution)
- **LLM Providers**: Groq (Intent routing & reflection) and OpenRouter (Planner reasoning)
- **Unit Testing**: Pytest

---

## Model Selection
| Agent | Model Option | Provider | Purpose | Reason |
| :--- | :--- | :--- | :--- | :--- |
| **Router** | Configured via `GROQ_ROUTER_MODEL` | Groq | Intent & structured param extraction | High speed and JSON schema support. |
| **Planner** | Configured via `OPENROUTER_REASONING_MODEL` | OpenRouter | Day-by-day itinerary synthesis | High reasoning ability for logistics. |
| **Critic** | Configured via `CRITIC_MODEL` | Groq / OpenRouter | Quality verification & scorecard | Adaptable for structural inspection. |

---

## RAG Ingestion & Retrieval Pipeline
```
Raw Guides (.md) -> loaders.py (frontmatter parse) -> chunking.py (800 char overlap) -> embeddings.py (MiniLM) -> vector_store.py (ChromaDB) -> retriever.py (retrieval)
```

The database indexes **22 Sri Lankan destinations** (Colombo, Ella, Yala, Galle, Jaffna, etc.) loaded from [data/raw/](file:///C:/Users/kesha/Desktop/ceylonguide-ai/data/raw/).

> [!IMPORTANT]
> **Academic Data Disclosure**: The tourism guide documents in the corpus contain unverified academic template placeholder information. They do not represent official tourist board documentation and must be manually verified before live travel use.

---

## Installation & Setup

1. **Clone & Set Up Environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Settings**:
   Copy `.env.example` to `.env` and fill in API keys:
   ```bash
   copy .env.example .env
   ```
   Provide valid `GROQ_API_KEY` and `OPENROUTER_API_KEY` in your `.env`.

4. **Ingest Documents**:
   Index the 22 tourism documents into ChromaDB:
   ```bash
   python scripts/ingest.py
   ```

5. **Start Application**:
   ```bash
   streamlit run app.py
   ```

---

## Testing & Quality Assurance
Run the unit test suite containing 31 test assertions (covering metrics, loaders, chunkers, routing logic, safety, and LangGraph state passing):
```bash
python -m pytest -v
```

---

## Evaluation Results
The project contains formal, reproducible evaluation results (detailed in [evaluation/REPORT.md](file:///C:/Users/kesha/Desktop/ceylonguide-ai/evaluation/REPORT.md)):
- **RAG Retrieval Evaluation** (12 queries, K=5):
  - Hit@5: `100.0%`
  - Recall@5: `93.1%`
  - Mean Reciprocal Rank (MRR): `1.000`
- **Agent/Workflow Evaluation**: 7/7 constraints checks passed.
- **Robustness/Anti-Misuse Tests**: Passed safety boundaries for prompt injections, key exposure, and loop limits.

*Note: Live LLM evaluations were bypassed using mocks during the automated suite run to prevent rate limits.*

---

## Known Limitations
- Guide content consists of template academic placeholder data.
- Itinerary suggestions (travel times, budget levels) use illustrative heuristics, not real-time flight, weather, or hotel APIs.
- Real API runs require active Groq/OpenRouter keys and network access.

---

## Academic Integrity & AI Usage Declaration
Generative AI tools (including Claude, Gemini, and GitHub Copilot) were utilized as development assistants. The student has reviewed, modified, and tested all generated codes to ensure design validity.
