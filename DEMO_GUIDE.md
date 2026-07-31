# CeylonGuide AI – Demonstration Guide

This guide is designed to assist you in demonstrating CeylonGuide AI to your university lecturer, walking through the architecture, agentic design patterns, and potential examination questions.

---

## 1. 30-Second Introduction
> "CeylonGuide AI is a RAG-grounded multi-agent travel planner for Sri Lanka. Instead of acting as a generic chatbot that hallucinates routes, it uses a structured state-machine workflow coordinated by LangGraph. It routes requests, extracts travel constraints, fetches local tourism guides, structures day-by-day itineraries, and runs a self-correction loop where a Critic Agent evaluates the draft plan and requests revisions if necessary—all while keeping system API keys and secrets secure."

---

## 2. Architecture Walkthrough
1. **User Input**: Input is submitted through the Streamlit interface (Structured Form or Natural Language Prompt).
2. **Router Agent** (Groq): Extracts intent, starting location, budget, and duration.
3. **Retrieval Agent** (Local RAG): Queries ChromaDB using MiniLM embeddings to retrieve top relevant local guide chunks.
4. **Trip Planner Agent** (OpenRouter): Generates a structured JSON travel plan matching day counts and interests, grounded in RAG text.
5. **Critic Agent** (Groq/OpenRouter): Scores the draft (0-10) and checks constraints.
6. **Revision Loop**: If score < 7.5 and revision count < 1, the Planner revises the plan incorporating the Critic's suggestions.
7. **Synthesizer**: Compiles the final plan into formatted Markdown and lists reference source guides.

---

## 3. Four Agentic Patterns
- **Routing**: Router Agent classifies if a query is a greeting, general query, or planning request, avoiding unnecessary RAG queries.
- **Tool Use**: Use of deterministic Python helpers in `travel_tools.py` (Interest Coverage, Day Allocator, Itinerary Validator).
- **Planning**: Decomposition of the trip into daily schedules.
- **Reflection / Self-Critique**: Critic reviews the output and sends feedback back to the Planner, restricted to a maximum of one revision loop to prevent terminal infinite runs.

---

## 4. RAG Explanation
- **Why RAG?** Large Language Models do not natively know the exact geography, attraction hours, or travel notes of Sri Lankan cities. RAG retrieves local document chunks and inserts them into the planner's context, forcing the model to generate grounded schedules and reducing hallucinations.
- **Components**: 22 local Markdown documents chunked at 800 characters with 120 character overlap, indexed using the local `sentence-transformers/all-MiniLM-L6-v2` model in a persistent ChromaDB instance.

---

## 5. Suggested Demonstration Steps
1. **Show the Codebase**: Point to the clean structure under `src/` (separation of agents, graph workflow, RAG, tools, and configurations).
2. **Start the App**: Run `streamlit run app.py` and show the sidebar showing the API system connection status.
3. **Run a Demo Query**:
   - Use the sample prompt: *"I am visiting Sri Lanka for 5 days starting from Colombo. I like wildlife, beaches, and cultural attractions."*
4. **Show the Agent Trace**: Expand the "Agent Process Details" section. Explain each step (Router decision, Retrieval result count, Critic scorecard, and revision cycle).
5. **Show RAG Sources**: Show the referenced guide filenames at the bottom of the itinerary.
6. **Run Pytest**: Show the unit test execution (`python -m pytest -v`) demonstrating 31 passing assertions.

---

## 6. Likely Lecturer Questions + Short Answers

### Q1: What makes this application "agentic" rather than a standard RAG pipeline?
* **Answer**: A standard RAG pipeline is single-step (Retrieve → Generate). This application uses autonomous, specialized nodes (Router, Planner, Critic) with distinct model roles, tool dependencies, state modifications, and a conditional loop (Reflection) that reviews and corrects its own plans.

### Q2: Why did you use two different LLM providers (Groq and OpenRouter)?
* **Answer**: We use a fast, lightweight model on **Groq** for high-speed intent routing and JSON extraction, and a deep-reasoning reasoning model on **OpenRouter** to handle complex logistical travel scheduling.

### Q3: how does the Critic Agent communicate revisions to the Planner?
* **Answer**: Agents share a typed state dict (`AgentState` in `src/graph/state.py`). The Critic writes its review to `critique` in the state. If the plan is rejected, the workflow transitions back to the Planner, which reads the `critique` field to modify the itinerary.

### Q4: How do you prevent the Critic and Planner from looping infinitely?
* **Answer**: The LangGraph conditional edge `route_critic_decision` in `workflow.py` checks `revision_count`. If it reaches `max_revisions` (1 cycle), the loop terminates and forwards the plan directly to the Synthesizer node, even if the score is still low.

### Q5: How are API keys protected?
* **Answer**: API keys are loaded dynamically via a config helper checking environment variables (`.env`) or Streamlit secrets (`secrets.toml`). Both files, along with `data/chroma/`, are explicitly excluded in `.gitignore` to prevent leaks to Git repositories.

---

## 7. Commands to Remember
- Ingest documents: `python scripts/ingest.py`
- Run Streamlit: `streamlit run app.py`
- Run Unit Tests: `python -m pytest -v`
- Run RAG evaluation metrics: `python evaluation/rag_eval.py`
- Run Agent evaluation checks: `python evaluation/agent_eval.py`
