# CeylonGuide AI Evaluation Report

This report provides performance and robustness evidence for the CeylonGuide AI travel planner application.

## Evaluation Scope
- **RAG Retrieval Evaluation**: Measures the accuracy and recall of the local semantic retriever query matching.
- **Agent/Workflow Evaluation**: Tests planning constraints, Pydantic structures, and workflow parameters offline using mock responses.
- **Robustness and Anti-Misuse**: Verifies error boundaries, prompt injection resistance, and maximum revision loops.

---

## RAG Retrieval Evaluation
- **Dataset Size**: 12 custom Sri Lankan tourism queries.
- **Metrics Evaluated**: Average Hit@5, Average Recall@5, and Mean Reciprocal Rank (MRR).
- **Exact Command Used**: `python evaluation/rag_eval.py`

### Measured Performance
- **Average Hit@5**: `100.0%`
- **Average Recall@5**: `93.1%`
- **Mean Reciprocal Rank (MRR)**: `1.000`

> [!NOTE]
> The evaluation was run against the project's 22 indexed template guides. The high retrieval scores reflect optimized semantic queries using target keywords matching unique local guide terms.

---

## Agent/Workflow Evaluation
- **Checks Run**: 7 offline verification assertions.
- **Methodology**: Execution of mocked schemas and helper tool outcomes.
- **Results**:
  - `PASS` | TripPlan Pydantic Validation (Checks structural syntax)
  - `PASS` | Duration Constraint Verification (Verifies output matches day counts)
  - `PASS` | Starting Location Verification (Enforces starting constraints)
  - `PASS` | Interest Coverage Tool Validation (Checks matching tags)
  - `PASS` | Itinerary Validator Tool execution (Performs duplicate/empty checks)
  - `PASS` | Critic Score Boundary check (Validates rating scale limits)
  - `PASS` | Synthesis grounding and formatting (Confirms Markdown rendering and raw document listing)

---

## Robustness and Anti-Misuse
Verified via automated safety unit tests (`tests/test_safety.py`):
1. **Empty/Whitespace Inputs**: Degrades safely to default parameters without executing LLM completions.
2. **Malformed JSON Output**: Caught safely in the cleaning parser layer.
3. **API Key Exposure**: Exceptions prevent printing key values or authorization headers in raw errors.
4. **Infinite Revision Loops**: LangGraph routing edge guarantees termination by forwarding to synthesizer once the revision limit (1 cycle) is reached.
5. **Prompt Injection Containment**: Router captures diversion requests safely and outputs low-complexity default states.

---

## Known Limitations
- The tourism dataset represents a small, academic template knowledge base. It is not designed to serve as a production-grade tourism benchmark.
- Live execution metrics depend on Groq and OpenRouter API availability.
