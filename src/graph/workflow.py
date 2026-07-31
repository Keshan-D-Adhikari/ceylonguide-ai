import logging
from langgraph.graph import StateGraph, START, END
from src.graph.state import AgentState
from src.agents.router import route_query
from src.agents.retrieval import retrieve_tourism_info
from src.agents.planner import plan_trip
from src.agents.critic import evaluate_itinerary
from src.agents.synthesizer import synthesize_final_response
from src.rag.retriever import TourismRetriever

logger = logging.getLogger(__name__)

# Define Node functions

def router_node(state: AgentState) -> dict:
    logger.info("Starting Router Node")
    query = state["user_request"]
    
    try:
        decision = route_query(query)
        event = f"✓ Router Agent identified user intent as '{decision.intent}'"
        return {
            "intent": decision.intent,
            "duration_days": decision.duration_days,
            "starting_location": decision.starting_location,
            "interests": decision.interests,
            "budget": decision.budget,
            "complexity": decision.complexity,
            "requires_rag": decision.requires_rag,
            "requires_planning": decision.requires_planning,
            "workflow_events": state.get("workflow_events", []) + [event],
            "errors": state.get("errors", [])
        }
    except Exception as e:
        logger.error(f"Router node failed: {e}")
        return {
            "intent": "unknown",
            "requires_rag": False,
            "requires_planning": False,
            "workflow_events": state.get("workflow_events", []) + ["✗ Router Agent failed to categorize request."],
            "errors": state.get("errors", []) + [f"Router error: {str(e)}"]
        }

def retrieval_node(state: AgentState) -> dict:
    logger.info("Starting Retrieval Node")
    
    if not state.get("requires_rag"):
        event = "✓ Skipping Retrieval (RAG not required for this request)"
        return {
            "retrieved_documents": [],
            "workflow_events": state.get("workflow_events", []) + [event]
        }
        
    query = state["user_request"]
    interests = state.get("interests", [])
    
    try:
        results = retrieve_tourism_info(query, interests)
        
        # Structure the results for the state
        docs = []
        for r in results:
            docs.append({
                "content": r["content"],
                "metadata": r["metadata"],
                "score": r.get("score", 0.0)
            })
            
        event = f"✓ Retrieval Agent retrieved {len(docs)} relevant knowledge chunk(s)"
        return {
            "retrieved_documents": docs,
            "workflow_events": state.get("workflow_events", []) + [event]
        }
    except Exception as e:
        logger.error(f"Retrieval node failed: {e}")
        return {
            "retrieved_documents": [],
            "workflow_events": state.get("workflow_events", []) + ["✗ Retrieval Agent failed to query database."],
            "errors": state.get("errors", []) + [f"Retrieval error: {str(e)}"]
        }

def planner_node(state: AgentState) -> dict:
    logger.info("Starting Planner Node")
    
    if not state.get("requires_planning"):
        # Synthesize simple queries directly
        return {
            "workflow_events": state.get("workflow_events", []) + ["✓ Skipping Planner (planning not required)"]
        }
        
    query = state["user_request"]
    duration = state.get("duration_days") or 3
    start_loc = state.get("starting_location") or "Colombo"
    interests = state.get("interests", [])
    budget = state.get("budget", "moderate")
    docs = state.get("retrieved_documents", [])
    
    # Format retrieved context
    from src.rag.retriever import TourismRetriever
    # Mock/dummy formatting since we don't have retriever instance, format directly:
    evidence_blocks = []
    for idx, doc in enumerate(docs):
        evidence_blocks.append(
            f"Source [{idx+1}]: {doc['metadata'].get('title')} (File: {doc['metadata'].get('source_file')})\n"
            f"Content: {doc['content']}"
        )
    evidence = "\n---\n".join(evidence_blocks) if evidence_blocks else "No evidence available."
    
    # Check if this is a revision cycle
    critic_feedback = None
    critique = state.get("critique")
    if critique and not critique.get("approved"):
        critic_feedback = f"Issues found: {', '.join(critique.get('issues', []))}\nSuggestions: {', '.join(critique.get('suggestions', []))}"
        event = f"✓ Planner Agent revising itinerary (Cycle {state.get('revision_count', 0) + 1})"
    else:
        event = f"✓ Planner Agent generating {duration}-day itinerary draft"
        
    try:
        plan = plan_trip(
            user_request=query,
            duration_days=duration,
            starting_location=start_loc,
            interests=interests,
            budget=budget,
            evidence=evidence,
            critic_feedback=critic_feedback
        )
        
        return {
            "draft_itinerary": plan.model_dump(),
            "workflow_events": state.get("workflow_events", []) + [event]
        }
    except Exception as e:
        logger.error(f"Planner node failed: {e}")
        return {
            "workflow_events": state.get("workflow_events", []) + ["✗ Planner Agent failed to create itinerary."],
            "errors": state.get("errors", []) + [f"Planner error: {str(e)}"]
        }

def critic_node(state: AgentState) -> dict:
    logger.info("Starting Critic Node")
    draft = state.get("draft_itinerary")
    
    if not draft or not state.get("requires_planning"):
        return {
            "critique": {"approved": True, "score": 10.0, "issues": [], "suggestions": []}
        }
        
    query = state["user_request"]
    duration = state.get("duration_days") or 3
    start_loc = state.get("starting_location") or "Colombo"
    interests = state.get("interests", [])
    docs = state.get("retrieved_documents", [])
    
    # Format evidence
    evidence_blocks = []
    for idx, doc in enumerate(docs):
        evidence_blocks.append(
            f"Source [{idx+1}]: {doc['metadata'].get('title')} (File: {doc['metadata'].get('source_file')})\n"
            f"Content: {doc['content']}"
        )
    evidence = "\n---\n".join(evidence_blocks) if evidence_blocks else "No evidence available."
    
    try:
        evaluation = evaluate_itinerary(
            user_request=query,
            duration_days=duration,
            starting_location=start_loc,
            interests=interests,
            draft_itinerary=draft,
            evidence=evidence
        )
        
        new_revision_count = state.get("revision_count", 0) + 1
        
        status_symbol = "✓" if evaluation.approved else "⚠"
        event = f"{status_symbol} Critic Agent rated itinerary {evaluation.score}/10 (Approved: {evaluation.approved})"
        
        return {
            "critique": evaluation.model_dump(),
            "revision_count": new_revision_count,
            "workflow_events": state.get("workflow_events", []) + [event]
        }
    except Exception as e:
        logger.error(f"Critic node failed: {e}")
        return {
            "critique": {"approved": True, "score": 5.0, "issues": [], "suggestions": []},
            "revision_count": state.get("revision_count", 0) + 1,
            "workflow_events": state.get("workflow_events", []) + ["✗ Critic Agent check encountered an error. Proceeding anyway."],
            "errors": state.get("errors", []) + [f"Critic error: {str(e)}"]
        }

def synthesizer_node(state: AgentState) -> dict:
    logger.info("Starting Synthesizer Node")
    draft = state.get("draft_itinerary")
    docs = state.get("retrieved_documents", [])
    
    try:
        if state.get("requires_planning") and draft:
            final_text = synthesize_final_response(draft, docs)
            event = "✓ Synthesizer formatted and grounded final travel plan"
        else:
            # Fallback for general queries without planning
            if docs:
                source_list = [f"- {d['metadata'].get('title')} (`{d['metadata'].get('source_file')}`)" for d in docs]
                joined_sources = "\n".join(source_list)
                final_text = (
                    f"### Answer to query:\n\n{docs[0]['content']}\n\n"
                    f"**Sources Referenced:**\n{joined_sources}"
                )
            else:
                final_text = "I'm sorry, I couldn't find enough information in the local guides to resolve your query."
            event = "✓ Synthesizer generated query answer"
            
        return {
            "final_answer": final_text,
            "workflow_events": state.get("workflow_events", []) + [event]
        }
    except Exception as e:
        logger.error(f"Synthesizer node failed: {e}")
        return {
            "final_answer": "An error occurred while compiling the final response.",
            "workflow_events": state.get("workflow_events", []) + ["✗ Synthesizer failed."],
            "errors": state.get("errors", []) + [f"Synthesizer error: {str(e)}"]
        }

# Define Conditional Edge

def route_critic_decision(state: AgentState) -> str:
    critique = state.get("critique")
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 1)
    
    if not critique:
        return "synthesizer"
        
    approved = critique.get("approved", True)
    
    if approved:
        logger.info("Itinerary approved by Critic. Directing to Synthesizer.")
        return "synthesizer"
    elif revision_count >= max_revisions:
        logger.info(f"Itinerary revision limit reached ({revision_count}/{max_revisions}). Directing to Synthesizer.")
        return "synthesizer"
    else:
        logger.info(f"Itinerary needs revision ({revision_count}/{max_revisions} completed). Directing back to Planner.")
        return "planner"

# Build LangGraph Workflow

def create_workflow() -> StateGraph:
    workflow = StateGraph(AgentState)
    
    # Register Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("synthesizer", synthesizer_node)
    
    # Establish Edges
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "retrieval")
    workflow.add_edge("retrieval", "planner")
    workflow.add_edge("planner", "critic")
    
    # Conditional edge from Critic based on reflection evaluation
    workflow.add_conditional_edges(
        "critic",
        route_critic_decision,
        {
            "planner": "planner",
            "synthesizer": "synthesizer"
        }
    )
    
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()
