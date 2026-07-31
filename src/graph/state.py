from typing import TypedDict, List, Optional, Any

class AgentState(TypedDict):
    """
    Shared state container passed between agents in the LangGraph workflow.
    This enables structured agent communication.
    """
    # Inputs & Router decisions
    user_request: str
    intent: str
    duration_days: Optional[int]
    starting_location: Optional[str]
    interests: List[str]
    budget: str
    complexity: str
    requires_rag: bool
    requires_planning: bool
    
    # Retrieval outputs
    retrieved_documents: List[dict]
    
    # Planner outputs
    draft_itinerary: Optional[dict]
    
    # Critic outputs
    critique: Optional[dict]
    revision_count: int
    max_revisions: int
    
    # Final synthesized answer
    final_answer: Optional[str]
    
    # Execution tracing & error handling
    workflow_events: List[str]
    errors: List[str]
