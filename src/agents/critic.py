import json
import logging
from pydantic import BaseModel, Field
from typing import List
from src.models import config
from src.models.factory import generate_groq_completion, generate_openrouter_completion
from src.agents.router import clean_and_parse_json

logger = logging.getLogger(__name__)

class CriticResult(BaseModel):
    """Structured evaluation feedback from the Critic reflection agent."""
    score: float = Field(..., description="Evaluation score from 0.0 (poor) to 10.0 (perfect)")
    approved: bool = Field(..., description="True if the plan meets all expectations and requires no revision")
    issues: List[str] = Field(default_factory=list, description="List of identified issues or errors in the draft")
    suggestions: List[str] = Field(default_factory=list, description="Constructive suggestions for improvement")

def evaluate_itinerary(
    user_request: str,
    duration_days: int,
    starting_location: str,
    interests: list[str],
    draft_itinerary: dict,
    evidence: str
) -> CriticResult:
    """
    Critic reflection agent evaluating the plan constraints, grounding, and coverage.
    """
    system_prompt = f"""You are the Critic Agent for CeylonGuide AI. Your job is to objectively review the generated travel plan draft.

CRITERIA TO CHECK:
1. Duration Match: Does the draft have exactly {duration_days} day(s)?
2. Starting Location Match: Does the draft begin at {starting_location or 'Colombo'}?
3. Interest Alignment: Does the itinerary actually include locations/activities aligned with: {', '.join(interests) if interests else 'General sightseeing'}?
4. Evidence Grounding: Are the recommended destinations supported by the retrieved RAG evidence? Flag any hallucinated/unsupported attractions.
5. Logistical Feasibility: Are the daily schedules realistic? (e.g. not proposing to travel between opposite ends of Sri Lanka in a single afternoon).

KNOWLEDGE BASE EVIDENCE:
{evidence}

DRAFT ITINERARY TO CRITIQUE:
{json.dumps(draft_itinerary, indent=2)}

You MUST output a JSON response matching the following schema:
{{
  "score": float (between 0.0 and 10.0),
  "approved": boolean (true if score >= 7.5 and no major errors, otherwise false),
  "issues": [string],
  "suggestions": [string]
}}

Respond ONLY with valid JSON. Do not write anything outside the JSON object."""

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Determine provider dynamically based on the CRITIC_MODEL name
    critic_model = config.CRITIC_MODEL
    
    try:
        # OpenRouter models typically contain a slash '/' (e.g., deepseek/deepseek-r1)
        if "/" in critic_model:
            raw_response = generate_openrouter_completion(
                messages=messages,
                model=critic_model,
                temperature=0.1
            )
        else:
            raw_response = generate_groq_completion(
                messages=messages,
                model=critic_model,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
        parsed_dict = clean_and_parse_json(raw_response)
        return CriticResult(**parsed_dict)
    except Exception as e:
        logger.error(f"Critic Agent evaluation failed: {e}")
        # Graceful fallback: return a default rejection to attempt a recovery or proceed
        return CriticResult(
            score=5.0,
            approved=False,
            issues=[f"Critic evaluation encountered an internal error: {e}"],
            suggestions=["Please check model connectivity and re-verify itinerary structure."]
        )
