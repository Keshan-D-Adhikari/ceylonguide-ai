import json
import logging
from pydantic import BaseModel, Field
from typing import List, Optional
from src.models.factory import generate_openrouter_completion
from src.models import config
from src.agents.router import clean_and_parse_json

logger = logging.getLogger(__name__)

class TripDay(BaseModel):
    day: int = Field(..., description="Day number of the itinerary (e.g., 1, 2, 3)")
    location: str = Field(..., description="Main city, national park, or site visited")
    activities: List[str] = Field(..., description="Detailed points of things to do/see")
    travel_notes: str = Field(..., description="Logistics, transfer suggestions, or travel advice")
    evidence_sources: List[str] = Field(..., description="List of raw files from metadata used as evidence (e.g., '08_ella.md')")

class TripPlan(BaseModel):
    """Structured Pydantic schema for the generated travel plan."""
    title: str = Field(..., description="A short, catchy title for the travel plan")
    duration_days: int = Field(..., description="Total days matching the request")
    starting_location: str = Field(..., description="Starting location of the plan")
    summary: str = Field(..., description="A brief narrative summary explaining the route concept")
    days: List[TripDay] = Field(..., description="Day-by-day breakdown")

def plan_trip(
    user_request: str, 
    duration_days: int, 
    starting_location: str, 
    interests: list[str], 
    budget: str, 
    evidence: str,
    critic_feedback: str = None
) -> TripPlan:
    """
    Trip Planner Agent using the OpenRouter reasoning model.
    Generates a structured day-by-day itinerary grounded in RAG evidence.
    """
    starting_city = starting_location or "Colombo"
    
    # Base prompt instructions
    system_prompt = f"""You are the Trip Planner Agent for CeylonGuide AI. Your job is to create a structured day-by-day travel itinerary for Sri Lanka.
    
Your itinerary MUST be strictly grounded in the provided EVIDENCE. Do not invent destinations, travel times, or attractions that are not mentioned in the evidence.
Make sure you respect the travel days constraint and the starting location.

CONSTRAINTS:
- Duration: {duration_days} Day(s)
- Starting Location: {starting_city}
- User Interests: {', '.join(interests) if interests else 'General sightseeing'}
- Travel Style / Budget: {budget}

EVIDENCE FROM KNOWLEDGE BASE:
{evidence}
"""

    if critic_feedback:
        system_prompt += f"\n\nCRITIC REVISION FEEDBACK TO INCORPORATE:\n{critic_feedback}\n\nPlease revise the previous draft to resolve all points raised by the Critic."

    system_prompt += """

You MUST return a JSON response matching the following schema:
{
  "title": "catchy title of trip",
  "duration_days": int,
  "starting_location": "starting city name",
  "summary": "overall plan summary",
  "days": [
    {
      "day": 1,
      "location": "location name",
      "activities": ["activity 1", "activity 2"],
      "travel_notes": "travel/logistical notes",
      "evidence_sources": ["source_filename.md"]
    }
  ]
}

Ensure the output is valid JSON. Do not include extra conversational text outside the JSON object."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request}
    ]
    
    # Run completion using OpenRouter reasoning model
    raw_response = generate_openrouter_completion(
        messages=messages,
        model=config.OPENROUTER_REASONING_MODEL,
        temperature=0.2
    )
    
    parsed_dict = clean_and_parse_json(raw_response)
    
    # Basic field overrides to enforce safety
    parsed_dict["duration_days"] = duration_days
    parsed_dict["starting_location"] = starting_city
    
    # Validate structure using Pydantic
    return TripPlan(**parsed_dict)
