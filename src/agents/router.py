import re
import json
import logging
from pydantic import BaseModel, Field
from typing import List, Optional
from src.models.factory import generate_groq_completion
from src.models import config

logger = logging.getLogger(__name__)

# Supported interest options
SUPPORTED_INTERESTS = [
    "beaches", "wildlife", "culture", "history", 
    "nature", "hiking", "adventure", "food", "relaxation"
]

class RouterDecision(BaseModel):
    """
    Structured response schema representing the Router Agent's decision.
    This encapsulates the 'Routing' agentic design pattern.
    """
    intent: str = Field(
        ..., 
        description="Classified user intent: 'itinerary_planning', 'general_query', 'greeting', or 'unknown'"
    )
    duration_days: Optional[int] = Field(
        None, 
        description="Number of days requested for the trip, constrained to 1-14 days"
    )
    starting_location: Optional[str] = Field(
        None, 
        description="Starting town/city in Sri Lanka (e.g. Colombo, Negombo)"
    )
    interests: List[str] = Field(
        default_factory=list, 
        description="List of detected tourist interests"
    )
    budget: str = Field(
        "moderate", 
        description="Budget level, must be either 'budget', 'moderate', or 'premium'"
    )
    requires_rag: bool = Field(
        ..., 
        description="Whether local database search is required to ground the answer"
    )
    requires_planning: bool = Field(
        ..., 
        description="Whether a multi-day day-by-day plan is requested"
    )
    complexity: str = Field(
        "medium", 
        description="Calculated complexity: 'low', 'medium', or 'high' based on days & preferences"
    )

def clean_and_parse_json(text: str) -> dict:
    """
    Resilient JSON parsing layer to extract structured information.
    Handles markdown fences, leading/trailing conversational filler, and basic errors.
    """
    # 1. Clean markdown code fences if present
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    json_candidate = match.group(1) if match else text
    
    # 2. Try to locate outer braces
    braces_match = re.search(r"(\{.*\})", json_candidate, re.DOTALL)
    if braces_match:
        json_candidate = braces_match.group(1)
        
    json_candidate = json_candidate.strip()
    
    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed for string: {json_candidate}. Error: {e}")
        raise ValueError("Model output did not contain valid JSON.")

def normalize_router_output(data: dict) -> dict:
    """Normalizes interests, duration constraints, and budget levels."""
    # Normalize interests (lower case, check against supported list)
    raw_interests = data.get("interests", [])
    if isinstance(raw_interests, list):
        normalized = []
        for item in raw_interests:
            item_str = str(item).lower().strip()
            if item_str in SUPPORTED_INTERESTS:
                normalized.append(item_str)
        data["interests"] = list(set(normalized))
    else:
        data["interests"] = []
        
    # Constrain duration to 1-14 days
    duration = data.get("duration_days")
    if duration is not None:
        try:
            d_val = int(duration)
            if d_val < 1:
                data["duration_days"] = 1
            elif d_val > 14:
                data["duration_days"] = 14
            else:
                data["duration_days"] = d_val
        except (ValueError, TypeError):
            data["duration_days"] = None
            
    # Normalize budget
    budget = str(data.get("budget", "moderate")).lower().strip()
    if budget not in ["budget", "moderate", "premium"]:
        data["budget"] = "moderate"
        
    # Normalize complexity
    complexity = str(data.get("complexity", "medium")).lower().strip()
    if complexity not in ["low", "medium", "high"]:
        data["complexity"] = "medium"
        
    return data

def route_query(user_query: str) -> RouterDecision:
    """
    Router Agent executing the 'Routing' design pattern.
    Classifies intent, extracts constraints, and outputs validated configuration state.
    """
    # Guard against empty/whitespace queries
    if not user_query or not user_query.strip():
        return RouterDecision(
            intent="unknown",
            duration_days=None,
            starting_location=None,
            interests=[],
            budget="moderate",
            requires_rag=False,
            requires_planning=False,
            complexity="low"
        )
        
    system_prompt = f"""You are the Router Agent for CeylonGuide AI. Your task is to analyze the user's travel request for Sri Lanka and output a structured JSON decision.

SUPPORTED INTERESTS:
{", ".join(SUPPORTED_INTERESTS)}

CLASSIFICATION RULES:
1. intent: Set to 'itinerary_planning' if the user asks for a day-by-day plan or route. Set to 'general_query' if they ask single specific questions (e.g. "What is Kandy famous for?"). Set to 'greeting' if it's just a greeting (e.g. "hi"). Otherwise 'unknown'.
2. duration_days: Extract any integer representing the duration of the trip (e.g., 5 days). If not specified, set to null.
3. starting_location: Extract the starting city or town in Sri Lanka if mentioned (e.g. Colombo, Negombo). If not specified, set to null.
4. interests: Identify interests from the user request that map to our SUPPORTED INTERESTS.
5. budget: Classify the budget level if mentioned. Choose from: 'budget', 'moderate', 'premium'. Default is 'moderate'.
6. requires_rag: Set to true if the request requires factual knowledge about Sri Lankan tourism destinations, geography, or guides.
7. requires_planning: Set to true if the intent is 'itinerary_planning'.
8. complexity: Assign 'low', 'medium', or 'high'. High complexity is for trips longer than 7 days or having 4+ specific interests.

You MUST respond with a single JSON object matching this schema:
{{
    "intent": "itinerary_planning" | "general_query" | "greeting" | "unknown",
    "duration_days": int | null,
    "starting_location": string | null,
    "interests": [string],
    "budget": "budget" | "moderate" | "premium",
    "requires_rag": boolean,
    "requires_planning": boolean,
    "complexity": "low" | "medium" | "high"
}}

Respond ONLY with valid JSON. Do not include introductory text, conversational filler, or final explanations."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    # Perform completion using fast Groq router model
    raw_response = generate_groq_completion(
        messages=messages,
        model=config.GROQ_ROUTER_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    parsed_dict = clean_and_parse_json(raw_response)
    normalized_dict = normalize_router_output(parsed_dict)
    
    # Validate structure using Pydantic
    return RouterDecision(**normalized_dict)
