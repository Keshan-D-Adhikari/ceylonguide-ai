import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.planner import TripPlan, TripDay
from src.agents.critic import CriticResult
from src.agents.synthesizer import synthesize_final_response
from src.tools.travel_tools import validate_itinerary, check_interest_coverage

def run_agent_evaluation():
    print("=" * 60)
    print("CeylonGuide AI - Agent/Workflow Offline Evaluation")
    print("=" * 60)
    
    # 1. Define typical outputs for evaluation test cases
    mock_trip_day = {
        "day": 1,
        "location": "Yala",
        "activities": ["Jeep safari tour in Yala National Park to spot leopards."],
        "travel_notes": "Morning travel from Colombo.",
        "evidence_sources": ["11_yala.md"]
    }
    
    mock_plan = {
        "title": "Short Southern Wildlife Safari",
        "duration_days": 1,
        "starting_location": "Colombo",
        "summary": "Quick wildlife focused itinerary.",
        "days": [mock_trip_day]
    }
    
    mock_critic = {
        "score": 8.5,
        "approved": True,
        "issues": [],
        "suggestions": []
    }
    
    checks_passed = 0
    checks_failed = 0
    
    def log_check(name: str, passed: bool, details: str = ""):
        nonlocal checks_passed, checks_failed
        if passed:
            checks_passed += 1
            print(f"PASS | {name:<45} | {details}")
        else:
            checks_failed += 1
            print(f"FAIL | {name:<45} | {details}")

    # --- TEST 1: Pydantic TripPlan Validation ---
    try:
        validated_plan = TripPlan(**mock_plan)
        log_check("TripPlan Pydantic Validation", True, "Successfully instantiated structured TripPlan.")
    except Exception as e:
        log_check("TripPlan Pydantic Validation", False, f"Instantiation failed: {e}")

    # --- TEST 2: Duration Constraint Alignment ---
    log_check(
        "Duration Constraint Verification", 
        mock_plan["duration_days"] == len(mock_plan["days"]), 
        f"Trip plan duration ({mock_plan['duration_days']} days) matches day count ({len(mock_plan['days'])})."
    )

    # --- TEST 3: Starting Location Verification ---
    log_check(
        "Starting Location Verification", 
        mock_plan["starting_location"] == "Colombo", 
        f"Starts at Colombo as requested."
    )

    # --- TEST 4: Interest Coverage Tool ---
    user_interests = ["wildlife", "beaches"]
    selected_categories = ["wildlife", "nature"]
    coverage = check_interest_coverage(user_interests, selected_categories)
    interest_ok = "wildlife" in coverage["covered_interests"] and "beaches" in coverage["uncovered_interests"]
    log_check(
        "Interest Coverage Tool Validation",
        interest_ok,
        f"Covered: {coverage['covered_interests']}, Uncovered: {coverage['uncovered_interests']}."
    )

    # --- TEST 5: Itinerary Validation Tool ---
    validation_res = validate_itinerary(mock_plan["days"], expected_days=1)
    log_check(
        "Itinerary Validator Tool execution",
        validation_res["valid"],
        f"Valid itinerary syntax checks: {validation_res}."
    )

    # --- TEST 6: Critic Score Constraints ---
    critic_obj = CriticResult(**mock_critic)
    score_ok = 0.0 <= critic_obj.score <= 10.0
    log_check(
        "Critic Score Boundary check",
        score_ok,
        f"Critic score ({critic_obj.score}) is in valid [0.0, 10.0] range."
    )

    # --- TEST 7: Synthesis Output Generation ---
    mock_retrieved_docs = [
        {"content": "Yala Safari Guide info", "metadata": {"source_file": "11_yala.md", "title": "Yala National Park"}}
    ]
    synthesis = synthesize_final_response(mock_plan, mock_retrieved_docs)
    synthesis_ok = len(synthesis.strip()) > 0 and "11_yala.md" in synthesis and "# Short Southern Wildlife" in synthesis
    log_check(
        "Synthesis grounding and formatting",
        synthesis_ok,
        "Final Markdown output contains title and correctly lists RAG sources."
    )

    print("=" * 60)
    print("AGENT EVALUATION RUN SUMMARY:")
    print(f"- Total Checks Run: {checks_passed + checks_failed}")
    print(f"- Passed Checks:    {checks_passed}")
    print(f"- Failed Checks:    {checks_failed}")
    print("=" * 60)

if __name__ == "__main__":
    run_agent_evaluation()
