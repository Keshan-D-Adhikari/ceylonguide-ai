import unittest
from unittest.mock import patch, MagicMock
from src.graph.workflow import create_workflow, route_critic_decision

class TestWorkflow(unittest.TestCase):
    
    def test_route_critic_decision_approved(self):
        state = {
            "critique": {"approved": True, "score": 8.0, "issues": []},
            "revision_count": 0,
            "max_revisions": 1
        }
        next_step = route_critic_decision(state)
        self.assertEqual(next_step, "synthesizer")

    def test_route_critic_decision_needs_revision(self):
        state = {
            "critique": {"approved": False, "score": 5.0, "issues": ["Incorrect days"]},
            "revision_count": 0,
            "max_revisions": 1
        }
        next_step = route_critic_decision(state)
        self.assertEqual(next_step, "planner")

    def test_route_critic_decision_max_revisions_limit(self):
        state = {
            "critique": {"approved": False, "score": 5.0, "issues": ["Incorrect days"]},
            "revision_count": 1,
            "max_revisions": 1
        }
        next_step = route_critic_decision(state)
        self.assertEqual(next_step, "synthesizer")

    @patch("src.graph.workflow.route_query")
    @patch("src.graph.workflow.retrieve_tourism_info")
    @patch("src.graph.workflow.plan_trip")
    @patch("src.graph.workflow.evaluate_itinerary")
    @patch("src.graph.workflow.synthesize_final_response")
    def test_workflow_state_passing_mock(
        self, mock_synth, mock_critic, mock_plan, mock_retrieve, mock_route
    ):
        # 1. Setup mocks
        mock_route.return_value = MagicMock(
            intent="itinerary_planning",
            duration_days=2,
            starting_location="Colombo",
            interests=["beaches"],
            budget="budget",
            complexity="low",
            requires_rag=True,
            requires_planning=True
        )
        mock_retrieve.return_value = [
            {"content": "Colombo beaches", "metadata": {"source_file": "01_colombo.md", "title": "Colombo"}}
        ]
        
        mock_trip_day = MagicMock()
        mock_trip_day.model_dump.return_value = {
            "day": 1, "location": "Colombo", "activities": ["beach walk"], "travel_notes": "None", "evidence_sources": ["01_colombo.md"]
        }
        
        mock_trip_plan = MagicMock()
        mock_trip_plan.model_dump.return_value = {
            "title": "Quick beach tour",
            "duration_days": 2,
            "starting_location": "Colombo",
            "summary": "Short stay",
            "days": [mock_trip_day.model_dump.return_value]
        }
        mock_plan.return_value = mock_trip_plan
        
        # First return not approved, then second approved (testing loop)
        mock_critic.side_effect = [
            MagicMock(approved=False, score=4.0, model_dump=lambda: {"approved": False, "score": 4.0, "issues": ["Needs more activities"], "suggestions": []}),
            MagicMock(approved=True, score=9.0, model_dump=lambda: {"approved": True, "score": 9.0, "issues": [], "suggestions": []})
        ]
        
        mock_synth.return_value = "Formatted Final Plan markdown output."
        
        # 2. Initialize and run compiled graph workflow
        app = create_workflow()
        
        initial_state = {
            "user_request": "I want a 2 day beach trip from Colombo",
            "revision_count": 0,
            "max_revisions": 1,
            "errors": [],
            "workflow_events": []
        }
        
        final_state = app.invoke(initial_state)
        
        # 3. Assert outputs
        self.assertEqual(final_state["intent"], "itinerary_planning")
        self.assertEqual(final_state["final_answer"], "Formatted Final Plan markdown output.")
        # Ensure revision count incremented to 1 (max_revisions limit stopped it from loop 2)
        self.assertEqual(final_state["revision_count"], 1)
        self.assertTrue(len(final_state["workflow_events"]) > 0)
        
if __name__ == "__main__":
    unittest.main()
