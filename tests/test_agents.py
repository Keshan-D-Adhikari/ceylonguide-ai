import unittest
from unittest.mock import patch, MagicMock
from src.agents.retrieval import retrieve_tourism_info
from src.agents.planner import plan_trip, TripPlan
from src.agents.critic import evaluate_itinerary, CriticResult
from src.agents.synthesizer import synthesize_final_response

class TestAgents(unittest.TestCase):

    @patch("src.agents.retrieval._retriever")
    def test_retrieval_agent_metadata_preservation(self, mock_retriever):
        mock_retriever.retrieve.return_value = [
            {
                "content": "Ella is famous for hiking.",
                "metadata": {
                    "source_file": "08_ella.md",
                    "title": "Ella",
                    "province": "Uva"
                },
                "score": 0.95
            }
        ]
        
        results = retrieve_tourism_info("hiking in Ella", interests=["hiking"])
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["metadata"]["source_file"], "08_ella.md")
        self.assertEqual(results[0]["metadata"]["title"], "Ella")
        self.assertEqual(results[0]["score"], 0.95)

    @patch("src.agents.planner.generate_openrouter_completion")
    def test_planner_structured_output_and_duration(self, mock_completion):
        mock_response = """{
            "title": "Adventure in Ella",
            "duration_days": 3,
            "starting_location": "Colombo",
            "summary": "Scenic mountain hiking",
            "days": [
                {
                    "day": 1,
                    "location": "Ella",
                    "activities": ["Climbing Little Adam's Peak"],
                    "travel_notes": "Take scenic train",
                    "evidence_sources": ["08_ella.md"]
                }
            ]
        }"""
        mock_completion.return_value = mock_response
        
        plan = plan_trip(
            user_request="3 days in Ella starting from Colombo",
            duration_days=3,
            starting_location="Colombo",
            interests=["hiking"],
            budget="moderate",
            evidence="Evidence text"
        )
        
        self.assertIsInstance(plan, TripPlan)
        self.assertEqual(plan.duration_days, 3)
        self.assertEqual(plan.starting_location, "Colombo")
        self.assertEqual(len(plan.days), 1)
        self.assertEqual(plan.days[0].location, "Ella")
        self.assertEqual(plan.days[0].evidence_sources, ["08_ella.md"])

    @patch("src.agents.critic.generate_groq_completion")
    def test_critic_structured_parsing(self, mock_completion):
        mock_response = """{
            "score": 8.5,
            "approved": true,
            "issues": [],
            "suggestions": ["Add some restaurant tips"]
        }"""
        mock_completion.return_value = mock_response
        
        # Override config.CRITIC_MODEL to ensure it targets Groq mock path
        with patch("src.models.config.CRITIC_MODEL", "llama-3.3-70b-versatile"):
            result = evaluate_itinerary(
                user_request="Plan a trip",
                duration_days=3,
                starting_location="Colombo",
                interests=["wildlife"],
                draft_itinerary={"days": []},
                evidence="Evidence"
            )
            
            self.assertIsInstance(result, CriticResult)
            self.assertEqual(result.score, 8.5)
            self.assertTrue(result.approved)
            self.assertEqual(result.suggestions, ["Add some restaurant tips"])

    def test_synthesizer_produces_grounded_answer(self):
        draft = {
            "title": "Wildlife & Beach Escape",
            "duration_days": 2,
            "starting_location": "Colombo",
            "summary": "Quick loop around Yala and Mirissa",
            "days": [
                {
                    "day": 1,
                    "location": "Yala",
                    "activities": ["Afternoon Safari"],
                    "travel_notes": "Private transfer from airport",
                    "evidence_sources": ["11_yala.md"]
                }
            ]
        }
        retrieved = [
            {
                "content": "Yala is great for safari",
                "metadata": {"source_file": "11_yala.md", "title": "Yala National Park"}
            }
        ]
        
        result = synthesize_final_response(draft, retrieved)
        self.assertIn("# Wildlife & Beach Escape", result)
        self.assertIn("Yala National Park", result)
        self.assertIn("11_yala.md", result)
        self.assertIn("Disclaimer", result)

if __name__ == "__main__":
    unittest.main()
