import unittest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

from src.models import config
from src.models.factory import get_groq_client, generate_groq_completion, APIKeyError, LLMProviderError
from src.agents.router import route_query, clean_and_parse_json, normalize_router_output, RouterDecision

class TestModelConfigAndFactory(unittest.TestCase):
    
    @patch("src.models.config.GROQ_API_KEY", "")
    def test_missing_api_key_raises_error(self):
        with self.assertRaises(APIKeyError):
            get_groq_client()

    @patch("src.models.config.GROQ_API_KEY", "mock_key")
    @patch("groq.Groq")
    def test_get_groq_client_with_key(self, mock_groq):
        # Should initialize without raising an exception
        client = get_groq_client()
        self.assertIsNotNone(client)

class TestRouterAgent(unittest.TestCase):
    
    def test_clean_and_parse_json_valid(self):
        text = '{"intent": "itinerary_planning", "requires_rag": true, "requires_planning": true}'
        res = clean_and_parse_json(text)
        self.assertEqual(res["intent"], "itinerary_planning")

    def test_clean_and_parse_json_fenced(self):
        text = """Here is your output:
```json
{
    "intent": "general_query",
    "requires_rag": true,
    "requires_planning": false
}
```
Explanations are here."""
        res = clean_and_parse_json(text)
        self.assertEqual(res["intent"], "general_query")

    def test_clean_and_parse_json_invalid(self):
        text = "This is not JSON at all."
        with self.assertRaises(ValueError):
            clean_and_parse_json(text)

    def test_normalize_router_output(self):
        # Test lowercasing and validation of interests
        raw = {
            "intent": "itinerary_planning",
            "duration_days": 25, # Out of bounds
            "starting_location": "Colombo",
            "interests": ["Wildlife", "Swimming", "beaches"], # Swimming is unsupported
            "budget": "ultra-luxury", # Unsupported
            "requires_rag": True,
            "requires_planning": True,
            "complexity": "super-hard" # Unsupported
        }
        normalized = normalize_router_output(raw)
        
        self.assertEqual(normalized["duration_days"], 14) # Capped at 14
        self.assertIn("wildlife", normalized["interests"])
        self.assertIn("beaches", normalized["interests"])
        self.assertNotIn("swimming", normalized["interests"])
        self.assertEqual(normalized["budget"], "moderate") # Defaulted
        self.assertEqual(normalized["complexity"], "medium") # Defaulted

    def test_empty_query_handling(self):
        # Empty string should yield a default RouterDecision without hitting the LLM
        decision = route_query("")
        self.assertEqual(decision.intent, "unknown")
        self.assertEqual(decision.interests, [])
        self.assertFalse(decision.requires_rag)
        
        decision_spaces = route_query("   \n  ")
        self.assertEqual(decision_spaces.intent, "unknown")

    @patch("src.agents.router.generate_groq_completion")
    def test_route_query_success_mock(self, mock_completion):
        mock_response_json = """{
            "intent": "itinerary_planning",
            "duration_days": 5,
            "starting_location": "Colombo",
            "interests": ["wildlife", "beaches"],
            "budget": "budget",
            "requires_rag": true,
            "requires_planning": true,
            "complexity": "medium"
        }"""
        mock_completion.return_value = mock_response_json
        
        decision = route_query("I want to visit Yala and Mirissa in 5 days starting from Colombo")
        
        self.assertEqual(decision.intent, "itinerary_planning")
        self.assertEqual(decision.duration_days, 5)
        self.assertEqual(decision.starting_location, "Colombo")
        self.assertIn("wildlife", decision.interests)
        self.assertTrue(decision.requires_rag)
        self.assertTrue(decision.requires_planning)

if __name__ == "__main__":
    unittest.main()
