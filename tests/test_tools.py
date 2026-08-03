import unittest
from src.tools.travel_tools import (
    trip_day_allocator,
    estimate_budget,
    check_interest_coverage,
    validate_itinerary
)

class TestTripDayAllocator(unittest.TestCase):

    def test_zero_duration_returns_error(self):
        result = trip_day_allocator(0, 3)
        self.assertIn("error", result)

    def test_zero_destinations_returns_empty_allocation(self):
        result = trip_day_allocator(5, 0)
        self.assertEqual(result["allocation"], [])

    def test_even_split(self):
        result = trip_day_allocator(6, 3)
        self.assertEqual(result["allocation_suggestion"], [2, 2, 2])

    def test_uneven_split_distributes_remainder(self):
        result = trip_day_allocator(7, 3)
        self.assertEqual(result["allocation_suggestion"], [3, 2, 2])
        self.assertEqual(sum(result["allocation_suggestion"]), 7)


class TestEstimateBudget(unittest.TestCase):

    def test_unknown_budget_level_defaults_to_moderate(self):
        result = estimate_budget(3, "luxury-ish")
        self.assertEqual(result["daily_rate_usd"], 75)

    def test_budget_level_is_case_and_whitespace_insensitive(self):
        result = estimate_budget(2, "  PREMIUM ")
        self.assertEqual(result["daily_rate_usd"], 200)
        self.assertEqual(result["total_estimated_usd"], 400)

    def test_zero_duration_yields_zero_total(self):
        result = estimate_budget(0, "budget")
        self.assertEqual(result["total_estimated_usd"], 0)


class TestCheckInterestCoverage(unittest.TestCase):

    def test_empty_interests_reports_full_coverage(self):
        result = check_interest_coverage([], ["beach", "wildlife"])
        self.assertEqual(result["coverage_percentage"], 100.0)
        self.assertEqual(result["covered_interests"], [])

    def test_partial_coverage(self):
        result = check_interest_coverage(["beach", "history"], ["Beach", "wildlife"])
        self.assertEqual(set(result["covered_interests"]), {"beach"})
        self.assertEqual(set(result["uncovered_interests"]), {"history"})
        self.assertAlmostEqual(result["coverage_percentage"], 50.0)


class TestValidateItinerary(unittest.TestCase):

    def test_valid_itinerary_has_no_errors(self):
        days = [
            {"day": 1, "location": "Kandy", "activities": ["Temple visit"]},
            {"day": 2, "location": "Ella", "activities": ["Hiking"]}
        ]
        result = validate_itinerary(days, 2)
        self.assertTrue(result["valid"])
        self.assertEqual(result["errors"], [])

    def test_day_count_mismatch_is_reported(self):
        days = [{"day": 1, "location": "Kandy", "activities": ["Temple visit"]}]
        result = validate_itinerary(days, 3)
        self.assertFalse(result["valid"])
        self.assertTrue(any("day count mismatch" in e.lower() for e in result["errors"]))

    def test_duplicate_day_is_reported(self):
        days = [
            {"day": 1, "location": "Kandy", "activities": ["Temple visit"]},
            {"day": 1, "location": "Ella", "activities": ["Hiking"]}
        ]
        result = validate_itinerary(days, 1)
        self.assertFalse(result["valid"])
        self.assertTrue(any("duplicate" in e.lower() for e in result["errors"]))

    def test_missing_intermediate_day_is_reported(self):
        days = [
            {"day": 1, "location": "Kandy", "activities": ["Temple visit"]},
            {"day": 3, "location": "Ella", "activities": ["Hiking"]}
        ]
        result = validate_itinerary(days, 3)
        self.assertFalse(result["valid"])
        self.assertTrue(any("Day 2 is missing" in e for e in result["errors"]))

    def test_missing_location_or_activities_yields_warnings(self):
        days = [{"day": 1, "location": "", "activities": []}]
        result = validate_itinerary(days, 1)
        self.assertEqual(len(result["warnings"]), 2)


if __name__ == "__main__":
    unittest.main()
