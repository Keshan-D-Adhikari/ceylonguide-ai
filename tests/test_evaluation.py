import unittest
from evaluation.rag_eval import calculate_metrics

class TestEvaluationMetrics(unittest.TestCase):

    def test_calculate_metrics_perfect_match(self):
        retrieved = ["01_colombo.md", "02_negombo.md"]
        expected = ["01_colombo.md", "02_negombo.md"]
        
        hit, recall, rr = calculate_metrics(retrieved, expected, k=2)
        
        self.assertEqual(hit, 1.0)
        self.assertEqual(recall, 1.0)
        self.assertEqual(rr, 1.0) # Rank 1 match

    def test_calculate_metrics_partial_match(self):
        retrieved = ["02_negombo.md", "08_ella.md"]
        expected = ["08_ella.md", "11_yala.md"]
        
        hit, recall, rr = calculate_metrics(retrieved, expected, k=2)
        
        self.assertEqual(hit, 1.0)
        # 1 out of 2 expected retrieved
        self.assertEqual(recall, 0.5)
        # First match is at rank 2 (index 1) -> RR = 1/2 = 0.5
        self.assertEqual(rr, 0.5)

    def test_calculate_metrics_no_match(self):
        retrieved = ["01_colombo.md", "02_negombo.md"]
        expected = ["08_ella.md"]
        
        hit, recall, rr = calculate_metrics(retrieved, expected, k=5)
        
        self.assertEqual(hit, 0.0)
        self.assertEqual(recall, 0.0)
        self.assertEqual(rr, 0.0)

    def test_calculate_metrics_multiple_chunks_same_file(self):
        # Retrieved list contains duplicate files (representing different chunks of the same file)
        retrieved = ["11_yala.md", "11_yala.md", "12_wilpattu.md"]
        expected = ["11_yala.md", "12_wilpattu.md"]
        
        hit, recall, rr = calculate_metrics(retrieved, expected, k=3)
        
        self.assertEqual(hit, 1.0)
        # Recall must be 1.0 (both unique expected files retrieved), not inflated
        self.assertEqual(recall, 1.0)
        self.assertEqual(rr, 1.0)

if __name__ == "__main__":
    unittest.main()
