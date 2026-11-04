import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import run  # noqa: E402
from src.tools import calculator, lookup_record, search_documents  # noqa: E402


class TestTools(unittest.TestCase):
    def test_calculator(self):
        self.assertEqual(calculator("15 * 4 + 20")["result"], 80)

    def test_calculator_rejects_text(self):
        self.assertIn("error", calculator("drop table"))

    def test_lookup(self):
        self.assertEqual(lookup_record("aurora-laptop")["record"]["value"], "899")

    def test_lookup_missing(self):
        self.assertIn("error", lookup_record("zephyr-drone"))

    def test_search(self):
        out = search_documents("refund window unopened")
        self.assertEqual(out["results"][0]["source"], "refund_policy.md")


class TestAgent(unittest.TestCase):
    def test_multi_step(self):
        trace = run("What is the total cost of the aurora laptop and the orbit headphones combined?")
        self.assertIn("1048", trace["final_answer"])

    def test_recovery(self):
        trace = run("What is twelve times five plus 3?")
        self.assertTrue(trace["recovered_from_error"])
        self.assertIn("63", trace["final_answer"])

    def test_no_hallucination_on_missing(self):
        trace = run("What is the price of the zephyr drone?")
        self.assertNotIn("zephyr", trace["final_answer"].lower().replace("no local document matched", ""))


if __name__ == "__main__":
    unittest.main()
