"""CPU unit tests for probes/corrupt_solution.py."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from corrupt_solution import corrupt_solution  # noqa: E402


class TestCorruptSolution(unittest.TestCase):
    def test_numeric_boxed_and_body(self):
        sol = ("We compute the total and find 204 minutes. "
               "Since 2048 is unrelated, the answer is $\\boxed{204}$.")
        out, n = corrupt_solution(sol, "204", "217")
        self.assertEqual(n, 2)                      # body '204' + boxed{204}
        self.assertIn("\\boxed{217}", out)
        self.assertIn("find 217 minutes", out)
        self.assertIn("2048 is unrelated", out)     # 2048 NOT touched (boundary)
        self.assertNotIn("204 minutes", out)

    def test_latex_answer(self):
        sol = ("Simplifying gives \\frac{3\\sqrt{3}}{2}. "
               "Thus \\boxed{\\frac{3\\sqrt{3}}{2}}.")
        out, n = corrupt_solution(sol, "\\frac{3\\sqrt{3}}{2}", "\\frac{5\\sqrt{3}}{2}")
        self.assertEqual(n, 2)
        self.assertIn("\\boxed{\\frac{5\\sqrt{3}}{2}}", out)
        self.assertNotIn("\\frac{3\\sqrt{3}}{2}", out)

    def test_numeric_gt_latex_corrupted(self):
        # regression: numeric gt with a backslash-bearing corruption must not be
        # interpreted as a regex replacement template ('\f'/'\s' bad escape).
        sol = "the answer is $\\boxed{343}$ and also 343 elsewhere."
        out, n = corrupt_solution(sol, "343", "\\frac{1}{343}")
        self.assertEqual(n, 2)
        self.assertIn("\\boxed{\\frac{1}{343}}", out)
        self.assertIn("also \\frac{1}{343} elsewhere", out)

    def test_no_match_skip(self):
        sol = "The reasoning concludes with a value expressed only in words."
        out, n = corrupt_solution(sol, "42", "43")
        self.assertEqual(n, 0)                       # skip: answer form absent
        self.assertEqual(out, sol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
