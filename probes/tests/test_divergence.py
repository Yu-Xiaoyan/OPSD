"""CPU unit tests for probes/divergence.py (+ aggregate).

Run either way:
    ~/.conda/envs/opsd/bin/python probes/tests/test_divergence.py
    ~/.conda/envs/opsd/bin/python -m unittest discover -s probes/tests
"""
import math
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from divergence import (  # noqa: E402
    token_jsd, token_kl, clipped_kl_view, token_classifier,
    load_token_categories,
)
import aggregate  # noqa: E402

LOG2 = math.log(2.0)


def _logits(probs):
    return torch.log(torch.tensor(probs, dtype=torch.float32))


class TestJSD(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_jsd_self_is_zero(self):
        p = torch.randn(6, 10)
        jsd = token_jsd(p, p)
        self.assertTrue(torch.allclose(jsd, torch.zeros(6), atol=1e-6))

    def test_jsd_symmetric(self):
        p = torch.randn(5, 8)
        q = torch.randn(5, 8)
        self.assertTrue(torch.allclose(token_jsd(p, q), token_jsd(q, p), atol=1e-6))

    def test_jsd_upper_bound_log2(self):
        # even for near-disjoint distributions, JSD <= log 2 (nats)
        p = torch.randn(20, 32) * 5.0
        q = torch.randn(20, 32) * 5.0
        jsd = token_jsd(p, q)
        self.assertTrue(torch.all(jsd <= LOG2 + 1e-6))
        self.assertTrue(torch.all(jsd >= -1e-6))


class TestClosedForm(unittest.TestCase):
    """V=4 distributions with hand-computed KL/JSD (nats)."""

    def setUp(self):
        self.p = _logits([[0.25, 0.25, 0.25, 0.25]])   # uniform
        self.q = _logits([[0.40, 0.30, 0.20, 0.10]])

    def test_forward_kl(self):
        val = token_kl(self.p, self.q, "forward").item()
        self.assertAlmostEqual(val, 0.12177727, places=5)

    def test_reverse_kl(self):
        val = token_kl(self.p, self.q, "reverse").item()
        self.assertAlmostEqual(val, 0.10644014, places=5)

    def test_jsd_value(self):
        val = token_jsd(self.p, self.q).item()
        self.assertAlmostEqual(val, 0.02786561, places=5)

    def test_reverse_equals_forward_swapped(self):
        fwd = token_kl(self.p, self.q, "forward")
        rev = token_kl(self.q, self.p, "reverse")
        self.assertTrue(torch.allclose(fwd, rev, atol=1e-6))

    def test_kl_matches_numpy_oracle(self):
        torch.manual_seed(3)
        pl = torch.randn(4, 4)
        ql = torch.randn(4, 4)
        p = torch.softmax(pl, -1).numpy()
        q = torch.softmax(ql, -1).numpy()
        oracle_fwd = (p * (np.log(p) - np.log(q))).sum(-1)
        got = token_kl(pl, ql, "forward").numpy()
        self.assertTrue(np.allclose(got, oracle_fwd, atol=1e-6))


class TestClippedView(unittest.TestCase):
    def test_clip_inf_reduces_to_forward_kl(self):
        torch.manual_seed(1)
        p = torch.randn(7, 5)
        q = torch.randn(7, 5)
        clipped = clipped_kl_view(p, q, clip=float("inf"))
        forward = token_kl(p, q, "forward")
        self.assertTrue(torch.allclose(clipped, forward, atol=1e-6))

    def test_clip_actually_caps(self):
        torch.manual_seed(2)
        p = torch.randn(4, 6)
        q = torch.randn(4, 6)
        capped = clipped_kl_view(p, q, clip=0.01)
        forward = token_kl(p, q, "forward")
        # clamping element contributions from above can only lower the sum
        self.assertTrue(torch.all(capped <= forward + 1e-6))


class TestTokenClassifier(unittest.TestCase):
    def test_labels(self):
        cats = load_token_categories()
        toks = ["123", " +", "\\boxed", "\\frac", "boxed", "wait", " So",
                "wait,", "the", "cat", "=", "▁let", "x^2"]
        expected = ["math", "math", "math", "math", "math", "style", "style",
                    "style", "other", "other", "math", "style", "math"]
        got = token_classifier(toks, cats)
        self.assertEqual(got, expected)


class TestAggregate(unittest.TestCase):
    def test_position_curve_shapes_and_values(self):
        seqs = [np.array([0.0, 1.0]), np.array([0.0, 0.5, 1.0, 2.0])]
        curve = aggregate.relative_position_curve(seqs, n_bins=2)
        self.assertEqual(curve["mean"].shape, (2,))
        self.assertEqual(int(curve["count"].sum()), 6)
        # first-half tokens: 0.0,0.0,0.5 -> mean 1/6 ; second-half: 1.0,1.0,2.0
        self.assertAlmostEqual(curve["mean"][0], (0.0 + 0.0 + 0.5) / 3, places=6)
        self.assertAlmostEqual(curve["mean"][1], (1.0 + 1.0 + 2.0) / 3, places=6)

    def test_group_stats(self):
        vals = [1.0, 3.0, 10.0]
        labs = ["math", "math", "style"]
        st = aggregate.group_stats(vals, labs)
        self.assertEqual(st["math"]["count"], 2)
        self.assertAlmostEqual(st["math"]["mean"], 2.0, places=6)
        self.assertAlmostEqual(st["style"]["mean"], 10.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
