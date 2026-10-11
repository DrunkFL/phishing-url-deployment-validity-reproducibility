from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_frozen import expected_calibration_error, select_fpr_threshold


class EvaluationContractTests(unittest.TestCase):
    def test_low_fpr_threshold_respects_empirical_budget(self) -> None:
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
        probabilities = np.array([0.01, 0.02, 0.03, 0.8, 0.2, 0.6, 0.7, 0.9])
        threshold = select_fpr_threshold(labels, probabilities, 0.0)
        false_positive_rate = float(((probabilities[labels == 0] >= threshold).sum()) / 4)
        self.assertLessEqual(false_positive_rate, 0.0)

    def test_ece_is_zero_for_exact_binary_probabilities(self) -> None:
        labels = np.array([0, 0, 1, 1], dtype=np.int8)
        probabilities = labels.astype(float)
        self.assertEqual(expected_calibration_error(labels, probabilities, 15, None), 0.0)


if __name__ == "__main__":
    unittest.main()
