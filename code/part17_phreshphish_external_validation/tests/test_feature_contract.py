from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT.parent
PART2_SCRIPTS = ARCHIVE_ROOT / "part2_url_normalization_leakage_audit" / "scripts"
sys.path.insert(0, str(PART2_SCRIPTS))

from feature_extraction import FEATURE_NAMES, extract_url_features, feature_vector_sha256
from url_normalization import normalize_url


class FeatureContractTests(unittest.TestCase):
    def test_feature_order_matches_frozen_config(self) -> None:
        config = json.loads((ROOT / "config" / "experiment.json").read_text(encoding="utf-8"))
        self.assertEqual(len(FEATURE_NAMES), config["feature_count"])
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))

    def test_normalization_and_features_are_deterministic(self) -> None:
        url = " HTTPS://Sub.Example.CO.UK:443/a/b?x=1&empty=#frag "
        first_normalized = normalize_url(url, keep_fragment=False)
        second_normalized = normalize_url(url, keep_fragment=False)
        self.assertEqual(first_normalized, second_normalized)
        self.assertEqual(first_normalized.host, "sub.example.co.uk")
        self.assertEqual(first_normalized.registrable_domain, "example.co.uk")
        first = extract_url_features(url)
        second = extract_url_features(url)
        self.assertEqual(first, second)
        self.assertEqual(tuple(first), FEATURE_NAMES)
        self.assertEqual(first["has_port"], 0)
        self.assertEqual(first["path_segment_count"], 2)
        self.assertEqual(first["query_parameter_count"], 2)
        self.assertEqual(feature_vector_sha256(first), feature_vector_sha256(second))


if __name__ == "__main__":
    unittest.main()
