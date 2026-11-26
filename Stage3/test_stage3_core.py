"""Focused regression tests for Stage 3 preprocessing and success criteria."""

from __future__ import annotations

import unittest

import numpy as np

from dataset_loaders import clip_negative_reflectance
from classical_unmixing_baselines import run_baseline
from logdet_volume_autoencoder import train_logdet_ae
from synthetic_purity_phase_transition import (
    is_identifiable_within_sad,
    sample_purity_constrained_abundances,
)


class Stage3CoreTests(unittest.TestCase):
    def test_negative_reflectance_is_clipped_and_counted(self) -> None:
        raw = np.array([[0.2, -50.0], [-0.01, 0.8]], dtype=np.float32)
        clipped, count = clip_negative_reflectance(raw)
        np.testing.assert_allclose(clipped, [[0.2, 0.0], [0.0, 0.8]])
        self.assertEqual(count, 2)
        self.assertTrue(np.all(clipped >= 0))

    def test_identifiability_requires_all_matched_endmembers_to_pass(self) -> None:
        metrics = {"mean_sad_deg": 3.75, "max_sad_deg": 15.0}
        self.assertFalse(is_identifiable_within_sad(metrics, 5.0))
        self.assertTrue(is_identifiable_within_sad({"max_sad_deg": 4.99}, 5.0))

    def test_near_pure_cap_is_not_silently_changed_to_pure_pixels(self) -> None:
        abundances = sample_purity_constrained_abundances(
            100, 4, 0.999, np.random.default_rng(3)
        )
        self.assertLessEqual(float(abundances.max()), 0.999)
        self.assertFalse(np.any(abundances == 1.0))

    def test_pure_setting_includes_each_endmember_when_pixel_count_is_small(self) -> None:
        abundances = sample_purity_constrained_abundances(
            4, 4, 1.0, np.random.default_rng(3)
        )
        self.assertTrue(np.all(np.any(abundances == 1.0, axis=0)))
        with self.assertRaises(ValueError):
            sample_purity_constrained_abundances(3, 4, 1.0, np.random.default_rng(3))

    def test_invalid_purity_caps_fail_fast(self) -> None:
        for purity in (1.01, float("nan"), 0.25):
            with self.subTest(purity=purity), self.assertRaises(ValueError):
                sample_purity_constrained_abundances(
                    20, 4, purity, np.random.default_rng(3)
                )
        with self.assertRaises(ValueError):
            sample_purity_constrained_abundances(0, 4, 0.8, np.random.default_rng(3))
        with self.assertRaises(ValueError):
            sample_purity_constrained_abundances(20, 1, 1.0, np.random.default_rng(3))

    def test_extremely_tight_feasible_cap_fails_instead_of_hanging(self) -> None:
        with self.assertRaisesRegex(ValueError, "increase the purity cap"):
            sample_purity_constrained_abundances(
                2000, 4, 0.252, np.random.default_rng(3)
            )

    def test_training_and_baseline_record_clipped_negative_samples(self) -> None:
        rng = np.random.default_rng(9)
        x = rng.uniform(0.1, 1.0, size=(24, 6)).astype(np.float32)
        x[0, 0] = -50.0

        ae_payload, _ = train_logdet_ae(
            x, 2, epochs=1, lambda_vol=0.0, seed=9, device="cpu", verbose=False
        )
        baseline_payload = run_baseline("spa", x, 2, seed=9, device="cpu")
        self.assertEqual(ae_payload["settings"]["negative_samples_clipped"], 1)
        self.assertEqual(baseline_payload["settings"]["negative_samples_clipped"], 1)
        self.assertTrue(np.all(ae_payload["e"].numpy() >= 0))
        self.assertTrue(np.allclose(baseline_payload["a"].numpy().sum(axis=1), 1.0, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
