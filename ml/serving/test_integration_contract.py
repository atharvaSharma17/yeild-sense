"""Offline integration fixtures; generated untrained weights are not evaluation."""
import math
import unittest
from unittest.mock import patch

import worker
import test_worker


class IntegrationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_worker.WorkerTests.setUpClass()

    def payload(self):
        return test_worker.WorkerTests().payload()

    def test_dual_head_modules_and_forward_are_available(self):
        self.assertIsNotNone(worker.DualHeadEfficientNet)
        model = worker.DualHeadEfficientNet(num_categories=2, pretrained=False)
        model.eval()
        payload = self.payload()
        payload["model"]["architecture"] = "dualhead_efficientnet_b0"
        payload["model"]["conformal_battery_threshold"] = 0.0
        bundle = {"type": "dualhead_efficientnet", "model": model, "labels": payload["model"]["labels"], "calibration": {}}
        with patch("worker.artifact_model", return_value=bundle):
            result = worker.predict(payload)
        self.assertEqual(result["request_id"], payload["request_id"])
        self.assertEqual(result["battery_alert"], "suspected")
        self.assertEqual(set(result["probabilities"]), set(payload["model"]["labels"]))
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0)
        self.assertTrue(all(math.isfinite(value) for value in result["probabilities"].values()))

    def test_missing_track_a_models_never_produce_fake_composition(self):
        self.assertIsNotNone(worker.ParticleMassRegressor)
        self.assertIsNotNone(worker.ParticleMaterialClassifier)
        payload = self.payload()
        payload["model"].update(track="A", task="particle_composition", architecture="gbr_particle_mass_regressor")
        with patch("worker.artifact_model", return_value={"type": "gbr_particle_mass_regressor"}):
            with self.assertRaisesRegex(ValueError, "requires trained classifier"):
                worker.predict(payload)


if __name__ == "__main__":
    unittest.main()
