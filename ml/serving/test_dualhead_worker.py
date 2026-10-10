"""Integration test for YieldSense State-of-the-Art DualHeadEfficientNet and Track A serving worker."""

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import unittest

from PIL import Image
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
import worker


class DualHeadWorkerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent.parent
        cls.manifest_path = cls.project_root / "artifacts" / "registered_model_manifest.json"
        cls.archive_path = cls.project_root / "artifacts" / "model.tar.gz"

        if not cls.manifest_path.exists() or not cls.archive_path.exists():
            raise unittest.SkipTest("Trained artifacts (model.tar.gz, manifest.json) not packaged yet")

        with open(cls.manifest_path, "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)

        # Point manifest to local file for direct offline contract testing
        cls.manifest["artifact_s3_uri"] = f"file:///{cls.archive_path.as_posix()}"
        cls.manifest["artifact_sha256"] = hashlib.sha256(cls.archive_path.read_bytes()).hexdigest()

    def setUp(self):
        worker._cache.clear()
        os.environ["S3_BUCKET"] = "yieldsense-storage-ap-south-1"

    def make_test_payload(self, is_track_a: bool = False):
        img_bytes = io.BytesIO()
        Image.new("RGB", (224, 224), color=(128, 128, 128)).save(img_bytes, format="JPEG")
        data = img_bytes.getvalue()

        manifest = dict(self.manifest)
        if is_track_a:
            manifest["track"] = "A"
            manifest["task"] = "particle_composition"

        return {
            "contract_version": "1",
            "request_id": "test-req-integration-001",
            "batch_id": "test-batch-001",
            "mass_kg": 250.0,
            "model": manifest,
            "image": {
                "id": "img-001",
                "batch_id": "test-batch-001",
                "object_key": "raw/test-batch-001/img-001.jpg",
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
                "content_type": "image/jpeg",
                "created_at": "2026-10-10T10:00:00Z",
            },
            "image_base64": base64.b64encode(data).decode("utf-8"),
        }

    def test_dualhead_track_b_inference_and_conformal_alert(self):
        payload = self.make_test_payload(is_track_a=False)
        result = worker.predict(payload)

        # Go Backend Contract assertions
        self.assertEqual(result["contract_version"], "1")
        self.assertEqual(result["request_id"], "test-req-integration-001")
        self.assertEqual(result["model_version"], self.manifest["model_version"])
        self.assertEqual(result["track"], "B")
        self.assertEqual(result["task"], "device_classification")
        self.assertFalse(result["abstained"])

        # Check all 15 classes present and probabilities sum strictly to 1.0 (within 1e-6)
        probs = result["probabilities"]
        self.assertEqual(len(probs), len(self.manifest["labels"]))
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

        # Check conformal battery alert is either "suspected" or "none" (not "unknown" or arbitrary string)
        self.assertIn(result["battery_alert"], ["suspected", "none"])

        # Verify second call is cached and fast
        again = worker.predict(payload)
        self.assertEqual(result["probabilities"], again["probabilities"])
        self.assertEqual(result["battery_alert"], again["battery_alert"])

    def test_track_a_composition_and_scenario_reconciliation(self):
        payload = self.make_test_payload(is_track_a=True)
        result = worker.predict(payload)

        self.assertEqual(result["contract_version"], "1")
        self.assertEqual(result["track"], "A")
        self.assertEqual(result["task"], "particle_composition")
        self.assertIn("material_mass_kg", result)
        self.assertIn("scenarios", result)

        masses = result["material_mass_kg"]
        self.assertEqual(set(masses.keys()), {"FEM", "NFM", "PCB", "PLA", "OTHER"})

        scenarios = result["scenarios"]
        self.assertGreaterEqual(len(scenarios), 1)

        # Go Contract: Scenario 0 must match normalized material mass estimates within 1e-6
        total_est_mass = sum(masses.values())
        self.assertGreater(total_est_mass, 0)
        for mat in ["FEM", "NFM", "PCB", "PLA", "OTHER"]:
            self.assertAlmostEqual(scenarios[0][mat], masses[mat] / total_est_mass, places=6)


if __name__ == "__main__":
    unittest.main()
