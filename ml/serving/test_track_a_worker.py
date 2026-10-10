"""Unit tests for Track A Particle Composition Serving Worker (`ml/serving/test_track_a_worker.py`).

Verifies:
1. Strict adherence to Go inference contract (backend/internal/inference/contract.go)
2. S3 artifact loading and bundle caching
3. Five-material composition output (FEM, NFM, PCB, PLA, OTHER)
4. Strict scenario[0] alignment with normalized material_mass_kg (within 1e-6)
5. Model validation boundaries and error rejection
"""

from __future__ import annotations
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from PIL import Image

_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent
for p in (str(_CURRENT_DIR), str(_PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import worker


class TrackAWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        artifacts_dir = _PROJECT_ROOT / "artifacts" / "track_a"
        tar_path = artifacts_dir / "model.tar.gz"
        manifest_path = artifacts_dir / "registered_model_manifest.json"

        if not tar_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                f"Track A artifacts not found at {artifacts_dir}. Run scripts/package_track_a.py first."
            )

        cls.artifact_bytes = tar_path.read_bytes()
        cls.artifact_sha256 = hashlib.sha256(cls.artifact_bytes).hexdigest()
        with open(manifest_path, "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)

    def setUp(self):
        worker._cache.clear()
        os.environ["S3_BUCKET"] = "yieldsense-storage-ap-south-1"

    def s3_client_mock(self):
        class MockClient:
            def get_object(inner, **kwargs):
                return {
                    "ContentLength": len(self.artifact_bytes),
                    "Body": io.BytesIO(self.artifact_bytes),
                }
        return MockClient()

    def build_payload(self, mass_kg: float = 100.0, image_color: tuple = (40, 150, 40)):
        # Greenish synthetic particle image (suggestive of PCB)
        img = Image.new("RGB", (64, 64), color=image_color)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()

        return {
            "contract_version": "1",
            "request_id": "track-a-test-req-001",
            "mass_kg": mass_kg,
            "model": dict(self.manifest),
            "image": {
                "id": "img-001",
                "batch_id": "batch-001",
                "object_key": "raw/img-001.png",
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
            "image_base64": base64.b64encode(data).decode("ascii"),
        }

    def test_track_a_forward_and_go_contract_compliance(self):
        payload = self.build_payload(mass_kg=250.0)

        with patch("worker.boto3.client", return_value=self.s3_client_mock()) as client:
            resp = worker.predict(payload)
            # Second call should hit the cache
            cached_resp = worker.predict(payload)

        self.assertEqual(client.call_count, 1, "Expected second call to use LRU cache")
        self.assertEqual(resp, cached_resp)

        # 1. Top-level contract validation
        self.assertEqual(resp["contract_version"], "1")
        self.assertEqual(resp["request_id"], "track-a-test-req-001")
        self.assertEqual(resp["model_version"], self.manifest["model_version"])
        self.assertEqual(resp["track"], "A")
        self.assertEqual(resp["task"], "particle_composition")
        self.assertFalse(resp["abstained"])
        self.assertEqual(resp["battery_alert"], "none")
        self.assertNotIn("probabilities", resp, "Track A must not return device probabilities")

        # 2. Material masses validation
        expected_materials = ["FEM", "NFM", "PCB", "PLA", "OTHER"]
        self.assertEqual(set(resp["material_mass_kg"].keys()), set(expected_materials))
        for m in expected_materials:
            val = resp["material_mass_kg"][m]
            self.assertTrue(math.isfinite(val))
            self.assertGreater(val, 0.0, f"Material mass for {m} must be positive")

        # Total estimated mass
        tot_mass = sum(resp["material_mass_kg"].values())
        self.assertGreater(tot_mass, 0.0)
        self.assertLess(tot_mass, 1e9)

        # 3. Joint Scenarios validation
        scenarios = resp["scenarios"]
        self.assertGreaterEqual(len(scenarios), 1)
        self.assertLessEqual(len(scenarios), 20)

        for s_idx, scen in enumerate(scenarios):
            self.assertEqual(set(scen.keys()), set(expected_materials))
            s_sum = sum(scen.values())
            self.assertAlmostEqual(
                s_sum, 1.0, places=5,
                msg=f"Scenario {s_idx} fractions must sum to 1.0 within 1e-5 (sum={s_sum})"
            )
            for m in expected_materials:
                self.assertGreaterEqual(scen[m], 0.0)
                self.assertLessEqual(scen[m], 1.0)

        # 4. Strict Scenario 0 alignment with normalized material_mass_kg
        scen_0 = scenarios[0]
        for m in expected_materials:
            expected_frac = resp["material_mass_kg"][m] / tot_mass
            diff = abs(scen_0[m] - expected_frac)
            self.assertLess(
                diff, 1e-6,
                msg=f"Material {m} scenario 0 fraction ({scen_0[m]}) deviates from normalized mass ({expected_frac}) by {diff}"
            )

    def test_track_a_rejection_of_incompatible_task_or_track(self):
        # 1. Track B with Track A architecture
        payload = self.build_payload()
        payload["model"]["track"] = "B"
        payload["model"]["task"] = "device_classification"
        with patch("worker.boto3.client", return_value=self.s3_client_mock()):
            with self.assertRaises(ValueError):
                worker.predict(payload)

        # 2. Track A architecture with unsupported task
        payload2 = self.build_payload()
        payload2["model"]["task"] = "invalid_task"
        with patch("worker.boto3.client", return_value=self.s3_client_mock()):
            with self.assertRaises(ValueError):
                worker.predict(payload2)

    def test_manifest_schema_compliance_with_go_domain(self):
        m = self.manifest
        self.assertEqual(m["contract_version"], "1")
        self.assertEqual(m["track"], "A")
        self.assertEqual(m["task"], "particle_composition")
        self.assertEqual(m["architecture"], "gbr_particle_mass_regressor")
        self.assertTrue(m["artifact_s3_uri"].startswith("s3://yieldsense-storage-ap-south-1/"))
        self.assertTrue(m["artifact_s3_uri"].endswith(".tar.gz"))
        self.assertEqual(len(m["artifact_sha256"]), 64)
        self.assertEqual(len(m["dataset_manifest_sha256"]), 64)
        self.assertEqual(m["labels"], ["FEM", "NFM", "PCB", "PLA", "OTHER"])
        self.assertTrue(len(m["paired_mass_evidence"]) > 0)
        self.assertIn("particle_mass_mae_g", m["evaluation"])
        self.assertIn("mixed_batch_composition_mae", m["evaluation"])


if __name__ == "__main__":
    unittest.main()
