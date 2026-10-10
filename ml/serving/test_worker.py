"""Contract tests use untrained weights only; they are not ML evaluation."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import unittest
from unittest.mock import patch

_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent
for p in (str(_CURRENT_DIR), str(_PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from PIL import Image
import torch
from torchvision.models import resnet18
import worker


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        state = io.BytesIO()
        torch.save(model.state_dict(), state)
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            for name, data in (("model.pth", state.getvalue()), ("labels.json", b'["device-a","device-b"]')):
                member = tarfile.TarInfo(name)
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
        cls.artifact = archive.getvalue()

    def setUp(self):
        worker._cache.clear()
        os.environ["S3_BUCKET"] = "test-bucket"

    def payload(self):
        image = io.BytesIO()
        Image.new("RGB", (10, 10)).save(image, format="PNG")
        data = image.getvalue()
        return {
            "contract_version": "1", "request_id": "test-request",
            "model": {"architecture": "resnet18", "model_version": "test-untrained",
                "task": "device_classification", "track": "B",
                "artifact_s3_uri": "s3://test-bucket/models/model.tar.gz",
                "artifact_sha256": hashlib.sha256(self.artifact).hexdigest(),
                "labels": ["device-a", "device-b"],
                "preprocessing": {"width": 224, "height": 224, "color_mode": "RGB",
                    "scale": 1 / 255, "mean": [.485, .456, .406], "std": [.229, .224, .225]}},
            "image": {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()},
            "image_base64": base64.b64encode(data).decode(),
        }

    def s3(self):
        class Client:
            def get_object(inner, **kwargs):
                return {"ContentLength": len(self.artifact), "Body": io.BytesIO(self.artifact)}
        return Client()

    def test_real_cpu_forward_and_cache(self):
        with patch("worker.boto3.client", return_value=self.s3()) as client:
            result = worker.predict(self.payload())
            again = worker.predict(self.payload())
        self.assertEqual(result["request_id"], "test-request")
        self.assertEqual(result["model_version"], "test-untrained")
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1, places=6)
        self.assertEqual(result["battery_alert"], "unknown")
        self.assertEqual(result, again)
        self.assertEqual(client.call_count, 1)

    def test_artifact_hash_and_label_mismatch(self):
        payload = self.payload()
        payload["model"]["artifact_sha256"] = "0" * 64
        with patch("worker.boto3.client", return_value=self.s3()), self.assertRaises(ValueError):
            worker.predict(payload)
        payload = self.payload()
        payload["model"]["labels"].reverse()
        with patch("worker.boto3.client", return_value=self.s3()), self.assertRaises(ValueError):
            worker.predict(payload)

    def test_device_model_cannot_be_used_for_track_a(self):
        payload = self.payload()
        payload["model"]["track"] = "A"
        with self.assertRaises(ValueError):
            worker.predict(payload)


if __name__ == "__main__":
    unittest.main()
