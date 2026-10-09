import base64
from collections import OrderedDict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
import math
import os
import secrets
import tarfile
import threading
from urllib.parse import urlparse

import boto3
from botocore.config import Config
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18

torch.set_num_threads(1)
Image.MAX_IMAGE_PIXELS = 16_000_000
LOG = logging.getLogger("inference")
MAX_BODY = 15 * 1024 * 1024
MAX_ARTIFACT = 128 * 1024 * 1024
_cache = OrderedDict()
_lock = threading.Lock()
_slots = threading.BoundedSemaphore(2)


def artifact_model(manifest):
    if manifest["architecture"] != "resnet18" or manifest["task"] != "device_classification" or manifest["track"] != "B":
        raise ValueError("This worker supports the Track B ResNet-18 classifier only")
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    with _lock:
        if identity in _cache:
            _cache.move_to_end(identity)
            return _cache[identity]
        uri = urlparse(manifest["artifact_s3_uri"])
        key = uri.path.lstrip("/")
        if uri.scheme != "s3" or uri.netloc != os.environ["S3_BUCKET"] or not key.startswith(("models/", "model-artifacts/")):
            raise ValueError("Artifact must be in the configured bucket's model prefix")
        client = boto3.client("s3", config=Config(connect_timeout=3, read_timeout=5, retries={"max_attempts": 1}))
        obj = client.get_object(Bucket=uri.netloc, Key=key)
        try:
            if obj["ContentLength"] > MAX_ARTIFACT:
                raise ValueError("Model archive exceeds 128 MiB")
            data = obj["Body"].read(MAX_ARTIFACT + 1)
        finally:
            obj["Body"].close()
        if len(data) > MAX_ARTIFACT or hashlib.sha256(data).hexdigest() != manifest["artifact_sha256"]:
            raise ValueError("Model artifact hash mismatch or archive too large")
        # Read two exact regular-file members; never extract archive paths.
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            found = {}
            expanded = 0
            for index, member in enumerate(archive):
                expanded += member.size
                if expanded > MAX_ARTIFACT:
                    raise ValueError("Expanded model archive exceeds 128 MiB")
                if index > 1000:
                    raise ValueError("Too many archive entries")
                name = member.name.removeprefix("./")
                if name not in ("model.pth", "labels.json"):
                    continue
                if name in found or not member.isfile() or member.size > (100 * 1024 * 1024 if name == "model.pth" else 65536):
                    raise ValueError("Invalid model archive member")
                found[name] = archive.extractfile(member).read()
        if set(found) != {"model.pth", "labels.json"}:
            raise ValueError("Archive must contain root model.pth and labels.json")
        labels = json.loads(found["labels.json"])
        if labels != manifest["labels"] or not 1 <= len(labels) <= 128:
            raise ValueError("Artifact class order differs from registered labels")
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
        model.load_state_dict(torch.load(io.BytesIO(found["model.pth"]), map_location="cpu", weights_only=True), strict=True)
        model.eval()
        _cache[identity] = model
        while len(_cache) > 2:
            _cache.popitem(last=False)
        return model


def predict(payload):
    if payload["contract_version"] != "1":
        raise ValueError("Unsupported contract version")
    manifest = payload["model"]
    model = artifact_model(manifest)
    data = base64.b64decode(payload["image_base64"], validate=True)
    image_meta = payload["image"]
    if len(data) > 10 * 1024 * 1024 or len(data) != image_meta["size_bytes"] or hashlib.sha256(data).hexdigest() != image_meta["sha256"]:
        raise ValueError("Image integrity mismatch")
    p = manifest["preprocessing"]
    if p["color_mode"] != "RGB" or not 1 <= p["width"] <= 4096 or not 1 <= p["height"] <= 4096:
        raise ValueError("Invalid preprocessing")
    if len(p["mean"]) != 3 or len(p["std"]) != 3 or not math.isfinite(p["scale"]) or p["scale"] <= 0 or any(not math.isfinite(v) or v <= 0 for v in p["std"]):
        raise ValueError("Invalid normalization")
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in ("PNG", "JPEG") or image.width * image.height > 16_000_000:
            raise ValueError("Unsupported image")
        rgb = image.convert("RGB").resize((p["width"], p["height"]), Image.Resampling.BILINEAR)
        array = np.array(rgb, dtype=np.float32, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1) * p["scale"]
    tensor = (tensor - torch.tensor(p["mean"]).view(3, 1, 1)) / torch.tensor(p["std"]).view(3, 1, 1)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0].tolist()
    labels = manifest["labels"]
    top = labels[max(range(len(labels)), key=lambda i: probabilities[i])]
    return {
        "contract_version": "1", "request_id": payload["request_id"],
        "model_version": manifest["model_version"], "track": "B", "task": "device_classification",
        "probabilities": dict(zip(labels, probabilities)), "abstained": False,
        "battery_alert": "suspected" if top in manifest.get("battery_review_labels", []) else "unknown",
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, code, value):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/health":
            return self.send_json(404, {"error": "not found"})
        self.send_json(200, {"ok": True, "cached_models": len(_cache)})

    def do_POST(self):
        if self.path != "/predict":
            return self.send_json(404, {"error": "not found"})
        token = os.environ.get("ML_INFERENCE_TOKEN", "")
        if not token or not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
            return self.send_json(401, {"error": "unauthorized"})
        if not _slots.acquire(blocking=False):
            return self.send_json(503, {"error": "busy"})
        try:
            self.connection.settimeout(15)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                return self.send_json(413, {"error": "invalid body length"})
            if self.headers.get("Content-Type") != "application/json":
                return self.send_json(415, {"error": "JSON required"})
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Incomplete request")
            result = predict(json.loads(body))
            self.send_json(200, result)
        except (ValueError, KeyError, TypeError, tarfile.TarError):
            LOG.exception("Invalid model or inference input")
            self.send_json(422, {"error": "invalid model or inference input"})
        except Exception:
            LOG.exception("Inference failed")
            self.send_json(503, {"error": "inference unavailable"})
        finally:
            _slots.release()

    def log_message(self, format, *args):
        LOG.info(format, *args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not os.environ.get("ML_INFERENCE_TOKEN") or not os.environ.get("S3_BUCKET"):
        raise SystemExit("S3_BUCKET and ML_INFERENCE_TOKEN are required")
    host, port = os.environ.get("ML_LISTEN_ADDR", "127.0.0.1:9000").rsplit(":", 1)
    ThreadingHTTPServer((host, int(port)), Handler).serve_forever()
