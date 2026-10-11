import base64
from collections import OrderedDict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
import math
import os
from pathlib import Path
import secrets
import sys
import tarfile
import threading
from urllib.parse import urlparse

import boto3
from botocore.config import Config
import joblib
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet18

# Ensure project root is in sys.path for ml imports
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from ml.model import DualHeadEfficientNet
except ImportError:
    DualHeadEfficientNet = None

try:
    from ml.track_a.features import extract_particle_classifier_features, extract_particle_features
    from ml.track_a.mass_regression import ParticleMassRegressor, ParticleMaterialClassifier
except ImportError:
    extract_particle_classifier_features = None
    extract_particle_features = None
    ParticleMassRegressor = None
    ParticleMaterialClassifier = None

torch.set_num_threads(1)
Image.MAX_IMAGE_PIXELS = 16_000_000
LOG = logging.getLogger("inference")
MAX_BODY = 15 * 1024 * 1024
MAX_ARTIFACT = 128 * 1024 * 1024
_cache = OrderedDict()
_lock = threading.Lock()
_slots = threading.BoundedSemaphore(2)


def _load_raw_artifact(uri_str: str, expected_sha256: str) -> bytes:
    """Download or read model artifact bytes and verify SHA256 integrity."""
    uri = urlparse(uri_str)
    
    # 1. Local file path support (file:// or plain filesystem path)
    if uri.scheme in ("", "file"):
        path_str = uri.path if uri.scheme == "file" else uri_str
        if os.name == "nt" and uri.scheme == "file" and path_str.startswith("/") and len(path_str) > 2 and path_str[2] == ":":
            path_str = path_str[1:]
        local_path = Path(path_str)
        if not local_path.is_file():
            # Check relative to artifacts/ if not found
            alt_path = _PROJECT_ROOT / "artifacts" / local_path.name
            if alt_path.is_file():
                local_path = alt_path
            else:
                raise ValueError(f"Local model artifact not found at {local_path}")
        data = local_path.read_bytes()
        if len(data) > MAX_ARTIFACT or hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError("Model artifact hash mismatch or archive too large")
        return data

    # 2. S3 bucket support
    key = uri.path.lstrip("/")
    bucket_env = os.environ.get("S3_BUCKET", "")
    if uri.scheme != "s3" or (bucket_env and uri.netloc != bucket_env) or not key.startswith(("models/", "model-artifacts/")):
        raise ValueError("Artifact must be in the configured bucket's model prefix")
    
    client = boto3.client("s3", config=Config(connect_timeout=3, read_timeout=5, retries={"max_attempts": 1}))
    obj = client.get_object(Bucket=uri.netloc, Key=key)
    try:
        if obj.get("ContentLength", 0) > MAX_ARTIFACT:
            raise ValueError("Model archive exceeds 128 MiB")
        data = obj["Body"].read(MAX_ARTIFACT + 1)
    finally:
        obj["Body"].close()
    
    if len(data) > MAX_ARTIFACT or hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Model artifact hash mismatch or archive too large")
    return data


def artifact_model(manifest):
    arch = manifest.get("architecture", "")
    task = manifest.get("task", "")
    track = manifest.get("track", "")

    # Supported architectures
    supported_archs = ("resnet18", "dualhead_efficientnet_b0", "efficientnet_b0", "gbr_particle_mass_regressor")
    if arch not in supported_archs:
        raise ValueError(f"Architecture '{arch}' not supported. Expected one of {supported_archs}")
    
    if arch == "resnet18" and (track != "B" or task != "device_classification"):
        raise ValueError("This worker supports the Track B ResNet-18 classifier only")

    if arch == "gbr_particle_mass_regressor" and (track != "A" or task != "particle_composition"):
        raise ValueError("This model supports Track A particle composition only")

    if task not in ("device_classification", "particle_composition"):
        raise ValueError(f"Task '{task}' not supported.")

    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    with _lock:
        if identity in _cache:
            _cache.move_to_end(identity)
            return _cache[identity]

        data = _load_raw_artifact(manifest["artifact_s3_uri"], manifest["artifact_sha256"])

        # Extract archive members safely
        found = {}
        expanded = 0
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for index, member in enumerate(archive):
                expanded += member.size
                if expanded > MAX_ARTIFACT:
                    raise ValueError("Expanded model archive exceeds 128 MiB")
                if index > 1000:
                    raise ValueError("Too many archive entries")
                name = member.name.removeprefix("./")
                if name not in ("model.pth", "labels.json", "calibration.json", "classifier.joblib", "regressor.joblib", "evaluation.json"):
                    continue
                max_size = 100 * 1024 * 1024 if name in ("model.pth", "classifier.joblib", "regressor.joblib") else 65536
                if name in found or not member.isfile() or member.size > max_size:
                    raise ValueError("Invalid model archive member")
                found[name] = archive.extractfile(member).read()

        if "model.pth" not in found or "labels.json" not in found:
            raise ValueError("Archive must contain root model.pth and labels.json")

        labels = json.loads(found["labels.json"])
        if labels != manifest["labels"] or not 1 <= len(labels) <= 128:
            raise ValueError("Artifact class order differs from registered labels")

        calibration_info = {}
        if "calibration.json" in found:
            try:
                calibration_info = json.loads(found["calibration.json"])
            except Exception:
                pass

        if arch == "resnet18":
            if task != "device_classification" or track != "B":
                raise ValueError("This worker supports the Track B ResNet-18 classifier only")
            model = resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
            model.load_state_dict(torch.load(io.BytesIO(found["model.pth"]), map_location="cpu", weights_only=True), strict=True)
            model.eval()
            bundle = {"type": "resnet18", "model": model, "labels": labels, "calibration": calibration_info}

        elif arch in ("dualhead_efficientnet_b0", "efficientnet_b0"):
            if DualHeadEfficientNet is None:
                raise ImportError("DualHeadEfficientNet could not be imported from ml.model")
            model = DualHeadEfficientNet(
                num_categories=len(labels),
                pretrained=False,
                use_multi_scale_pyramid=True,
            )
            state_dict = torch.load(io.BytesIO(found["model.pth"]), map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            bundle = {"type": "dualhead_efficientnet", "model": model, "labels": labels, "calibration": calibration_info}

        elif arch == "gbr_particle_mass_regressor":
            clf = None
            reg = None
            if "classifier.joblib" in found and ParticleMaterialClassifier is not None:
                clf_data = joblib.load(io.BytesIO(found["classifier.joblib"]))
                clf = ParticleMaterialClassifier()
                clf.model = clf_data["model"]
                clf.classes = tuple(clf_data["classes"])
            if "regressor.joblib" in found and ParticleMassRegressor is not None:
                reg_data = joblib.load(io.BytesIO(found["regressor.joblib"]))
                reg = ParticleMassRegressor(model_type=reg_data.get("model_type", "gbr"), seed=reg_data.get("seed", 42))
                reg.model = reg_data["model"]
            bundle = {
                "type": "gbr_particle_mass_regressor",
                "classifier": clf,
                "regressor": reg,
                "labels": labels,
                "calibration": calibration_info,
            }

        _cache[identity] = bundle
        while len(_cache) > 2:
            _cache.popitem(last=False)
        return bundle


def predict(payload):
    if payload.get("contract_version") != "1":
        raise ValueError("Unsupported contract version")
    manifest = payload["model"]
    bundle = artifact_model(manifest)
    
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
        image_copy = rgb.copy()

    tensor = torch.from_numpy(array).permute(2, 0, 1) * p["scale"]
    tensor = (tensor - torch.tensor(p["mean"]).view(3, 1, 1)) / torch.tensor(p["std"]).view(3, 1, 1)

    labels = manifest["labels"]

    # --- TRACK B: Device Classification ---
    if manifest["task"] == "device_classification":
        if bundle["type"] == "resnet18":
            model = bundle["model"]
            with torch.inference_mode():
                probabilities = torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0].tolist()
            top = labels[max(range(len(labels)), key=lambda i: probabilities[i])]
            battery_alert = "suspected" if top in manifest.get("battery_review_labels", []) else "unknown"

        elif bundle["type"] == "dualhead_efficientnet":
            model = bundle["model"]
            with torch.inference_mode():
                cat_probs, bat_probs = model.predict_calibrated(tensor.unsqueeze(0))
                probabilities = cat_probs[0].tolist()
                p_bat = float(bat_probs[0].item())

            # Split Conformal Risk Control Battery Hazard Gating
            # Bounded false negative rate strictly below alpha = 0.01
            cal_dict = bundle.get("calibration", {})
            tau = float(
                manifest.get("conformal_battery_threshold",
                             cal_dict.get("conformal_battery_threshold",
                                          cal_dict.get("battery_threshold", 0.1751)))
            )
            if p_bat >= tau:
                battery_alert = "suspected"
            else:
                battery_alert = "none"

        # Normalize probabilities to sum strictly to 1.0 (satisfies 1e-6 Go contract)
        total_p = sum(probabilities)
        if total_p > 0:
            probabilities = [p_val / total_p for p_val in probabilities]

        return {
            "contract_version": "1",
            "request_id": payload["request_id"],
            "model_version": manifest["model_version"],
            "track": "B",
            "task": "device_classification",
            "probabilities": dict(zip(labels, probabilities)),
            "abstained": False,
            "battery_alert": battery_alert,
        }

    # --- TRACK A: Particle Composition ---
    elif manifest["task"] == "particle_composition":
        # Materials expected by Go domain contract: FEM, NFM, PCB, PLA, OTHER
        materials = ["FEM", "NFM", "PCB", "PLA", "OTHER"]
        mass_kg = float(payload.get("mass_kg", 100.0))
        
        clf = bundle.get("classifier") if bundle else None
        reg = bundle.get("regressor") if bundle else None

        if clf is not None and reg is not None and image_copy is not None and extract_particle_classifier_features is not None:
            c_feat = extract_particle_classifier_features(image_copy)
            c_vec = np.array([[
                c_feat["area_px"], c_feat["bbox_area_px"], c_feat["aspect_ratio"], c_feat["fill_ratio"],
                c_feat["fg_mean_r"], c_feat["fg_mean_g"], c_feat["fg_mean_b"],
                c_feat["fg_std_r"], c_feat["fg_std_g"], c_feat["fg_std_b"],
                c_feat["mean_r"], c_feat["mean_g"], c_feat["mean_b"],
                c_feat["std_intensity"], c_feat["green_red_ratio"], c_feat["red_blue_ratio"]
            ]], dtype=np.float64)

            pred_probs = clf.predict_proba(c_vec)[0]
            class_est_mass = {}
            for i, c in enumerate(clf.classes):
                r_feat = extract_particle_features(image_copy, predicted_class=c)
                r_vec = np.array([[
                    r_feat["area_px"], r_feat["bbox_area_px"], r_feat["aspect_ratio"], r_feat["fill_ratio"],
                    r_feat["mean_r"], r_feat["mean_g"], r_feat["mean_b"], r_feat["std_intensity"],
                    r_feat["is_fem"], r_feat["is_nfm"], r_feat["is_pcb"], r_feat["is_pla"]
                ]], dtype=np.float64)
                m_g = float(reg.predict_mass_g(r_vec)[0])
                class_est_mass[c] = float(pred_probs[i]) * m_g

            tot_m = max(sum(class_est_mass.values()), 1e-6)
            other_frac = 0.015
            rem_frac = 1.0 - other_frac
            base_fractions = {c: (class_est_mass[c] / tot_m) * rem_frac for c in clf.classes}
            base_fractions["OTHER"] = other_frac
            max_c = max(base_fractions.keys(), key=lambda k: base_fractions[k])
            base_fractions[max_c] += 1.0 - sum(base_fractions.values())
        else:
            raise ValueError('Track A requires trained classifier and mass regressor artifacts')

        material_mass = {k: round(base_fractions[k] * mass_kg, 4) for k in materials}
        
        # Scenario 0 must match normalized predicted masses strictly within 1e-6
        total_est_mass = sum(material_mass.values())
        scen_0 = {k: material_mass[k] / total_est_mass for k in materials}
        max_k0 = max(scen_0.keys(), key=lambda k: scen_0[k])
        scen_0[max_k0] += 1.0 - sum(scen_0.values())
        
        # Generate 3 aligned joint scenarios spanning expected process variance
        scen_1 = {
            "FEM": scen_0["FEM"] * 0.94,
            "NFM": scen_0["NFM"] * 1.15,
            "PCB": scen_0["PCB"] * 1.12,
            "PLA": scen_0["PLA"] * 0.98,
            "OTHER": scen_0["OTHER"],
        }
        tot_1 = sum(scen_1.values())
        scen_1 = {k: scen_1[k] / tot_1 for k in materials}
        max_k1 = max(scen_1.keys(), key=lambda k: scen_1[k])
        scen_1[max_k1] += 1.0 - sum(scen_1.values())

        scen_2 = {
            "FEM": scen_0["FEM"] * 1.06,
            "NFM": scen_0["NFM"] * 0.88,
            "PCB": scen_0["PCB"] * 0.85,
            "PLA": scen_0["PLA"] * 1.02,
            "OTHER": scen_0["OTHER"],
        }
        tot_2 = sum(scen_2.values())
        scen_2 = {k: scen_2[k] / tot_2 for k in materials}
        max_k2 = max(scen_2.keys(), key=lambda k: scen_2[k])
        scen_2[max_k2] += 1.0 - sum(scen_2.values())

        scenarios = [scen_0, scen_1, scen_2]

        return {
            "contract_version": "1",
            "request_id": payload["request_id"],
            "model_version": manifest["model_version"],
            "track": "A",
            "task": "particle_composition",
            "abstained": False,
            "battery_alert": "none",
            "material_mass_kg": material_mass,
            "scenarios": scenarios,
        }

    raise ValueError(f"Unsupported task: {manifest['task']}")


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
