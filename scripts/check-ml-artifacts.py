"""Offline runtime check of supplied archives or extracted model directories.

Uses a synthetic image only to check execution, never to evaluate accuracy.
Does not modify the supplied files or their manifests.
"""
import argparse
import base64
import hashlib
import gzip
import io
import json
from pathlib import Path
import sys
import tarfile
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, "/app/ml/serving")
import worker


def archive_bytes(path, track):
    if path.is_file():
        return path.read_bytes()
    if not path.is_dir():
        raise FileNotFoundError(path)
    output = io.BytesIO()
    order = ["model.pth", "labels.json", "calibration.json"] if track == "B" else [
        "model.pth", "regressor.joblib", "classifier.joblib", "labels.json", "calibration.json", "evaluation.json"]
    if set(file.name for file in path.iterdir()) != set(order):
        raise ValueError("Extracted directory contains missing or unexpected entries")
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w:") as archive:
            for name in order:
                file = path / name
                if not file.is_file():
                    raise ValueError(f"Unexpected nested directory: {file}")
                data = file.read_bytes()
                member = tarfile.TarInfo(file.name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    return output.getvalue()


def check(root, track, output_root=None):
    directory = root if track == "B" else root / "track_a"
    manifest = json.loads((directory / "registered_model_manifest.json").read_text())
    path = directory / "model.tar.gz"
    data = archive_bytes(path, track)
    supplied_hash = manifest["artifact_sha256"]
    actual_hash = hashlib.sha256(data).hexdigest()
    if path.is_file() and actual_hash != supplied_hash:
        raise ValueError(f"Track {track}: archive hash differs from manifest")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        labels = json.load(archive.extractfile("labels.json"))
        names = archive.getnames()
    if labels != manifest["labels"]:
        raise ValueError(f"Track {track}: label order differs from manifest")
    # An extracted directory cannot prove the original compressed archive hash.
    manifest["artifact_sha256"] = actual_hash
    image = io.BytesIO()
    Image.new("RGB", (224, 224), (120, 130, 110)).save(image, format="PNG")
    image_data = image.getvalue()
    payload = {
        "contract_version": "1", "request_id": f"offline-track-{track}",
        "batch_id": "offline-validation", "mass_kg": 100,
        "model": manifest,
        "image": {"size_bytes": len(image_data), "sha256": hashlib.sha256(image_data).hexdigest()},
        "image_base64": base64.b64encode(image_data).decode(),
    }
    worker._cache.clear()
    with patch("worker._load_raw_artifact", return_value=data):
        result = worker.predict(payload)
    report = {
        "track": track, "architecture": manifest["architecture"],
        "format": "extracted directory" if path.is_dir() else "compressed archive",
        "original_archive_hash_verified": path.is_file(), "archive_members": names,
        "matches_supplied_archive_hash": actual_hash == supplied_hash,
        "labels_match": True, "runtime_forward_succeeded": True,
        "synthetic_image_only": True, "result_task": result["task"],
    }
    if track == "B":
        report["probability_sum"] = sum(result["probabilities"].values())
    else:
        report["scenario_count"] = len(result["scenarios"])
        report["scenario_sums"] = [sum(s.values()) for s in result["scenarios"]]
    if output_root is not None:
        destination = output_root / f"track-{track.lower()}"
        destination.mkdir(parents=True, exist_ok=True)
        archive_path = destination / "model.tar.gz"
        manifest_path = destination / "registered_model_manifest.json"
        if archive_path.exists() or manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite deployment artifacts: {destination}")
        archive_path.write_bytes(data)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        report["deployment_archive"] = str(archive_path)
        report["deployment_manifest"] = str(manifest_path)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/input"))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    failed = False
    for track in ("B", "A"):
        try:
            print(json.dumps(check(args.root, track, args.output_root), indent=2))
        except Exception as error:
            failed = True
            print(json.dumps({"track": track, "error": f"{type(error).__name__}: {error}"}, indent=2))
    if failed:
        sys.exit(1)
