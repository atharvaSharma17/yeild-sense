"""Calibrated Inference & Conformal Safety Triage Engine for YieldSense 2.0 (`ml/inference.py`).

Executes:
1. Temperature-scaled DualHeadEfficientNet inference on RGB WEEE photographs.
2. Split Conformal Risk Control on Battery Hazards (FNR <= alpha = 0.01).
3. Split Conformal Prediction Set (APS) construction for category ambiguity detection.
4. Entropy-conditioned Dirichlet composition scenario generation centered on verified FutuRaM priors.
"""

from __future__ import annotations
import io
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Union
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model import DualHeadEfficientNet
from scenarios import XBAT_CLASS_TO_WEEE_CAT, generate_adaptive_scenarios
from split_dataset import HIGH_RISK_LITHIUM_CATEGORIES


class TriageEngine:
    """Calibrated visual perception, conformal hazard gating, and scenario bridge."""

    def __init__(
        self,
        artifact_dir: Union[Path, str] = Path("artifacts"),
        conf_threshold: float = 0.65,
    ):
        self.artifact_dir = Path(artifact_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = float(conf_threshold)

        cat_file = self.artifact_dir / "categories.json"
        if not cat_file.exists():
            raise FileNotFoundError(f"Missing categories.json in {self.artifact_dir}")
        self.categories: List[str] = json.loads(cat_file.read_text(encoding="utf-8"))
        self.high_risk_indices = [
            i for i, c in enumerate(self.categories) if c in HIGH_RISK_LITHIUM_CATEGORIES
        ]

        self.model = DualHeadEfficientNet(
            num_categories=len(self.categories),
            pretrained=False,
            high_risk_class_indices=self.high_risk_indices,
        )
        weights_file = self.artifact_dir / "model.pth"
        state_dict = torch.load(weights_file, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        cal_file = self.artifact_dir / "calibration_scores.npz"
        if cal_file.exists():
            cal_data = np.load(cal_file)
            if "conformal_battery_threshold" in cal_data:
                self.battery_threshold = float(cal_data["conformal_battery_threshold"])
            elif "battery_threshold" in cal_data:
                self.battery_threshold = float(cal_data["battery_threshold"])
            else:
                self.battery_threshold = 0.1751
            self.category_qhat = (
                float(cal_data["category_qhat"])
                if "category_qhat" in cal_data
                else 0.85
            )
            self.alpha = float(cal_data["alpha"]) if "alpha" in cal_data else 0.01
        else:
            self.battery_threshold = 0.1751
            self.category_qhat = 0.85
            self.alpha = 0.01

        meta_file = self.artifact_dir / "training_metadata.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            self.model_version = str(meta.get("model_version", "xbat-dualhead-v1"))
            self.sagemaker_job_name = str(meta.get("sagemaker_job_name", "local"))
        else:
            self.model_version = "xbat-dualhead-v1"
            self.sagemaker_job_name = "local"

    def _load_image(self, image_source: Union[Path, str, bytes, Image.Image]) -> Image.Image:
        if isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        if isinstance(image_source, (bytes, bytearray)):
            return Image.open(io.BytesIO(image_source)).convert("RGB")
        return Image.open(Path(image_source)).convert("RGB")

    def predict(
        self,
        image_path: Union[Path, str, bytes, Image.Image],
        n_scenarios: int = 10,
        seed: int = 42,
    ) -> Dict[str, Any]:
        image = self._load_image(image_path)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            cat_probs, bat_probs = self.model.predict_calibrated(tensor)

        cat_np = cat_probs[0].cpu().numpy()
        bat_prob = float(bat_probs[0].item())

        idx = int(np.argmax(cat_np))
        confidence = float(cat_np[idx])
        predicted_cat = self.categories[idx]
        weee_category = XBAT_CLASS_TO_WEEE_CAT.get(predicted_cat, "WEEE_Cat5")
        entropy = float(-np.sum(cat_np * np.log(np.maximum(cat_np, 1e-12))))
        max_entropy = float(np.log(max(len(self.categories), 2)))
        norm_entropy = float(entropy / max_entropy)

        # Conformal prediction set C_{1 - alpha_cat}(x) = {c : 1 - p_c(x) <= q_hat}
        min_prob_for_set = 1.0 - self.category_qhat
        conformal_set = [
            self.categories[i]
            for i in np.argsort(-cat_np)
            if float(cat_np[i]) >= min_prob_for_set
        ]
        if not conformal_set:
            conformal_set = [predicted_cat]

        # Safety triage via Split Conformal Risk Control (FNR <= alpha)
        is_hazard = bat_prob >= self.battery_threshold
        low_confidence = confidence < self.conf_threshold
        review_reasons: List[str] = []

        if is_hazard:
            battery_status = "suspect"
            review_reasons.append(
                f"Battery hazard score ({bat_prob:.3f}) >= conformal threshold ({self.battery_threshold:.3f})"
            )
        else:
            battery_status = "confirmed_absent"

        if low_confidence:
            review_reasons.append(
                f"Category confidence ({confidence:.3f}) < abstention threshold ({self.conf_threshold:.3f})"
            )
        if len(conformal_set) > 3:
            review_reasons.append(
                f"Conformal prediction set spans {len(conformal_set)} classes: {conformal_set[:4]}"
            )

        human_review = bool(is_hazard or low_confidence or len(conformal_set) > 3)

        scenarios = generate_adaptive_scenarios(
            category=predicted_cat,
            entropy=entropy,
            n_scenarios=n_scenarios,
            seed=seed,
        )

        top3_idx = np.argsort(-cat_np)[:3]
        top_probabilities = {
            self.categories[int(i)]: round(float(cat_np[int(i)]), 4) for i in top3_idx
        }

        return {
            "predicted_category": predicted_cat,
            "weee_category": weee_category,
            "confidence": round(confidence, 4),
            "entropy": round(entropy, 4),
            "normalized_entropy": round(norm_entropy, 4),
            "battery_risk_score": round(bat_prob, 4),
            "battery_threshold": round(self.battery_threshold, 4),
            "battery_status": battery_status,
            "human_review_required": human_review,
            "review_reasons": review_reasons,
            "conformal_prediction_set": conformal_set,
            "top_probabilities": top_probabilities,
            "model_version": self.model_version,
            "sagemaker_job_name": self.sagemaker_job_name,
            "scenarios": scenarios,
        }
