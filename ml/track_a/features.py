"""Conditional Track A Particle Feature Extraction (`ml/track_a/features.py`).

Used only when paired shredded-particle RGB crops and physical particle masses
(TU Darmstadt RGB-MFCO) are verified and available.
"""

from __future__ import annotations
from typing import Dict
import numpy as np
from PIL import Image


def extract_particle_features(image: Image.Image, predicted_class: str = "FEM") -> Dict[str, float]:
    """Extract morphological, color-space, and texture summary features from a particle crop."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    h, w, _ = rgb.shape
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # Foreground mask assuming non-black / non-uniform conveyor background
    fg_mask = gray > 0.08
    area_px = float(np.sum(fg_mask))
    aspect_ratio = float(w / max(h, 1))
    fill_ratio = float(area_px / max(float(h * w), 1.0))

    mean_r, mean_g, mean_b = (
        float(np.mean(rgb[:, :, 0])),
        float(np.mean(rgb[:, :, 1])),
        float(np.mean(rgb[:, :, 2])),
    )
    std_intensity = float(np.std(gray))

    return {
        "area_px": area_px,
        "bbox_area_px": float(h * w),
        "aspect_ratio": aspect_ratio,
        "fill_ratio": fill_ratio,
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "std_intensity": std_intensity,
        "is_fem": 1.0 if predicted_class == "FEM" else 0.0,
        "is_nfm": 1.0 if predicted_class == "NFM" else 0.0,
        "is_pcb": 1.0 if predicted_class == "PCB" else 0.0,
        "is_pla": 1.0 if predicted_class == "PLA" else 0.0,
    }
