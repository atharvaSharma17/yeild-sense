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
    if np.sum(fg_mask) == 0:
        fg_mask = gray > 0.01
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


def extract_particle_classifier_features(image: Image.Image) -> Dict[str, float]:
    """Extract class-independent visual features (morphology, color moments, ratios) for material classification."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    h, w, _ = rgb.shape
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    fg_mask = gray > 0.08
    if np.sum(fg_mask) == 0:
        fg_mask = gray > 0.01

    area_px = float(np.sum(fg_mask))
    aspect_ratio = float(w / max(h, 1))
    fill_ratio = float(area_px / max(float(h * w), 1.0))

    fg_r = rgb[:, :, 0][fg_mask] if np.sum(fg_mask) > 0 else rgb[:, :, 0].flatten()
    fg_g = rgb[:, :, 1][fg_mask] if np.sum(fg_mask) > 0 else rgb[:, :, 1].flatten()
    fg_b = rgb[:, :, 2][fg_mask] if np.sum(fg_mask) > 0 else rgb[:, :, 2].flatten()

    mean_fg_r = float(np.mean(fg_r))
    mean_fg_g = float(np.mean(fg_g))
    mean_fg_b = float(np.mean(fg_b))
    std_fg_r = float(np.std(fg_r))
    std_fg_g = float(np.std(fg_g))
    std_fg_b = float(np.std(fg_b))

    mean_r = float(np.mean(rgb[:, :, 0]))
    mean_g = float(np.mean(rgb[:, :, 1]))
    mean_b = float(np.mean(rgb[:, :, 2]))
    std_intensity = float(np.std(gray))

    # Color space ratios
    green_red_ratio = float(mean_fg_g / max(mean_fg_r, 1e-4))
    red_blue_ratio = float(mean_fg_r / max(mean_fg_b, 1e-4))

    return {
        "area_px": area_px,
        "bbox_area_px": float(h * w),
        "aspect_ratio": aspect_ratio,
        "fill_ratio": fill_ratio,
        "fg_mean_r": mean_fg_r,
        "fg_mean_g": mean_fg_g,
        "fg_mean_b": mean_fg_b,
        "fg_std_r": std_fg_r,
        "fg_std_g": std_fg_g,
        "fg_std_b": std_fg_b,
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "std_intensity": std_intensity,
        "green_red_ratio": green_red_ratio,
        "red_blue_ratio": red_blue_ratio,
    }

