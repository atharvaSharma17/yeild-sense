"""Dual-Head Multi-Scale Spatial Pyramid Vision Architecture (`ml/model.py`).

Key Zero-Cost Accuracy & Performance Optimizations:
1. Automatic Letterbox Detection & 3-Scale ROI Pyramid (`Active Conveyor Strip`,
   `Medium Device ROI`, `Tight Core ROI`): Resolves the XBAT+ geometry bottleneck where
   small devices occupy only 2%–9% of the 640x640 letterboxed frame.
2. Multi-Stage Feature Fusion: Extracts and L2-normalizes both Stage-5 mid-level texture
   descriptors (112-d) and Stage-8 high-level semantic descriptors (1280-d) across all 3
   spatial scales (total 4,176-d pyramid representation).
3. Exact Convex L-BFGS / Warm-Start Head Initialization + Hierarchical Battery Prior Fusion
   + Post-Hoc Temperature Calibration.
"""

from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DualHeadEfficientNet(nn.Module):
    """Multi-scale pyramid Dual-Head WEEE category and battery hazard perception network."""

    def __init__(
        self,
        num_categories: int,
        pretrained: bool = True,
        backbone_name: str = "efficientnet_b0",
        high_risk_class_indices: Optional[Iterable[int]] = None,
        use_multi_scale_pyramid: bool = True,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.backbone_name = backbone_name
        self.use_multi_scale_pyramid = use_multi_scale_pyramid

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Stage-5 (112-d) + Stage-8 (1280-d) = 1392-d per scale; x3 scales = 4176-d
        self.single_scale_dim = 1280 + 112
        self.pyramid_dim = self.single_scale_dim * 3 if use_multi_scale_pyramid else 1280

        self.category_head = nn.Linear(self.pyramid_dim, num_categories)
        self.battery_head = nn.Linear(self.pyramid_dim, 1)

        # Learnable post-hoc calibration temperatures
        self.temperature = nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.battery_temperature = nn.Parameter(torch.ones(1, dtype=torch.float32))

        mask = torch.zeros(num_categories, dtype=torch.float32)
        if high_risk_class_indices is not None:
            for idx in high_risk_class_indices:
                if 0 <= idx < num_categories:
                    mask[idx] = 1.0
        self.register_buffer("high_risk_mask", mask)

    def freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_top_stages(self, n_blocks: int = 3) -> None:
        children = list(self.features.children())
        for block in children[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    def _run_backbone_stages(self, x_224: torch.Tensor) -> torch.Tensor:
        h = x_224
        mid_f = None
        for idx, layer in enumerate(self.features):
            h = layer(h)
            if idx == 5:
                mid_f = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)
                mid_f = F.normalize(mid_f, dim=-1)
        high_f = self.avgpool(h).flatten(1)
        high_f = F.normalize(high_f, dim=-1)
        if mid_f is None:
            return high_f
        return torch.cat([high_f, mid_f], dim=-1)

    def extract_pyramid_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 3-scale (Active Strip, Center ROI, Core ROI) multi-stage features."""
        if not self.use_multi_scale_pyramid:
            v = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            h = self.avgpool(self.features(v)).flatten(1)
            return F.normalize(h, dim=-1)

        _, _, H, W = x.shape
        # Detect if top/bottom 10% borders are white Roboflow letterbox bars
        top_mean = float(x[:, :, : max(1, int(0.10 * H)), :].mean().item())
        bot_mean = float(x[:, :, min(H - 1, int(0.90 * H)) :, :].mean().item())
        is_letterboxed = (top_mean > 1.0) and (bot_mean > 1.0)

        if is_letterboxed:
            c1 = x[:, :, int(0.215 * H) : int(0.785 * H), int(0.06 * W) : int(0.94 * W)]
            c2 = x[:, :, int(0.28 * H) : int(0.66 * H), int(0.22 * W) : int(0.82 * W)]
            c3 = x[:, :, int(0.33 * H) : int(0.61 * H), int(0.32 * W) : int(0.72 * W)]
        else:
            c1 = x
            c2 = x[:, :, int(0.14 * H) : int(0.86 * H), int(0.16 * W) : int(0.84 * W)]
            c3 = x[:, :, int(0.26 * H) : int(0.74 * H), int(0.28 * W) : int(0.72 * W)]

        v1 = F.interpolate(c1, size=(224, 224), mode="bilinear", align_corners=False)
        v2 = F.interpolate(c2, size=(224, 224), mode="bilinear", align_corners=False)
        v3 = F.interpolate(c3, size=(224, 224), mode="bilinear", align_corners=False)

        stacked = torch.cat([v1, v2, v3], dim=0)
        comb = self._run_backbone_stages(stacked)
        f1, f2, f3 = torch.chunk(comb, 3, dim=0)
        return torch.cat([f1, f2, f3], dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_pyramid_features(x)
        cat_logits = self.category_head(feat)
        bat_logits = self.battery_head(feat).squeeze(-1)
        return cat_logits, bat_logits

    def predict_calibrated(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cat_logits, bat_logits = self.forward(x)
        temp_cat = torch.clamp(self.temperature, min=0.05, max=10.0)
        temp_bat = torch.clamp(self.battery_temperature, min=0.05, max=10.0)

        cat_probs = torch.softmax(cat_logits / temp_cat, dim=-1)
        direct_bat_probs = torch.sigmoid(bat_logits / temp_bat)

        if torch.any(self.high_risk_mask > 0):
            hierarchical_bat_prob = torch.matmul(cat_probs, self.high_risk_mask)
            bat_probs = 0.65 * direct_bat_probs + 0.35 * hierarchical_bat_prob
        else:
            bat_probs = direct_bat_probs

        return cat_probs, torch.clamp(bat_probs, 0.0, 1.0)

    def fit_temperature(
        self,
        cal_loader,
        device: torch.device,
        max_iter: int = 50,
    ) -> dict:
        """Optimize self.temperature and self.battery_temperature on calibration logits."""
        self.eval()
        cat_logits_list: List[torch.Tensor] = []
        bat_logits_list: List[torch.Tensor] = []
        cat_targets_list: List[torch.Tensor] = []
        bat_targets_list: List[torch.Tensor] = []

        with torch.no_grad():
            for images, cat_t, bat_t in cal_loader:
                images = images.to(device)
                c_log, b_log = self.forward(images)
                cat_logits_list.append(c_log.detach())
                bat_logits_list.append(b_log.detach())
                cat_targets_list.append(cat_t.to(device))
                bat_targets_list.append(bat_t.to(device))

        cat_logits = torch.cat(cat_logits_list, dim=0)
        bat_logits = torch.cat(bat_logits_list, dim=0)
        cat_targets = torch.cat(cat_targets_list, dim=0)
        bat_targets = torch.cat(bat_targets_list, dim=0)

        ce_fn = nn.CrossEntropyLoss()
        bce_fn = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            pre_nll_cat = float(ce_fn(cat_logits, cat_targets).item())
            pre_nll_bat = float(bce_fn(bat_logits, bat_targets).item())

            best_t_cat = 1.0
            best_obj_cat = pre_nll_cat
            for cand_t in torch.linspace(0.25, 3.50, 66, device=device):
                nll_val = float(ce_fn(cat_logits / cand_t, cat_targets).item())
                if nll_val < best_obj_cat:
                    best_obj_cat = nll_val
                    best_t_cat = float(cand_t.item())

            best_t_bat = 1.0
            best_obj_bat = pre_nll_bat
            for cand_t in torch.linspace(0.25, 3.50, 66, device=device):
                nll_val = float(bce_fn(bat_logits / cand_t, bat_targets).item())
                if nll_val < best_obj_bat:
                    best_obj_bat = nll_val
                    best_t_bat = float(cand_t.item())

            self.temperature.copy_(torch.tensor([best_t_cat], dtype=torch.float32, device=device))
            self.battery_temperature.copy_(torch.tensor([best_t_bat], dtype=torch.float32, device=device))
            post_nll_cat = float(ce_fn(cat_logits / self.temperature, cat_targets).item())
            post_nll_bat = float(bce_fn(bat_logits / self.battery_temperature, bat_targets).item())

        return {
            "temperature_cat": round(float(self.temperature.item()), 4),
            "temperature_bat": round(float(self.battery_temperature.item()), 4),
            "pre_nll_cat": round(pre_nll_cat, 4),
            "post_nll_cat": round(post_nll_cat, 4),
            "pre_nll_bat": round(pre_nll_bat, 4),
            "post_nll_bat": round(post_nll_bat, 4),
        }
