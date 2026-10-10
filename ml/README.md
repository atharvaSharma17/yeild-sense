# YieldSense Machine Learning & Intelligence Pipeline

PyTorch 2.x, 3-Scale Spatial ROI Pyramid Vision, Split Conformal Risk Control, and Entropy-Conditioned Dirichlet Simplex Scenarios.

This directory houses the complete intelligence pipeline for YieldSense:
1. **Track B (Intake Visual Triage):** Whole-device category classification across 15 WEEE categories and battery hazard detection with distribution-free risk guarantees.
2. **Statistical Uncertainty Bridge:** Maps visual prediction entropy to dynamic Dirichlet concentration parameters $\kappa(H(p))$ centered on empirical FutuRaM priors.
3. **Serving Worker (`ml/serving/`):** Lightweight HTTP inference worker compliant with the Go backend contract (`backend/internal/inference/contract.go`).
4. **Track A (Particle Composition):** Particle feature extraction, mass reconciliation, and scenario generation for shredded material.

---

## 1. Architecture & Methodology

### 1.1 Discovery: Letterbox Padding & Small-Device Bottleneck
Analysis of the official XBAT+ v1.0 dataset revealed that **43% of every 640x640 intake image** (`y < 138` and `y > 502`) is dead white letterbox padding from acquisition framing. Furthermore, small critical devices (`Thermometer`, `FlashLight`, `FireAlarm`, `E-Razor`) occupy only **2%–9%** of the frame. Standard 224x224 isotropic downsampling caused severe loss of fine texture and battery seam cues.

### 1.2 The 3-Scale Spatial ROI Pyramid (`DualHeadEfficientNet` v2)
To resolve this without GPU overhead, [`ml/model.py`](model.py) implements a 3-Scale Spatial ROI Pyramid:
* **Scale 1 (Active Conveyor Strip):** Automatically detects and crops the active conveyor area, stripping dead white margins.
* **Scale 2 (Medium Device ROI):** Focuses on the central 60% device bounding region.
* **Scale 3 (Tight Core ROI):** Crops the inner 40% core, preserving fine surface texture and battery compartment lines.

### 1.3 Multi-Stage Feature Fusion
Across all 3 spatial crops, the model extracts and L2-normalizes:
* **Stage-5 Mid-Level Texture Descriptors (`112-d`)**
* **Stage-8 High-Level Semantic Descriptors (`1280-d`)**

Concatenating across the 3 scales produces a unified **4,176-dimensional representation** driving two prediction heads:
1. **Category Classification Head:** $g_\psi(f(x)) = z_{\text{cat}} \in \mathbb{R}^{15}$, producing calibrated probabilities scaled by temperature $T$.
2. **Battery Safety Hazard Head:** $h_\phi(f(x)) = z_{\text{bat}} \in \mathbb{R}$, producing battery existence probability $p_{\text{bat}}(x) = \sigma(z_{\text{bat}})$.

### 1.4 Fast Convex Training Solver
Rather than expensive backpropagation across the entire network, [`ml/train.py`](train.py) performs single-pass feature extraction through the pretrained backbone, caching the 4,176-d representations and solving both heads via an exact convex L-BFGS solver in **< 30 seconds on CPU**.

---

## 2. Mathematical Safety: Split Conformal Risk Control

Under battery handling and CPCB guidelines, non-quarantined lithium batteries present thermal runaway and fire risks during mechanical shredding. Battery detection cannot rely on arbitrary softmax thresholds.

Let $\mathcal{D}_{\text{cal}} = \{(x_i, y_{\text{bat}, i})\}_{i=1}^n$ be a held-out calibration partition ($n_{\text{haz}} = 36$ hazardous devices) disjoint from training. The nonconformity score for hazardous instances is $S_i = 1 - p_{\text{bat}}(x_i)$. To bound the expected False Negative Rate (FNR) strictly below $\alpha = 0.01$ with distribution-free finite-sample guarantees:

$$\hat{\tau} = 1 - \text{Quantile}\left(\{S_i\}_{i: y_{\text{bat}, i}=1}, \frac{\lceil (n_{\text{haz}} + 1)(1 - \alpha) \rceil}{n_{\text{haz}}}\right)$$

* **Calibrated Threshold:** $\hat{\tau}_{\text{bat}} = 0.1744$
* **Safety Rule:** Any sample with $p_{\text{bat}}(x) \ge \hat{\tau}_{\text{bat}}$ emits `battery_alert: "suspected"`, which the Go backend uses to automatically divert the lot to quarantine hold.
* **Held-Out Test Result:** **100% Hazard Recall (39/39 hazards caught, 0 missed)**.

---

## 3. Statistical Uncertainty Bridge & Scenarios

The downstream optimizer requires discrete composition scenarios $\mathbf{p}_s = [p_{s, \text{FEM}}, p_{s, \text{NFM}}, p_{s, \text{PCB}}, p_{s, \text{PLA}}, p_{s, \text{OTHER}}]$ satisfying $\sum_k p_{s, k} = 1$.

Visual triage uncertainty is quantified via Shannon entropy:
$$H(p_{\text{cat}}) = -\sum_{c=1}^C p_{\text{cat}}(c \mid x) \log p_{\text{cat}}(c \mid x)$$

The Dirichlet concentration parameter $\kappa$ scales inversely with visual uncertainty:
$$\kappa(x) = \kappa_{\min} + (\kappa_{\max} - \kappa_{\min}) \exp(-\beta \cdot H(p_{\text{cat}}))$$

Given the category prior $\mathbf{\mu}_c$ from FutuRaM v14.2:
$$\mathbf{\alpha}(x) = \max(\mathbf{\mu}_c, 10^{-4}) \cdot \kappa(x)$$
$$\mathbf{p}_s \sim \text{Dirichlet}(\mathbf{\alpha}(x)), \quad s \in \{1, \dots, S\}$$

* **High visual certainty:** Concentrates scenarios tightly around the historical mean.
* **High visual ambiguity:** Widens scenario variance, forcing the optimizer to hedge against downside material risk.
* **Unit Simplex Enforcement:** Employs exact simplex projection ensuring $|\sum_k p_{s, k} - 1.0| < 10^{-12}$.

---

## 4. Data Provenance & Dataset Inventory

All datasets are checksum-verified against public Zenodo records:

| Dataset | Zenodo Record | File | Size | Role |
|---|---|---|---|---|
| **FutuRaM WEEE Composition v14.2** | [Zenodo 17789750](https://zenodo.org/records/17789750) | `consolidated composition v14.2.csv` | 8.4 MB | Empirical material priors across `WEEE_Cat1`–`WEEE_Cat6` |
| **XBAT+ v1.0 Metadata** | [Zenodo 18022530](https://zenodo.org/records/18022530) | `res_XBAT+_v1.0_metadata.zip` | 1.4 KB | 15-category taxonomy & annotation metadata |
| **XBAT+ v1.0 RGB (Train)** | [Zenodo 18022530](https://zenodo.org/records/18022530) | `res_XBAT+_v1.0_RGB_train.zip` | 6.0 MB | Grouped physical split into Train (247) & Calibration (83) |
| **XBAT+ v1.0 RGB (Test)** | [Zenodo 18022530](https://zenodo.org/records/18022530) | `res_XBAT+_v1.0_RGB_test.zip` | 1.6 MB | 91 held-out test images |

Data partitions are strictly grouped by physical `device_id` ([`scripts/split_xbat.py`](../scripts/split_xbat.py)) to prevent cross-partition leakage.

---

## 5. Experimental Results (Held-Out Test Set)

Evaluated on the locked 91-image held-out test split:

| Metric | Baseline ResNet-18 (1-Crop) | 2-Stage Unfreeze | **DualHeadEfficientNet v2 (3-Scale Pyramid)** |
|---|---:|---:|---:|
| **Top-1 Category Accuracy** | 57.14% | 68.13% | **78.02%** (+20.88%) |
| **Macro F1** | 0.3580 | 0.4681 | **0.5346** (+17.66%) |
| **Battery Hazard ROC-AUC** | 0.8570 | 0.9221 | **0.9408** |
| **Conformal Hazard Recall ($\alpha=0.01$)** | 100.0% (39/39) | 100.0% (39/39) | **100.0% (39/39)** |
| **Calibration ECE (Post)** | 0.1205 | 0.1143 | **0.0761** |
| **Training Time (CPU)** | ~140 s | ~195 s | **< 35 s** |

---

## 6. Assumptions & Scope Boundaries

1. **Intake Triage vs. Chemical Assay:** RGB images detect device categories and battery hazards from exterior visual features; they cannot measure internal precious metal purities or chemical compositions.
2. **Fail-Closed Safety:** Suspect or unknown battery lots are restricted from mechanical shredding routes and diverted exclusively to quarantine.
3. **Scenario Alignment:** Scenario indices across batches represent shared joint macroeconomic/recovery outcomes.

---

## 7. Serving Worker & Backend Contract

The serving worker in [`ml/serving/worker.py`](serving/worker.py) implements `POST /predict` on `http://127.0.0.1:9000` (or ECR sidecar):
* Accepts Go backend requests containing authorization token, request ID, image bytes, and model manifest.
* Supports `dualhead_efficientnet_b0` and `resnet18`.
* Returns calibrated class probabilities summing strictly to 1.0 ($\pm 10^{-6}$), within the 256 KiB response budget.
* Emits `battery_alert: "suspected"` when $p_{\text{bat}} \ge \hat{\tau}_{\text{bat}}$, triggering Go quarantine routing.
* Supports Track A (`task: "particle_composition"`) returning normalized material masses and scenarios matching scenario 0.

---

## 8. Reproduction & Commands

```powershell
# 1. Generate splits & priors
python scripts/split_xbat.py
python scripts/transform_futuram.py

# 2. Train and calibrate (CPU, <35s)
python ml/train.py --train_dir data/processed --val_dir data/processed --model_dir artifacts

# 3. Evaluate held-out test performance
python ml/evaluate.py --model_dir artifacts --test_csv data/processed/test.csv

# 4. Package model archive & Go-compliant manifest
python scripts/package_model.py

# 5. Run serving and contract integration tests
python -m unittest ml/serving/test_dualhead_worker.py
$env:PYTHONPATH="ml/serving"; python -m unittest ml/serving/test_worker.py
```
