# YieldSense: Intelligent WEEE Recycling Dispatch & Safety Optimization

YieldSense combines **Go-powered linear programming optimization** with **dual-track machine learning intelligence** and **AWS serverless/container infrastructure** to automate e-waste recycling triage, hazardous lithium battery containment, post-shredder material recovery, and optimal economic dispatch.

---

## 1. System Architecture

```
                                 +--------------------------------------------+
                                 |         Weighed Batch Intake (Scale)       |
                                 +--------------------------------------------+
                                                       |
                        +------------------------------+------------------------------+
                        |                                                             |
                        v (Pre-Shredder Visual Triage)                                v (Post-Shredder Separation)
         +-------------------------------+                             +-------------------------------+
         |    Track B: Whole Device      |                             |   Track A: Shred Particles    |
         |  DualHeadEfficientNet v2      |                             |  Material Classifier + GBR    |
         |  3-Scale Spatial ROI Pyramid  |                             |  TU Darmstadt RGB-MFCO v1.0   |
         |  Split Conformal Gating       |                             |  Weighbridge Reconciliation   |
         +-------------------------------+                             +-------------------------------+
                        |                                                             |
                        v                                                             v
             +---------------------+                                       +---------------------+
             |  100% Hazard Recall |                                       |  2.76% Batch MAE    |
             |  Quarantine Routing |                                       |  Joint Scenarios    |
             +---------------------+                                       +---------------------+
                        \                                                             /
                         \                                                           /
                          v                                                         v
         +------------------------------------------------------------------------------------+
         |                 YieldSense Go Backend & Linear Programming Optimizer               |
         |       (Robust Expected Value Maximization, CPCB Compliance, Route Allocation)      |
         +------------------------------------------------------------------------------------+
                                                   |
                                                   v
         +------------------------------------------------------------------------------------+
         |                   AWS Infrastructure (DynamoDB, S3, ECS Fargate)                   |
         +------------------------------------------------------------------------------------+
```

---

## 2. Core Subsystems

* **[Machine Learning & Intelligence Pipeline](ml/README.md):**
  * **Track B (Intake Triage & Safety):** `DualHeadEfficientNet` v2 with a 3-Scale Spatial ROI Pyramid (4,176-d representation) resolving acquisition letterbox padding. Uses **Split Conformal Risk Control** ($\hat{\tau} = 0.1744$) providing mathematically guaranteed $\le 1\%$ hazard error rate (**100% recall, 39/39 lithium hazards caught** on held-out test data).
  * **Track A (Particle Composition & Mass Regression):** Trained on TU Darmstadt WEEE RGB-MFCO dataset (1,976 paired particle images with calibrated scale masses). Evaluated on 2,789 unlabelled particles across 3 real mixed shred batches, achieving **2.76% composition MAE** and tracking industrial weighbridge scale tonnage within 2.8%–6.4%.
  * **Serving Sidecar (`ml/serving/worker.py`):** Low-latency HTTP inference worker compliant with the Go backend contract, generating exact Dirichlet joint scenarios matching scenario 0 within $10^{-6}$.
* **[Go Backend & Optimizer](backend/README.md):**
  * Built with **Go 1.26.1 and Echo v5**.
  * Pure serverless persistence using **Amazon DynamoDB and S3 only**.
  * Provides batch lifecycle management, route economics, robust optimization, immutable JSON audit trails, and versioned optimistic concurrency.
* **[AWS Infrastructure Guide](infra/README.md):**
  * CloudFormation templates (`infra/storage.yaml`, `infra/service.yaml`) provisioning encrypted S3, pay-per-request DynamoDB, ECR, Secrets Manager, and ECS Fargate Express Mode.
* **[ML Integration Handoff](docs/ml-integration.md):**
  * Wire format contracts, packaging specifications, and sidecar deployment guide.

---

## 3. Quickstart & Verification

```powershell
# 1. Run all serving worker & contract tests (8/8 passed)
python -m unittest ml/serving/test_dualhead_worker.py ml/serving/test_worker.py ml/serving/test_track_a_worker.py

# 2. Re-verify Track B held-out evaluation
python ml/evaluate.py --model_dir artifacts --test_csv data/processed/test.csv

# 3. Re-verify Track A physical particle training & DS3 mixed-batch validation
python ml/track_a/train.py --zip_path RGB-MFCO-dataset-v1.0.0.zip --output_dir artifacts/track_a

# 4. Deploy AWS infrastructure (ap-south-1)
./infra/deploy.ps1 -Region ap-south-1
```

