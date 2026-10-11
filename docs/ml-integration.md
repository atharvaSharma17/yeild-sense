# ML integration handoff

For HLD **Track B**, use the implementation guide's PyTorch ResNet-18 classifier. The included `ml/serving/worker.py` loads and serves its real weights on CPU; it does not train a model or return mock predictions.

Deliver:

1. A real SageMaker training job and held-out evaluation evidence.
2. Its `model.tar.gz`, uploaded to the deployed S3 bucket under `models/` or `model-artifacts/`. It must contain root-level `model.pth` and `labels.json`, using the guide's `torch.save(model.state_dict(), ...)` format. The final ResNet-18 linear layer must match the number of labels.
3. A model manifest with actual labels in class-index order, preprocessing, archive SHA-256, dataset-manifest SHA-256, data version, training-job ARN and evaluation metrics.

Register the manifest with `POST /v1/models`, using the backend bearer key. Versions are immutable; register a new version for each changed model. The worker verifies the archive hash and compares `labels.json` with the registered class order. Training-job/evaluation claims are recorded, not independently verified by the backend.

This example shows the manifest shape; replace placeholders and use a real numeric `macro_f1`:

```json
{
  "contract_version":"1",
  "model_version":"xbat-classifier-v1",
  "track":"B",
  "task":"device_classification",
  "architecture":"resnet18",
  "artifact_s3_uri":"s3://ACTUAL_BUCKET/model-artifacts/ACTUAL_JOB/output/model.tar.gz",
  "artifact_sha256":"ACTUAL_64_HEX_HASH_OF_COMPRESSED_ARCHIVE",
  "dataset_manifest_sha256":"ACTUAL_64_HEX_HASH_OF_DATA_MANIFEST",
  "data_version":"ACTUAL_DATA_AND_SPLIT_VERSION",
  "training_job_arn":"arn:aws:sagemaker:ap-south-1:ACTUAL_ACCOUNT:training-job/ACTUAL_JOB",
  "labels":["ACTUAL_CLASS_AT_INDEX_0","ACTUAL_CLASS_AT_INDEX_1"],
  "preprocessing":{
    "width":224,"height":224,"color_mode":"RGB","scale":0.00392156862745098,
    "mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]
  },
  "confidence_threshold":0.7,
  "evaluation":{"macro_f1":"REPLACE_WITH_REAL_NUMBER"},
  "battery_review_labels":[]
}
```

The wrapper implements RGB conversion, bilinear resize, scaling and channel normalization. Supply exactly the training preprocessing. Different architectures/preprocessing require a custom worker implementing the contract below. `battery_review_labels` may contain only actual model labels whose inspected dataset semantics justify a battery review alert. An empty list makes no battery-specific class claim. The confidence threshold is an abstention policy, not statistically calibrated confidence.

## Deployment

Build the CPU wrapper for linux/amd64, then tag/push it to the **same foundation ECR repository** as the API, under a unique tag. Obtain its digest using `aws ecr describe-images`. The execution role is intentionally scoped to that repository.

```powershell
docker build --platform linux/amd64 -f ml/serving/Dockerfile -t yieldsense-inference .
# Authenticate to ECR, tag/push the worker, then use its actual digest:
./infra/deploy.ps1 -MLWorkerDigest 'sha256:ACTUAL_64_HEX_DIGEST' -DefaultModelVersion xbat-classifier-v1
```

CloudFormation runs the worker at `127.0.0.1:9000/predict` beside Go. Its port is not exposed by the load balancer. Both receive `ML_INFERENCE_TOKEN`; the worker also receives `S3_BUCKET`, `AWS_REGION` and `ML_LISTEN_ADDR`. The worker can read the authorized model prefixes. It is nonessential, so its failure does not stop the API's manual-composition workflow.

The first request downloads/loads the model; a cold load may exceed the 12-second budget. Retry using the unchanged batch version after a timeout. The worker caches at most two models and does not download pretrained weights. No real-time SageMaker endpoint is provisioned or claimed.

Alternatively configure `ML_INFERENCE_URL`, `ML_INFERENCE_TOKEN` and `DEFAULT_MODEL_VERSION` locally, or deploy with `-MLInferenceURL 'https://worker.example/predict'`. The deployment uses the generated API secret as the worker bearer token; configure the remote worker to accept it. Choose the worker digest or remote URL, not both.

## Public API flow

All `/v1` calls require the backend bearer key.

1. Create the batch through `/v1/batches`.
2. Upload JPEG/PNG bytes to `/v1/batches/{id}/images`; retain its returned image ID. `/v1/images/{id}` retrieves metadata.
3. Call `POST /v1/predict-device` with JSON and `If-Match: <current batch version>`:

```json
{"batch_id":"stored-batch-id","image_id":"stored-image-id","model_version":"xbat-classifier-v1"}
```

The model version may be omitted when a default is configured. Callers cannot supply arbitrary S3 keys or model URLs. The response is `{prediction, batch}` and includes class probabilities, top class, confidence, abstention/review flags, image hash, model provenance and the new batch version. `/v1/predictions/{id}` retrieves saved evidence. If a response was lost, reload the batch and follow `latest_prediction_id` before retrying.

4. Supply validated category-prior/operator composition separately through `/v1/composition`, record human review, then optimize. Track B does not invent material fractions or change category automatically.

Abstention or a suspected battery resets review to pending. A suspected alert sets battery status to suspected. `unknown`/`none` never clear batteries or override existing human inspection. Prediction evidence and batch changes commit atomically in DynamoDB; concurrent edits return 409 with neither write committed. Reports preserve the exact model evidence used.

Missing worker configuration returns 503 `ML_NOT_CONFIGURED`; unavailable worker 503, timeout 504, invalid output 502. These failures leave the batch unchanged. `/health` exposes `ml_configured`, which indicates configuration, not model readiness.

## Custom worker contract

Implement `POST /predict`, accepting JSON and `Authorization: Bearer <ML_INFERENCE_TOKEN>`. Go sends:

```json
{
  "contract_version":"1","request_id":"prediction-id","batch_id":"batch-id","mass_kg":100,
  "model":{"...":"complete registered manifest"},
  "image":{"id":"image-id","batch_id":"batch-id","object_key":"raw/...","sha256":"...","size_bytes":1234,"content_type":"image/png","created_at":"..."},
  "image_base64":"BASE64_OF_ORIGINAL_IMAGE_BYTES"
}
```

Return JSON with the exact request/model identity. Example probabilities are illustrative:

```json
{
  "contract_version":"1","request_id":"prediction-id","model_version":"registered-version",
  "track":"B","task":"device_classification",
  "probabilities":{"ACTUAL_CLASS_AT_INDEX_0":0.8,"ACTUAL_CLASS_AT_INDEX_1":0.2},
  "abstained":false,"battery_alert":"unknown"
}
```

Return all registered probabilities, finite, 0-1, summing to one within `1e-6`. Unsupported inputs may return `abstained:true`, a nonempty `abstention_reason`, and no probabilities. Allowed battery alerts are `unknown`, `suspected`, `none`. Go recomputes top class and applies the threshold. Unsupported output labels cause abstention; identity mismatch, extra fields or invalid numbers are rejected. Device outputs cannot include material composition.

Budgets: two concurrent inference requests, 12-second timeout, 256-KiB response, up to 15-MiB worker request (10-MiB/16-megapixel image). Go verifies the saved image hash before forwarding and does not follow redirects. Workers own artifact loading/preprocessing and should cache model versions.

## Optional Track A

Register `track:A`, `task:particle_composition`, the five material labels and real `paired_mass_evidence`. A custom mass-inference worker is required; the included ResNet-18 device wrapper does not estimate mass. Call `/v1/predict-composition` with the same request shape/header. Worker response example with synthetic numbers:

```json
{
  "contract_version":"1","request_id":"prediction-id","model_version":"registered-version",
  "track":"A","task":"particle_composition","abstained":false,"battery_alert":"unknown",
  "material_mass_kg":{"FEM":5,"NFM":2,"PCB":1,"PLA":1,"OTHER":1},
  "scenarios":[{"FEM":0.5,"NFM":0.2,"PCB":0.1,"PLA":0.1,"OTHER":0.1}]
}
```

Go validates positive total mass, all five material classes, and 1-20 normalized scenarios. Scenario zero must match normalized predicted masses. It records predicted-vs-measured mass discrepancy and scaling factor, and attaches attributed composition. The optimizer uses measured batch weight. Reconciliation does not prove physical accuracy. Across batches, scenario indices must represent aligned joint outcomes; the ML pipeline owns generation/calibration. Manual changes must be marked operator-entered instead of falsely retaining model attribution.

## Tests

`go test ./...` covers contract validation, inference failures, safety gating, immutable registry, provenance, concurrent edits and Track A reconciliation. Wrapper tests run actual CPU forward passes with **untrained test-only weights**; they are not training-job evidence or accuracy claims:

```powershell
docker run --rm --network none --mount "type=bind,source=$PWD/ml/serving,target=/tests,readonly" --workdir /tests yieldsense-inference python -m unittest -v test_worker
```

The team's real model still needs a live end-to-end check with its actual S3 artifact and evaluation inputs. Training roles, dataset preparation, training jobs and metrics remain ML pipeline responsibilities.

References: [HLD](../HLD.md), [implementation guide](../Implementation_Guide.md), [PyTorch CPU versions](https://pytorch.org/get-started/previous-versions/), [state-dict loading](https://docs.pytorch.org/docs/stable/generated/torch.load.html).
