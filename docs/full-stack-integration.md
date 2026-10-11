# Frontend, Go backend, and ML worker

The browser calls the Next.js server proxy, which adds the backend bearer key. Go owns S3/DynamoDB persistence, version checks and human safety decisions. The CPU ML sidecar runs the selected registered model and returns contract-version-1 JSON. Both Track B device triage and Track A particle composition are available in the batch detail page.

## Connect the frontend now

From the repository root, with the AWS profile authenticated:

```powershell
./scripts/connect-frontend.ps1 -Profile yieldsense-dev -Region ap-south-1 -StackPrefix yieldsense-dev
cd frontend
npm ci
npm run dev
```

Open http://localhost:3000/workspace. The script creates ignored `frontend/.env.local`, selecting API mode and saving the generated backend secret only in server configuration. It refuses to overwrite an existing environment file. When one already exists, set `NEXT_PUBLIC_YIELDSENSE_DATA_MODE=api`, `YIELDSENSE_API_BASE_URL=https://<backend-host>` and `YIELDSENSE_API_KEY=<secret>` manually, then restart/rebuild Next.js. Do not use NEXT_PUBLIC_ for the key.

The existing proxy uses a shared backend credential; individual operator login is not implemented. Keep the frontend behind trusted access before hosting it for operators. Server-side secret storage alone does not authenticate browser users.

## Deploy the ML runtime and updated Go backend

```powershell
./infra/deploy.ps1 -Profile yieldsense-dev -Region ap-south-1 -StackPrefix yieldsense-dev -TableName yieldsense-dev -MaxTasks 1 -UseTemporaryDockerConfig -BuildMLWorker
```

This builds the worker from the repository root, including the dual-head EfficientNet and Track A modules, pushes it into the existing ECR repository and deploys it beside Go. Model files are downloaded from the authorized S3 prefixes on demand. Building/deploying this runtime does not create trained weights or register any model. Both tracks use explicit frontend model selection; a single default model is not needed.

The updated Go backend must be deployed before model registration or using the new image-list/content endpoints. Images uploaded after this update receive a batch-scoped DynamoDB index. Older uploads have no such index; re-upload them to make them available in the new image picker.

## ML developer handoff

Request each real `model.tar.gz` and matching JSON manifest. The files must agree on architecture, label order, preprocessing and archive SHA-256. The received extracted model directories have now been checked with the runtime. Upload-ready compressed copies are in `artifacts/deployment/track-b/` and `artifacts/deployment/track-a/`; both compressed hashes exactly match the original manifests. Both tracks execute with a synthetic smoke-test image; this does not verify accuracy. Track A requires scikit-learn 1.7.2, now pinned in the serving Dockerfile. Existing manifest bucket/account placeholders are not usable deployment targets.

Track B supports ResNet-18 or `dualhead_efficientnet_b0`; its archive includes model.pth, labels.json and optional calibration.json. Track A `gbr_particle_mass_regressor` requires classifier.joblib and regressor.joblib plus the five labels. Joblib deserialization requires trusted artifacts from the ML developer; match the training scikit-learn version with the runtime before deployment. Track A no longer substitutes heuristic fractions if trained components are absent. Its current worker expects compatible particle imagery, not arbitrary whole-lot photos; bulk particle aggregation is not implemented by this UI.

For a locally trained model, record actual run/evaluation provenance:

```powershell
./scripts/register-ml-model.ps1 -ManifestPath artifacts/deployment/track-b/registered_model_manifest.json -ArchivePath artifacts/deployment/track-b/model.tar.gz -LocalTrainingEvidence 'Actual training run log location, dataset split and evaluation report reference'

./scripts/register-ml-model.ps1 -ManifestPath artifacts/deployment/track-a/registered_model_manifest.json -ArchivePath artifacts/deployment/track-a/model.tar.gz -LocalTrainingEvidence 'Actual paired-particle training and held-out evaluation evidence'
```

For SageMaker training, supply `-TrainingJobArn <real-arn>` instead. The registration script discovers the real deployed bucket, checks the archive hash, uploads it under an immutable model-version prefix and registers the manifest through Go. It rejects placeholder job account IDs. It never generates a training job or invents evaluation metrics. Local training is explicitly recorded and does not satisfy an HLD requirement for a real SageMaker training run. Go validates provenance fields, not their truth.

Models are immutable. A changed artifact requires a new version and matching manifest. A cold model load can exceed the current 12-second inference budget; reload the batch before retrying to check its latest saved prediction.

## Operator flow

1. Create a batch with its weighed mass.
2. Upload a compatible JPEG/PNG image. Images persist in S3 and remain selectable after reload.
3. Choose the Track B or Track A model and image in Image analysis.
4. Run inference. Go validates the result and commits prediction evidence and batch changes atomically.
5. For device triage, supply measured/category-prior/operator composition separately. Successful Track A output attaches its composition automatically.
6. Record human review and battery inspection evidence, then use route planning and reports.

Predictions never clear batteries or grant authorization. If the worker/artifact is unavailable, the UI displays a failure and manual workflows remain available.
