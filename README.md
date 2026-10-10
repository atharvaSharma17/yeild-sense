# YieldSense

The backend is implemented in **Go with Echo v5**, using **Amazon DynamoDB and S3 only** for runtime persistence. It provides weighed-batch intake, human review, composition scenarios, route management, robust/expected optimization, baselines, shutdown/price simulations, image uploads and immutable JSON audit reports.

See [backend setup and API documentation](backend/README.md), [machine learning & intelligence pipeline](ml/README.md), [storage infrastructure](infra/storage.yaml), and the [end-to-end demo](scripts/demo-backend.ps1).

Deploy the backend and provision storage with `./infra/deploy.ps1 -Region ap-south-1`. See the [infrastructure deployment guide](infra/README.md) for AWS setup, configuration and cleanup.

`HLD.md` and `Implementation_Guide.md` contain the original design, including Python/FastAPI, OR-Tools, and SageMaker proposals. The API and optimizer are implemented in Go. A CPU serving wrapper connects the PyTorch dual-head vision model through a versioned contract adapter; see the [ML developer handoff](docs/ml-integration.md) and the [ML intelligence architecture](ml/README.md). The frontend remains a placeholder.
