# YieldSense

The backend is implemented in **Go with Echo v5**, using **Amazon DynamoDB and S3 only** for runtime persistence. It provides weighed-batch intake, human review, composition scenarios, route management, robust/expected optimization, baselines, shutdown/price simulations, image uploads and immutable JSON audit reports.

See [backend setup and API documentation](backend/README.md), [machine learning & intelligence pipeline](ml/README.md), and [storage infrastructure](infra/storage.yaml).

Deploy the backend and provision storage with `./infra/deploy.ps1 -Region ap-south-1`. See the [infrastructure deployment guide](infra/README.md) for AWS setup, configuration and cleanup.

`HLD.md` and `Implementation_Guide.md` contain the original design, including Python/FastAPI, OR-Tools, and SageMaker proposals. The API and optimizer are implemented in Go. The current deployment runs Track A and Track B inference in a Python worker on ECS/Fargate; it does not provision SageMaker. The Next.js frontend connects through a server-side proxy and supports stored images and registered model inference. See [full-stack setup and the ML contract](docs/full-stack-integration.md) and [ML pipeline documentation](ml/README.md).
