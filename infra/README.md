# Deploy YieldSense

CloudFormation provisions the Go backend and its storage. Application records remain in **DynamoDB**, and images/reports remain in **S3**. ECR holds the container image; Secrets Manager holds only the API credential.

| File | Creates or does |
|---|---|
| `storage.yaml` | Private encrypted/versioned S3 bucket, DynamoDB table with `pk`/`sk`, immutable-tag ECR repository, generated 48-character API key |
| `service.yaml` | VPC, two public subnets, internet routing, S3/DynamoDB gateway endpoints, ECS cluster, IAM roles, log group, Fargate Express Mode API with HTTPS and autoscaling |
| `deploy.ps1` | Validates templates, creates foundation, builds/pushes image, deploys service pinned to the uploaded digest |
| `validate.ps1` | Runs CloudFormation schema lint locally through Docker; requires no AWS credentials |

ECS Express Mode manages the HTTPS load balancer, certificate, service security groups and scaling infrastructure. The application health check is `/health` on port 8080. The service defaults to one 1-vCPU/2-GiB task and scales up to two tasks. Task ingress is managed by Express Mode; no direct public port-8080 ingress rule is declared. Public task IPs provide outbound access to ECR, Secrets Manager and CloudWatch without a NAT gateway. S3/DynamoDB use gateway endpoints. The backend does not need code changes or static AWS credentials.

## Deploy

Requirements: AWS CLI v2 with ECS Express Mode support, Docker running with Linux containers, PowerShell, and an authenticated AWS account. The default region is Mumbai (`ap-south-1`). Use a region with at least two available Availability Zones and ECS Fargate Express Mode support.

The deploy identity needs CloudFormation, EC2 networking, ECR upload, S3, DynamoDB, Secrets Manager, ECS, CloudWatch Logs and IAM provisioning permissions. It must be able to pass the execution, task and infrastructure roles to ECS. The application task role itself is limited to `GetItem`, `PutItem`, `Query` on its table and `GetObject`/`PutObject` under its bucket's `raw/` and `reports/` prefixes. The separate execution role pulls the image, writes logs and reads the API key. The infrastructure role uses AWS's managed Express Mode provisioning policy.

From the repository root:

```powershell
# Optional local validation (downloads the linter dependencies).
./infra/validate.ps1

# Creates billable AWS infrastructure and uploads the backend image.
./infra/deploy.ps1 -Region ap-south-1 -CorsOrigins 'https://your-frontend.example.com'

# Optional named AWS CLI profile and separate environment:
./infra/deploy.ps1 -Profile dev -StackPrefix yieldsense-dev -TableName yieldsense-dev -Region ap-south-1
```

Each invocation uses a unique image tag, then deploys the immutable ECR digest. The script stops when an AWS/Docker command fails and waits for each CloudFormation deployment to finish. Rerun with the same stack prefix/table to deploy a new backend revision. The generated API key is preserved on normal stack updates and is never printed by the script. If a deployment fails, created resources can remain; inspect CloudFormation stack events before retrying. Use unique table names for separate environments.

Two stacks avoid the first-deployment dependency between creating the ECR repository and uploading the image. Defaults are `yieldsense-storage` and `yieldsense-service`. The service imports foundation outputs; both stacks must use the same AWS account and region. If you already deployed the earlier `storage.yaml`, the same foundation stack can be updated to add ECR and Secrets Manager without replacing the existing bucket/table, provided `TableName` is unchanged.

## Call the deployed API

To deploy CPU inference alongside Go, push the ML worker image to the foundation ECR repository, then pass `-MLWorkerDigest <sha256:digest> -DefaultModelVersion <version>` to `deploy.ps1`. Alternatively pass `-MLInferenceURL <url>` for a separately hosted worker. The service now uses an explicit Fargate task definition with the required `Main` application container and optional inference sidecar. Models are registered through `/v1/models`; see [the complete ML handoff](../docs/ml-integration.md).

The worker receives the generated secret as `ML_INFERENCE_TOKEN`, and the task role has read access to `models/` and `model-artifacts/`. This does not grant model upload permissions or create a training job. The inference worker's loopback port is not exposed publicly; a remote worker must authenticate the same generated secret.

Use the profile/region used during deployment. Load the generated key into your current shell without printing it:

```powershell
$region = 'ap-south-1'
$secretArn = aws cloudformation describe-stacks --stack-name yieldsense-storage --region $region --query "Stacks[0].Outputs[?OutputKey=='APIKeySecretArn'].OutputValue | [0]" --output text
$env:API_KEY = aws secretsmanager get-secret-value --secret-id $secretArn --region $region --query SecretString --output text
$endpoint = aws cloudformation describe-stacks --stack-name yieldsense-service --region $region --query "Stacks[0].Outputs[?OutputKey=='Endpoint'].OutputValue | [0]" --output text
if (-not $endpoint.StartsWith('https://')) { $endpoint = "https://$endpoint" }
Invoke-RestMethod "$endpoint/health"
./scripts/demo-backend.ps1 -BaseUrl $endpoint
```

The caller retrieving the key needs `secretsmanager:GetSecretValue`; the public frontend should call through a trusted server proxy, keeping the shared key out of browser bundles. Secret rotation requires fresh ECS tasks to load the new value; use an ECS force deployment after changing the secret.

Logs go to the service stack's `LogGroup` output and expire after 14 days. `Endpoint` is the managed HTTPS URL; `ServiceArn` identifies the deployed service. CORS defaults to disabled; configure only the actual frontend origins, comma-separated.

## Remove compute

Delete the service stack first; it imports the storage stack's exports:

```powershell
aws cloudformation delete-stack --stack-name yieldsense-service --region ap-south-1
aws cloudformation wait stack-delete-complete --stack-name yieldsense-service --region ap-south-1
# Optional: remove the foundation stack's management after compute is gone.
aws cloudformation delete-stack --stack-name yieldsense-storage --region ap-south-1
```

S3, DynamoDB, ECR and the API secret have `Retain` policies: stack deletion does **not** remove their contents and these retained resources can continue to incur charges. Reusing retained resources in a fresh stack requires importing them; do not assume rerunning after deletion will adopt them automatically. There is no automatic image/report expiration, preserving audit history and rollback images.

The deployment creates ongoing Fargate, load balancer, public IPv4, logs, storage, registry and secret costs. Live AWS deployment is not part of local lint verification.

References: [ECS Express Mode](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-overview.html), [CloudFormation service schema](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-ecs-expressgatewayservice.html), [Express Mode networking](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/express-service-work.html).
