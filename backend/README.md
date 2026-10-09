# YieldSense Go backend

Go 1.26.1, Echo v5, AWS SDK v2. Runtime persistence uses **only DynamoDB and S3**. There is no Python process, SQL database, Redis, SageMaker endpoint, or external solver dependency.

## Setup

Provision the two storage services using `infra/storage.yaml` from the repository root:

```powershell
aws cloudformation deploy --template-file infra/storage.yaml --stack-name yieldsense-storage --region ap-south-1
aws cloudformation describe-stacks --stack-name yieldsense-storage --region ap-south-1 --query 'Stacks[0].Outputs'
```

The template creates a private, encrypted, versioned S3 bucket and a pay-per-request DynamoDB table with `pk` (string partition key) and `sk` (string sort key), with point-in-time recovery. Resources are retained when the stack is deleted.

Use AWS's default credential chain (local AWS profile or deployment role). The API needs `dynamodb:GetItem`, `dynamodb:PutItem`, and `dynamodb:Query` on the table, and `s3:PutObject` and `s3:GetObject` on the bucket's `raw/*` and `reports/*` objects. It does not create resources on startup.

```powershell
cd backend
$env:AWS_REGION = 'ap-south-1'
$env:DYNAMODB_TABLE = 'yieldsense'
$env:S3_BUCKET = 'bucket-name-from-stack-output'
$env:API_KEY = 'your-random-secret-with-at-least-32-characters'
$env:CORS_ORIGINS = 'http://localhost:3000'
go run ./cmd/server
```

`.env.example` lists configuration; the program reads environment variables directly and does not automatically load `.env`. Optional `DYNAMODB_ENDPOINT` and `S3_ENDPOINT` support compatible local emulators. A local emulator must also have the bucket and table created before requests.

All `/v1` endpoints require `Authorization: Bearer <API_KEY>`. `/health` is public and indicates process health, not AWS connectivity. Use HTTPS at your hosting ingress. The shared API key grants operator/reviewer access; use a trusted client or server proxy to keep the secret out of a public browser bundle. Role-separated authentication and multitenancy are outside this MVP.

## API

JSON endpoints require `Content-Type: application/json`. Lists return `{items, next_cursor}`; pass `?limit=25&cursor=...` to paginate (maximum 100). Updates require `If-Match: <current version>` and return 409 on concurrent modification.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Process health |
| POST / GET | `/v1/batches` | Create/list weighed batches |
| GET / PUT | `/v1/batches/:id` | Read/replace batch, including human review |
| POST | `/v1/composition` | Attach `{batch_id, composition}`, using `If-Match` |
| POST | `/v1/batches/:id/images` | Raw JPEG/PNG body; 10 MiB, 16 megapixels maximum |
| POST / GET | `/v1/routes` | Create/list processing routes |
| GET / PUT | `/v1/routes/:id` | Read/replace route |
| POST | `/v1/optimize` | Load stored records and create audited plan |
| POST | `/v1/simulate` | Derive a run from an immutable base snapshot |
| GET | `/v1/runs` | Paginated run metadata |
| GET | `/v1/runs/:id`, `/v1/reports/:id` | Full JSON audit report |

Batch creation always sets battery state `unknown` and review status `pending`. A subsequent versioned update records human inspection evidence. Sources for mass are `scale` or `illustrative`. Composition sources are `measured`, `category_prior`, `operator_entered`, or `model` (the latter also requires `model_version`). Every composition requires evidence and 1-20 scenarios with **all five** material fractions `FEM`, `NFM`, `PCB`, `PLA`, `OTHER`, nonnegative and summing to one.

Run the complete example against a running server:

```powershell
../scripts/demo-backend.ps1
```

Optimize request:

```json
{
  "batch_ids": ["stored-batch-id"],
  "route_ids": ["stored-route-id"],
  "objective": "ROBUST",
  "assumptions": ["Synthetic INR/kg prices; excludes overhead and storage"]
}
```

Simulation request (zero capacity represents a shutdown):

```json
{
  "base_run_id": "stored-run-id",
  "overrides": [{"route_id": "stored-route-id", "capacity_kg": 0}],
  "assumptions": ["Processor unavailable"]
}
```

Simulation supports capacity and full material price overrides only; it cannot override authorization or review rules. It returns a new run with `parent_id`; the base report and catalog remain unchanged. All batches must have the same number of aligned **joint** scenarios. Scenario index 0 in every batch is one shared possible outcome; index 1 is another, and so on. Supplied scenarios and assumptions are preserved in the snapshot.

## Optimization and safety

Each whole batch is assigned to exactly one authorized route accepting its category and battery status, within incoming mass capacity. Unknown/suspected batteries and pending reviews are restricted to explicitly approved hold routes. Cleared/approved states require human evidence. Missing composition blocks optimization. Image upload never clears batteries or estimates composition.

Modeled contribution is `mass_kg * (sum(fraction * recovery * price) - cost_per_kg) - transport_per_lot`, in INR. Recovery must be between zero and one. Economics and route authorization require provenance. Include residual treatment, energy, overhead and storage costs in supplied assumptions/costs when known.

The Go branch-and-bound solver maximizes `EXPECTED` (mean joint-scenario total) or `ROBUST` (minimum joint-scenario total). `GREEDY` ranks routes by mean value and processes the most constrained/largest batches first; `FIXED` requires `fixed_route_id`. All policies use identical eligibility and capacities. Baselines are separate runs on the same records; to preserve exactly identical inputs across later catalog edits, derive runs through `/v1/simulate` with an objective change and a no-change capacity override.

Search stops after three seconds or two million nodes, with at most two concurrent optimizations. `OPTIMAL` means search completed with a proven best assignment; `INFEASIBLE` means no eligible assignment exists. `TIME_LIMIT` may include a feasible incumbent and never implies proven optimality/infeasibility. `FEASIBLE` describes a completed baseline. `NO_SOLUTION_FOUND` means a baseline got stuck and does not prove global infeasibility. The implementation is not an OR-Tools MILP and makes no 30-batch latency guarantee.

Run reports include the complete inputs, their SHA-256, record versions, solver version/status, assumptions, units, assignments and evaluated scenario totals. Worst-case values cover only the provided scenarios. Each run is an independent plan and does not reserve shared capacity. Reads of several entities are not a transactional catalog snapshot; the report preserves the exact individually versioned records used.

## Persistence and validation

DynamoDB uses collection partitions `batches`, `routes`, `images`, and `runs`, with server-generated IDs as sort keys. Records hold JSON payloads plus a numeric version. Conditional writes prevent lost updates; collection queries paginate without table scans. Stored records are limited to 300 KiB. The collection-partition layout is suitable for the small single-facility MVP; scale-out would require partition sharding or tenant partitioning.

S3 holds validated images at `raw/<batch_id>/<image_id>.*` and immutable run snapshots at `reports/<run_id>.json`. Objects are written before metadata; if the DynamoDB write fails, an unreferenced S3 object can remain. S3 and DynamoDB have no shared transaction. Metadata never points to an object whose write failed. API retries can create duplicate runs/intake records; request idempotency is not implemented.

```powershell
go test ./...
go vet ./...
go build ./cmd/server
docker build -t yieldsense-backend .
```

Tests exercise robust/expected divergence, joint scenarios, safety/capacity, deadline handling, and 100 seeded comparisons with an independent exhaustive optimizer. HTTP tests cover authentication, intake defaults, version conflicts, body/image validation, immutable simulation and failed report writes. Test doubles exist only in tests. A live AWS/emulator check is still needed for deployment credentials, permissions and networking.

Official references: [Echo](https://echo.labstack.com/guide/quickstart/), [AWS SDK for Go v2](https://docs.aws.amazon.com/sdk-for-go/v2/developer-guide/welcome.html).
