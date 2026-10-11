[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$ManifestPath,
    [Parameter(Mandatory)][string]$ArchivePath,
    [string]$LocalTrainingEvidence = '',
    [string]$TrainingJobArn = '',
    [string]$Profile = 'yieldsense-dev',
    [string]$Region = 'ap-south-1',
    [string]$StackPrefix = 'yieldsense-dev'
)
$ErrorActionPreference = 'Stop'
if ($LocalTrainingEvidence -and $TrainingJobArn) { throw 'Choose local training evidence or a real SageMaker job ARN.' }
if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) { throw 'The ML developer must provide the trained model archive first.' }
if ((Get-Item -LiteralPath $ArchivePath).Length -gt 128MB) { throw 'Model archive exceeds the worker limit of 128 MiB.' }
$manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
if ($manifest.model_version -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$') { throw 'Invalid immutable model version.' }
$identityJSON = & aws sts get-caller-identity --profile $Profile --region $Region --output json --no-cli-pager
if ($LASTEXITCODE -ne 0) { throw 'AWS authentication failed.' }
$identity = $identityJSON | ConvertFrom-Json
if ($LocalTrainingEvidence) {
    $manifest | Add-Member -NotePropertyName training_source -NotePropertyValue 'local' -Force
    $manifest | Add-Member -NotePropertyName training_evidence -NotePropertyValue $LocalTrainingEvidence -Force
    $manifest.PSObject.Properties.Remove('training_job_arn')
} else {
    if ($TrainingJobArn) { $manifest | Add-Member -NotePropertyName training_job_arn -NotePropertyValue $TrainingJobArn -Force }
    $jobProperty = $manifest.PSObject.Properties['training_job_arn']
    if (-not $jobProperty -or $jobProperty.Value -notmatch "^arn:aws[a-z-]*:sagemaker:[a-z0-9-]+:$($identity.Account):training-job/[a-zA-Z0-9-]+$") {
        throw 'Provide a real SageMaker training job ARN, or use -LocalTrainingEvidence for a locally trained model. Placeholder ARNs are not accepted.'
    }
    $manifest | Add-Member -NotePropertyName training_source -NotePropertyValue 'sagemaker' -Force
}
function Read-Outputs([string]$Name) {
    $json = & aws cloudformation describe-stacks --stack-name $Name --profile $Profile --region $Region --output json --no-cli-pager
    if ($LASTEXITCODE -ne 0) { throw "Unable to read $Name" }
    $outputs = @{}
    foreach ($output in ($json | ConvertFrom-Json).Stacks[0].Outputs) { $outputs[$output.OutputKey] = $output.OutputValue }
    return $outputs
}
$storage = Read-Outputs "$StackPrefix-storage"
$service = Read-Outputs "$StackPrefix-service"
$archiveHash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($manifest.artifact_sha256 -ne $archiveHash) { throw 'Archive hash differs from the supplied manifest. Ask the ML developer to package matching files.' }
$uri = "s3://$($storage.S3Bucket)/models/$($manifest.model_version)/model.tar.gz"
$manifest.artifact_s3_uri = $uri
$endpoint = [string]$service.Endpoint
if (-not $endpoint.StartsWith('https://')) { $endpoint = "https://$endpoint" }
$key = & aws secretsmanager get-secret-value --secret-id $storage.APIKeySecretArn --profile $Profile --region $Region --query SecretString --output text --no-cli-pager
if ($LASTEXITCODE -ne 0) { throw 'Unable to retrieve backend credential.' }
try {
    $headers = @{ Authorization = "Bearer $($key.Trim())" }
    # Check registration before uploading, so an immutable version cannot be overwritten.
    $models = @()
    $cursor = ''
    do {
        $page = Invoke-RestMethod "$endpoint/v1/models?limit=100&cursor=$([Uri]::EscapeDataString($cursor))" -Headers $headers
        $models += @($page.items)
        $cursor = [string]$page.next_cursor
    } while ($cursor)
    if ($models | Where-Object model_version -eq $manifest.model_version) { throw 'Model version already registered. Use a new version for changed artifacts.' }
    & aws s3 cp $ArchivePath $uri --profile $Profile --region $Region --no-cli-pager --only-show-errors
    if ($LASTEXITCODE -ne 0) { throw 'Model upload failed.' }
    $registered = Invoke-RestMethod "$endpoint/v1/models" -Method Post -Headers $headers -ContentType 'application/json' -Body ($manifest | ConvertTo-Json -Depth 12)
    Write-Host "Registered Track $($registered.track) model: $($registered.model_version)"
} finally { $key = $null; $headers = $null }
