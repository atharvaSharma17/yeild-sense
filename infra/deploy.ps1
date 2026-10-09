[CmdletBinding()]
param(
    [ValidatePattern('^[a-zA-Z][a-zA-Z0-9-]{0,63}$')]
    [string]$StackPrefix = 'yieldsense',
    [string]$Region = 'ap-south-1',
    [string]$Profile = '',
    [ValidatePattern('^[a-zA-Z0-9_.-]{3,255}$')]
    [string]$TableName = 'yieldsense',
    [string]$CorsOrigins = '',
    [ValidateRange(1,10)]
    [int]$MaxTasks = 2,
    [switch]$UseTemporaryDockerConfig,
    [ValidatePattern('^$|^sha256:[a-f0-9]{64}$')]
    [string]$MLWorkerDigest = '',
    [string]$MLInferenceURL = '',
    [ValidatePattern('^$|^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$')]
    [string]$DefaultModelVersion = ''
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($MLWorkerDigest -and $MLInferenceURL) { throw 'Choose MLWorkerDigest or MLInferenceURL, not both.' }
foreach ($tool in @('aws', 'docker')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "Install $tool before deploying." }
}
$awsOptions = @('--region', $Region, '--no-cli-pager')
if ($Profile) { $awsOptions += @('--profile', $Profile) }
function Invoke-Aws {
    param([string[]]$Arguments)
    & aws @Arguments @awsOptions
    if ($LASTEXITCODE -ne 0) { throw "AWS command failed: $($Arguments[0]) $($Arguments[1])" }
}
function Invoke-Docker {
    param([string[]]$Arguments)
    & docker @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed: $($Arguments[0])" }
}
$foundation = "$StackPrefix-storage"
$service = "$StackPrefix-service"
$storageTemplate = Join-Path $PSScriptRoot 'storage.yaml'
$serviceTemplate = Join-Path $PSScriptRoot 'service.yaml'
$backend = Join-Path (Split-Path $PSScriptRoot -Parent) 'backend'

# Check identity and local Docker before provisioning any resources.
Invoke-Aws -Arguments @('sts', 'get-caller-identity') | Out-Host
Invoke-Docker -Arguments @('info', '--format', '{{.ServerVersion}}') | Out-Host
foreach ($template in @($storageTemplate, $serviceTemplate)) {
    Invoke-Aws -Arguments @('cloudformation', 'validate-template', '--template-body', "file://$template") | Out-Null
}
Write-Host 'Deploying storage, ECR and the generated API credential...'
Invoke-Aws -Arguments @('cloudformation', 'deploy', '--template-file', $storageTemplate,
    '--stack-name', $foundation, '--parameter-overrides', "TableName=$TableName",
    '--no-fail-on-empty-changeset') | Out-Host

$stack = (Invoke-Aws -Arguments @('cloudformation', 'describe-stacks', '--stack-name', $foundation, '--output', 'json')) | ConvertFrom-Json
$outputs = @{}
foreach ($output in $stack.Stacks[0].Outputs) { $outputs[$output.OutputKey] = $output.OutputValue }
$repository = [string]$outputs['RepositoryUri']
$registry = $repository.Split('/')[0]
$tag = 'deploy-' + [DateTime]::UtcNow.ToString('yyyyMMddHHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0,8)
$image = "${repository}:$tag"

Write-Host 'Building and pushing the linux/amd64 backend image...'
if ($UseTemporaryDockerConfig) {
    # Keep builds on the normal Docker context. Only the ECR push uses temporary
    # file credentials, bypassing a broken Windows credential helper.
    $dockerEndpoint = (Invoke-Docker -Arguments @('context', 'inspect', '--format', '{{.Endpoints.docker.Host}}')).Trim()
    if ($env:DOCKER_HOST -and -not $env:DOCKER_CONTEXT) { $dockerEndpoint = $env:DOCKER_HOST }
    if ($dockerEndpoint -notmatch '^(npipe|unix)://') {
        throw 'Temporary Docker authentication requires a local Docker Desktop/Engine socket.'
    }
    Invoke-Docker -Arguments @('build', '--platform', 'linux/amd64', '-t', $image, $backend) | Out-Host
    $temporaryDockerConfig = Join-Path ([IO.Path]::GetTempPath()) ('yieldsense-ecr-' + [Guid]::NewGuid().ToString('N'))
    $credentialFile = Join-Path $temporaryDockerConfig 'config.json'
    $previousDockerContext = $env:DOCKER_CONTEXT
    try {
        New-Item -ItemType Directory -Path $temporaryDockerConfig | Out-Null
        $password = (Invoke-Aws -Arguments @('ecr', 'get-login-password')).Trim()
        $auth = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes('AWS:' + $password))
        $configuration = @{ auths = @{ $registry = @{ auth = $auth } } } | ConvertTo-Json -Depth 4
        [IO.File]::WriteAllText($credentialFile, $configuration, [Text.UTF8Encoding]::new($false))
        $password = $null
        $auth = $null
        $configuration = $null
        # An inherited context would override --host and refer to context files
        # absent from the temporary directory.
        $env:DOCKER_CONTEXT = $null
        Invoke-Docker -Arguments @('--config', $temporaryDockerConfig, '--host', $dockerEndpoint, 'push', $image) | Out-Host
    } finally {
        $env:DOCKER_CONTEXT = $previousDockerContext
        $password = $null
        $auth = $null
        $configuration = $null
        if (Test-Path -LiteralPath $credentialFile) { Remove-Item -LiteralPath $credentialFile -Force }
        if (Test-Path -LiteralPath $temporaryDockerConfig) { Remove-Item -LiteralPath $temporaryDockerConfig }
    }
} else {
    # Pass the registry token only through stdin, never as a Docker argument.
    $password = Invoke-Aws -Arguments @('ecr', 'get-login-password')
    try {
        $password | & docker login --username AWS --password-stdin $registry
        if ($LASTEXITCODE -ne 0) { throw 'ECR login failed. If the Docker credential helper failed, rerun with -UseTemporaryDockerConfig.' }
    } finally { $password = $null }
    Invoke-Docker -Arguments @('build', '--platform', 'linux/amd64', '-t', $image, $backend) | Out-Host
    Invoke-Docker -Arguments @('push', $image) | Out-Host
}
$digest = (Invoke-Aws -Arguments @('ecr', 'describe-images', '--repository-name', [string]$outputs['RepositoryName'],
    '--image-ids', "imageTag=$tag", '--query', 'imageDetails[0].imageDigest', '--output', 'text')).Trim()
if ($digest -notmatch '^sha256:[a-f0-9]{64}$') { throw 'Could not resolve the uploaded image digest.' }

Write-Host 'Deploying the Go API with managed HTTPS ingress...'
Invoke-Aws -Arguments @('cloudformation', 'deploy', '--template-file', $serviceTemplate,
    '--stack-name', $service, '--capabilities', 'CAPABILITY_IAM', '--parameter-overrides',
    "FoundationStack=$foundation", "ImageDigest=$digest", "CorsOrigins=$CorsOrigins", "MaxTasks=$MaxTasks",
    "MLWorkerDigest=$MLWorkerDigest", "MLInferenceURL=$MLInferenceURL", "DefaultModelVersion=$DefaultModelVersion",
    '--no-fail-on-empty-changeset') | Out-Host
Invoke-Aws -Arguments @('cloudformation', 'describe-stacks', '--stack-name', $service,
    '--query', 'Stacks[0].Outputs', '--output', 'table') | Out-Host
Write-Host "API key secret ARN: $($outputs['APIKeySecretArn'])"
Write-Host 'The API key value is not printed. See infra/README.md to use it.'
