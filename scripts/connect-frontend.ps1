[CmdletBinding()]
param([string]$Profile = 'yieldsense-dev', [string]$Region = 'ap-south-1', [string]$StackPrefix = 'yieldsense-dev')
$ErrorActionPreference = 'Stop'
function Read-StackOutputs([string]$Name) {
    $json = & aws cloudformation describe-stacks --stack-name $Name --profile $Profile --region $Region --output json --no-cli-pager
    if ($LASTEXITCODE -ne 0) { throw "Unable to read stack $Name" }
    $values = @{}
    foreach ($output in ($json | ConvertFrom-Json).Stacks[0].Outputs) { $values[$output.OutputKey] = $output.OutputValue }
    return $values
}
$storage = Read-StackOutputs "$StackPrefix-storage"
$service = Read-StackOutputs "$StackPrefix-service"
$endpoint = [string]$service.Endpoint
if (-not $endpoint.StartsWith('https://')) { $endpoint = "https://$endpoint" }
$key = & aws secretsmanager get-secret-value --secret-id $storage.APIKeySecretArn --profile $Profile --region $Region --query SecretString --output text --no-cli-pager
if ($LASTEXITCODE -ne 0 -or -not $key) { throw 'Unable to retrieve backend credential.' }
try {
    $envPath = Join-Path (Split-Path $PSScriptRoot -Parent) 'frontend/.env.local'
    if (Test-Path -LiteralPath $envPath) { throw 'frontend/.env.local already exists. Update its three YieldSense settings manually to preserve your other configuration.' }
    $contents = "NEXT_PUBLIC_YIELDSENSE_DATA_MODE=api`nYIELDSENSE_API_BASE_URL=$endpoint`nYIELDSENSE_API_KEY=$($key.Trim())`n"
    [IO.File]::WriteAllText($envPath, $contents, [Text.UTF8Encoding]::new($false))
    Write-Host 'Frontend connected to the deployed backend. API key was saved server-side and was not printed.'
    Write-Host 'Restart the frontend development server, or rebuild it before hosting.'
} finally { $key = $null; $contents = $null }
