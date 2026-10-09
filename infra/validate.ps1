[CmdletBinding()]
param([string]$Region = 'ap-south-1')
$ErrorActionPreference = 'Stop'
& docker build -f (Join-Path $PSScriptRoot 'lint.Dockerfile') -t yieldsense-infra-lint $PSScriptRoot
if ($LASTEXITCODE -ne 0) { throw 'Failed to build the CloudFormation linter.' }
& docker run --rm --network none --mount "type=bind,source=$PSScriptRoot,target=/templates,readonly" yieldsense-infra-lint -r $Region -t /templates/storage.yaml /templates/service.yaml
if ($LASTEXITCODE -ne 0) { throw 'CloudFormation lint failed.' }
Write-Host 'CloudFormation templates passed lint.'
