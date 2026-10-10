[CmdletBinding()]
param(
    [string]$StackName = 'yieldsense-dev-service',
    [string]$Profile = 'yieldsense-dev',
    [string]$Region = 'ap-south-1'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$awsOptions = @('--profile', $Profile, '--region', $Region, '--no-cli-pager')
function Invoke-Aws {
    param([string[]]$Arguments)
    & aws @Arguments @awsOptions
    if ($LASTEXITCODE -ne 0) { throw "AWS command failed: $($Arguments[0]) $($Arguments[1])" }
}
function Test-IamRoleExists {
    param([string]$Name)
    # Windows PowerShell converts redirected native stderr to ErrorRecords.
    # Capture the expected NoSuchEntity result without terminating the script.
    $previousErrorPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $result = & aws iam get-role --role-name $Name @awsOptions 2>&1
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorPreference
    }
    if ($exitCode -eq 0) { return $true }
    $message = $result | Out-String
    if ($message -match 'NoSuchEntity') { return $false }
    throw $message
}
$identity = Invoke-Aws -Arguments @('sts', 'get-caller-identity', '--output', 'json') | ConvertFrom-Json
$stack = Invoke-Aws -Arguments @('cloudformation', 'describe-stacks', '--stack-name', $StackName, '--output', 'json') | ConvertFrom-Json
if ($stack.Stacks[0].StackStatus -ne 'ROLLBACK_FAILED') {
    throw 'This recovery script is only for a failed initial deployment in ROLLBACK_FAILED.'
}
$resources = Invoke-Aws -Arguments @('cloudformation', 'list-stack-resources', '--stack-name', $StackName, '--output', 'json') | ConvertFrom-Json
$roleResource = @($resources.StackResourceSummaries | Where-Object LogicalResourceId -eq 'InfrastructureRole')
$clusterResource = @($resources.StackResourceSummaries | Where-Object LogicalResourceId -eq 'Cluster')
if ($roleResource.Count -ne 1 -or $clusterResource.Count -ne 1) { throw 'Could not identify the failed stack role and cluster.' }
$roleName = [string]$roleResource[0].PhysicalResourceId
$clusterName = ([string]$clusterResource[0].PhysicalResourceId).Split('/')[-1]
if (-not $roleName -or -not $clusterName) { throw 'The failed stack has no physical role/cluster identifiers.' }
$partition = ([string]$identity.Arn).Split(':')[1]
$serviceArn = "arn:${partition}:ecs:${Region}:$($identity.Account):service/${clusterName}/${StackName}-api"
$policyArn = "arn:${partition}:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices"
$temporaryDirectory = Join-Path ([IO.Path]::GetTempPath()) ('yieldsense-recovery-' + [Guid]::NewGuid().ToString('N'))
$trustFile = Join-Path $temporaryDirectory 'trust.json'
$policyFile = Join-Path $temporaryDirectory 'policy.json'
try {
    New-Item -ItemType Directory -Path $temporaryDirectory | Out-Null
    $trust = @{ Version = '2012-10-17'; Statement = @(@{
        Effect = 'Allow'; Principal = @{ Service = 'ecs.amazonaws.com' }; Action = 'sts:AssumeRole'
    }) }
    $policy = @{ Version = '2012-10-17'; Statement = @(
        @{ Effect = 'Allow'; Action = 'ec2:DescribeAccountAttributes'; Resource = '*' },
        @{ Effect = 'Allow'; Action = @('ecs:DescribeServices', 'ecs:UpdateService'); Resource = $serviceArn },
        @{ Effect = 'Allow'; Action = 'cloudwatch:DescribeAlarms'; Resource = "arn:${partition}:cloudwatch:${Region}:$($identity.Account):alarm:*" }
    ) }
    [IO.File]::WriteAllText($trustFile, ($trust | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
    [IO.File]::WriteAllText($policyFile, ($policy | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
    if (-not (Test-IamRoleExists -Name $roleName)) {
        Write-Host "Restoring the deleted infrastructure role: $roleName"
        Invoke-Aws -Arguments @('iam', 'create-role', '--role-name', $roleName, '--assume-role-policy-document', "file://$trustFile") | Out-Null
    }
    Invoke-Aws -Arguments @('iam', 'attach-role-policy', '--role-name', $roleName, '--policy-arn', $policyArn) | Out-Null
    Invoke-Aws -Arguments @('iam', 'put-role-policy', '--role-name', $roleName, '--policy-name', 'ExpressDeploymentOperations', '--policy-document', "file://$policyFile") | Out-Null
    foreach ($linkedRole in @(
        @{ Name = 'AWSServiceRoleForApplicationAutoScaling_ECSService'; Service = 'ecs.application-autoscaling.amazonaws.com' },
        @{ Name = 'AWSServiceRoleForElasticLoadBalancing'; Service = 'elasticloadbalancing.amazonaws.com' }
    )) {
        if (-not (Test-IamRoleExists -Name $linkedRole.Name)) {
            Invoke-Aws -Arguments @('iam', 'create-service-linked-role', '--aws-service-name', $linkedRole.Service) | Out-Null
        }
    }
    Write-Host 'Recovery permissions restored. Allow about one minute for IAM propagation.'
    Write-Host "Service ARN: $serviceArn"
    Write-Host "Restored role: $roleName (remove this recovery role after failed-service cleanup completes)."
    Write-Host 'This script does not delete resources or modify the storage stack.'
} finally {
    foreach ($file in @($trustFile, $policyFile)) {
        if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file -Force }
    }
    if (Test-Path -LiteralPath $temporaryDirectory) { Remove-Item -LiteralPath $temporaryDirectory }
}

