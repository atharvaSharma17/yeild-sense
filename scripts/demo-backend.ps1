param([string]$BaseUrl = 'http://localhost:8080')
$ErrorActionPreference = 'Stop'
if (-not $env:API_KEY) { throw 'Set API_KEY to the running server API key first.' }
$headers = @{ Authorization = "Bearer $env:API_KEY" }
$fractions = @{ FEM = 0.4; NFM = 0.2; PCB = 0.1; PLA = 0.2; OTHER = 0.1 }
$body = @{
  category = 'small_devices'; mass_kg = 100; mass_source = 'scale'
  composition = @{ source = 'operator_entered'; evidence = 'Synthetic demo fractions, not an assay'; scenarios = @($fractions) }
}
$batch = Invoke-RestMethod -Uri "$BaseUrl/v1/batches" -Method Post -Headers $headers -ContentType 'application/json' -Body ($body | ConvertTo-Json -Depth 10)
$batch.battery_status = 'cleared'
$batch.review_status = 'approved'
$batch.review_evidence = 'Synthetic demo inspection record; replace with actual human evidence'
$reviewHeaders = @{ Authorization = "Bearer $env:API_KEY"; 'If-Match' = [string]$batch.version }
$batch = Invoke-RestMethod -Uri "$BaseUrl/v1/batches/$($batch.id)" -Method Put -Headers $reviewHeaders -ContentType 'application/json' -Body ($batch | ConvertTo-Json -Depth 10)
$routeBody = @{
  name = 'Demo mechanical processing'; capacity_kg = 500; cost_per_kg = 8; transport_per_lot = 100
  recovery = @{ FEM = 0.9; NFM = 0.8; PCB = 0.5; PLA = 0.4; OTHER = 0 }
  price = @{ FEM = 30; NFM = 100; PCB = 200; PLA = 10; OTHER = 0 }
  accepted_categories = @('small_devices'); allowed_battery_states = @('cleared')
  authorized = $true; authorization_evidence = 'Synthetic demo route approval'; approved_hold = $false
  economics_source = 'Synthetic INR/kg assumptions; excludes overhead and storage'
}
$route = Invoke-RestMethod -Uri "$BaseUrl/v1/routes" -Method Post -Headers $headers -ContentType 'application/json' -Body ($routeBody | ConvertTo-Json -Depth 10)
$planBody = @{ batch_ids = @($batch.id); route_ids = @($route.id); objective = 'ROBUST'; assumptions = @('Synthetic demo; not measured savings; excludes overhead and storage') }
$run = Invoke-RestMethod -Uri "$BaseUrl/v1/optimize" -Method Post -Headers $headers -ContentType 'application/json' -Body ($planBody | ConvertTo-Json -Depth 10)
$run | ConvertTo-Json -Depth 20
$simulation = @{ base_run_id = $run.id; overrides = @(@{ route_id = $route.id; capacity_kg = 0 }); assumptions = @('Demo processor shutdown') }
Invoke-RestMethod -Uri "$BaseUrl/v1/simulate" -Method Post -Headers $headers -ContentType 'application/json' -Body ($simulation | ConvertTo-Json -Depth 10) | ConvertTo-Json -Depth 20
