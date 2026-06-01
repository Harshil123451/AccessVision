# run_load_test.ps1
# PowerShell script to execute Locust load testing scenarios on the AccessVision API.

param (
    [string]$TargetUrl = "http://127.0.0.1:8000",
    [string]$Scenario = "sanity", # sanity, load, stress
    [string]$Duration = "1m",
    [bool]$Headless = $true
)

# Set Locust parameters based on selected scenario
switch ($Scenario) {
    "sanity" {
        $Users = 5
        $SpawnRate = 1
    }
    "load" {
        $Users = 50
        $SpawnRate = 5
    }
    "stress" {
        $Users = 200
        $SpawnRate = 10
    }
    default {
        $Users = 5
        $SpawnRate = 1
    }
}

Write-Host "=============================================" -ForegroundColor Green
Write-Host "AccessVision Locust Load Test Execution" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host "Target Host:  $TargetUrl"
Write-Host "Scenario:     $Scenario (Users: $Users, Spawn Rate: $SpawnRate/sec)"
Write-Host "Duration:     $Duration"
Write-Host "Headless Mode:$Headless"
Write-Host "============================================="

# Ensure Locust CSV output directory exists
$CsvDir = Join-Path $PSScriptRoot "reports"
if (!(Test-Path $CsvDir)) {
    New-Item -ItemType Directory -Path $CsvDir -Force | Out-Null
}

$CsvPrefix = Join-Path $CsvDir "report_${Scenario}"

# Construct command arguments
$Args = @(
    "-f", (Join-Path $PSScriptRoot "locustfile.py"),
    "--host", $TargetUrl
)

if ($Headless) {
    $Args += @(
        "--headless",
        "-u", $Users,
        "-r", $SpawnRate,
        "-t", $Duration,
        "--csv", $CsvPrefix
    )
}

# Execute Locust
Write-Host "Launching Locust process..." -ForegroundColor Yellow
locust @Args
Write-Host "Load test finished. Reports saved to: $CsvDir" -ForegroundColor Green
