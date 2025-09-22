# GCP Authentication Setup Script for Windows
param(
    [string]$ServiceAccountPath = "",
    [switch]$UseExistingCredentials = $false,
    [switch]$Quiet = $false,
    [switch]$Help = $false
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$ConfigDir = Join-Path $ProjectRoot "config"
$EnvFile = Join-Path $ProjectRoot ".env"
$SAName = "gcp-profile"
$SAFilePatterns = @("gcp-profile*.json", "gcp-profile*credentials*.json", "*gcp-profile*.json")

function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    if (-not $Quiet) {
        switch ($Color) {
            "Red" { Write-Host $Message -ForegroundColor Red }
            "Green" { Write-Host $Message -ForegroundColor Green }
            "Yellow" { Write-Host $Message -ForegroundColor Yellow }
            "Blue" { Write-Host $Message -ForegroundColor Blue }
            "Cyan" { Write-Host $Message -ForegroundColor Cyan }
            default { Write-Host $Message }
        }
    }
}

function Write-Header {
    param([string]$Title)
    Write-ColorOutput ""
    Write-ColorOutput "================================================" -Color Blue
    Write-ColorOutput "    $Title" -Color Blue
    Write-ColorOutput "================================================" -Color Blue
    Write-ColorOutput ""
}

function Show-Help {
    Write-ColorOutput "GCP Authentication Setup Script" -Color Cyan
    Write-ColorOutput "USAGE: .\setup-gcp-auth.ps1 [OPTIONS]"
    Write-ColorOutput "OPTIONS:"
    Write-ColorOutput "  -ServiceAccountPath <path>  Path to service account JSON file"
    Write-ColorOutput "  -UseExistingCredentials     Use existing Application Default Credentials"
    Write-ColorOutput "  -Quiet                      Suppress non-error output"
    Write-ColorOutput "  -Help                       Show this help message"
}

function Find-GcpProfileServiceAccount {
    $locations = @(
        "$env:USERPROFILE\.config\gcloud",
        "$env:USERPROFILE\.gcp",
        "$env:USERPROFILE\Downloads",
        "$env:USERPROFILE\Documents",
        $ProjectRoot,
        $ConfigDir,
        "$env:APPDATA\gcloud"
    )

    $foundFiles = @()
    foreach ($location in $locations) {
        if (Test-Path $location) {
            foreach ($pattern in $SAFilePatterns) {
                $files = Get-ChildItem -Path $location -Filter $pattern -Recurse -Depth 1 -ErrorAction SilentlyContinue
                $foundFiles += $files.FullName
            }
        }
    }
    return $foundFiles | Where-Object { Test-Path $_ } | Sort-Object | Get-Unique
}

function Test-ServiceAccountFile {
    param([string]$FilePath)
    if (-not (Test-Path $FilePath)) {
        return $false, "File not found"
    }
    try {
        $content = Get-Content $FilePath -Raw | ConvertFrom-Json
        $requiredFields = @("type", "project_id", "private_key_id", "private_key", "client_email", "client_id")
        $missingFields = @()
        foreach ($field in $requiredFields) {
            if (-not $content.PSObject.Properties.Name.Contains($field)) {
                $missingFields += $field
            }
        }
        if ($missingFields.Count -gt 0) {
            return $false, "Missing required fields: $($missingFields -join ', ')"
        }
        if ($content.type -ne "service_account") {
            return $false, "Not a service account key"
        }
        $clientEmail = $content.client_email
        $isGcpProfile = $clientEmail -like "*gcp-profile*" -or $clientEmail -like "*business*"
        return $true, @{
            ProjectId = $content.project_id
            ClientEmail = $clientEmail
            IsGcpProfile = $isGcpProfile
        }
    }
    catch {
        return $false, "Invalid JSON file"
    }
}

function Set-ServiceAccountAuth {
    param([string]$ServiceAccountPath)
    if (-not (Test-Path $ConfigDir)) {
        New-Item -ItemType Directory -Path $ConfigDir -Force | Out-Null
    }
    $configSAFile = Join-Path $ConfigDir "gcp-profile-sa.json"
    try {
        Copy-Item $ServiceAccountPath $configSAFile -Force
        Write-ColorOutput "✓ Service account key copied to: $configSAFile" -Color Green
        $ServiceAccountPath = $configSAFile
    }
    catch {
        Write-ColorOutput "⚠ Could not copy to config directory" -Color Yellow
    }

    $envContent = @()
    if (Test-Path $EnvFile) {
        $envContent = Get-Content $EnvFile | Where-Object { $_ -notmatch "^GOOGLE_APPLICATION_CREDENTIALS=" }
    }
    try {
        $saContent = Get-Content $ServiceAccountPath -Raw | ConvertFrom-Json
        $envContent += "GOOGLE_APPLICATION_CREDENTIALS=`"$ServiceAccountPath`""
        $envContent += "GCP_PROJECT_ID=`"$($saContent.project_id)`""
        $envContent += "GCP_CLIENT_EMAIL=`"$($saContent.client_email)`""
        $envContent += "GCP_AUTH_METHOD=`"service_account`""
        $envContent | Set-Content $EnvFile -Encoding UTF8
        Write-ColorOutput "✓ Environment configured successfully" -Color Green
        return $true
    }
    catch {
        Write-ColorOutput "✗ Error processing service account" -Color Red
        return $false
    }
}

# Main execution
if ($Help) {
    Show-Help
    exit 0
}

try {
    if ($ServiceAccountPath) {
        Write-Header "GCP Authentication Setup"
        $validationResult = Test-ServiceAccountFile $ServiceAccountPath
        if ($validationResult[0]) {
            Set-ServiceAccountAuth $ServiceAccountPath
        }
        else {
            Write-ColorOutput "✗ Invalid service account file: $($validationResult[1])" -Color Red
            exit 1
        }
    }
    else {
        Write-Header "GCP Authentication Setup"
        Write-ColorOutput "Searching for gcp-profile service account files..." -Color Blue
        $foundFiles = Find-GcpProfileServiceAccount
        if ($foundFiles.Count -gt 0) {
            $validFiles = @()
            foreach ($file in $foundFiles) {
                $validationResult = Test-ServiceAccountFile $file
                if ($validationResult[0]) {
                    $validFiles += $file
                    $info = $validationResult[1]
                    Write-ColorOutput "✓ Valid: $file" -Color Green
                }
            }
            if ($validFiles.Count -gt 0) {
                Set-ServiceAccountAuth $validFiles[0]
            }
        }
        else {
            Write-ColorOutput "No gcp-profile service account files found." -Color Yellow
            Write-ColorOutput "Run: .\setup-gcp-auth.ps1 -ServiceAccountPath <path-to-json>" -Color Yellow
        }
    }
}
catch {
    Write-ColorOutput "✗ An error occurred: $($_.Exception.Message)" -Color Red
    exit 1
}
