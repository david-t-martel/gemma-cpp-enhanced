# Orchestrate PR creation for changed repos under C:\codedev\llm
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-GhProtocol {
  try { (gh config get git_protocol).Trim() } catch { 'ssh' }
}

$GitHubUser = 'david-t-martel'
$Protocol   = Get-GhProtocol

function Get-ForkUrl {
  param([Parameter(Mandatory)][string]$RepoName)
  if ($Protocol -eq 'ssh') { "git@github.com:$GitHubUser/$RepoName.git" } else { "https://github.com/$GitHubUser/$RepoName.git" }
}

function Ensure-ForkAndRemote {
  param([Parameter(Mandatory)][string]$RepoPath)
  Push-Location $RepoPath
  try {
    $originUrl = (git remote get-url origin).Trim()
    $clean = $originUrl -replace '^(git@github.com:|https://github.com/)', '' -replace '\.git$', ''
    $parts = $clean.Split('/')
    if ($parts.Count -lt 2) { return }
    $owner = $parts[0]; $name = $parts[1]

    if ($owner -ne $GitHubUser) {
      # Ensure fork exists
      $forkExists = $false
      try { gh api "repos/$GitHubUser/$name" 1>$null 2>$null; $forkExists = $true } catch { $forkExists = $false }
      if (-not $forkExists) {
        Write-Host "Creating fork of $owner/$name..."
        gh repo fork "$owner/$name" --clone=false | Out-Null
      }

      # Keep original as upstream and set origin to the fork
      if (-not (git remote | Select-String -SimpleMatch 'upstream')) {
        Write-Host "Renaming origin to upstream and adding fork as origin..."
        git remote rename origin upstream | Out-Null
        git remote add origin (Get-ForkUrl -RepoName $name) | Out-Null
      } else {
        git remote set-url origin (Get-ForkUrl -RepoName $name) | Out-Null
      }
      git fetch --all --prune | Out-Null
    }
  } finally {
    Pop-Location
  }
}

function Get-DefaultBranch {
  param([Parameter(Mandatory)][string]$RepoPath)
  Push-Location $RepoPath
  try {
    $def = gh repo view --json defaultBranchRef -q .defaultBranchRef.name 2>$null
    if ([string]::IsNullOrWhiteSpace($def)) { 'main' } else { $def.Trim() }
  } finally {
    Pop-Location
  }
}

function New-FeatureBranchName {
  param([Parameter(Mandatory)][string]$RepoSlug)
  $date = Get-Date -Format 'yyyyMMdd-HHmmss'
  "feature/code-review-$date-$RepoSlug"
}

function Get-RepoSlug {
  param([Parameter(Mandatory)][string]$RepoPath)
  (Split-Path $RepoPath -Leaf).ToLowerInvariant()
}

function CommitAndCreatePR {
  param([Parameter(Mandatory)][string]$RepoPath)
  $res = [ordered]@{ Path=$RepoPath; Branch=''; PR=''; Status=''; Error='' }
  Push-Location $RepoPath
  try {
    Write-Host "`n=== Processing: $RepoPath ==="
    
    # Ensure fork & remote (specifically needed for sentencepiece; safe for others)
    Ensure-ForkAndRemote -RepoPath $RepoPath

    # Check if there are any working changes; skip if clean
    $porcelain = git status --porcelain
    if ([string]::IsNullOrWhiteSpace($porcelain)) {
      $res.Status = 'Clean; no changes to commit'
      Write-Host "  Status: Clean, skipping"
      return $res
    }

    $baseBranch = Get-DefaultBranch -RepoPath $RepoPath
    $slug = Get-RepoSlug -RepoPath $RepoPath
    $feature = New-FeatureBranchName -RepoSlug $slug

    # Create feature branch
    Write-Host "  Creating branch: $feature"
    git switch -c $feature 2>$null | Out-Null

    # Stage everything and commit
    Write-Host "  Staging changes..."
    git add -A
    $staged = git diff --cached --name-only
    if ([string]::IsNullOrWhiteSpace($staged)) {
      $res.Status = 'No staged changes after add; skipping'
      Write-Host "  Status: No staged changes, skipping"
      return $res
    }

    $subject = "chore($slug): prepare PR with local changes"
    if ($slug -like 'sentencepiece*') { $subject = "build($slug): update CMake configuration for Windows build" }
    $body    = "Automated commit to stage current local changes for PR"
    
    Write-Host "  Committing: $subject"
    git commit -m $subject -m $body | Out-Null

    # Push feature branch
    Write-Host "  Pushing to origin/$feature..."
    git push -u origin $feature

    # Create PR; mention literal @gemini and @codex in the body
    $commitSubject = (git log -1 --pretty=%s).Trim()
    $changed = (git diff --name-status HEAD~1 HEAD) -join "`n"
    $prTitle = "[Auto PR] $commitSubject"
    $prBody = @"
This PR was created automatically to request review on local changes.

Base: $baseBranch
Head: $feature

Changes included:
$changed

Requesting review: @gemini @codex
"@

    Write-Host "  Creating PR..."
    $prUrl = gh pr create --base $baseBranch --head $feature --title $prTitle --body $prBody --json url -q .url
    $res.Branch = $feature
    $res.PR     = $prUrl
    $res.Status = 'Created'
    Write-Host "  Success: $prUrl"
    return $res
  } catch {
    $res.Error  = $_.Exception.Message
    $res.Status = 'Failed'
    Write-Host "  Failed: $($_.Exception.Message)" -ForegroundColor Red
    return $res
  } finally {
    Pop-Location
  }
}

# Discover repos with changes
Write-Host "Discovering git repositories with changes..."
$root = 'C:\codedev\llm'
$repos = Get-ChildItem -Path $root -Directory -Recurse | Where-Object { Test-Path "$($_.FullName)\.git" }

$targets = @()
foreach ($r in $repos) {
  Push-Location $r.FullName
  try {
    $dirty = git status --porcelain
    if ($dirty) { $targets += $r.FullName }
  } catch { } finally { Pop-Location }
}

# If discovery returns none, fall back to known changed repos from context
if (-not $targets -or $targets.Count -eq 0) {
  Write-Host "No dirty repos found via discovery, using known changed repos..."
  $targets = @(
    'C:\codedev\llm\gemma',
    'C:\codedev\llm\build\_deps\sentencepiece-src'
  ) | Where-Object { Test-Path $_ }
}

Write-Host "Found $($targets.Count) repositories with changes"

# Process each repo and collect results
$results = @()
foreach ($t in $targets) {
  $results += CommitAndCreatePR -RepoPath $t
}

# Summary
Write-Host "`n=== Summary ==="
foreach ($r in $results) {
  if ($r.Status -eq 'Created') {
    Write-Host "OK: $($r.Path) -> $($r.PR)" -ForegroundColor Green
  } elseif ($r.Status -like 'Clean*' -or $r.Status -like 'No staged*') {
    Write-Host "SKIP: $($r.Path) -> $($r.Status)" -ForegroundColor Yellow
  } else {
    Write-Host "FAIL: $($r.Path) -> $($r.Error)" -ForegroundColor Red
  }
}
