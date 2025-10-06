<#!
.SYNOPSIS
  Helper script to prepare a Windows development environment for Gemma.cpp (tests + backends).

.DESCRIPTION
  - Detects and prints locations of CMake, Ninja, Python, oneAPI (SYCL), CUDA, and Vulkan.
  - Optionally appends missing tools to PATH for the current session.
  - Generates a summary block friendly for LLM agents parsing plain text.

.USAGE
  pwsh -ExecutionPolicy Bypass -File scripts/setup_env_windows.ps1 [-AddToPath]

.NOTES
  This does NOT persist environment changes; it only updates the current session when -AddToPath is used.
!#>
param(
    [switch]$AddToPath
)

function Write-Section($title) { Write-Host "`n=== $title ===" }
function Test-Cmd($cmd) { try { Get-Command $cmd -ErrorAction Stop | Select-Object -First 1 } catch { $null } }

Write-Section "Toolchain Detection"
$cmake    = Test-Cmd cmake
$ninja    = Test-Cmd ninja
$python   = Test-Cmd python

Write-Host "CMake : " ($cmake?.Source ?? 'NOT FOUND')
Write-Host "Ninja : " ($ninja?.Source ?? 'NOT FOUND')
Write-Host "Python: " ($python?.Source ?? 'NOT FOUND')

Write-Section "Intel oneAPI (SYCL)"
$oneApiRoots = @(
  'C:/Program Files (x86)/Intel/oneAPI',
  'D:/Intel/oneAPI'
) | Where-Object { Test-Path $_ }
if($oneApiRoots){
  $dpcpp = Get-ChildItem -Recurse -Filter dpcpp.exe -ErrorAction SilentlyContinue -Path $oneApiRoots | Select-Object -First 1
  Write-Host "oneAPI roots: $($oneApiRoots -join ';')"
  if($dpcpp){ Write-Host "Found dpcpp: $($dpcpp.FullName)" }
} else { Write-Host "oneAPI not detected" }

Write-Section "CUDA Toolkit"
$cudaRoot = Get-ChildItem 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA' -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
if($cudaRoot){
  Write-Host "CUDA root: $($cudaRoot.FullName)"
  Write-Host "Include: $($cudaRoot.FullName)/include"
  Write-Host "Lib x64: $($cudaRoot.FullName)/lib/x64"
} else { Write-Host "CUDA not detected" }

Write-Section "Vulkan SDK"
$vulkanRoot = Get-ChildItem 'C:/' -Directory -Filter 'VulkanSDK' -ErrorAction SilentlyContinue | Select-Object -First 1
if($env:VULKAN_SDK){ Write-Host "VULKAN_SDK env: $env:VULKAN_SDK" }
elseif($vulkanRoot){ Write-Host "Candidate Vulkan SDK path root (set VULKAN_SDK to specific version subdir): $($vulkanRoot.FullName)" }
else { Write-Host "Vulkan SDK not detected" }

if($AddToPath){
  if($cmake){ $env:PATH = (Split-Path $cmake.Source) + ";" + $env:PATH }
  if($ninja){ $env:PATH = (Split-Path $ninja.Source) + ";" + $env:PATH }
  Write-Host "Updated PATH for this session." -ForegroundColor Green
}

Write-Section "LLM Summary (machine-parse)"
Write-Host "SUMMARY_BEGIN"
Write-Host "CMAKE_PATH=$($cmake?.Source ?? 'NONE')"
Write-Host "NINJA_PATH=$($ninja?.Source ?? 'NONE')"
Write-Host "PYTHON_PATH=$($python?.Source ?? 'NONE')"
Write-Host "ONEAPI_ROOTS=$($oneApiRoots -join ',')"
Write-Host "CUDA_ROOT=$($cudaRoot?.FullName ?? 'NONE')"
Write-Host "VULKAN_SDK=$($env:VULKAN_SDK ?? 'NONE')"
Write-Host "SUMMARY_END"
