# Gemma.cpp Benchmark Suite - Quick Start Guide

## 🚀 Immediate Testing (No Setup Required)

### Option 1: PowerShell (Recommended)

Open PowerShell and run:

```powershell
cd C:\codedev\llm\benchmarks

# Quick test (2-3 minutes)
.\Run-QuickBenchmark.ps1

# Full benchmark (10-15 minutes)
.\Benchmark-Gemma.ps1 -FullBenchmark
```

### Option 2: WSL/Bash

Open WSL terminal and run:

```bash
cd /mnt/c/codedev/llm/benchmarks
./test_gemma.sh
```

### Option 3: Double-Click

Simply double-click `RunBenchmark.bat` in Windows Explorer.

## 📊 What Gets Tested

### Performance Metrics
- **Token Generation Speed**: Measures tokens per second
- **Response Latency**: First token and total generation time
- **Memory Usage**: Peak and average memory consumption
- **Batch Processing**: Efficiency with multiple prompts
- **Concurrent Load**: Performance under stress

### Test Configurations

| Test Type | Duration | What It Tests |
|-----------|----------|---------------|
| Quick Test | 2-3 min | Basic functionality and baseline performance |
| Standard | 5-7 min | Core performance metrics |
| Full Suite | 10-15 min | Comprehensive analysis with all models |
| Stress Test | Custom | Performance under load |

## 🔍 Understanding Results

### Good Performance Indicators

**Gemma 2B Model:**
- Token Speed: 40-60 tokens/sec ✅
- Memory: < 2GB ✅
- Latency: < 500ms first token ✅

**Gemma 4B Model:**
- Token Speed: 20-30 tokens/sec ✅
- Memory: < 4GB ✅
- Latency: < 800ms first token ✅

### Performance Warnings

- Token Speed < 10/sec: Check CPU/memory constraints ⚠️
- Memory > 8GB: May cause system slowdown ⚠️
- Latency > 2s: Model loading issues ⚠️

## 📈 Viewing Results

### HTML Report
After running benchmarks:
```powershell
.\Generate-BenchmarkReport.ps1
```
Opens an interactive HTML report with charts.

### Raw Data
Results are saved in `results\` folder:
- `TokenGeneration_*.csv` - Speed metrics
- `MemoryUsage_*.json` - Memory profiles
- `Latency_*.csv` - Response times

## 🎯 Recommended Test Sequence

1. **First Run**: Quick test to verify setup
2. **Baseline**: Standard benchmark for baseline metrics
3. **Optimization**: Adjust based on results
4. **Validation**: Stress test for production readiness

## ⚡ Performance Tips

### For Maximum Speed
```powershell
# Use 2B model with optimized settings
.\Benchmark-Gemma.ps1 -QuickTest
```

### For Quality Output
```powershell
# Use 4B model with higher token count
.\Compare-Models.ps1 -RunTests
```

### For Production Testing
```powershell
# Stress test with realistic load
.\Run-StressTest.ps1 -Model 2b -Duration 300 -Concurrency 8
```

## 🛠️ Troubleshooting

### "WSL not found"
- Install WSL: `wsl --install` in admin PowerShell
- Restart computer after installation

### "Binary not found"
- Build the binary first:
```bash
cd /mnt/c/codedev/llm/gemma/gemma.cpp
cmake --preset make
cmake --build --preset make -j $(nproc)
```

### "Model not found"
- Ensure models are in: `C:\codedev\llm\.models\`
- Download models if missing

### Permission Errors
- Run PowerShell as Administrator
- Or use: `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`

## 📋 Benchmark Scripts Overview

| Script | Purpose | Usage Time |
|--------|---------|------------|
| `Run-QuickBenchmark.ps1` | Fast validation | 2-3 min |
| `Benchmark-Gemma.ps1` | Full benchmarking | 10-15 min |
| `Run-StressTest.ps1` | Load testing | Variable |
| `Compare-Models.ps1` | Model comparison | 5-10 min |
| `Generate-BenchmarkReport.ps1` | Create reports | 10 sec |
| `Start-Benchmark.ps1` | Interactive menu | Variable |

## 🎉 Ready to Benchmark!

You're all set! Start with:

```powershell
.\Run-QuickBenchmark.ps1
```

For questions or issues, check the full README.md documentation.