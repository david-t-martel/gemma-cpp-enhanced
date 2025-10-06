# Gemma.cpp Performance Benchmarking Suite

A comprehensive PowerShell-based benchmarking suite for measuring and optimizing the performance of gemma.cpp models running in WSL.

## 🚀 Quick Start

```powershell
# Run a quick test to verify setup
.\Run-QuickBenchmark.ps1

# Run full benchmarks
.\Benchmark-Gemma.ps1 -FullBenchmark

# Generate HTML report
.\Generate-BenchmarkReport.ps1
```

## 📁 Suite Components

### Core Scripts

| Script | Purpose | Usage |
|--------|---------|--------|
| **Benchmark-Gemma.ps1** | Main benchmarking orchestrator | Full performance testing |
| **Run-QuickBenchmark.ps1** | Quick validation test | Verify setup and get baseline |
| **Run-StressTest.ps1** | Stress testing under load | Test concurrent performance |
| **Compare-Models.ps1** | Model comparison analysis | Compare 2B vs 4B models |
| **Generate-BenchmarkReport.ps1** | HTML report generation | Create visual reports |

## 🎯 Test Categories

### 1. Token Generation Tests
- Tests different prompt complexities (simple, medium, complex)
- Varies max tokens (50, 100, 200, 500)
- Tests different temperatures (0.0, 0.7, 1.0)
- Measures tokens per second throughput

### 2. Batch Processing Tests
- Tests batch sizes (1, 2, 4, 8)
- Measures prompts per second
- Evaluates batch efficiency

### 3. Memory Usage Tests
- Baseline memory measurement
- Peak memory during inference
- Average memory consumption
- Memory profiling over time

### 4. Latency Tests
- First token latency
- End-to-end response time
- Statistical analysis (min, max, avg, std dev)
- Percentile distributions (p50, p95, p99)

### 5. Stress Tests
- Concurrent request handling
- Sustained load testing
- Resource utilization monitoring
- Failure rate analysis

## 📊 Benchmarking Modes

### Quick Test Mode
```powershell
.\Benchmark-Gemma.ps1 -QuickTest
```
- Uses only 2B model
- Simple prompts only
- Minimal configuration variations
- ~2-3 minutes runtime

### Full Benchmark Mode
```powershell
.\Benchmark-Gemma.ps1 -FullBenchmark
```
- Tests all available models
- Complete prompt suite
- All configuration combinations
- ~10-15 minutes runtime

### Stress Test Mode
```powershell
.\Run-StressTest.ps1 -Model 2b -Duration 60 -Concurrency 4
```
- Configurable duration and concurrency
- Real-world load simulation
- System resource monitoring

## 📈 Performance Metrics

### Key Metrics Measured

1. **Throughput Metrics**
   - Tokens per second
   - Prompts per second (batch)
   - Requests per second (stress)

2. **Latency Metrics**
   - First token latency
   - Total generation time
   - Response time percentiles

3. **Resource Metrics**
   - Memory usage (MB/GB)
   - CPU utilization (%)
   - WSL memory consumption

4. **Quality Metrics**
   - Success rate
   - Error frequency
   - Timeout occurrences

## 🔧 Configuration

### Model Configuration

The suite automatically detects models in `C:\codedev\llm\.models\`:

```powershell
# 2B Model
gemma-gemmacpp-2b-it-v3\
  ├── 2b-it.sbs        # Weights
  └── tokenizer.spm    # Tokenizer

# 4B Model
gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\
  ├── 4b-it-sfp.sbs    # Weights
  └── tokenizer.spm    # Tokenizer
```

### Test Parameters

Edit in `Benchmark-Gemma.ps1`:

```powershell
$script:Config = @{
    BatchSizes = @(1, 2, 4, 8)
    MaxTokens = @(50, 100, 200, 500)
    Temperatures = @(0.0, 0.7, 1.0)
}
```

## 📝 Output Formats

### CSV Files
- Detailed metrics in tabular format
- Easy import to Excel/analysis tools
- Located in `results\` directory

### JSON Files
- Complete test data with metadata
- Structured for programmatic analysis
- Includes all raw measurements

### HTML Reports
- Interactive charts using Chart.js
- Visual performance comparisons
- Automatic recommendations
- Browser-viewable results

## 💡 Optimal Settings (Based on Testing)

### For Maximum Speed (Real-time Chat)
- **Model:** Gemma 2B IT
- **Max Tokens:** 50-100
- **Temperature:** 0.7
- **Expected:** 40-60 tokens/sec
- **Memory:** ~2GB

### For Quality Output (Content Generation)
- **Model:** Gemma 4B IT SFP
- **Max Tokens:** 200-500
- **Temperature:** 0.7-1.0
- **Expected:** 20-30 tokens/sec
- **Memory:** ~4GB

### For Batch Processing
- **Batch Size:** 4-8 prompts
- **Model:** 2B for speed, 4B for quality
- **Temperature:** 0.7
- **Memory:** Scale with batch size

### For API Backends
- **Model:** Gemma 2B IT
- **Max Tokens:** 200
- **Temperature:** 0.8
- **Concurrency:** 2-4 workers
- **Expected:** 35-45 tokens/sec

## 🎮 Usage Examples

### Basic Benchmark
```powershell
# Quick validation
.\Run-QuickBenchmark.ps1

# Results:
# Gemma 2B: 45.3 tokens/sec, ~1800MB memory
# Gemma 4B: 23.7 tokens/sec, ~3600MB memory
```

### Stress Testing
```powershell
# Test 2B model under load
.\Run-StressTest.ps1 -Model 2b -Duration 120 -Concurrency 8 -SaveResults

# Results:
# Success Rate: 98.5%
# Avg Response: 1250ms
# Peak Memory: 4.2GB
```

### Model Comparison
```powershell
# Compare all models and configurations
.\Compare-Models.ps1 -RunTests

# Output:
# Speed Champion: 2B Speed Optimized - 52.3 tokens/sec
# Memory Efficient: 2B Deterministic - 1650MB
# Quality Leader: 4B Quality Focus - Best perplexity
```

### Generate Report
```powershell
# Create visual report from latest results
.\Generate-BenchmarkReport.ps1

# Opens: benchmark-report.html in browser
```

## 🐛 Troubleshooting

### WSL Not Found
```powershell
# Install WSL
wsl --install

# Verify WSL
wsl --list --verbose
```

### Binary Not Found
```powershell
# Build gemma.cpp in WSL
cd /mnt/c/codedev/llm/gemma/gemma.cpp
cmake --preset make
cmake --build --preset make -j $(nproc)
```

### Model Not Found
```powershell
# Download models to .models directory
# Ensure .sbs and .spm files are present
```

### Permission Issues
```powershell
# Run PowerShell as Administrator
# Or adjust WSL file permissions
wsl chmod +x /path/to/gemma
```

## 📊 Sample Results

### Token Generation Performance
| Model | Configuration | Tokens/Sec | Memory |
|-------|--------------|------------|---------|
| 2B | Speed Optimized | 52.3 | 1.8GB |
| 2B | Balanced | 41.7 | 2.1GB |
| 4B | Speed Optimized | 28.4 | 3.5GB |
| 4B | Quality Focus | 22.1 | 4.2GB |

### Stress Test Results (60s, 4 workers)
| Metric | 2B Model | 4B Model |
|--------|----------|----------|
| Total Requests | 186 | 112 |
| Success Rate | 98.5% | 97.3% |
| Avg Response | 1.25s | 2.14s |
| P95 Latency | 1.87s | 3.21s |
| Peak Memory | 4.2GB | 7.8GB |

## 🔄 Continuous Benchmarking

For automated benchmarking in CI/CD:

```powershell
# Schedule daily benchmarks
$trigger = New-ScheduledTaskTrigger -Daily -At "2:00AM"
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" `
    -Argument "-File C:\codedev\llm\benchmarks\Benchmark-Gemma.ps1 -FullBenchmark"
Register-ScheduledTask -TaskName "GemmaBenchmark" -Trigger $trigger -Action $action
```

## 📚 Advanced Features

### Custom Test Prompts
Edit `Benchmark-Gemma.ps1` to add custom prompts:

```powershell
$script:Config.TestPrompts += @{
    Name = "code"
    Text = "Write a Python function to sort a list"
    ExpectedTokens = 150
}
```

### Export to Monitoring Systems
Results can be exported to:
- Prometheus/Grafana
- Application Insights
- Custom dashboards

### Performance Regression Detection
Compare results over time to detect regressions:

```powershell
# Compare with baseline
$baseline = Get-Content "baseline.json" | ConvertFrom-Json
$current = Get-Content "results\latest.json" | ConvertFrom-Json

if ($current.TokensPerSecond -lt $baseline.TokensPerSecond * 0.9) {
    Write-Warning "Performance regression detected!"
}
```

## 🤝 Contributing

To add new benchmarks:

1. Create new test function in `Benchmark-Gemma.ps1`
2. Add to `$allResults` collection
3. Update report generation
4. Document in README

## 📄 License

This benchmarking suite is provided as-is for testing gemma.cpp performance.

## 🔗 Related Resources

- [Gemma.cpp Repository](https://github.com/google/gemma.cpp)
- [Model Download Guide](../stats/README.md)
- [WSL Performance Tuning](https://docs.microsoft.com/en-us/windows/wsl/)