# Security Critical TODOs Added

## Summary
Added TODO comments for critical security vulnerabilities in the stats/ directory. Each TODO follows the format: `# TODO: SECURITY CRITICAL - [description of issue and fix]`

## Files Modified

### 1. C:/codedev/llm/stats/src/application/agents/orchestrator.py
- **Line 392**: Added TODO for eval() usage vulnerability
- **Issue**: `eval()` allows arbitrary code execution
- **Fix**: Replace with `ast.literal_eval()` or a safe expression evaluator library like `simpleeval`

### 2. C:/codedev/llm/stats/src/server/auth.py
- **Line 82**: Added TODO for insecure JWT secret generation
- **Issue**: Temporary JWT secrets are regenerated on every restart, invalidating existing tokens
- **Fix**: Persist secrets across restarts using a secure key vault or protected file

### 3. C:/codedev/llm/stats/src/server/middleware.py
- **Line 599**: Added TODO for permissive CORS configuration
- **Issue**: `allow_headers=["*"]` is too permissive and `allow_credentials=True` with origins is dangerous
- **Fix**: Specify exact headers needed and validate origins properly

## Additional Security Findings (No TODO needed)

### Safe Usage Verified:
1. **subprocess.run in gemma_download.py**: Uses fixed command (nvidia-smi) with no user input
2. **asyncio.create_subprocess_exec in process.py**: Uses fixed pip commands with controlled inputs
3. **torch.compile in gemma.py**: Standard PyTorch compilation, not code execution

## Recommendations

1. **Immediate Priority**: Fix the eval() usage in orchestrator.py as it allows arbitrary code execution
2. **High Priority**: Implement proper JWT secret persistence to maintain token validity
3. **Medium Priority**: Tighten CORS configuration to specify exact headers

## Security Best Practices Applied

- All TODOs marked as "SECURITY CRITICAL" for visibility
- Each TODO includes both the problem description and suggested fix
- Comments placed directly above the vulnerable code for maximum visibility