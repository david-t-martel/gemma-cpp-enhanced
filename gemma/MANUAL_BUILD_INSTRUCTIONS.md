# Manual Build Instructions for gemma.exe with Session Management

## Quick Start (Visual Studio 2022 IDE) ✅ RECOMMENDED

### Step 1: Open Project
1. Launch Visual Studio 2022
2. Go to: **File → Open → CMake...**
3. Navigate to and select: `C:\codedev\llm\gemma\CMakeLists.txt`
4. Click **Open**

### Step 2: Wait for Configuration
- Visual Studio will automatically configure the CMake project
- Watch the **Output** window for "CMake generation finished" message
- This may take 2-3 minutes on first run

### Step 3: Select Configuration
- At the top toolbar, find the configuration dropdown (default: "x64-Debug")
- Change to: **x64-Release** for optimized binary

### Step 4: Build
- Method A: Click **Build → Build All** from menu
- Method B: Press **Ctrl+Shift+B**
- Method C: Right-click on **gemma (executable)** in Solution Explorer → Build

### Step 5: Locate Binary
- Build output directory: `C:\codedev\llm\gemma\out\build\x64-Release\`
- Binary location: `out\build\x64-Release\gemma.exe`
- Size: Should be ~2-3 MB

### Step 6: Verify Session Support
Open a terminal and run:
```cmd
cd C:\codedev\llm\gemma\out\build\x64-Release
gemma.exe --help
```

Look for session-related flags:
- `--load_session` - Load session from file
- `--save_session` - Save session to file

### Step 7: Deploy
```cmd
copy out\build\x64-Release\gemma.exe deploy\gemma.exe
```

## Alternative: Command Line Build (After IDE Configuration)

If Visual Studio IDE successfully configured the project once, you can use command line builds afterward:

```cmd
cd C:\codedev\llm\gemma
cmake --build out\build\x64-Release --config Release -j 10 --target gemma
```

## Troubleshooting

### "CMake Error: No CMAKE_CXX_COMPILER"
- **Solution**: Use Visual Studio IDE as described above
- The IDE handles compiler detection automatically

### Build fails with linker errors
- **Solution**: Clean and rebuild
  - In IDE: **Build → Clean Solution**, then **Build → Rebuild All**
  - Command line: Delete `out\build\` directory and reconfigure

### Binary doesn't have session flags
- **Verify**: Check that you're building from the correct source tree
- **Check**: Ensure `gemma.cpp/run.cc` was modified on Oct 23 06:19+
- **View**: Open `gemma.cpp/run.cc` in VS and search for "load_session"

### Out of memory during build
- **Solution**: Reduce parallel jobs
  - Close other applications
  - In IDE: Tools → Options → Projects and Solutions → Build and Run → Maximum number of parallel project builds = 4

## Build Specifications

### Recommended Configuration
- **Configuration**: Release
- **Platform**: x64
- **Toolset**: v143 (MSVC 2022)
- **C++ Standard**: C++20
- **Parallel Jobs**: 10 (reduce if out of memory)

### Expected Optimizations
- **Level**: /O2 (Maximize Speed)
- **Architecture**: AVX2 if supported by CPU
- **Link-Time Optimization**: Yes
- **Whole Program Optimization**: Yes

### Intel oneAPI Alternative (If Available)
If Intel oneAPI is properly installed and configured:

```powershell
# From PowerShell with oneAPI initialized
cd C:\codedev\llm\gemma
.\build_oneapi.ps1 -Config perfpack -Jobs 10
```

This provides:
- Intel C++ Compiler (ICX) with advanced optimizations
- Intel MKL (Math Kernel Library)
- Intel IPP (Integrated Performance Primitives)
- Intel TBB (Threading Building Blocks)
- Intel DNNL (Deep Neural Network Library)

## Verification Checklist

- [ ] Visual Studio 2022 successfully opened CMakeLists.txt
- [ ] CMake configuration completed without errors
- [ ] Release configuration selected
- [ ] Build completed successfully (0 errors)
- [ ] Binary exists at `out\build\x64-Release\gemma.exe`
- [ ] Binary size is ~2-3 MB
- [ ] `--help` output includes session flags
- [ ] Binary copied to `deploy\gemma.exe`
- [ ] Timestamp of deploy/gemma.exe is newer than Oct 23 00:20

## Session Management Features to Test

After building, test session functionality:

```cmd
# Start with session save
gemma.exe --weights path\to\model.sbs --tokenizer path\to\tokenizer.spm --save_session session.bin

# Load previous session
gemma.exe --weights path\to\model.sbs --tokenizer path\to\tokenizer.spm --load_session session.bin
```

## Next Steps After Successful Build

1. Update `BUILD_DEPLOYMENT_REPORT.md` with success status
2. Document build time and any warnings encountered
3. Note which optimizations were applied (check compiler flags in build log)
4. Run benchmark to compare performance with previous binary
5. Test session save/load functionality
6. Deploy to target environment
