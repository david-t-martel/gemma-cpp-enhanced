# Gemma.cpp Project - Quick Reference

## 🚨 CRITICAL ISSUE
**Build blocked by griffin.obj file locking (LNK1104)**
- File: `build\gemma\CMakeFiles\libgemma.dir\griffin.obj` (0 bytes, locked)
- Cause: Windows Defender/antivirus real-time scanning
- **PRIMARY WORKAROUND**: Build in WSL2 environment

## 📍 Key Locations
```
Project Root:  C:\codedev\llm\gemma
Source Code:   C:\codedev\llm\gemma\gemma.cpp
Build Dir:     C:\codedev\llm\gemma\gemma.cpp\build
Model Dir:     C:\codedev\llm\.models\gemma-gemmacpp-2b-it-v3
Scripts:       C:\codedev\llm\gemma\scripts
```

## 🎯 Current Status
- **Phase**: Build Configuration (BLOCKED)
- **Model**: Downloaded and ready (2b-it.sbs + tokenizer.spm)
- **Scripts**: Created (download-kaggle-model.sh, build-gemma.ps1)
- **Next Step**: Resolve griffin.obj issue via WSL2 or antivirus exclusion

## 🔧 Build Commands
```powershell
# PowerShell build
.\build-gemma.ps1

# Direct CMake
cd gemma.cpp
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

# WSL2 alternative (recommended)
wsl
cd /mnt/c/codedev/llm/gemma/gemma.cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 🚀 Quick Start (once built)
```bash
# Test inference
./gemma --model /c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs \
        --tokenizer /c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/tokenizer.spm \
        --prompt "Hello, world"
```

## 🛠️ Workarounds for Build Issue

### Option 1: WSL2 Build (Recommended)
```bash
# In WSL2 Ubuntu
sudo apt update
sudo apt install cmake g++ build-essential
cd /mnt/c/codedev/llm/gemma/gemma.cpp
rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Option 2: Windows Defender Exclusion
1. Open Windows Security
2. Virus & threat protection → Manage settings
3. Add exclusion for: `C:\codedev\llm\gemma\gemma.cpp\build`
4. Retry build

### Option 3: Alternative Build Directory
```powershell
# Build in temp location
cd $env:TEMP
git clone <repo>
cd gemma.cpp
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

## 📦 Dependencies Status
- ✅ Highway SIMD: Included
- ✅ SentencePiece: Included
- ✅ nlohmann/json: Included
- ✅ Google Test: Included
- ❌ ClangCL: Not installed (optional but recommended)
- ❓ vcpkg: Configured but optional

## 🎯 Agent Coordination Notes
- **Build Specialist**: Focus on WSL2 workaround
- **Performance Specialist**: Wait for successful build before benchmarking
- **Test Specialist**: Model files ready at `.models/gemma-gemmacpp-2b-it-v3/`

## 📝 Key Files Created
1. **`.env`**: Environment configuration
2. **`scripts/download-kaggle-model.sh`**: Model downloader
3. **`build-gemma.ps1`**: PowerShell build script
4. **`build-msvc.bat`**: Batch build script
5. **`.claude/context/project-context-2025-09-16.json`**: Full context

## 🔄 Last Updated
- Date: 2025-09-16
- Status: Build blocked, workarounds documented
- Next Action: Implement WSL2 build