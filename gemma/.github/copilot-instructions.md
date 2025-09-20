# AI Coding Agent Instructions

Concise, project-specific guidance to be productive quickly in this repository (gemma.cpp + Windows integration layer). Keep edits focused, portable, and aligned with existing patterns.

## 1. Project Essence
- Core: `gemma.cpp/` = C++ CPU inference engine for Gemma 2 / 3, Griffin (recurrent), and PaliGemma (vision-language).
- Windows Layer: Python + batch tooling (`gemma-cli.py`, `run_gemma.bat`, `demo_working.py`) wraps the C++ binary (native or WSL) and provides fallbacks when model load fails.
- Optimization pillars: Highway SIMD (multi-target via `foreach_target.h`), compressed weight formats (.sbs SFP/NUQ), batched query execution (`AllQueries`, `QBatch`), memory-mapped I/O.

## 2. Source Structure (edit hotspots)
- `gemma/` – Core model logic: attention, recurrent (griffin), kv cache, configs. Example: `griffin.cc` shows multi-arch SIMD patterns (CallUpcasted, HWY macros, ThreadPool usage).
- `ops/` – Low-level kernels (matvec, fused transforms). Touch only if adding new primitive; follow existing inl header inclusion style.
- `compression/` – Weight format handling. Add new format by mirroring pattern: types -> loader -> quantized matmul path.
- `io/` – Streaming / mmap; keep cross-platform guards minimal; prefer existing abstractions when adding file access.
- `python/*.py` (root scripts) – User-facing wrappers; preserve Windows↔WSL path translation logic.
- `examples/` – Reference usage; add minimal, runnable examples (avoid duplicating core logic).

## 3. Build & Run Workflows
- Preferred dev build (Linux/WSL): `cmake --preset make && cmake --build --preset make -j $(nproc)`.
- Windows native: `cmake --preset windows && cmake --build --preset windows -j 4` (Griffin may be conditionally stubbed—avoid regressing this; see `FINAL_WORKING_STATUS.md`).
- Benchmarks: `./build/single_benchmark` or `./build/benchmarks` with `--weights` & `--tokenizer`.
- CLI chat (Python wrapper): `python gemma-cli.py --model <path>.sbs [--tokenizer <tok>.spm]`.
- Migration (embed tokenizer): `./build/migrate_weights --tokenizer <tok> --weights in.sbs --output_weights out-single.sbs`.

## 4. Model + Runtime Configuration
- Configs centralized in `gemma/configs.{h,cc}` via `ModelConfig::Create()`; when adding a model: update enum, creation logic, ensure tokenizer mapping.
- Runtime knobs (sampling, batching): Check `RuntimeConfig` struct and ensure new flags are plumbed through CLI + examples consistently.
- Griffin / recurrent path relies on per-layer caches (`rglru_cache`, `conv1d_cache`); when modifying, maintain layout expectations (stride = model_dim, modulo indexing for conv cache).

## 5. SIMD & Highway Patterns (critical for edits)
- Every arch-specific file uses: undef + redefine `HWY_TARGET_INCLUDE`, then `#include "hwy/foreach_target.h"` + namespace blocks (`HWY_BEFORE_NAMESPACE` / `HWY_AFTER_NAMESPACE`). Preserve this or multi-target build breaks.
- Vector loops assert lane divisibility: keep `HWY_DASSERT(model_dim % hn::Lanes(df) == 0)` style guards.
- Parallel per-head work uses `ThreadPool.Run(0, heads, lambda)`; keep side-effect confinement per head to avoid races.

## 6. Error Handling & Fallbacks
- Python layer: provide graceful degradation (simulate output / status) when native binary fails—do not remove these pathways; extend with feature flags instead.
- C++: Prefer assertions (`HWY_ASSERT_M`, `HWY_DASSERT`) internally over throwing; external API surfaces return status or rely on caller validation.

## 7. Adding Features Safely
1. Identify layer (config / kernel / high-level orchestration).
2. Follow existing naming (lower_snake for functions, PascalCase for structs, Capitalized directories).
3. Extend minimal public surface—avoid leaking internal tensor layouts.
4. Add a tiny example in `examples/` if user-facing.
5. If new weight attribute: update loader + serialization in one change set.

## 8. Testing & Verification
- If `GEMMA_ENABLE_TESTS` is on: use `ctest --test-dir build` (some tests require real `.sbs` + tokenizer files—document assumptions instead of embedding large fixtures).
- For performance-sensitive changes: run `./build/single_benchmark --weights <model> --tokenizer <tok>` before & after—avoid >2–3% regression without note.
- Prefer micro-bench pattern in `benchmarks/` when adding kernels.

## 9. Windows Specific Notes
- Griffin instabilities historically caused link issues; guard new Griffin changes with platform checks if they introduce new symbols.
- Path translation logic in wrappers: always accept native Windows paths but convert to WSL form when invoking Linux binaries.
- Model load error code 3221226356: treat as compatibility/runtime dependency issue—avoid masking it silently; surface diagnostic.

## 10. Performance Guidance (respect existing heuristics)
- Encourage SFP models (`-sfp`) first; NUQ only if explicitly requested.
- Warm-up auto-tuning: do not reset between sequential queries unless config explicitly forces.
- Maintain streaming-friendly token emission (callback pattern preserved in new generation APIs).

## 11. When Unsure
- Search similar implementation in `gemma/` or `ops/` and mirror structure.
- Keep diffs small; avoid reformatting unrelated SIMD blocks.
- Defer large refactors; propose in comments or docs first.

## 12. Non-Goals for Agents
- Do NOT introduce GPU code paths here.
- Avoid adding heavy external dependencies (keep standalone C++ + lightweight Python only).
- Do not embed proprietary model weights or large binaries in repo.

---
Provide a brief rationale in PR descriptions for kernel or config changes (what metric improved, dataset if benchmarked). Ask maintainers if adding new compression formats or model families.
