# AI Coding Agent Instructions

Concise, project-specific guidance for fast, safe contribution (gemma.cpp core + Windows & emerging hardware backends). Keep diffs minimal; mirror existing idioms.

## 1. Scope & Core Concepts
- Core engine (`gemma.cpp/`): Gemma 2/3, Griffin (recurrent), PaliGemma 2 (vision) with streaming generation & batched (`AllQueries` / `QBatch`).
- Weights: custom `.sbs` (SFP8, NUQ4, BF16/F32). Newer single-file weights embed tokenizer (omit `--tokenizer`). Use `migrate_weights` to convert.
- Critical pillars: Highway multi-target SIMD, on-the-fly GEMM autotuning (cached between queries), memory-mapped or parallel I/O, compressed-weight fused matmul.

## 2. Layout Hotspots
- `gemma/`: `gemma.cc`, `attention.cc`, `griffin.cc`, `vit.cc`, `kv_cache.cc`, `configs.{h,cc}` (`ModelConfig::Create`).
- `compression/`: format types -> loader -> fused GEMM path; replicate pattern when adding format in ONE PR (types + load + matmul branch + serialization).
- `ops/`: low-level kernels; only touch when adding a primitive – preserve HWY include pattern.
- `backends/`: GPU/NPU (CUDA, SYCL, Vulkan, etc.) via `BackendInterface` + `BackendManager`. Prefer extending existing interface; avoid introducing new global singletons.
- `python/*.py`: Windows/WSL bridge (`gemma-cli.py`) – DO NOT remove fallback / simulated output pathways.
- `examples/`: keep minimal; add new public feature sample here (not in core).

## 3. Build & Run (verify before changing docs)
- CPU dev: `cmake --preset make && cmake --build --preset make -j $(nproc)`.
- Windows native: `cmake --preset windows && cmake --build --preset windows -j 4` (watch Griffin stubs + path translation).
- Enable backends: configure with corresponding `GEMMA_BUILD_*_BACKEND=ON` (see `backends/README.md`). Keep CPU path working first.
- Benchmarks: `./build/single_benchmark --weights <w> --tokenizer <tok>` or `./build/benchmarks ...`.
- CLI: `python gemma-cli.py --model <file>.sbs [--tokenizer <tok>.spm]`.
- Migrate: `./build/migrate_weights --tokenizer <tok> --weights in.sbs --output_weights out-single.sbs`.

## 4. SIMD & Performance Patterns
- Preserve Highway boilerplate: redefine `HWY_TARGET_INCLUDE`, include `hwy/foreach_target.h`, use `HWY_BEFORE/AFTER_NAMESPACE` blocks.
- Always keep lane divisibility asserts (e.g. `HWY_DASSERT(dim % hn::Lanes(df) == 0)`).
- Per-head parallelism via `ThreadPool.Run(0, heads, ...)`; no shared mutable state outside head scope.
- Auto-tuning results persist across queries—do not reset unless adding an explicit config flag.

## 5. Runtime / Model Extension
- Add model: extend enum + `ModelConfig::Create` + tokenizer mapping; update any size-dependent cache layout (griffin conv / recurrent caches maintain stride = model_dim).
- New compression format: mirror existing SFP/NUQ structure; add loader, dispatch in GEMM, serialization, tiny example and mention in docs.
- Backends: implement `BackendInterface`, register in `BackendRegistry`; ensure graceful fallback to CPU if unsupported. Benchmark locally before enabling by default.

## 6. Testing & Benchmarks
- Enable tests: configure with `GEMMA_ENABLE_TESTS`; run `ctest --test-dir build` (some tests assume real `.sbs` + tokenizer—document if skipped).
- Add microbench in `benchmarks/` for new kernel; avoid >2–3% regression (justify otherwise).
- Validate correctness first on CPU; then backend variant; use existing sample invocations as smoke tests.

## 7. Windows / WSL Notes
- Maintain path translation & fallback logic; accept native paths, convert for WSL when spawning Linux binary.
- Error 3221226356: treat as dependency/runtime failure—surface diagnostic, don't mask.
- Avoid large `-j` in WSL if build instability appears.

## 8. Safe Change Checklist
1. Identify layer (config / compression / kernel / backend / high-level API).
2. Preserve Highway + threading idioms; no drive-by reformatting.
3. Touch only minimal surfaces; avoid leaking internal tensor layouts.
4. Provide tiny example if user-facing; update README snippet only after verifying commands.
5. Include perf note in PR body for kernel/compression/backend changes.

## 9. Non-Goals / Cautions
- Do not add unrelated heavy deps or fetch large model artifacts into repo.
- Do not introduce alternative backend abstraction layers; extend `BackendInterface` instead.
- Avoid speculative refactors of SIMD blocks; propose first if structural.
- Leave merge-conflict markers out of new docs (root `README.md` currently contains markers – don't replicate).

When unsure: grep similar code in `gemma/` or `compression/`, mimic pattern, keep diff small. Provide rationale (what improved, metric & hw). Ask before adding entirely new model families or compression schemes.
