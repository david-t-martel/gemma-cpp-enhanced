# Session Framework Refactoring & Optimization Plan

Purpose: Decompose the current monolithic Session / SessionStorage / SessionManager implementation into smaller, testable, extensible units that follow SOLID principles while minimizing risky churn. This document is the authoritative task list; update it as items land. Keep diffs narrowly scoped (aim <800 lines per PR, ideally <400). Each task lists: Goal, Rationale (SOLID mapping), Actions, Acceptance Criteria, Risk / Mitigation, and Suggested PR Size.

## Architectural End-State (Target Shape)

```text
session/
  core/
    conversation_types.h        (ConversationMessage, enums)
    session.h / session.cc      (pure session state & context window logic)
  serialization/
    session_serialization.h/.cc (to_json, from_json only)
  cache/
    cache_interface.h           (ISessionCache)
    lru_cache.h/.cc             (LRUCache impl)
  storage/
    storage_interface.h         (ISessionStorage)
    sqlite/                     (conditional build if SQLite present)
      sqlite_storage.h/.cc      (SQLite implementation + RAII stmt wrapper)
  metrics/
    metrics_sink.h              (IMetricsSink)
    basic_metrics.h/.cc         (in‑memory impl)
  events/
    event_publisher.h           (IEventPublisher)
  manager/
    session_manager.h/.cc       (composition + façade only)
  tests/                        (unit tests per component)
```

`SessionManager` will depend on abstractions (interfaces) not concrete SQLite / LRU types (DIP). Storage + cache can be swapped / mocked.

## Guiding SOLID Mapping

| Principle | Current Issue                                         | Target Remedy                                                   |
| --------- | ----------------------------------------------------- | --------------------------------------------------------------- |
| SRP       | Files mix caching, persistence, metrics, events       | Split into focused components & interfaces                      |
| OCP       | Adding new storage backends requires changing manager | Introduce `ISessionStorage` registry & factory                  |
| LSP       | No abstractions; cannot substitute alt cache/storage  | Define minimal behavior contracts & mock in tests               |
| ISP       | Consumers forced to include huge headers              | Thin interfaces with forward declarations; include what you use |
| DIP       | High-level code depends on concrete SQLite & LRU      | Manager accepts interfaces (ctor injection / builder)           |

## Global Hardening Tasks (Phase 0 – Pre-Refactor)

1. Introduce build options

   - `GEMMA_ENABLE_SESSION` (OFF by default)
   - `GEMMA_BUILD_SESSION_BENCH` (OFF; depends on ENABLE_SESSION)
   - `GEMMA_BUILD_ENHANCED_CLI` (OFF; depends optionally on ENABLE_SESSION)
     Acceptance: CMake config succeeds with options toggled; disabling excludes targets from `all` but allows manual build.

2. Add license headers & TODO stub markers to current large files referencing this plan.
   Acceptance: All session/cli sources have consistent SPDX header; `grep -i todo session_refactor` finds markers.

3. Add GoogleTest migration scaffold (empty test `session_split_smoke.cpp`) wired into existing label system (`session;refactor`).

## Phase 1 – Interface Extraction (Small PRs)

Task 1.1: Extract `conversation_types.h`

- Move `ConversationMessage` enum + struct only.
- Remove JSON code from struct (SRP). Provide forward declarations of serialization functions.
- Acceptance: Existing tests compile; include graph shrinks (measure by `clang -MM` or include-what-you-use trial).

Task 1.2: Introduce `ISessionCache`

- Interface: `get`, `put`, `remove`, `clear`, `size`, `capacity`.
- Wrap current LRU implementation under new header `lru_cache.h/.cc` implementing interface.
- Acceptance: Manager compiles referencing interface; existing behavior unchanged (functional tests still pass).

Task 1.3: Introduce `ISessionStorage`

- Interface minimal CRUD: `save(std::shared_ptr<Session>)`, `load(id)`, `remove(id)`, `exists(id)`, `list(query struct)`, `export_all(out)`, `import(in, overwrite)`.
- Provide adapter wrapping current SQLite logic; rename large code to `sqlite_storage.cc`.

Task 1.4: Introduce RAII wrapper `SqliteStatement` (ctor prepares, dtor finalizes); replace raw `sqlite3_stmt*` in storage.

Task 1.5: Metrics abstraction `IMetricsSink`

- Methods: `onSessionCreated()`, `onSessionDeleted()`, `onMessageAdded(tokens)`, `snapshot()`.
- Manager composes one; default no‑op implementation.

Task 1.6: Event publisher abstraction.

- Replace direct `std::function` with `IEventPublisher` supporting `publish(event, json)`.

## Phase 2 – Serialization & Separation

Task 2.1: Move JSON (de)serialization to `session_serialization.{h,cc}` using free functions:

- `nlohmann::json ToJson(const Session&)` and `void FromJson(Session&, const json&)`.
- Remove direct dependency from `session.h` on `<nlohmann/json.hpp>` (ISP: reduce heavyweight includes).

Task 2.2: Optional streaming export/import

- Replace building whole `json::array` in memory with incremental write (ostream) to reduce peak memory.

## Phase 3 – Manager Slimming

Task 3.1: Rewrite `SessionManager` to delegate:

- Constructor injects `ISessionStorage&`, `ISessionCache&`, `IMetricsSink&`, `IEventPublisher&`.
- Provide factory function `CreateDefaultSessionSubsystem(const Config&)` which builds concrete objects and returns an aggregate struct for convenience.

Task 3.2: Move UUID generation to separate utility `uuid.h/.cc` (SRP). Manager just calls function.

Task 3.3: Replace ad‑hoc cleanup timing logic with `CleanupPolicy` strategy (configurable intervals / TTL enforcement) – DIP for future persistent stores.

## Phase 4 – Testing & Quality Gates

Task 4.1: Convert manual `test_session.cpp` to GoogleTest suite:

- Split into: `session_core_test.cpp`, `session_storage_sqlite_test.cpp`, `session_manager_test.cpp`, `session_concurrency_test.cpp`.
- Use fixtures; remove custom macros; apply labels: `session;core`, `session;storage`, `session;manager`, `session;concurrency`.

Task 4.2: Add mock storage & mock cache tests to validate DIP (use GoogleMock).

Task 4.3: Add performance microbench (Google Benchmark if already present) for:

- `ContextWindowCalculation` (warm cache O(1) vs cold) – assert upper bound timing (soft check, maybe just log).
- `StorageRoundTrip` size scaling (N messages).

Task 4.4: Add clang-tidy configuration overrides: limit function length (readability-function-size), enable `modernize-*` checks but suppress for `sqlite_storage.cc` where needed.

## Phase 5 – Performance & Memory Optimizations

Task 5.1: Replace repeated `std::accumulate` in trimming with maintained rolling counters (already partial) – ensure complexity remains O(k) for removals.

Task 5.2: Implement small-string optimization shuttle: store short messages inline (optional) OR document deliberate choice to rely on `std::string` SSO (decide and justify).

Task 5.3: Introduce `ReserveHint` API allowing caller to pre-size conversation for expected message count.

Task 5.4: Add metrics sampling for cache hit ratio and expose through manager (avoid recomputation each call by using periodic snapshot).

## Phase 6 – Extensibility / Open-Closed Enablement

Task 6.1: Storage registry

- Static registry keyed by string ("sqlite", "memory") returning `unique_ptr<ISessionStorage>`.
- Manager config chooses backend via `SessionManager::Config::storage_backend`.

Task 6.2: Pluggable cache (LFU variant) – demonstration of OCP.

Task 6.3: Optional metrics exporter interface (Prometheus-style textual format or JSON). Provide no‑op + simple JSON exporter.

## Phase 7 – Documentation & Cleanup

Task 7.1: Replace giant `tools/session/README.md` with concise overview (<200 lines) linking to new docs:

- `docs/session/architecture.md`
- `docs/session/performance.md`
- `docs/session/extensibility.md`

Task 7.2: Add architecture diagram (ASCII + optional Mermaid) showing dependency direction (manager -> interfaces -> concrete impls; arrows not reversed).

Task 7.3: Update root README to mention optional session subsystem & build flag.

Task 7.4: Add CHANGELOG entries per phase (group minor steps into single release notes section).

## Non-Goals (Explicit)

Avoid during this refactor:

- Introducing a heavy JSON Schema validator library (manual structural checks fine).
- Adding asynchronous IO threads before profiling justifies need.
- Implementing distributed or networked storage backends (future separate proposal).
- Over-optimizing micro allocations (premature until benchmark proves need).

## Prioritization & Sequencing

1. Phase 0 (unblocks safe incremental work) – single PR.
2. Phase 1 tasks 1.1–1.3 can be separate small PRs; 1.4–1.6 can batch if under size limit.
3. Phase 2 after interfaces stable.
4. Phase 3 (manager rewrite) only after tests from Phase 4 scaffolding exist (write tests first to lock behavior).
5. Phases 5–6 optional after functional parity achieved.

## Risk Matrix (High-Level)

| Risk                                       | Impact | Mitigation                                                   |
| ------------------------------------------ | ------ | ------------------------------------------------------------ |
| Large diffs hide regressions               | High   | Enforce small PR size & incremental gating tests             |
| Behavior drift during split                | Med    | Snapshot current tests before split; add golden JSON exports |
| Performance regression (extra indirection) | Med    | Benchmarks before/after; inline simple getters               |
| Windows SQLite linkage break               | Med    | Gate behind feature flag & CI matrix job with option ON      |
| Interface over-design                      | Low    | Start minimal, extend only with concrete use-cases           |

## Acceptance Metrics (Minimum)

By completion of Phases 1–4:

- No source file > 800 lines in session subsystem (except third-party or generated code).
- All public headers compile standalone (include self-sufficiency test target).
- GoogleTest suite covers ≥90% of lines in new session core (excluding storage backend specifics).
- Cache hit ratio surfaced via metrics snapshot.
- clang-tidy report introduces no new warnings vs baseline (except documented NOLINT blocks).

## Tracking Checklist (Mark with [x] as tasks complete)

Phase 0:

- [x] Build options added
- [ ] License headers standardized
- [x] Refactor smoke test added

Phase 1:

- [x] conversation_types extracted
- [x] ISessionCache introduced
- [x] ISessionStorage introduced
- [ ] RAII SqliteStatement
- [ ] IMetricsSink abstraction
- [ ] IEventPublisher abstraction

Phase 2:

- [x] Serialization split (initial separation of session & message JSON)
- [ ] Streaming export/import

Phase 3:

- [ ] Manager slimmed (constructor DI)
- [ ] UUID utility extracted
- [ ] CleanupPolicy strategy

Phase 4:

- [ ] Tests migrated to GTest (core/storage/manager/concurrency)
- [ ] Mock-based substitution tests
- [ ] Performance microbench
- [ ] clang-tidy function-size enforcement

Phase 5:

- [ ] Rolling counters for trimming
- [ ] Message storage optimization decision documented
- [ ] ReserveHint API
- [ ] Cache hit ratio metrics

Phase 6:

- [ ] Storage registry
- [ ] Alternate cache implementation
- [ ] Metrics exporter interface

Phase 7:

- [ ] Docs split & architecture diagrams
- [ ] Root README updated
- [ ] CHANGELOG entries

---

Maintainer Note: Keep each PR laser-focused. If a later task requires revisiting an earlier interface, document rationale in the PR description (link to benchmark or test demonstrating the need). Avoid speculative abstraction proliferation.
