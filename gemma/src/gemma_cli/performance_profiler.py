#!/usr/bin/env python
"""Performance profiling script for Gemma CLI optimization analysis."""

import asyncio
import cProfile
import io
import json
import pstats
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Lazy import analysis
import_times: Dict[str, float] = {}

def timed_import(module_name: str):
    """Import a module and track time."""
    start = time.perf_counter()
    module = __import__(module_name)
    import_times[module_name] = time.perf_counter() - start
    return module


class PerformanceProfiler:
    """Performance profiler for Gemma CLI."""

    def __init__(self):
        self.metrics = {
            "startup": {},
            "first_token": {},
            "rag_operations": {},
            "memory": {},
            "imports": {}
        }
        self.profiler = cProfile.Profile()

    def profile_imports(self):
        """Profile import times for all major modules."""
        print("Profiling imports...")
        modules_to_test = [
            "click",
            "pydantic",
            "rich",
            "toml",
            "numpy",
            "aiofiles",
            "redis",
        ]

        for module in modules_to_test:
            try:
                start = time.perf_counter()
                __import__(module)
                self.metrics["imports"][module] = time.perf_counter() - start
            except ImportError:
                self.metrics["imports"][module] = None

    def profile_cli_startup(self):
        """Profile CLI initialization and startup time."""
        print("Profiling CLI startup...")

        # Memory tracking
        tracemalloc.start()
        start_time = time.perf_counter()

        # Import CLI module
        from gemma_cli import cli

        # Measure config loading
        config_start = time.perf_counter()
        from gemma_cli.config.settings import load_config
        config_path = Path.home() / ".gemma_cli" / "config.toml"

        if config_path.exists():
            try:
                config = load_config(config_path)
                config_time = time.perf_counter() - config_start
                self.metrics["startup"]["config_load_time"] = config_time
            except:
                self.metrics["startup"]["config_load_time"] = None

        # Measure CLI initialization
        self.metrics["startup"]["total_import_time"] = time.perf_counter() - start_time

        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.metrics["memory"]["startup_current_mb"] = current / 1024 / 1024
        self.metrics["memory"]["startup_peak_mb"] = peak / 1024 / 1024

    async def profile_gemma_interface(self):
        """Profile GemmaInterface subprocess communication."""
        print("Profiling Gemma interface...")

        try:
            from gemma_cli.core.gemma import GemmaInterface, GemmaRuntimeParams

            # Find a test model
            test_models = [
                "C:\\codedev\\llm\\.models\\gemma-gemmacpp-2b-it-v3\\2b-it.sbs",
                "C:\\codedev\\llm\\.models\\gemma-3-gemmaCpp-3.0-4b-it-sfp-v1\\4b-it-sfp.sbs"
            ]

            model_path = None
            tokenizer_path = None

            for model in test_models:
                if Path(model).exists():
                    model_path = model
                    tokenizer_dir = Path(model).parent
                    tokenizer_candidates = list(tokenizer_dir.glob("*.spm"))
                    if tokenizer_candidates:
                        tokenizer_path = str(tokenizer_candidates[0])
                    break

            if not model_path:
                self.metrics["first_token"]["error"] = "No model found"
                return

            # Initialize Gemma
            params = GemmaRuntimeParams(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                max_tokens=50,
                temperature=0.7
            )

            # Measure initialization time
            init_start = time.perf_counter()
            gemma = GemmaInterface(params)
            self.metrics["first_token"]["init_time"] = time.perf_counter() - init_start

            # Measure first token latency
            prompt = "Hello, how are you?"
            first_token_time = None

            async def token_callback(chunk: str):
                nonlocal first_token_time
                if first_token_time is None and chunk.strip():
                    first_token_time = time.perf_counter()

            generation_start = time.perf_counter()
            response = await gemma.generate_response(prompt, token_callback)
            total_generation_time = time.perf_counter() - generation_start

            if first_token_time:
                self.metrics["first_token"]["latency"] = first_token_time - generation_start
            self.metrics["first_token"]["total_generation_time"] = total_generation_time
            self.metrics["first_token"]["response_length"] = len(response)

        except Exception as e:
            self.metrics["first_token"]["error"] = str(e)

    async def profile_rag_operations(self):
        """Profile RAG system operations."""
        print("Profiling RAG operations...")

        try:
            from gemma_cli.rag.hybrid_rag import HybridRAGManager, StoreMemoryParams, RecallMemoriesParams
            from gemma_cli.rag.embedded_vector_store import EmbeddedVectorStore

            # Test embedded store (default)
            embedded_start = time.perf_counter()
            rag_manager = HybridRAGManager(use_embedded_store=True)
            await rag_manager.initialize()
            self.metrics["rag_operations"]["embedded_init_time"] = time.perf_counter() - embedded_start

            # Test store operation
            store_start = time.perf_counter()
            store_params = StoreMemoryParams(
                content="Test memory content for performance profiling.",
                memory_type="working",
                importance=0.8
            )
            entry_id = await rag_manager.store_memory(params=store_params)
            self.metrics["rag_operations"]["store_time"] = time.perf_counter() - store_start

            # Test recall operation
            recall_start = time.perf_counter()
            recall_params = RecallMemoriesParams(
                query="performance profiling",
                limit=5
            )
            memories = await rag_manager.recall_memories(params=recall_params)
            self.metrics["rag_operations"]["recall_time"] = time.perf_counter() - recall_start
            self.metrics["rag_operations"]["memories_returned"] = len(memories)

            # Test file persistence
            if hasattr(rag_manager.backend, 'embedded_store'):
                persist_start = time.perf_counter()
                await rag_manager.backend.embedded_store.persist()
                self.metrics["rag_operations"]["persist_time"] = time.perf_counter() - persist_start

                # Get store stats
                store = rag_manager.backend.embedded_store
                self.metrics["rag_operations"]["store_size"] = len(store.store)
                if store.STORE_FILE.exists():
                    self.metrics["rag_operations"]["file_size_kb"] = store.STORE_FILE.stat().st_size / 1024

            await rag_manager.close()

        except Exception as e:
            self.metrics["rag_operations"]["error"] = str(e)

    def analyze_bottlenecks(self):
        """Analyze and identify performance bottlenecks."""
        bottlenecks = []

        # Check import times
        slow_imports = {k: v for k, v in self.metrics["imports"].items()
                       if v and v > 0.1}  # Imports taking > 100ms
        if slow_imports:
            bottlenecks.append({
                "type": "slow_imports",
                "details": slow_imports,
                "impact": "high" if any(v > 0.5 for v in slow_imports.values()) else "medium"
            })

        # Check startup memory
        if self.metrics["memory"].get("startup_peak_mb", 0) > 100:
            bottlenecks.append({
                "type": "high_startup_memory",
                "value": self.metrics["memory"]["startup_peak_mb"],
                "impact": "medium"
            })

        # Check first token latency
        first_token_latency = self.metrics["first_token"].get("latency")
        if first_token_latency and first_token_latency > 2.0:
            bottlenecks.append({
                "type": "high_first_token_latency",
                "value": first_token_latency,
                "impact": "high"
            })

        # Check RAG operations
        rag_recall_time = self.metrics["rag_operations"].get("recall_time")
        if rag_recall_time and rag_recall_time > 0.1:
            bottlenecks.append({
                "type": "slow_rag_recall",
                "value": rag_recall_time,
                "impact": "medium"
            })

        return bottlenecks

    def generate_report(self) -> str:
        """Generate performance analysis report."""
        report = []
        report.append("=" * 60)
        report.append("GEMMA CLI PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 60)

        # Import times
        report.append("\n📦 IMPORT TIMES:")
        for module, time_taken in sorted(self.metrics["imports"].items(),
                                        key=lambda x: x[1] if x[1] else 0,
                                        reverse=True):
            if time_taken:
                report.append(f"  - {module}: {time_taken*1000:.1f}ms")

        # Startup metrics
        report.append("\n🚀 STARTUP METRICS:")
        if "total_import_time" in self.metrics["startup"]:
            report.append(f"  - Total import time: {self.metrics['startup']['total_import_time']*1000:.1f}ms")
        if "config_load_time" in self.metrics["startup"]:
            report.append(f"  - Config load time: {self.metrics['startup']['config_load_time']*1000:.1f}ms")

        # Memory metrics
        report.append("\n💾 MEMORY USAGE:")
        if "startup_current_mb" in self.metrics["memory"]:
            report.append(f"  - Startup current: {self.metrics['memory']['startup_current_mb']:.1f}MB")
        if "startup_peak_mb" in self.metrics["memory"]:
            report.append(f"  - Startup peak: {self.metrics['memory']['startup_peak_mb']:.1f}MB")

        # First token metrics
        report.append("\n⚡ FIRST TOKEN METRICS:")
        if "error" in self.metrics["first_token"]:
            report.append(f"  - Error: {self.metrics['first_token']['error']}")
        else:
            if "init_time" in self.metrics["first_token"]:
                report.append(f"  - Gemma init time: {self.metrics['first_token']['init_time']*1000:.1f}ms")
            if "latency" in self.metrics["first_token"]:
                report.append(f"  - First token latency: {self.metrics['first_token']['latency']*1000:.1f}ms")
            if "total_generation_time" in self.metrics["first_token"]:
                report.append(f"  - Total generation time: {self.metrics['first_token']['total_generation_time']:.2f}s")

        # RAG metrics
        report.append("\n🔍 RAG OPERATIONS:")
        if "error" in self.metrics["rag_operations"]:
            report.append(f"  - Error: {self.metrics['rag_operations']['error']}")
        else:
            if "embedded_init_time" in self.metrics["rag_operations"]:
                report.append(f"  - Embedded store init: {self.metrics['rag_operations']['embedded_init_time']*1000:.1f}ms")
            if "store_time" in self.metrics["rag_operations"]:
                report.append(f"  - Store operation: {self.metrics['rag_operations']['store_time']*1000:.1f}ms")
            if "recall_time" in self.metrics["rag_operations"]:
                report.append(f"  - Recall operation: {self.metrics['rag_operations']['recall_time']*1000:.1f}ms")
            if "persist_time" in self.metrics["rag_operations"]:
                report.append(f"  - Persist to disk: {self.metrics['rag_operations']['persist_time']*1000:.1f}ms")

        # Bottleneck analysis
        bottlenecks = self.analyze_bottlenecks()
        report.append("\n🔥 IDENTIFIED BOTTLENECKS:")
        if bottlenecks:
            for bottleneck in bottlenecks:
                impact_emoji = "🔴" if bottleneck["impact"] == "high" else "🟡"
                report.append(f"  {impact_emoji} {bottleneck['type']}")
                if "details" in bottleneck:
                    for k, v in bottleneck["details"].items():
                        report.append(f"     - {k}: {v*1000:.1f}ms")
                elif "value" in bottleneck:
                    report.append(f"     Value: {bottleneck['value']}")
        else:
            report.append("  ✅ No significant bottlenecks found")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    async def run_full_profile(self):
        """Run complete performance profiling."""
        print("\n" + "=" * 60)
        print("Starting Gemma CLI Performance Profiling...")
        print("=" * 60)

        # Profile imports
        self.profile_imports()

        # Profile CLI startup
        self.profile_cli_startup()

        # Profile async operations
        await self.profile_gemma_interface()
        await self.profile_rag_operations()

        # Generate report
        report = self.generate_report()
        print(report)

        # Save metrics to JSON
        output_file = Path("performance_metrics.json")
        with open(output_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\n📊 Detailed metrics saved to: {output_file}")

        return self.metrics


async def main():
    """Main entry point for performance profiling."""
    profiler = PerformanceProfiler()
    await profiler.run_full_profile()


if __name__ == "__main__":
    asyncio.run(main())