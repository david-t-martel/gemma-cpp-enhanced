"""Gemini-based code optimization tool for performance and efficiency improvements.

This module provides AI-powered optimization suggestions for code performance,
memory usage, algorithmic efficiency, and resource utilization.
"""

import asyncio
import json
import re
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

from .client import GeminiClient
from .client import GeminiConfig
from .client import GeminiModel

# Configure logging


class OptimizationType(str, Enum):
    """Types of optimizations."""

    PERFORMANCE = "performance"
    MEMORY = "memory"
    ALGORITHM = "algorithm"
    DATABASE = "database"
    NETWORK = "network"
    CONCURRENCY = "concurrency"
    CACHING = "caching"
    CODE_SIZE = "code_size"
    ENERGY = "energy"
    STARTUP_TIME = "startup_time"


class OptimizationPriority(str, Enum):
    """Priority levels for optimizations."""

    CRITICAL = "critical"  # Major performance bottleneck
    HIGH = "high"  # Significant improvement potential
    MEDIUM = "medium"  # Moderate improvement
    LOW = "low"  # Minor improvement
    TRIVIAL = "trivial"  # Micro-optimization


class ResourceType(str, Enum):
    """Types of resources to optimize."""

    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    GPU = "gpu"
    BATTERY = "battery"


@dataclass
class PerformanceMetric:
    """Performance metric measurement."""

    name: str
    current_value: float
    optimized_value: float
    improvement_percentage: float
    unit: str


@dataclass
class OptimizationSuggestion:
    """A single optimization suggestion."""

    type: OptimizationType
    priority: OptimizationPriority
    title: str
    description: str
    current_code: str | None = None
    optimized_code: str | None = None
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    estimated_improvement: PerformanceMetric | None = None
    complexity: str = "simple"  # simple, moderate, complex
    risk_level: str = "low"  # low, medium, high
    prerequisites: list[str] = field(default_factory=list)
    trade_offs: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Result of a benchmark test."""

    name: str
    baseline_time: float
    optimized_time: float
    speedup: float
    memory_before: int | None = None
    memory_after: int | None = None
    memory_reduction: float | None = None


@dataclass
class OptimizationProfile:
    """Profile defining optimization preferences."""

    target_metrics: list[OptimizationType]
    resource_constraints: dict[ResourceType, float]
    acceptable_trade_offs: list[str]
    preserve_readability: bool = True
    maintain_compatibility: bool = True
    risk_tolerance: str = "medium"  # low, medium, high


@dataclass
class OptimizationResult:
    """Complete result of optimization analysis."""

    suggestions: list[OptimizationSuggestion]
    benchmarks: list[BenchmarkResult]
    total_potential_speedup: float
    total_memory_savings: int
    implementation_plan: list[str]
    warnings: list[str]
    estimated_effort_hours: float
    confidence_score: float  # 0-1


class GeminiOptimizer:
    """Code optimizer powered by Gemini AI."""

    def __init__(
        self,
        client: GeminiClient | None = None,
        config: GeminiConfig | None = None,
    ):
        """Initialize the optimizer.

        Args:
            client: Existing Gemini client to use
            config: Configuration for creating a new client
        """
        self.client = client or GeminiClient(
            config
            or GeminiConfig(
                model=GeminiModel.GEMINI_1_5_PRO,
                temperature=0.2,  # Low temperature for consistent optimization
                max_output_tokens=8192,
            )
        )

        # Common optimization patterns
        self.optimization_patterns = {
            OptimizationType.PERFORMANCE: [
                "loop optimization",
                "vectorization",
                "parallel processing",
                "algorithm complexity reduction",
                "lazy evaluation",
            ],
            OptimizationType.MEMORY: [
                "memory pooling",
                "object recycling",
                "data structure optimization",
                "memory leak prevention",
                "buffer management",
            ],
            OptimizationType.CACHING: [
                "memoization",
                "result caching",
                "query caching",
                "CDN usage",
                "browser caching",
            ],
        }

    async def optimize_code(
        self,
        code: str,
        language: str | None = None,
        profile: OptimizationProfile | None = None,
        context: str | None = None,
    ) -> OptimizationResult:
        """Optimize a code snippet or file.

        Args:
            code: The code to optimize
            language: Programming language
            profile: Optimization profile with preferences
            context: Additional context about the code

        Returns:
            OptimizationResult with suggestions and benchmarks
        """
        # Use default profile if not provided
        if profile is None:
            profile = OptimizationProfile(
                target_metrics=[OptimizationType.PERFORMANCE, OptimizationType.MEMORY],
                resource_constraints={},
                acceptable_trade_offs=["slight readability decrease"],
            )

        # Generate optimization suggestions
        suggestions = await self._generate_suggestions(code, language, profile, context)

        # Analyze potential improvements
        benchmarks = await self._estimate_improvements(code, suggestions, language)

        # Create implementation plan
        plan = self._create_implementation_plan(suggestions)

        # Calculate overall metrics
        total_speedup = self._calculate_total_speedup(benchmarks)
        total_memory = self._calculate_memory_savings(benchmarks)

        # Estimate effort
        effort_hours = self._estimate_effort(suggestions)

        # Calculate confidence
        confidence = self._calculate_confidence(suggestions, benchmarks)

        return OptimizationResult(
            suggestions=suggestions,
            benchmarks=benchmarks,
            total_potential_speedup=total_speedup,
            total_memory_savings=total_memory,
            implementation_plan=plan,
            warnings=self._generate_warnings(suggestions),
            estimated_effort_hours=effort_hours,
            confidence_score=confidence,
        )

    async def optimize_algorithm(
        self,
        code: str,
        algorithm_description: str | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """Optimize an algorithm for better time/space complexity.

        Args:
            code: The algorithm implementation
            algorithm_description: Description of what the algorithm does
            constraints: Constraints like input size, memory limits

        Returns:
            OptimizationResult with algorithmic improvements
        """
        prompt = f"""
        Optimize the following algorithm for better time and space complexity:

        Description: {algorithm_description or "Analyze and understand the algorithm"}

        Current Implementation:
        ```
        {code}
        ```

        Constraints: {json.dumps(constraints) if constraints else "None specified"}

        Provide:
        1. Current time complexity analysis
        2. Current space complexity analysis
        3. Optimized algorithm with better complexity
        4. Trade-offs of the optimization
        5. When the optimization is most beneficial

        Format as JSON with detailed explanations and code.
        """

        try:
            response = await self.client.generate(
                prompt=prompt,
                system_prompt=self._get_algorithm_expert_prompt(),
                temperature=0.3,
            )

            return self._parse_algorithm_optimization(response.content, code)

        except Exception as e:
            logger.error(f"Algorithm optimization failed: {e}")
            return OptimizationResult(
                suggestions=[],
                benchmarks=[],
                total_potential_speedup=1.0,
                total_memory_savings=0,
                implementation_plan=[],
                warnings=[f"Optimization failed: {e!s}"],
                estimated_effort_hours=0,
                confidence_score=0,
            )

    async def optimize_database_queries(
        self,
        queries: list[str],
        schema: str | None = None,
        database_type: str = "postgresql",
    ) -> list[OptimizationSuggestion]:
        """Optimize database queries for better performance.

        Args:
            queries: List of SQL queries to optimize
            schema: Database schema information
            database_type: Type of database (postgresql, mysql, etc.)

        Returns:
            List of optimization suggestions for queries
        """
        suggestions = []

        for query in queries:
            prompt = f"""
            Optimize this {database_type} query for better performance:

            Query:
            ```sql
            {query}
            ```

            {f"Schema: {schema}" if schema else ""}

            Provide optimizations for:
            1. Index usage
            2. Query structure
            3. Join optimization
            4. Subquery elimination
            5. Aggregation optimization

            Return JSON with original query, optimized query, and explanations.
            """

            try:
                response = await self.client.generate(
                    prompt=prompt,
                    system_prompt="You are a database optimization expert.",
                    temperature=0.2,
                )

                suggestion = self._parse_query_optimization(response.content, query)
                if suggestion:
                    suggestions.append(suggestion)

            except Exception as e:
                logger.error(f"Query optimization failed: {e}")

        return suggestions

    async def optimize_for_parallelism(
        self,
        code: str,
        language: str = "python",
        target_cores: int = 4,
    ) -> OptimizationResult:
        """Optimize code for parallel execution.

        Args:
            code: Sequential code to parallelize
            language: Programming language
            target_cores: Number of cores to optimize for

        Returns:
            OptimizationResult with parallelization suggestions
        """
        prompt = f"""
        Optimize the following {language} code for parallel execution on {target_cores} cores:

        ```{language}
        {code}
        ```

        Identify:
        1. Parallelizable sections
        2. Data dependencies
        3. Synchronization requirements
        4. Optimal parallelization strategy (threads, processes, async)
        5. Expected speedup

        Provide parallelized version with explanations.
        """

        try:
            response = await self.client.generate(
                prompt=prompt,
                system_prompt=self._get_parallelism_expert_prompt(),
                temperature=0.2,
            )

            return self._parse_parallelization(response.content, code)

        except Exception as e:
            logger.error(f"Parallelization optimization failed: {e}")
            return self._empty_result()

    async def optimize_memory_usage(
        self,
        code: str,
        memory_profile: dict[str, Any] | None = None,
        target_reduction: float = 0.3,
    ) -> OptimizationResult:
        """Optimize code for reduced memory usage.

        Args:
            code: Code to optimize for memory
            memory_profile: Current memory usage profile
            target_reduction: Target memory reduction (0.3 = 30%)

        Returns:
            OptimizationResult with memory optimization suggestions
        """
        prompt = f"""
        Optimize the following code to reduce memory usage by {target_reduction * 100}%:

        ```
        {code}
        ```

        Current Memory Profile: {json.dumps(memory_profile) if memory_profile else "Not provided"}

        Focus on:
        1. Data structure optimization
        2. Object pooling and recycling
        3. Lazy loading strategies
        4. Memory leak prevention
        5. Buffer optimization

        Provide optimized code with memory savings estimates.
        """

        try:
            response = await self.client.generate(
                prompt=prompt,
                system_prompt="You are an expert in memory optimization and efficient data structures.",
                temperature=0.2,
            )

            return self._parse_memory_optimization(response.content, code)

        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return self._empty_result()

    async def profile_and_optimize(
        self,
        code: str,
        test_inputs: list[Any] | None = None,
        language: str = "python",
    ) -> OptimizationResult:
        """Profile code and provide targeted optimizations.

        Args:
            code: Code to profile and optimize
            test_inputs: Sample inputs for profiling
            language: Programming language

        Returns:
            Comprehensive OptimizationResult
        """
        # First, analyze the code structure
        analysis_prompt = f"""
        Analyze this {language} code for performance bottlenecks:

        ```{language}
        {code}
        ```

        Test inputs: {test_inputs if test_inputs else "Not provided"}

        Identify:
        1. Time complexity of each function
        2. Space complexity
        3. Potential bottlenecks
        4. Inefficient patterns
        5. Optimization opportunities

        Provide detailed analysis in JSON format.
        """

        try:
            # Get analysis
            analysis_response = await self.client.generate(
                prompt=analysis_prompt,
                system_prompt=self._get_profiling_expert_prompt(),
                temperature=0.1,
            )

            analysis = self._parse_json_response(analysis_response.content)

            # Generate optimizations based on analysis
            optimization_prompt = f"""
            Based on this analysis: {json.dumps(analysis)}

            Optimize the code:
            ```{language}
            {code}
            ```

            Provide specific optimizations for each identified bottleneck.
            """

            optimization_response = await self.client.generate(
                prompt=optimization_prompt,
                system_prompt=self._get_optimization_expert_prompt(),
                temperature=0.2,
            )

            return self._parse_comprehensive_optimization(
                optimization_response.content,
                code,
                analysis,
            )

        except Exception as e:
            logger.error(f"Profile and optimize failed: {e}")
            return self._empty_result()

    async def suggest_caching_strategy(
        self,
        code: str,
        access_patterns: dict[str, Any] | None = None,
    ) -> list[OptimizationSuggestion]:
        """Suggest caching strategies for improved performance.

        Args:
            code: Code to analyze for caching opportunities
            access_patterns: Information about data access patterns

        Returns:
            List of caching suggestions
        """
        prompt = f"""
        Analyze this code for caching opportunities:

        ```
        {code}
        ```

        Access Patterns: {json.dumps(access_patterns) if access_patterns else "Analyze from code"}

        Suggest caching strategies for:
        1. Frequently computed results
        2. Database query results
        3. API responses
        4. File I/O operations
        5. Expensive computations

        Include cache invalidation strategies and TTL recommendations.
        """

        try:
            response = await self.client.generate(
                prompt=prompt,
                system_prompt="You are an expert in caching strategies and performance optimization.",
                temperature=0.3,
            )

            return self._parse_caching_suggestions(response.content)

        except Exception as e:
            logger.error(f"Caching strategy suggestion failed: {e}")
            return []

    def _get_algorithm_expert_prompt(self) -> str:
        """Get system prompt for algorithm optimization."""
        return """You are an expert in algorithms and data structures with deep knowledge of:
        - Time and space complexity analysis
        - Algorithm design paradigms (divide & conquer, dynamic programming, greedy, etc.)
        - Data structure selection and optimization
        - Trade-offs between time and space
        - Practical performance considerations

        Provide detailed, actionable optimizations with complexity analysis."""

    def _get_parallelism_expert_prompt(self) -> str:
        """Get system prompt for parallelization."""
        return """You are an expert in parallel and concurrent programming with expertise in:
        - Thread-level parallelism
        - Process-level parallelism
        - Asynchronous programming
        - GPU acceleration
        - Synchronization and race condition prevention
        - Load balancing strategies

        Provide safe, efficient parallelization strategies."""

    def _get_profiling_expert_prompt(self) -> str:
        """Get system prompt for code profiling."""
        return """You are an expert in performance profiling and analysis with skills in:
        - Identifying performance bottlenecks
        - Complexity analysis
        - Memory profiling
        - CPU profiling
        - I/O analysis
        - Cache behavior analysis

        Provide detailed profiling insights with actionable recommendations."""

    def _get_optimization_expert_prompt(self) -> str:
        """Get system prompt for general optimization."""
        return """You are a code optimization expert with comprehensive knowledge of:
        - Performance optimization techniques
        - Memory optimization strategies
        - Compiler optimizations
        - Platform-specific optimizations
        - Trade-off analysis
        - Benchmark-driven development

        Provide practical, measurable optimizations with clear benefits."""

    async def _generate_suggestions(
        self,
        code: str,
        language: str | None,
        profile: OptimizationProfile,
        context: str | None,
    ) -> list[OptimizationSuggestion]:
        """Generate optimization suggestions based on profile."""
        suggestions = []

        # Build targeted prompt based on profile
        target_metrics_str = ", ".join(m.value for m in profile.target_metrics)

        prompt = f"""
        Optimize the following {language or "code"} focusing on: {target_metrics_str}

        ```{language or ""}
        {code}
        ```

        Context: {context or "General optimization"}
        Constraints: Preserve readability: {profile.preserve_readability}, Maintain compatibility: {profile.maintain_compatibility}
        Risk tolerance: {profile.risk_tolerance}

        Provide specific optimization suggestions with:
        1. Type of optimization
        2. Priority level
        3. Original vs optimized code
        4. Expected improvements
        5. Implementation complexity
        6. Potential trade-offs

        Format as JSON array of suggestions.
        """

        try:
            response = await self.client.generate(
                prompt=prompt,
                system_prompt=self._get_optimization_expert_prompt(),
                temperature=0.2,
            )

            suggestions_data = self._parse_json_response(response.content)

            if isinstance(suggestions_data, list):
                for item in suggestions_data:
                    suggestion = self._create_suggestion_from_data(item)
                    if suggestion:
                        suggestions.append(suggestion)

        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")

        return suggestions

    async def _estimate_improvements(
        self,
        original_code: str,
        suggestions: list[OptimizationSuggestion],
        language: str | None,
    ) -> list[BenchmarkResult]:
        """Estimate performance improvements from suggestions."""
        benchmarks = []

        for suggestion in suggestions:
            if suggestion.optimized_code:
                # Estimate improvement based on optimization type
                speedup = self._estimate_speedup(suggestion)
                memory_reduction = self._estimate_memory_reduction(suggestion)

                benchmark = BenchmarkResult(
                    name=suggestion.title,
                    baseline_time=1.0,  # Normalized
                    optimized_time=1.0 / speedup if speedup > 0 else 1.0,
                    speedup=speedup,
                    memory_before=100,  # Normalized
                    memory_after=int(100 * (1 - memory_reduction)),
                    memory_reduction=memory_reduction,
                )

                benchmarks.append(benchmark)

        return benchmarks

    def _estimate_speedup(self, suggestion: OptimizationSuggestion) -> float:
        """Estimate speedup from optimization suggestion."""
        # Heuristic-based estimation
        speedup_estimates = {
            OptimizationType.ALGORITHM: 2.0,
            OptimizationType.PERFORMANCE: 1.5,
            OptimizationType.CACHING: 3.0,
            OptimizationType.CONCURRENCY: 2.5,
            OptimizationType.DATABASE: 2.0,
            OptimizationType.NETWORK: 1.8,
        }

        base_speedup = speedup_estimates.get(suggestion.type, 1.2)

        # Adjust based on priority
        priority_multipliers = {
            OptimizationPriority.CRITICAL: 1.5,
            OptimizationPriority.HIGH: 1.3,
            OptimizationPriority.MEDIUM: 1.1,
            OptimizationPriority.LOW: 1.05,
            OptimizationPriority.TRIVIAL: 1.01,
        }

        return base_speedup * priority_multipliers.get(suggestion.priority, 1.0)

    def _estimate_memory_reduction(self, suggestion: OptimizationSuggestion) -> float:
        """Estimate memory reduction from optimization suggestion."""
        if suggestion.type == OptimizationType.MEMORY:
            return 0.3  # 30% reduction for memory optimizations
        elif suggestion.type == OptimizationType.ALGORITHM:
            return 0.2  # 20% for better algorithms
        elif suggestion.type == OptimizationType.CODE_SIZE:
            return 0.1  # 10% for code size optimizations
        return 0.05  # 5% for other optimizations

    def _create_implementation_plan(
        self,
        suggestions: list[OptimizationSuggestion],
    ) -> list[str]:
        """Create an ordered implementation plan."""
        plan = []

        # Sort by priority and complexity
        sorted_suggestions = sorted(
            suggestions,
            key=lambda s: (
                self._priority_to_number(s.priority),
                self._complexity_to_number(s.complexity),
            ),
        )

        for i, suggestion in enumerate(sorted_suggestions, 1):
            plan.append(
                f"{i}. [{suggestion.priority.upper()}] {suggestion.title} "
                f"(Complexity: {suggestion.complexity}, Risk: {suggestion.risk_level})"
            )

        return plan

    def _priority_to_number(self, priority: OptimizationPriority) -> int:
        """Convert priority to number for sorting."""
        priority_map = {
            OptimizationPriority.CRITICAL: 0,
            OptimizationPriority.HIGH: 1,
            OptimizationPriority.MEDIUM: 2,
            OptimizationPriority.LOW: 3,
            OptimizationPriority.TRIVIAL: 4,
        }
        return priority_map.get(priority, 5)

    def _complexity_to_number(self, complexity: str) -> int:
        """Convert complexity to number for sorting."""
        complexity_map = {
            "simple": 0,
            "moderate": 1,
            "complex": 2,
        }
        return complexity_map.get(complexity, 3)

    def _calculate_total_speedup(self, benchmarks: list[BenchmarkResult]) -> float:
        """Calculate total potential speedup."""
        if not benchmarks:
            return 1.0

        # Geometric mean of speedups
        product = 1.0
        for benchmark in benchmarks:
            product *= benchmark.speedup

        return product ** (1 / len(benchmarks))

    def _calculate_memory_savings(self, benchmarks: list[BenchmarkResult]) -> int:
        """Calculate total memory savings in bytes."""
        total_savings = 0

        for benchmark in benchmarks:
            if benchmark.memory_before and benchmark.memory_after:
                savings = benchmark.memory_before - benchmark.memory_after
                total_savings += savings

        return total_savings

    def _estimate_effort(self, suggestions: list[OptimizationSuggestion]) -> float:
        """Estimate implementation effort in hours."""
        effort = 0.0

        complexity_hours = {
            "simple": 0.5,
            "moderate": 2.0,
            "complex": 8.0,
        }

        for suggestion in suggestions:
            effort += complexity_hours.get(suggestion.complexity, 1.0)

        return effort

    def _calculate_confidence(
        self,
        suggestions: list[OptimizationSuggestion],
        benchmarks: list[BenchmarkResult],
    ) -> float:
        """Calculate confidence score for optimizations."""
        if not suggestions:
            return 0.0

        # Base confidence on number of suggestions and benchmarks
        confidence = min(1.0, len(suggestions) * 0.1 + len(benchmarks) * 0.15)

        # Adjust based on risk levels
        high_risk_count = sum(1 for s in suggestions if s.risk_level == "high")
        confidence *= 1 - high_risk_count * 0.1

        return max(0.0, min(1.0, confidence))

    def _generate_warnings(self, suggestions: list[OptimizationSuggestion]) -> list[str]:
        """Generate warnings about potential issues."""
        warnings = []

        # Check for high-risk optimizations
        high_risk = [s for s in suggestions if s.risk_level == "high"]
        if high_risk:
            warnings.append(f"{len(high_risk)} high-risk optimizations require careful testing")

        # Check for complex optimizations
        complex_opts = [s for s in suggestions if s.complexity == "complex"]
        if complex_opts:
            warnings.append(
                f"{len(complex_opts)} complex optimizations may require significant refactoring"
            )

        # Check for trade-offs
        with_tradeoffs = [s for s in suggestions if s.trade_offs]
        if with_tradeoffs:
            warnings.append(
                "Some optimizations involve trade-offs that should be carefully considered"
            )

        return warnings

    def _create_suggestion_from_data(self, data: dict[str, Any]) -> OptimizationSuggestion | None:
        """Create OptimizationSuggestion from parsed data."""
        try:
            return OptimizationSuggestion(
                type=OptimizationType(data.get("type", "performance")),
                priority=OptimizationPriority(data.get("priority", "medium")),
                title=data.get("title", "Optimization"),
                description=data.get("description", ""),
                current_code=data.get("current_code"),
                optimized_code=data.get("optimized_code"),
                file_path=data.get("file_path"),
                line_start=data.get("line_start"),
                line_end=data.get("line_end"),
                complexity=data.get("complexity", "moderate"),
                risk_level=data.get("risk_level", "medium"),
                prerequisites=data.get("prerequisites", []),
                trade_offs=data.get("trade_offs", []),
                references=data.get("references", []),
            )
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to create suggestion from data: {e}")
            return None

    def _parse_json_response(self, response: str) -> Any:
        """Parse JSON from response content."""
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response

            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response")
            return {}

    def _parse_algorithm_optimization(
        self, response: str, original_code: str
    ) -> OptimizationResult:
        """Parse algorithm optimization response."""
        data = self._parse_json_response(response)

        suggestions = []
        if "optimized_algorithm" in data:
            suggestion = OptimizationSuggestion(
                type=OptimizationType.ALGORITHM,
                priority=OptimizationPriority.HIGH,
                title="Algorithm Optimization",
                description=data.get("explanation", "Optimized algorithm"),
                current_code=original_code,
                optimized_code=data.get("optimized_algorithm"),
                trade_offs=data.get("trade_offs", []),
            )
            suggestions.append(suggestion)

        return OptimizationResult(
            suggestions=suggestions,
            benchmarks=[],
            total_potential_speedup=data.get("expected_speedup", 1.0),
            total_memory_savings=0,
            implementation_plan=["Implement optimized algorithm"],
            warnings=data.get("warnings", []),
            estimated_effort_hours=2.0,
            confidence_score=0.8,
        )

    def _parse_query_optimization(
        self, response: str, original_query: str
    ) -> OptimizationSuggestion | None:
        """Parse database query optimization response."""
        data = self._parse_json_response(response)

        if "optimized_query" in data:
            return OptimizationSuggestion(
                type=OptimizationType.DATABASE,
                priority=OptimizationPriority.HIGH,
                title="Query Optimization",
                description=data.get("explanation", "Optimized SQL query"),
                current_code=original_query,
                optimized_code=data.get("optimized_query"),
                trade_offs=data.get("trade_offs", []),
            )
        return None

    def _parse_parallelization(self, response: str, original_code: str) -> OptimizationResult:
        """Parse parallelization optimization response."""
        data = self._parse_json_response(response)

        suggestions = []
        if "parallel_code" in data:
            suggestion = OptimizationSuggestion(
                type=OptimizationType.CONCURRENCY,
                priority=OptimizationPriority.HIGH,
                title="Parallelization",
                description=data.get("strategy", "Parallel execution"),
                current_code=original_code,
                optimized_code=data.get("parallel_code"),
                trade_offs=["Increased complexity", "Thread synchronization overhead"],
            )
            suggestions.append(suggestion)

        return OptimizationResult(
            suggestions=suggestions,
            benchmarks=[],
            total_potential_speedup=data.get("expected_speedup", 2.0),
            total_memory_savings=0,
            implementation_plan=data.get("implementation_steps", []),
            warnings=data.get("warnings", []),
            estimated_effort_hours=4.0,
            confidence_score=0.7,
        )

    def _parse_memory_optimization(self, response: str, original_code: str) -> OptimizationResult:
        """Parse memory optimization response."""
        data = self._parse_json_response(response)

        suggestions = []
        if "optimized_code" in data:
            suggestion = OptimizationSuggestion(
                type=OptimizationType.MEMORY,
                priority=OptimizationPriority.MEDIUM,
                title="Memory Optimization",
                description=data.get("description", "Memory usage optimization"),
                current_code=original_code,
                optimized_code=data.get("optimized_code"),
                trade_offs=data.get("trade_offs", []),
            )
            suggestions.append(suggestion)

        memory_savings = data.get("memory_savings_bytes", 0)

        return OptimizationResult(
            suggestions=suggestions,
            benchmarks=[],
            total_potential_speedup=1.0,
            total_memory_savings=memory_savings,
            implementation_plan=data.get("steps", []),
            warnings=data.get("warnings", []),
            estimated_effort_hours=2.0,
            confidence_score=0.75,
        )

    def _parse_comprehensive_optimization(
        self,
        response: str,
        original_code: str,
        analysis: dict[str, Any],
    ) -> OptimizationResult:
        """Parse comprehensive optimization response."""
        data = self._parse_json_response(response)

        suggestions = []
        if "optimizations" in data:
            for opt in data["optimizations"]:
                suggestion = self._create_suggestion_from_data(opt)
                if suggestion:
                    suggestions.append(suggestion)

        return OptimizationResult(
            suggestions=suggestions,
            benchmarks=[],
            total_potential_speedup=analysis.get("potential_speedup", 1.5),
            total_memory_savings=analysis.get("memory_savings", 0),
            implementation_plan=data.get("implementation_plan", []),
            warnings=analysis.get("warnings", []),
            estimated_effort_hours=self._estimate_effort(suggestions),
            confidence_score=0.8,
        )

    def _parse_caching_suggestions(self, response: str) -> list[OptimizationSuggestion]:
        """Parse caching strategy suggestions."""
        data = self._parse_json_response(response)
        suggestions = []

        if isinstance(data, list):
            for item in data:
                suggestion = OptimizationSuggestion(
                    type=OptimizationType.CACHING,
                    priority=OptimizationPriority(item.get("priority", "medium")),
                    title=item.get("title", "Caching Strategy"),
                    description=item.get("description", ""),
                    current_code=item.get("current_code"),
                    optimized_code=item.get("cached_implementation"),
                    trade_offs=["Memory usage for cache", "Cache invalidation complexity"],
                )
                suggestions.append(suggestion)

        return suggestions

    def _empty_result(self) -> OptimizationResult:
        """Return an empty optimization result."""
        return OptimizationResult(
            suggestions=[],
            benchmarks=[],
            total_potential_speedup=1.0,
            total_memory_savings=0,
            implementation_plan=[],
            warnings=[],
            estimated_effort_hours=0,
            confidence_score=0,
        )
