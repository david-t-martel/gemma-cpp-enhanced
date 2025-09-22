"""Gemini-based code analysis tool for deep code understanding and insights.

This module provides advanced code analysis capabilities including complexity analysis,
dependency mapping, design pattern detection, and architectural insights.
"""

import ast
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


class ComplexityLevel(str, Enum):
    """Code complexity levels."""

    SIMPLE = "simple"  # Cyclomatic complexity < 5
    MODERATE = "moderate"  # Cyclomatic complexity 5-10
    COMPLEX = "complex"  # Cyclomatic complexity 11-20
    VERY_COMPLEX = "very_complex"  # Cyclomatic complexity > 20


class DesignPattern(str, Enum):
    """Common design patterns to detect."""

    SINGLETON = "singleton"
    FACTORY = "factory"
    OBSERVER = "observer"
    STRATEGY = "strategy"
    DECORATOR = "decorator"
    ADAPTER = "adapter"
    FACADE = "facade"
    PROXY = "proxy"
    COMMAND = "command"
    ITERATOR = "iterator"
    TEMPLATE_METHOD = "template_method"
    BUILDER = "builder"
    PROTOTYPE = "prototype"
    CHAIN_OF_RESPONSIBILITY = "chain_of_responsibility"
    MEDIATOR = "mediator"
    MEMENTO = "memento"
    STATE = "state"
    VISITOR = "visitor"


class CodeSmell(str, Enum):
    """Types of code smells to detect."""

    LONG_METHOD = "long_method"
    LARGE_CLASS = "large_class"
    LONG_PARAMETER_LIST = "long_parameter_list"
    DUPLICATE_CODE = "duplicate_code"
    DEAD_CODE = "dead_code"
    FEATURE_ENVY = "feature_envy"
    DATA_CLUMPS = "data_clumps"
    PRIMITIVE_OBSESSION = "primitive_obsession"
    SWITCH_STATEMENTS = "switch_statements"
    PARALLEL_INHERITANCE = "parallel_inheritance"
    LAZY_CLASS = "lazy_class"
    SPECULATIVE_GENERALITY = "speculative_generality"
    MESSAGE_CHAINS = "message_chains"
    MIDDLE_MAN = "middle_man"
    INAPPROPRIATE_INTIMACY = "inappropriate_intimacy"
    INCOMPLETE_LIBRARY_CLASS = "incomplete_library_class"
    REFUSED_BEQUEST = "refused_bequest"
    COMMENTS = "excessive_comments"


@dataclass
class FunctionAnalysis:
    """Analysis results for a single function."""

    name: str
    file_path: str
    line_start: int
    line_end: int
    complexity: int
    complexity_level: ComplexityLevel
    parameters: list[str]
    returns: str | None
    calls: list[str]  # Functions this function calls
    called_by: list[str] = field(default_factory=list)  # Functions that call this
    docstring: str | None = None
    test_coverage: float | None = None
    code_smells: list[CodeSmell] = field(default_factory=list)


@dataclass
class ClassAnalysis:
    """Analysis results for a single class."""

    name: str
    file_path: str
    line_start: int
    line_end: int
    methods: list[FunctionAnalysis]
    attributes: list[str]
    base_classes: list[str]
    metaclass: str | None = None
    decorators: list[str] = field(default_factory=list)
    design_patterns: list[DesignPattern] = field(default_factory=list)
    cohesion_score: float = 0.0  # 0-1, higher is better
    coupling_score: float = 0.0  # 0-1, lower is better
    code_smells: list[CodeSmell] = field(default_factory=list)


@dataclass
class ModuleAnalysis:
    """Analysis results for a single module/file."""

    file_path: str
    imports: list[str]
    exports: list[str]
    classes: list[ClassAnalysis]
    functions: list[FunctionAnalysis]
    global_variables: list[str]
    dependencies: set[str] = field(default_factory=set)
    dependents: set[str] = field(default_factory=set)
    complexity_score: float = 0.0
    maintainability_index: float = 0.0
    lines_of_code: int = 0
    comment_ratio: float = 0.0


@dataclass
class ArchitecturalInsight:
    """High-level architectural insights."""

    layer: str  # presentation, business, data, infrastructure
    responsibility: str
    dependencies: list[str]
    violations: list[str]  # Architectural violations
    suggestions: list[str]


@dataclass
class CodeAnalysisResult:
    """Complete result of code analysis."""

    modules: list[ModuleAnalysis]
    total_complexity: float
    average_complexity: float
    design_patterns_found: dict[DesignPattern, list[str]]  # Pattern -> locations
    code_smells_found: dict[CodeSmell, list[str]]  # Smell -> locations
    architectural_insights: list[ArchitecturalInsight]
    dependency_graph: dict[str, list[str]]  # Module -> dependencies
    hotspots: list[tuple[str, str, float]]  # (file, function, complexity)
    recommendations: list[str]
    metrics: dict[str, Any]


class GeminiCodeAnalyzer:
    """Advanced code analyzer powered by Gemini AI."""

    def __init__(
        self,
        client: GeminiClient | None = None,
        config: GeminiConfig | None = None,
    ):
        """Initialize the code analyzer.

        Args:
            client: Existing Gemini client to use
            config: Configuration for creating a new client
        """
        self.client = client or GeminiClient(
            config
            or GeminiConfig(
                model=GeminiModel.GEMINI_2_5_PRO,
                temperature=0.2,  # Very low for consistent analysis
                max_output_tokens=8192,
            )
        )

    async def analyze_code(
        self,
        code: str,
        file_path: str | None = None,
        language: str | None = None,
        deep_analysis: bool = True,
    ) -> ModuleAnalysis:
        """Analyze a single code file.

        Args:
            code: The code to analyze
            file_path: Optional path to the file
            language: Programming language
            deep_analysis: Whether to perform deep AI-powered analysis

        Returns:
            ModuleAnalysis with detailed insights
        """
        # Start with static analysis
        module_analysis = self._static_analysis(code, file_path, language)

        if deep_analysis:
            # Enhance with AI analysis
            ai_insights = await self._ai_analysis(code, file_path, language)
            module_analysis = self._merge_analysis(module_analysis, ai_insights)

        return module_analysis

    async def analyze_project(
        self,
        project_path: Path,
        include_patterns: set[str] | None = None,
        exclude_patterns: set[str] | None = None,
        max_files: int = 100,
        parallel_analysis: int = 5,
    ) -> CodeAnalysisResult:
        """Analyze an entire project.

        Args:
            project_path: Path to the project directory
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            max_files: Maximum files to analyze
            parallel_analysis: Number of parallel analysis tasks

        Returns:
            Complete CodeAnalysisResult
        """
        project_path = Path(project_path)

        # Default patterns
        if include_patterns is None:
            include_patterns = {".py", ".js", ".ts", ".java", ".cpp", ".go"}

        if exclude_patterns is None:
            exclude_patterns = {"__pycache__", "node_modules", ".git", "venv", "dist", "build"}

        # Collect files
        files = self._collect_files(project_path, include_patterns, exclude_patterns, max_files)

        # Analyze files in parallel
        semaphore = asyncio.Semaphore(parallel_analysis)

        async def analyze_file(file_path: Path) -> ModuleAnalysis | None:
            async with semaphore:
                try:
                    code = file_path.read_text(encoding="utf-8", errors="ignore")
                    return await self.analyze_code(
                        code=code,
                        file_path=str(file_path),
                        language=self._detect_language(str(file_path)),
                    )
                except Exception as e:
                    logger.error(f"Failed to analyze {file_path}: {e}")
                    return None

        # Execute analysis
        tasks = [analyze_file(f) for f in files]
        results = await asyncio.gather(*tasks)

        # Filter out None results
        modules = [r for r in results if r is not None]

        # Build dependency graph
        dependency_graph = self._build_dependency_graph(modules)

        # Detect patterns and smells across project
        patterns = await self._detect_project_patterns(modules)
        smells = self._detect_project_smells(modules)

        # Get architectural insights
        insights = await self._analyze_architecture(modules, dependency_graph)

        # Calculate metrics
        metrics = self._calculate_project_metrics(modules)

        # Find complexity hotspots
        hotspots = self._find_hotspots(modules)

        # Generate recommendations
        recommendations = await self._generate_recommendations(
            modules, patterns, smells, insights, metrics
        )

        return CodeAnalysisResult(
            modules=modules,
            total_complexity=metrics["total_complexity"],
            average_complexity=metrics["average_complexity"],
            design_patterns_found=patterns,
            code_smells_found=smells,
            architectural_insights=insights,
            dependency_graph=dependency_graph,
            hotspots=hotspots,
            recommendations=recommendations,
            metrics=metrics,
        )

    async def analyze_complexity(
        self,
        code: str,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Analyze code complexity in detail.

        Args:
            code: Code to analyze
            language: Programming language

        Returns:
            Detailed complexity metrics
        """
        prompt = f"""
        Analyze the complexity of the following {language or "code"}:

        ```{language or ""}
        {code}
        ```

        Provide a detailed complexity analysis including:
        1. Cyclomatic complexity for each function/method
        2. Cognitive complexity scores
        3. Nesting depth analysis
        4. Coupling and cohesion metrics
        5. Suggestions for reducing complexity

        Format as JSON with structure:
        {{
            "functions": [
                {{
                    "name": "function_name",
                    "cyclomatic_complexity": 5,
                    "cognitive_complexity": 8,
                    "nesting_depth": 3,
                    "parameters": 4,
                    "lines": 25,
                    "complexity_level": "moderate"
                }}
            ],
            "overall_complexity": "moderate",
            "suggestions": ["..."]
        }}
        """

        try:
            response = await self.client.generate(
                prompt=prompt,
                system_prompt="You are an expert in code complexity analysis and software metrics.",
                temperature=0.1,
            )

            return self._parse_json_response(response.content)

        except Exception as e:
            logger.error(f"Complexity analysis failed: {e}")
            return {}

    async def detect_patterns(
        self,
        code: str,
        language: str | None = None,
    ) -> list[DesignPattern]:
        """Detect design patterns in code.

        Args:
            code: Code to analyze
            language: Programming language

        Returns:
            List of detected design patterns
        """
        prompt = f"""
        Identify design patterns in the following {language or "code"}:

        ```{language or ""}
        {code}
        ```

        Look for common patterns like:
        - Creational: Singleton, Factory, Builder, Prototype
        - Structural: Adapter, Decorator, Facade, Proxy
        - Behavioral: Observer, Strategy, Command, Iterator

        Return a JSON array of detected patterns with explanations:
        [
            {{
                "pattern": "pattern_name",
                "location": "class or function name",
                "confidence": 0.9,
                "explanation": "why this pattern was detected"
            }}
        ]
        """

        try:
            response = await self.client.generate(
                prompt=prompt,
                system_prompt="You are an expert in design patterns and software architecture.",
                temperature=0.1,
            )

            patterns_data = self._parse_json_response(response.content)
            patterns = []

            if isinstance(patterns_data, list):
                for item in patterns_data:
                    try:
                        pattern = DesignPattern(item.get("pattern", "").lower().replace(" ", "_"))
                        patterns.append(pattern)
                    except ValueError:
                        continue

            return patterns

        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return []

    async def suggest_refactoring(
        self,
        code: str,
        issues: list[str] | None = None,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """Suggest refactoring improvements.

        Args:
            code: Code to refactor
            issues: Specific issues to address
            language: Programming language

        Returns:
            List of refactoring suggestions
        """
        issues_text = (
            "\n".join(f"- {issue}" for issue in issues) if issues else "any issues you find"
        )

        prompt = f"""
        Suggest refactoring improvements for the following {language or "code"}:

        ```{language or ""}
        {code}
        ```

        Focus on addressing:
        {issues_text}

        Provide specific, actionable refactoring suggestions with:
        1. What to change
        2. Why it should be changed
        3. How to change it (with code examples)
        4. Expected benefits

        Format as JSON array.
        """

        try:
            response = await self.client.generate(
                prompt=prompt,
                system_prompt="You are an expert in code refactoring and clean code principles.",
                temperature=0.3,
            )

            return self._parse_json_response(response.content)

        except Exception as e:
            logger.error(f"Refactoring suggestions failed: {e}")
            return []

    def _static_analysis(
        self,
        code: str,
        file_path: str | None,
        language: str | None,
    ) -> ModuleAnalysis:
        """Perform static code analysis."""
        module = ModuleAnalysis(
            file_path=file_path or "unknown",
            imports=[],
            exports=[],
            classes=[],
            functions=[],
            global_variables=[],
            lines_of_code=len(code.split("\n")),
        )

        # Language-specific analysis
        if language == "python" or (file_path and file_path.endswith(".py")):
            module = self._analyze_python(code, module)

        # Calculate basic metrics
        module.comment_ratio = self._calculate_comment_ratio(code)

        return module

    def _analyze_python(self, code: str, module: ModuleAnalysis) -> ModuleAnalysis:
        """Analyze Python code using AST."""
        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module.imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    for alias in node.names:
                        module.imports.append(f"{module_name}.{alias.name}")
                elif isinstance(node, ast.FunctionDef):
                    func_analysis = self._analyze_python_function(node)
                    module.functions.append(func_analysis)
                elif isinstance(node, ast.ClassDef):
                    class_analysis = self._analyze_python_class(node)
                    module.classes.append(class_analysis)
                elif isinstance(node, ast.Assign):
                    # Global variables
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            module.global_variables.append(target.id)

        except SyntaxError as e:
            logger.warning(f"Failed to parse Python code: {e}")

        return module

    def _analyze_python_function(self, node: ast.FunctionDef) -> FunctionAnalysis:
        """Analyze a Python function node."""
        # Calculate cyclomatic complexity
        complexity = self._calculate_cyclomatic_complexity(node)

        # Determine complexity level
        if complexity < 5:
            level = ComplexityLevel.SIMPLE
        elif complexity < 11:
            level = ComplexityLevel.MODERATE
        elif complexity < 21:
            level = ComplexityLevel.COMPLEX
        else:
            level = ComplexityLevel.VERY_COMPLEX

        # Extract parameters
        parameters = [arg.arg for arg in node.args.args]

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Find function calls
        calls = []
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call):
                if isinstance(subnode.func, ast.Name):
                    calls.append(subnode.func.id)
                elif isinstance(subnode.func, ast.Attribute):
                    calls.append(subnode.func.attr)

        return FunctionAnalysis(
            name=node.name,
            file_path="",
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            complexity=complexity,
            complexity_level=level,
            parameters=parameters,
            returns=None,  # Would need type hints analysis
            calls=calls,
            docstring=docstring,
        )

    def _analyze_python_class(self, node: ast.ClassDef) -> ClassAnalysis:
        """Analyze a Python class node."""
        methods = []
        attributes = []

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(self._analyze_python_function(item))
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)

        # Extract base classes
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(base.attr)

        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(decorator.attr)

        return ClassAnalysis(
            name=node.name,
            file_path="",
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            methods=methods,
            attributes=attributes,
            base_classes=base_classes,
            decorators=decorators,
        )

    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of an AST node."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _calculate_comment_ratio(self, code: str) -> float:
        """Calculate the ratio of comment lines to total lines."""
        lines = code.split("\n")
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
        return comment_lines / len(lines) if lines else 0.0

    async def _ai_analysis(
        self,
        code: str,
        file_path: str | None,
        language: str | None,
    ) -> dict[str, Any]:
        """Perform AI-powered code analysis."""
        prompt = f"""
        Perform a comprehensive analysis of the following {language or "code"}:

        File: {file_path or "unknown"}

        ```{language or ""}
        {code}
        ```

        Analyze:
        1. Code structure and organization
        2. Design patterns used
        3. Code smells and anti-patterns
        4. Complexity hotspots
        5. Architectural concerns
        6. Testing considerations
        7. Performance implications

        Return a detailed JSON analysis.
        """

        try:
            response = await self.client.generate(
                prompt=prompt,
                system_prompt="You are an expert code analyst with deep knowledge of software engineering.",
                temperature=0.2,
            )

            return self._parse_json_response(response.content)

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {}

    def _merge_analysis(
        self,
        static: ModuleAnalysis,
        ai_insights: dict[str, Any],
    ) -> ModuleAnalysis:
        """Merge static and AI analysis results."""
        # Add AI insights to static analysis
        if "complexity_score" in ai_insights:
            static.complexity_score = ai_insights["complexity_score"]

        if "maintainability_index" in ai_insights:
            static.maintainability_index = ai_insights["maintainability_index"]

        # Merge additional insights as needed
        return static

    def _build_dependency_graph(
        self,
        modules: list[ModuleAnalysis],
    ) -> dict[str, list[str]]:
        """Build a dependency graph from module analyses."""
        graph = {}

        for module in modules:
            dependencies = []
            for imp in module.imports:
                # Extract module name from import
                module_name = imp.split(".")[0]
                if module_name not in ["os", "sys", "json", "re"]:  # Skip stdlib
                    dependencies.append(module_name)

            graph[module.file_path] = dependencies

        return graph

    async def _detect_project_patterns(
        self,
        modules: list[ModuleAnalysis],
    ) -> dict[DesignPattern, list[str]]:
        """Detect design patterns across the project."""
        patterns = {}

        for module in modules:
            for class_analysis in module.classes:
                # Simple heuristics for pattern detection
                if "singleton" in class_analysis.name.lower():
                    patterns.setdefault(DesignPattern.SINGLETON, []).append(
                        f"{module.file_path}:{class_analysis.name}"
                    )

                if "factory" in class_analysis.name.lower():
                    patterns.setdefault(DesignPattern.FACTORY, []).append(
                        f"{module.file_path}:{class_analysis.name}"
                    )

                if (
                    "observer" in class_analysis.name.lower()
                    or "listener" in class_analysis.name.lower()
                ):
                    patterns.setdefault(DesignPattern.OBSERVER, []).append(
                        f"{module.file_path}:{class_analysis.name}"
                    )

        return patterns

    def _detect_project_smells(
        self,
        modules: list[ModuleAnalysis],
    ) -> dict[CodeSmell, list[str]]:
        """Detect code smells across the project."""
        smells = {}

        for module in modules:
            # Check for long methods
            for func in module.functions:
                if func.line_end - func.line_start > 50:
                    smells.setdefault(CodeSmell.LONG_METHOD, []).append(
                        f"{module.file_path}:{func.name}"
                    )

                if len(func.parameters) > 5:
                    smells.setdefault(CodeSmell.LONG_PARAMETER_LIST, []).append(
                        f"{module.file_path}:{func.name}"
                    )

            # Check for large classes
            for class_analysis in module.classes:
                if len(class_analysis.methods) > 20:
                    smells.setdefault(CodeSmell.LARGE_CLASS, []).append(
                        f"{module.file_path}:{class_analysis.name}"
                    )

        return smells

    async def _analyze_architecture(
        self,
        modules: list[ModuleAnalysis],
        dependency_graph: dict[str, list[str]],
    ) -> list[ArchitecturalInsight]:
        """Analyze architectural patterns and violations."""
        insights = []

        # Detect layers based on file paths
        layers = {
            "presentation": [],
            "business": [],
            "data": [],
            "infrastructure": [],
        }

        for module in modules:
            path = module.file_path.lower()
            if "view" in path or "ui" in path or "presentation" in path:
                layers["presentation"].append(module.file_path)
            elif "service" in path or "business" in path or "domain" in path:
                layers["business"].append(module.file_path)
            elif "data" in path or "repository" in path or "dao" in path:
                layers["data"].append(module.file_path)
            elif "infra" in path or "util" in path or "helper" in path:
                layers["infrastructure"].append(module.file_path)

        # Check for violations
        for layer, files in layers.items():
            if files:
                insight = ArchitecturalInsight(
                    layer=layer,
                    responsibility=f"Handle {layer} concerns",
                    dependencies=[],
                    violations=[],
                    suggestions=[],
                )

                # Check dependencies
                for file in files:
                    if file in dependency_graph:
                        insight.dependencies.extend(dependency_graph[file])

                insights.append(insight)

        return insights

    def _calculate_project_metrics(
        self,
        modules: list[ModuleAnalysis],
    ) -> dict[str, Any]:
        """Calculate project-wide metrics."""
        total_functions = sum(len(m.functions) for m in modules)
        total_classes = sum(len(m.classes) for m in modules)
        total_complexity = 0
        total_loc = sum(m.lines_of_code for m in modules)

        for module in modules:
            for func in module.functions:
                total_complexity += func.complexity

        return {
            "total_modules": len(modules),
            "total_functions": total_functions,
            "total_classes": total_classes,
            "total_complexity": total_complexity,
            "average_complexity": total_complexity / total_functions if total_functions > 0 else 0,
            "total_lines_of_code": total_loc,
            "average_module_size": total_loc / len(modules) if modules else 0,
        }

    def _find_hotspots(
        self,
        modules: list[ModuleAnalysis],
        top_n: int = 10,
    ) -> list[tuple[str, str, float]]:
        """Find complexity hotspots in the codebase."""
        hotspots = []

        for module in modules:
            for func in module.functions:
                hotspots.append((module.file_path, func.name, func.complexity))

        # Sort by complexity and return top N
        hotspots.sort(key=lambda x: x[2], reverse=True)
        return hotspots[:top_n]

    async def _generate_recommendations(
        self,
        modules: list[ModuleAnalysis],
        patterns: dict[DesignPattern, list[str]],
        smells: dict[CodeSmell, list[str]],
        insights: list[ArchitecturalInsight],
        metrics: dict[str, Any],
    ) -> list[str]:
        """Generate project-wide recommendations."""
        recommendations = []

        # Complexity recommendations
        if metrics["average_complexity"] > 10:
            recommendations.append(
                "Consider refactoring complex functions to reduce average complexity"
            )

        # Code smell recommendations
        if CodeSmell.LONG_METHOD in smells:
            recommendations.append(f"Refactor {len(smells[CodeSmell.LONG_METHOD])} long methods")

        if CodeSmell.LARGE_CLASS in smells:
            recommendations.append(f"Break down {len(smells[CodeSmell.LARGE_CLASS])} large classes")

        # Pattern recommendations
        if not patterns:
            recommendations.append("Consider applying design patterns for better structure")

        # Architecture recommendations
        for insight in insights:
            if insight.violations:
                recommendations.append(f"Fix architectural violations in {insight.layer} layer")

        return recommendations

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

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext_to_lang = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
        }

        path = Path(file_path)
        return ext_to_lang.get(path.suffix.lower(), "text")

    def _collect_files(
        self,
        project_path: Path,
        include_patterns: set[str],
        exclude_patterns: set[str],
        max_files: int,
    ) -> list[Path]:
        """Collect files for analysis."""
        files = []

        for file_path in project_path.rglob("*"):
            if len(files) >= max_files:
                break

            if file_path.is_dir():
                continue

            # Check exclusions
            if any(pattern in str(file_path) for pattern in exclude_patterns):
                continue

            # Check inclusions
            if file_path.suffix in include_patterns:
                files.append(file_path)

        return files
