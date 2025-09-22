"""Gemini-based code review tool for automated code quality analysis.

This module provides intelligent code review capabilities using Gemini AI,
offering insights on code quality, best practices, and potential improvements.
"""

import asyncio
import json
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from .client import GeminiClient
from .client import GeminiConfig
from .client import GeminiModel

# Configure logging


class ReviewSeverity(str, Enum):
    """Severity levels for code review findings."""

    CRITICAL = "critical"  # Security issues, data loss risks
    HIGH = "high"  # Bugs, performance issues
    MEDIUM = "medium"  # Code quality, maintainability
    LOW = "low"  # Style, minor improvements
    INFO = "info"  # Suggestions, best practices


class ReviewCategory(str, Enum):
    """Categories of code review findings."""

    SECURITY = "security"
    PERFORMANCE = "performance"
    BUG = "bug"
    CODE_QUALITY = "code_quality"
    MAINTAINABILITY = "maintainability"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    ARCHITECTURE = "architecture"


@dataclass
class CodeIssue:
    """Represents a single code issue found during review."""

    severity: ReviewSeverity
    category: ReviewCategory
    message: str
    file_path: str | None = None
    line_number: int | None = None
    line_end: int | None = None
    code_snippet: str | None = None
    suggestion: str | None = None
    explanation: str | None = None
    references: list[str] = field(default_factory=list)


@dataclass
class ReviewMetrics:
    """Metrics collected during code review."""

    total_lines: int = 0
    reviewed_lines: int = 0
    total_files: int = 0
    reviewed_files: int = 0
    issues_by_severity: dict[ReviewSeverity, int] = field(default_factory=dict)
    issues_by_category: dict[ReviewCategory, int] = field(default_factory=dict)
    code_coverage_estimate: float = 0.0
    complexity_score: float = 0.0
    maintainability_index: float = 0.0


@dataclass
class CodeReviewResult:
    """Complete result of a code review."""

    issues: list[CodeIssue]
    metrics: ReviewMetrics
    summary: str
    overall_score: float  # 0-100
    recommendations: list[str]
    reviewed_files: list[str]
    skipped_files: list[str] = field(default_factory=list)
    error_messages: list[str] = field(default_factory=list)


class GeminiCodeReviewer:
    """Automated code reviewer powered by Gemini AI."""

    def __init__(
        self,
        client: GeminiClient | None = None,
        config: GeminiConfig | None = None,
    ):
        """Initialize the code reviewer.

        Args:
            client: Existing Gemini client to use
            config: Configuration for creating a new client
        """
        self.client = client or GeminiClient(
            config
            or GeminiConfig(
                model=GeminiModel.GEMINI_2_5_PRO,  # Use Pro model for better code understanding
                temperature=0.3,  # Lower temperature for more consistent reviews
                max_output_tokens=8192,
            )
        )

        # File patterns to review
        self.supported_extensions = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".cs",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".r",
            ".m",
            ".h",
            ".hpp",
            ".cc",
            ".cxx",
            ".vue",
            ".svelte",
        }

        # Patterns to exclude from review
        self.exclude_patterns = {
            "__pycache__",
            "node_modules",
            ".git",
            ".venv",
            "venv",
            "dist",
            "build",
            ".pytest_cache",
            ".mypy_cache",
            "coverage",
        }

    async def review_code(
        self,
        code: str,
        file_path: str | None = None,
        language: str | None = None,
        context: str | None = None,
        focus_areas: list[ReviewCategory] | None = None,
    ) -> CodeReviewResult:
        """Review a single code file or snippet.

        Args:
            code: The code to review
            file_path: Optional path to the file being reviewed
            language: Programming language (auto-detected if not provided)
            context: Additional context about the code
            focus_areas: Specific areas to focus the review on

        Returns:
            CodeReviewResult with findings and recommendations
        """
        # Auto-detect language if not provided
        if not language and file_path:
            language = self._detect_language(file_path)

        # Build the review prompt
        prompt = self._build_review_prompt(code, language, file_path, context, focus_areas)

        try:
            # Generate review using Gemini
            response = await self.client.generate(
                prompt=prompt,
                system_prompt=self._get_system_prompt(),
                temperature=0.3,
                max_output_tokens=8192,
            )

            # Parse the response
            review_result = self._parse_review_response(response.content, code, file_path)

            # Calculate metrics
            review_result.metrics = self._calculate_metrics(code, review_result.issues)

            return review_result

        except Exception as e:
            logger.error(f"Code review failed: {e}")
            return CodeReviewResult(
                issues=[],
                metrics=ReviewMetrics(),
                summary=f"Review failed: {e!s}",
                overall_score=0.0,
                recommendations=[],
                reviewed_files=[file_path] if file_path else [],
                error_messages=[str(e)],
            )

    async def review_project(
        self,
        project_path: Path,
        include_patterns: set[str] | None = None,
        exclude_patterns: set[str] | None = None,
        max_files: int = 50,
        parallel_reviews: int = 5,
    ) -> CodeReviewResult:
        """Review an entire project directory.

        Args:
            project_path: Path to the project directory
            include_patterns: Patterns to include (overrides defaults)
            exclude_patterns: Additional patterns to exclude
            max_files: Maximum number of files to review
            parallel_reviews: Number of parallel review tasks

        Returns:
            Aggregated CodeReviewResult for the entire project
        """
        project_path = Path(project_path)
        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")

        # Collect files to review
        files_to_review = self._collect_files(
            project_path,
            include_patterns or self.supported_extensions,
            exclude_patterns or set(),
            max_files,
        )

        # Review files in parallel batches
        all_issues = []
        reviewed_files = []
        skipped_files = []
        error_messages = []

        # Create review tasks
        semaphore = asyncio.Semaphore(parallel_reviews)

        async def review_file(file_path: Path) -> CodeReviewResult | None:
            async with semaphore:
                try:
                    code = file_path.read_text(encoding="utf-8", errors="ignore")
                    return await self.review_code(
                        code=code,
                        file_path=str(file_path),
                        language=self._detect_language(str(file_path)),
                    )
                except Exception as e:
                    logger.error(f"Failed to review {file_path}: {e}")
                    error_messages.append(f"{file_path}: {e}")
                    skipped_files.append(str(file_path))
                    return None

        # Execute reviews
        tasks = [review_file(f) for f in files_to_review]
        results = await asyncio.gather(*tasks)

        # Aggregate results
        for file_path, result in zip(files_to_review, results, strict=False):
            if result:
                all_issues.extend(result.issues)
                reviewed_files.append(str(file_path))
            else:
                skipped_files.append(str(file_path))

        # Generate project summary
        summary = await self._generate_project_summary(all_issues, reviewed_files)

        # Calculate overall metrics
        metrics = self._aggregate_metrics(all_issues, reviewed_files, skipped_files)

        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues, metrics)

        # Calculate overall score
        overall_score = self._calculate_overall_score(metrics, all_issues)

        return CodeReviewResult(
            issues=all_issues,
            metrics=metrics,
            summary=summary,
            overall_score=overall_score,
            recommendations=recommendations,
            reviewed_files=reviewed_files,
            skipped_files=skipped_files,
            error_messages=error_messages,
        )

    async def review_diff(
        self,
        diff_content: str,
        base_branch: str | None = None,
        target_branch: str | None = None,
    ) -> CodeReviewResult:
        """Review code changes from a diff or patch.

        Args:
            diff_content: Git diff or patch content
            base_branch: Optional base branch name
            target_branch: Optional target branch name

        Returns:
            CodeReviewResult for the changes
        """
        prompt = f"""
        Review the following code diff and identify any issues, improvements, or concerns:

        Base Branch: {base_branch or "unknown"}
        Target Branch: {target_branch or "unknown"}

        Diff Content:
        ```diff
        {diff_content}
        ```

        Focus on:
        1. New bugs or issues introduced
        2. Security vulnerabilities
        3. Performance regressions
        4. Breaking changes
        5. Missing tests or documentation
        """

        try:
            response = await self.client.generate(
                prompt=prompt,
                system_prompt=self._get_system_prompt(),
                temperature=0.3,
            )

            return self._parse_review_response(response.content, diff_content, "diff")

        except Exception as e:
            logger.error(f"Diff review failed: {e}")
            return CodeReviewResult(
                issues=[],
                metrics=ReviewMetrics(),
                summary=f"Diff review failed: {e!s}",
                overall_score=0.0,
                recommendations=[],
                reviewed_files=["diff"],
                error_messages=[str(e)],
            )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for code review."""
        return """You are an expert code reviewer with deep knowledge of software engineering best practices,
        security, performance optimization, and clean code principles. Your reviews should be:

        1. Constructive and actionable
        2. Prioritized by severity (critical > high > medium > low > info)
        3. Specific with line numbers and code snippets when possible
        4. Include concrete suggestions for fixes
        5. Reference industry standards and best practices

        Format your response as a structured JSON with the following schema:
        {
            "issues": [
                {
                    "severity": "critical|high|medium|low|info",
                    "category": "security|performance|bug|code_quality|maintainability|style|documentation|testing|architecture",
                    "message": "Clear description of the issue",
                    "line_number": 123,
                    "line_end": 125,
                    "code_snippet": "The problematic code",
                    "suggestion": "How to fix it",
                    "explanation": "Why this is an issue",
                    "references": ["Link to documentation or standard"]
                }
            ],
            "summary": "Overall assessment of the code",
            "recommendations": ["High-level recommendations for improvement"],
            "positive_aspects": ["What was done well"]
        }
        """

    def _build_review_prompt(
        self,
        code: str,
        language: str | None,
        file_path: str | None,
        context: str | None,
        focus_areas: list[ReviewCategory] | None,
    ) -> str:
        """Build the prompt for code review."""
        prompt_parts = []

        # Add file and language context
        if file_path:
            prompt_parts.append(f"File: {file_path}")
        if language:
            prompt_parts.append(f"Language: {language}")

        # Add context if provided
        if context:
            prompt_parts.append(f"Context: {context}")

        # Add focus areas
        if focus_areas:
            areas_str = ", ".join(area.value for area in focus_areas)
            prompt_parts.append(f"Focus on: {areas_str}")

        # Add the code
        prompt_parts.append(f"\nCode to review:\n```{language or ''}\n{code}\n```")

        # Add review instructions
        prompt_parts.append(
            "\nPlease provide a comprehensive code review following the specified format."
        )

        return "\n".join(prompt_parts)

    def _parse_review_response(
        self,
        response_content: str,
        original_code: str,
        file_path: str | None,
    ) -> CodeReviewResult:
        """Parse the review response from Gemini."""
        try:
            # Try to parse as JSON
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_str = response_content[json_start:json_end].strip()
            else:
                # Try to find JSON object in response
                json_str = response_content

            data = json.loads(json_str)

            # Parse issues
            issues = []
            for issue_data in data.get("issues", []):
                issues.append(
                    CodeIssue(
                        severity=ReviewSeverity(issue_data.get("severity", "info")),
                        category=ReviewCategory(issue_data.get("category", "code_quality")),
                        message=issue_data.get("message", ""),
                        file_path=file_path,
                        line_number=issue_data.get("line_number"),
                        line_end=issue_data.get("line_end"),
                        code_snippet=issue_data.get("code_snippet"),
                        suggestion=issue_data.get("suggestion"),
                        explanation=issue_data.get("explanation"),
                        references=issue_data.get("references", []),
                    )
                )

            return CodeReviewResult(
                issues=issues,
                metrics=ReviewMetrics(),
                summary=data.get("summary", "Code review completed"),
                overall_score=0.0,  # Will be calculated later
                recommendations=data.get("recommendations", []),
                reviewed_files=[file_path] if file_path else [],
            )

        except json.JSONDecodeError:
            # Fallback to text parsing
            logger.warning("Failed to parse JSON response, using text parsing")
            return self._parse_text_response(response_content, file_path)

    def _parse_text_response(
        self, response_content: str, file_path: str | None
    ) -> CodeReviewResult:
        """Fallback text parsing when JSON parsing fails."""
        # Simple text-based parsing logic
        issues = []
        lines = response_content.split("\n")

        for line in lines:
            line_lower = line.lower()
            issue = None

            if "critical:" in line_lower or "error:" in line_lower:
                issue = CodeIssue(
                    severity=ReviewSeverity.CRITICAL,
                    category=ReviewCategory.BUG,
                    message=line.split(":", 1)[1].strip() if ":" in line else line,
                    file_path=file_path,
                )
            elif "security:" in line_lower or "vulnerability:" in line_lower:
                issue = CodeIssue(
                    severity=ReviewSeverity.HIGH,
                    category=ReviewCategory.SECURITY,
                    message=line.split(":", 1)[1].strip() if ":" in line else line,
                    file_path=file_path,
                )
            elif "performance:" in line_lower:
                issue = CodeIssue(
                    severity=ReviewSeverity.MEDIUM,
                    category=ReviewCategory.PERFORMANCE,
                    message=line.split(":", 1)[1].strip() if ":" in line else line,
                    file_path=file_path,
                )

            if issue:
                issues.append(issue)

        return CodeReviewResult(
            issues=issues,
            metrics=ReviewMetrics(),
            summary=response_content[:500],
            overall_score=0.0,
            recommendations=[],
            reviewed_files=[file_path] if file_path else [],
        )

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext_to_lang = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "jsx",
            ".tsx": "tsx",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".m": "matlab",
            ".h": "c",
            ".hpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".vue": "vue",
            ".svelte": "svelte",
        }

        path = Path(file_path)
        return ext_to_lang.get(path.suffix.lower(), "text")

    def _calculate_metrics(self, code: str, issues: list[CodeIssue]) -> ReviewMetrics:
        """Calculate review metrics from code and issues."""
        lines = code.split("\n")
        metrics = ReviewMetrics(
            total_lines=len(lines),
            reviewed_lines=len(lines),
            total_files=1,
            reviewed_files=1,
        )

        # Count issues by severity and category
        for issue in issues:
            metrics.issues_by_severity[issue.severity] = (
                metrics.issues_by_severity.get(issue.severity, 0) + 1
            )
            metrics.issues_by_category[issue.category] = (
                metrics.issues_by_category.get(issue.category, 0) + 1
            )

        # Estimate complexity (simple heuristic)
        metrics.complexity_score = min(100, len(lines) / 10 + len(issues) * 5)

        # Calculate maintainability index (simplified)
        critical_issues = metrics.issues_by_severity.get(ReviewSeverity.CRITICAL, 0)
        high_issues = metrics.issues_by_severity.get(ReviewSeverity.HIGH, 0)
        metrics.maintainability_index = max(
            0, 100 - (critical_issues * 20 + high_issues * 10 + len(issues) * 2)
        )

        return metrics

    def _aggregate_metrics(
        self,
        issues: list[CodeIssue],
        reviewed_files: list[str],
        skipped_files: list[str],
    ) -> ReviewMetrics:
        """Aggregate metrics for multiple files."""
        metrics = ReviewMetrics(
            total_files=len(reviewed_files) + len(skipped_files),
            reviewed_files=len(reviewed_files),
        )

        for issue in issues:
            metrics.issues_by_severity[issue.severity] = (
                metrics.issues_by_severity.get(issue.severity, 0) + 1
            )
            metrics.issues_by_category[issue.category] = (
                metrics.issues_by_category.get(issue.category, 0) + 1
            )

        return metrics

    def _calculate_overall_score(self, metrics: ReviewMetrics, issues: list[CodeIssue]) -> float:
        """Calculate overall code quality score (0-100)."""
        score = 100.0

        # Deduct points based on issue severity
        severity_weights = {
            ReviewSeverity.CRITICAL: 15,
            ReviewSeverity.HIGH: 8,
            ReviewSeverity.MEDIUM: 4,
            ReviewSeverity.LOW: 2,
            ReviewSeverity.INFO: 0.5,
        }

        for severity, count in metrics.issues_by_severity.items():
            score -= count * severity_weights.get(severity, 0)

        return max(0, min(100, score))

    def _generate_recommendations(
        self, issues: list[CodeIssue], metrics: ReviewMetrics
    ) -> list[str]:
        """Generate high-level recommendations based on issues."""
        recommendations = []

        # Check for common patterns
        if metrics.issues_by_category.get(ReviewCategory.SECURITY, 0) > 0:
            recommendations.append("Prioritize fixing security vulnerabilities")

        if metrics.issues_by_category.get(ReviewCategory.TESTING, 0) > 2:
            recommendations.append("Improve test coverage and add more unit tests")

        if metrics.issues_by_category.get(ReviewCategory.DOCUMENTATION, 0) > 3:
            recommendations.append("Add or improve code documentation")

        if metrics.issues_by_severity.get(ReviewSeverity.CRITICAL, 0) > 0:
            recommendations.append("Address critical issues immediately before deployment")

        return recommendations

    async def _generate_project_summary(
        self,
        issues: list[CodeIssue],
        reviewed_files: list[str],
    ) -> str:
        """Generate a summary for the entire project review."""
        critical_count = sum(1 for i in issues if i.severity == ReviewSeverity.CRITICAL)
        high_count = sum(1 for i in issues if i.severity == ReviewSeverity.HIGH)

        summary = f"Reviewed {len(reviewed_files)} files and found {len(issues)} issues. "
        if critical_count > 0:
            summary += f"{critical_count} critical issues require immediate attention. "
        if high_count > 0:
            summary += f"{high_count} high-priority issues should be addressed soon."

        return summary

    def _collect_files(
        self,
        project_path: Path,
        include_patterns: set[str],
        exclude_patterns: set[str],
        max_files: int,
    ) -> list[Path]:
        """Collect files to review from a project directory."""
        files = []
        all_exclude = self.exclude_patterns.union(exclude_patterns)

        for file_path in project_path.rglob("*"):
            if len(files) >= max_files:
                break

            # Skip directories and excluded patterns
            if file_path.is_dir():
                continue

            # Check exclusions
            if any(pattern in str(file_path) for pattern in all_exclude):
                continue

            # Check inclusions
            if file_path.suffix in include_patterns:
                files.append(file_path)

        return files
