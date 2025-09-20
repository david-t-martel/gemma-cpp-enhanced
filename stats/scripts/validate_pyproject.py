#!/usr/bin/env python3
"""
Validate pyproject.toml configuration.

This script validates the pyproject.toml file for correctness, completeness,
and best practices compliance.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import tomllib
from typing import Any, Dict, List, Optional, Set, Union

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet

# Configure logging
s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("validate_pyproject.log"),
    ],
)

@dataclass
class ValidationError:
    """Represents a validation error."""

    level: str  # 'error', 'warning', 'info'
    section: str
    field: str
    message: str
    suggestion: str | None = None

    def __str__(self) -> str:
        """String representation of validation error."""
        suggestion = f" Suggestion: {self.suggestion}" if self.suggestion else ""
        return f"[{self.level.upper()}] {self.section}.{self.field}: {self.message}{suggestion}"

class PyProjectValidator:
    """Validates pyproject.toml configuration."""

    def __init__(self, project_root: Path):
        """Initialize validator with project root."""
        self.project_root = project_root
        self.pyproject_path = project_root / "pyproject.toml"
        self.errors: list[ValidationError] = []
        self.data: dict[str, Any] = {}

    def load_pyproject(self) -> bool:
        """Load and parse pyproject.toml."""
        if not self.pyproject_path.exists():
            self.errors.append(ValidationError(
                level="error",
                section="file",
                field="existence",
                message="pyproject.toml not found",
                suggestion="Create pyproject.toml file"
            ))
            return False

        try:
            with open(self.pyproject_path, "rb") as f:
                self.data = tomllib.load(f)
            logger.info("Successfully loaded pyproject.toml")
            return True
        except tomllib.TOMLDecodeError as e:
            self.errors.append(ValidationError(
                level="error",
                section="file",
                field="syntax",
                message=f"TOML syntax error: {e}",
                suggestion="Fix TOML syntax errors"
            ))
            return False
        except Exception as e:
            self.errors.append(ValidationError(
                level="error",
                section="file",
                field="reading",
                message=f"Failed to read file: {e}"
            ))
            return False

    def validate_build_system(self) -> None:
        """Validate [build-system] section."""
        if "build-system" not in self.data:
            self.errors.append(ValidationError(
                level="warning",
                section="build-system",
                field="missing",
                message="No build-system section found",
                suggestion="Add [build-system] section for better compatibility"
            ))
            return

        build_system = self.data["build-system"]

        # Check requires field
        if "requires" not in build_system:
            self.errors.append(ValidationError(
                level="error",
                section="build-system",
                field="requires",
                message="Missing 'requires' field",
                suggestion="Add 'requires' field with build dependencies"
            ))
        elif not isinstance(build_system["requires"], list):
            self.errors.append(ValidationError(
                level="error",
                section="build-system",
                field="requires",
                message="'requires' must be a list",
                suggestion="Convert 'requires' to list format"
            ))

        # Check build-backend field
        if "build-backend" not in build_system:
            self.errors.append(ValidationError(
                level="warning",
                section="build-system",
                field="build-backend",
                message="Missing 'build-backend' field",
                suggestion="Specify build backend (e.g., 'setuptools.build_meta')"
            ))

        # Validate build backend compatibility
        backend = build_system.get("build-backend", "")
        requires = build_system.get("requires", [])

        if "setuptools" in backend and not any("setuptools" in req for req in requires):
            self.errors.append(ValidationError(
                level="warning",
                section="build-system",
                field="requires",
                message="setuptools backend specified but not in requires",
                suggestion="Add 'setuptools>=61.0' to requires"
            ))

    def validate_project_metadata(self) -> None:
        """Validate [project] section."""
        if "project" not in self.data:
            self.errors.append(ValidationError(
                level="error",
                section="project",
                field="missing",
                message="No [project] section found",
                suggestion="Add [project] section with metadata"
            ))
            return

        project = self.data["project"]

        # Required fields
        required_fields = {
            "name": "Project name is required",
            "version": "Version is required (or use dynamic versioning)",
        }

        for field, message in required_fields.items():
            if field not in project:
                self.errors.append(ValidationError(
                    level="error",
                    section="project",
                    field=field,
                    message=message,
                    suggestion=f"Add '{field}' field to project metadata"
                ))

        # Validate name
        if "name" in project:
            name = project["name"]
            if not isinstance(name, str):
                self.errors.append(ValidationError(
                    level="error",
                    section="project",
                    field="name",
                    message="Project name must be a string"
                ))
            elif not re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?$", name):
                self.errors.append(ValidationError(
                    level="error",
                    section="project",
                    field="name",
                    message="Invalid project name format",
                    suggestion="Use only letters, numbers, hyphens, underscores, and dots"
                ))

        # Validate version
        if "version" in project:
            version = project["version"]
            if not isinstance(version, str):
                self.errors.append(ValidationError(
                    level="error",
                    section="project",
                    field="version",
                    message="Version must be a string"
                ))
            elif not re.match(r"^\d+(\.\d+)*", version):
                self.errors.append(ValidationError(
                    level="warning",
                    section="project",
                    field="version",
                    message="Version format should follow semantic versioning",
                    suggestion="Use format like '1.0.0'"
                ))

        # Validate Python version requirement
        if "requires-python" in project:
            req_python = project["requires-python"]
            try:
                SpecifierSet(req_python)
            except InvalidSpecifier:
                self.errors.append(ValidationError(
                    level="error",
                    section="project",
                    field="requires-python",
                    message="Invalid Python version specifier",
                    suggestion="Use format like '>=3.8'"
                ))

        # Validate authors/maintainers
        for field in ["authors", "maintainers"]:
            if field in project:
                authors = project[field]
                if not isinstance(authors, list):
                    self.errors.append(ValidationError(
                        level="error",
                        section="project",
                        field=field,
                        message=f"{field} must be a list"
                    ))
                else:
                    for i, author in enumerate(authors):
                        if not isinstance(author, dict):
                            self.errors.append(ValidationError(
                                level="error",
                                section="project",
                                field=f"{field}[{i}]",
                                message="Author entry must be a dictionary"
                            ))
                        elif "name" not in author and "email" not in author:
                            self.errors.append(ValidationError(
                                level="error",
                                section="project",
                                field=f"{field}[{i}]",
                                message="Author must have 'name' or 'email'",
                                suggestion="Add 'name' and/or 'email' fields"
                            ))

        # Validate classifiers
        if "classifiers" in project:
            classifiers = project["classifiers"]
            if not isinstance(classifiers, list):
                self.errors.append(ValidationError(
                    level="error",
                    section="project",
                    field="classifiers",
                    message="Classifiers must be a list"
                ))

        # Check for recommended fields
        recommended_fields = {
            "description": "Add a brief description",
            "readme": "Add README file reference",
            "license": "Specify license information",
            "keywords": "Add relevant keywords",
        }

        for field, suggestion in recommended_fields.items():
            if field not in project:
                self.errors.append(ValidationError(
                    level="info",
                    section="project",
                    field=field,
                    message=f"Missing recommended field: {field}",
                    suggestion=suggestion
                ))

    def validate_dependencies(self) -> None:
        """Validate project dependencies."""
        if "project" not in self.data:
            return

        project = self.data["project"]

        # Validate main dependencies
        if "dependencies" in project:
            deps = project["dependencies"]
            if not isinstance(deps, list):
                self.errors.append(ValidationError(
                    level="error",
                    section="project",
                    field="dependencies",
                    message="Dependencies must be a list"
                ))
            else:
                self._validate_dependency_list(deps, "project.dependencies")

        # Validate dependency groups
        if "dependency-groups" in project:
            groups = project["dependency-groups"]
            if not isinstance(groups, dict):
                self.errors.append(ValidationError(
                    level="error",
                    section="project",
                    field="dependency-groups",
                    message="dependency-groups must be a dictionary"
                ))
            else:
                for group_name, group_deps in groups.items():
                    if not isinstance(group_deps, list):
                        self.errors.append(ValidationError(
                            level="error",
                            section="project",
                            field=f"dependency-groups.{group_name}",
                            message="Dependency group must be a list"
                        ))
                    else:
                        self._validate_dependency_list(
                            group_deps, f"project.dependency-groups.{group_name}"
                        )

    def _validate_dependency_list(self, deps: list[str], context: str) -> None:
        """Validate a list of dependency specifications."""
        seen_packages = set()

        for i, dep in enumerate(deps):
            if not isinstance(dep, str):
                self.errors.append(ValidationError(
                    level="error",
                    section=context,
                    field=f"[{i}]",
                    message="Dependency must be a string"
                ))
                continue

            # Parse dependency
            try:
                req = Requirement(dep)

                # Check for duplicates
                pkg_name = req.name.lower()
                if pkg_name in seen_packages:
                    self.errors.append(ValidationError(
                        level="warning",
                        section=context,
                        field=f"[{i}]",
                        message=f"Duplicate dependency: {req.name}",
                        suggestion="Remove duplicate dependency"
                    ))
                seen_packages.add(pkg_name)

                # Validate version specifiers
                for spec in req.specifier:
                    if spec.operator in ("==", "~=") and not re.match(r"\d", spec.version):
                        self.errors.append(ValidationError(
                            level="warning",
                            section=context,
                            field=f"[{i}]",
                            message=f"Invalid version in {dep}",
                            suggestion="Use proper version format"
                        ))

                # Check for overly restrictive constraints
                if len([s for s in req.specifier if s.operator == "=="]) > 0:
                    self.errors.append(ValidationError(
                        level="info",
                        section=context,
                        field=f"[{i}]",
                        message=f"Exact version pinning for {req.name}",
                        suggestion="Consider using version ranges for flexibility"
                    ))

            except InvalidRequirement as e:
                self.errors.append(ValidationError(
                    level="error",
                    section=context,
                    field=f"[{i}]",
                    message=f"Invalid dependency format: {dep} ({e})",
                    suggestion="Use format: 'package>=version'"
                ))

    def validate_tool_config(self) -> None:
        """Validate [tool.*] sections."""
        if "tool" not in self.data:
            self.errors.append(ValidationError(
                level="info",
                section="tool",
                field="missing",
                message="No tool configuration found",
                suggestion="Add tool configurations for linters, formatters, etc."
            ))
            return

        tool_config = self.data["tool"]

        # Validate common tools
        self._validate_ruff_config(tool_config.get("ruff"))
        self._validate_mypy_config(tool_config.get("mypy"))
        self._validate_pytest_config(tool_config.get("pytest"))
        self._validate_coverage_config(tool_config.get("coverage"))

    def _validate_ruff_config(self, config: dict[str, Any] | None) -> None:
        """Validate ruff configuration."""
        if config is None:
            self.errors.append(ValidationError(
                level="info",
                section="tool.ruff",
                field="missing",
                message="No ruff configuration found",
                suggestion="Add ruff configuration for linting"
            ))
            return

        # Check for important settings
        recommended_settings = {
            "line-length": "Consider setting line length",
            "target-version": "Specify Python target version",
        }

        for setting, suggestion in recommended_settings.items():
            if setting not in config:
                self.errors.append(ValidationError(
                    level="info",
                    section="tool.ruff",
                    field=setting,
                    message=f"Missing recommended setting: {setting}",
                    suggestion=suggestion
                ))

    def _validate_mypy_config(self, config: dict[str, Any] | None) -> None:
        """Validate mypy configuration."""
        if config is None:
            self.errors.append(ValidationError(
                level="info",
                section="tool.mypy",
                field="missing",
                message="No mypy configuration found",
                suggestion="Add mypy configuration for type checking"
            ))
            return

        # Check for important settings
        if "python_version" not in config:
            self.errors.append(ValidationError(
                level="warning",
                section="tool.mypy",
                field="python_version",
                message="Python version not specified for mypy",
                suggestion="Add python_version setting"
            ))

        # Check for strict settings
        strict_settings = ["strict", "disallow_untyped_defs", "warn_return_any"]
        has_strict = any(config.get(setting, False) for setting in strict_settings)

        if not has_strict:
            self.errors.append(ValidationError(
                level="info",
                section="tool.mypy",
                field="strict",
                message="Consider enabling strict type checking",
                suggestion="Add 'strict = true' or individual strict flags"
            ))

    def _validate_pytest_config(self, config: dict[str, Any] | None) -> None:
        """Validate pytest configuration."""
        if config is None:
            self.errors.append(ValidationError(
                level="info",
                section="tool.pytest",
                field="missing",
                message="No pytest configuration found",
                suggestion="Add pytest configuration"
            ))
            return

        # Check ini_options
        if "ini_options" in config:
            ini_options = config["ini_options"]

            # Check for test paths
            if "testpaths" not in ini_options:
                self.errors.append(ValidationError(
                    level="info",
                    section="tool.pytest.ini_options",
                    field="testpaths",
                    message="Test paths not specified",
                    suggestion="Add testpaths = ['tests']"
                ))

            # Check for coverage
            addopts = ini_options.get("addopts", "")
            if "--cov" not in addopts:
                self.errors.append(ValidationError(
                    level="info",
                    section="tool.pytest.ini_options",
                    field="addopts",
                    message="Coverage not enabled",
                    suggestion="Add coverage options to addopts"
                ))

    def _validate_coverage_config(self, config: dict[str, Any] | None) -> None:
        """Validate coverage configuration."""
        if config is None:
            return

        # Validate run section
        if "run" in config:
            run_config = config["run"]
            if "source" not in run_config:
                self.errors.append(ValidationError(
                    level="warning",
                    section="tool.coverage.run",
                    field="source",
                    message="Coverage source not specified",
                    suggestion="Add source directories"
                ))

        # Validate report section
        if "report" in config:
            report_config = config["report"]
            if "fail_under" not in report_config:
                self.errors.append(ValidationError(
                    level="info",
                    section="tool.coverage.report",
                    field="fail_under",
                    message="Coverage threshold not set",
                    suggestion="Set minimum coverage percentage"
                ))

    def validate_scripts_and_entry_points(self) -> None:
        """Validate scripts and entry points."""
        if "project" not in self.data:
            return

        project = self.data["project"]

        # Validate scripts
        if "scripts" in project:
            scripts = project["scripts"]
            if not isinstance(scripts, dict):
                self.errors.append(ValidationError(
                    level="error",
                    section="project",
                    field="scripts",
                    message="Scripts must be a dictionary"
                ))
            else:
                for script_name, script_path in scripts.items():
                    if not isinstance(script_path, str):
                        self.errors.append(ValidationError(
                            level="error",
                            section="project.scripts",
                            field=script_name,
                            message="Script path must be a string"
                        ))
                    elif ":" not in script_path:
                        self.errors.append(ValidationError(
                            level="warning",
                            section="project.scripts",
                            field=script_name,
                            message="Script path should use module:function format",
                            suggestion="Use format: 'module.submodule:function'"
                        ))

        # Validate GUI scripts
        if "gui-scripts" in project:
            gui_scripts = project["gui-scripts"]
            if not isinstance(gui_scripts, dict):
                self.errors.append(ValidationError(
                    level="error",
                    section="project",
                    field="gui-scripts",
                    message="GUI scripts must be a dictionary"
                ))

    def validate_urls(self) -> None:
        """Validate project URLs."""
        if "project" not in self.data:
            return

        project = self.data["project"]

        if "urls" in project:
            urls = project["urls"]
            if not isinstance(urls, dict):
                self.errors.append(ValidationError(
                    level="error",
                    section="project",
                    field="urls",
                    message="URLs must be a dictionary"
                ))
            else:
                url_pattern = re.compile(
                    r"^https?://"  # http:// or https://
                    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain
                    r"localhost|"  # localhost
                    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # IP
                    r"(?::\d+)?"  # optional port
                    r"(?:/?|[/?]\S+)$", re.IGNORECASE
                )

                for url_type, url in urls.items():
                    if not isinstance(url, str):
                        self.errors.append(ValidationError(
                            level="error",
                            section="project.urls",
                            field=url_type,
                            message="URL must be a string"
                        ))
                    elif not url_pattern.match(url):
                        self.errors.append(ValidationError(
                            level="warning",
                            section="project.urls",
                            field=url_type,
                            message="Invalid URL format",
                            suggestion="Use full HTTP/HTTPS URL"
                        ))

    def check_file_references(self) -> None:
        """Check that referenced files exist."""
        if "project" not in self.data:
            return

        project = self.data["project"]

        # Check README file
        if "readme" in project:
            readme_spec = project["readme"]
            if isinstance(readme_spec, str):
                readme_path = self.project_root / readme_spec
                if not readme_path.exists():
                    self.errors.append(ValidationError(
                        level="error",
                        section="project",
                        field="readme",
                        message=f"README file not found: {readme_spec}",
                        suggestion="Create the README file or update the path"
                    ))
            elif isinstance(readme_spec, dict):
                if "file" in readme_spec:
                    readme_path = self.project_root / readme_spec["file"]
                    if not readme_path.exists():
                        self.errors.append(ValidationError(
                            level="error",
                            section="project",
                            field="readme.file",
                            message=f"README file not found: {readme_spec['file']}",
                            suggestion="Create the README file or update the path"
                        ))

        # Check license file
        if "license" in project:
            license_spec = project["license"]
            if isinstance(license_spec, dict) and "file" in license_spec:
                license_path = self.project_root / license_spec["file"]
                if not license_path.exists():
                    self.errors.append(ValidationError(
                        level="warning",
                        section="project",
                        field="license.file",
                        message=f"License file not found: {license_spec['file']}",
                        suggestion="Create the license file or update the path"
                    ))

    def run_validation(self) -> bool:
        """Run complete validation."""
        logger.info("Starting pyproject.toml validation")

        if not self.load_pyproject():
            return False

        try:
            self.validate_build_system()
            self.validate_project_metadata()
            self.validate_dependencies()
            self.validate_tool_config()
            self.validate_scripts_and_entry_points()
            self.validate_urls()
            self.check_file_references()

            logger.info(f"Validation complete. Found {len(self.errors)} issues.")
            return len([e for e in self.errors if e.level == "error"]) == 0

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def print_report(self, show_info: bool = True) -> None:
        """Print validation report."""
        if not self.errors:
            print("✅ pyproject.toml validation passed - no issues found!")
            return

        print("\n" + "="*80)
        print("PYPROJECT.TOML VALIDATION REPORT")
        print("="*80)

        # Count issues by level
        error_count = len([e for e in self.errors if e.level == "error"])
        warning_count = len([e for e in self.errors if e.level == "warning"])
        info_count = len([e for e in self.errors if e.level == "info"])

        print(f"\nSummary: {error_count} errors, {warning_count} warnings, {info_count} info")

        # Group by level
        for level in ["error", "warning", "info"]:
            if level == "info" and not show_info:
                continue

            level_errors = [e for e in self.errors if e.level == level]
            if not level_errors:
                continue

            icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[level]
            print(f"\n{icon} {level.upper()}S ({len(level_errors)})")
            print("-" * 40)

            for error in level_errors:
                print(f"  {error.section}.{error.field}: {error.message}")
                if error.suggestion:
                    print(f"    💡 {error.suggestion}")

        print("\n" + "="*80)

    def generate_fixes(self) -> dict[str, Any]:
        """Generate automatic fixes for common issues."""
        fixes = {}

        # Add missing build-system
        if not any(e.section == "build-system" and e.field == "missing"
                  for e in self.errors if e.level == "error") and "build-system" not in self.data:
            fixes["build-system"] = {
                "requires": ["setuptools>=61.0", "wheel"],
                "build-backend": "setuptools.build_meta"
            }

        # Add missing project fields
        if "project" in self.data:
            project_fixes = {}

            # Add missing description
            if not self.data["project"].get("description"):
                project_fixes["description"] = "A Python package"

            # Add missing keywords
            if not self.data["project"].get("keywords"):
                project_fixes["keywords"] = ["python", "package"]

            if project_fixes:
                fixes["project"] = project_fixes

        return fixes

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate pyproject.toml configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_pyproject.py                    # Validate current directory
  python validate_pyproject.py --project-root /path/to/project
  python validate_pyproject.py --no-info         # Hide info messages
  python validate_pyproject.py --fix             # Generate fixes
        """,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )

    parser.add_argument(
        "--no-info",
        action="store_true",
        help="Hide info-level messages",
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Generate automatic fixes",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:

    # Validate project root
    if not args.project_root.exists():
        logger.error(f"Project root does not exist: {args.project_root}")
        sys.exit(1)

    # Run validation
    validator = PyProjectValidator(args.project_root)
    success = validator.run_validation()

    # Print report
    validator.print_report(show_info=not args.no_info)

    # Generate fixes if requested
    if args.fix:
        fixes = validator.generate_fixes()
        if fixes:
            print("\n🔧 SUGGESTED FIXES")
            print("-" * 40)
            for section, section_fixes in fixes.items():
                print(f"[{section}]")
                for key, value in section_fixes.items():
                    if isinstance(value, list):
                        print(f"{key} = {value}")
                    elif isinstance(value, str):
                        print(f'{key} = "{value}"')
                    else:
                        print(f"{key} = {value}")
                print()

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
