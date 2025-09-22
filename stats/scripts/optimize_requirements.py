#!/usr/bin/env python3
"""
Optimize and deduplicate Python dependencies.

This script analyzes Python dependencies in both pyproject.toml and requirements.txt,
identifies duplicates, version conflicts, and optimization opportunities.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from typing import Any, Dict, List, Optional, Set, Tuple

# Configure logging
s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("optimize_requirements.log"),
    ],
)

@dataclass
class Dependency:
    """Represents a Python dependency with version constraints."""

    name: str
    version_spec: str = ""
    extras: set[str] = field(default_factory=set)
    source_files: set[str] = field(default_factory=set)
    is_dev: bool = False
    group: str | None = None

    def __post_init__(self):
        """Normalize dependency name."""
        self.name = self.name.lower().replace("_", "-")

    @property
    def normalized_name(self) -> str:
        """Get normalized package name."""
        return self.name.lower().replace("_", "-")

    def __str__(self) -> str:
        """String representation of the dependency."""
        extras_str = f"[{','.join(sorted(self.extras))}]" if self.extras else ""
        return f"{self.name}{extras_str}{self.version_spec}"

class DependencyOptimizer:
    """Optimizes and deduplicates Python dependencies."""

    def __init__(self, project_root: Path):
        """Initialize the optimizer with project root path."""
        self.project_root = project_root
        self.dependencies: dict[str, Dependency] = {}
        self.conflicts: list[tuple[Dependency, Dependency]] = []
        self.duplicates: dict[str, list[Dependency]] = defaultdict(list)

    def parse_version_spec(self, spec: str) -> tuple[str, str]:
        """Parse version specification into operator and version."""
        spec = spec.strip()
        if not spec:
            return "", ""

        # Match version operators
        match = re.match(r"([><=!~]+)(.+)", spec)
        if match:
            return match.group(1), match.group(2)

        # Exact version
        if re.match(r"^\d", spec):
            return "==", spec

        return "", spec

    def parse_requirement_line(self, line: str, source_file: str) -> Dependency | None:
        """Parse a single requirement line."""
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            return None

        # Handle inline comments
        if "#" in line:
            line = line.split("#")[0].strip()

        # Parse extras
        extras = set()
        if "[" in line and "]" in line:
            name_part, rest = line.split("[", 1)
            extras_part, version_part = rest.split("]", 1)
            extras = {e.strip() for e in extras_part.split(",")}
            line = name_part + version_part

        # Parse name and version
        parts = re.split(r"([><=!~]+)", line, 1)
        name = parts[0].strip()
        version_spec = "".join(parts[1:]).strip() if len(parts) > 1 else ""

        if not name:
            return None

        dep = Dependency(
            name=name,
            version_spec=version_spec,
            extras=extras,
            source_files={source_file}
        )

        return dep

    def load_requirements_txt(self) -> None:
        """Load dependencies from requirements.txt."""
        req_file = self.project_root / "requirements.txt"
        if not req_file.exists():
            logger.warning("requirements.txt not found")
            return

        logger.info("Loading requirements.txt")
        try:
            with open(req_file, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        dep = self.parse_requirement_line(line, "requirements.txt")
                        if dep:
                            self._add_dependency(dep)
                    except Exception as e:
                        logger.warning(
                            f"Error parsing line {line_num} in requirements.txt: {line.strip()}"
                        )
                        logger.debug(f"Parse error: {e}")
        except Exception as e:
            logger.error(f"Failed to load requirements.txt: {e}")

    def load_pyproject_toml(self) -> None:
        """Load dependencies from pyproject.toml."""
        pyproject_file = self.project_root / "pyproject.toml"
        if not pyproject_file.exists():
            logger.warning("pyproject.toml not found")
            return

        logger.info("Loading pyproject.toml")
        try:
            with open(pyproject_file, "rb") as f:
                data = tomllib.load(f)

            project = data.get("project", {})

            # Load main dependencies
            for dep_str in project.get("dependencies", []):
                dep = self.parse_requirement_line(dep_str, "pyproject.toml")
                if dep:
                    self._add_dependency(dep)

            # Load dependency groups
            dep_groups = project.get("dependency-groups", {})
            for group_name, deps in dep_groups.items():
                for dep_str in deps:
                    dep = self.parse_requirement_line(dep_str, f"pyproject.toml:{group_name}")
                    if dep:
                        dep.is_dev = group_name == "dev"
                        dep.group = group_name
                        self._add_dependency(dep)

        except Exception as e:
            logger.error(f"Failed to load pyproject.toml: {e}")

    def _add_dependency(self, dep: Dependency) -> None:
        """Add a dependency to the collection."""
        key = dep.normalized_name

        if key in self.dependencies:
            # Merge with existing dependency
            existing = self.dependencies[key]
            existing.source_files.update(dep.source_files)
            existing.extras.update(dep.extras)

            # Check for version conflicts
            if existing.version_spec != dep.version_spec and dep.version_spec:
                self.conflicts.append((existing, dep))
                logger.warning(
                    f"Version conflict for {dep.name}: "
                    f"{existing.version_spec} vs {dep.version_spec}"
                )

            # Update version spec if more specific
            if not existing.version_spec and dep.version_spec:
                existing.version_spec = dep.version_spec
        else:
            self.dependencies[key] = dep

        # Track duplicates
        self.duplicates[key].append(dep)

    def check_for_unused_dependencies(self) -> set[str]:
        """Check for potentially unused dependencies by scanning source code."""
        logger.info("Checking for unused dependencies")
        unused = set()

        # Get all Python files in src/
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            logger.warning("src/ directory not found")
            return unused

        python_files = list(src_dir.rglob("*.py"))
        if not python_files:
            logger.warning("No Python files found in src/")
            return unused

        # Read all Python files to find imports
        imports_found = set()
        for py_file in python_files:
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Find import statements
                import_matches = re.findall(
                    r"^(?:from\s+(\S+)\s+import|import\s+(\S+))",
                    content,
                    re.MULTILINE
                )

                for match in import_matches:
                    module = match[0] or match[1]
                    if module:
                        # Get top-level module name
                        top_module = module.split(".")[0].lower().replace("_", "-")
                        imports_found.add(top_module)

            except Exception as e:
                logger.debug(f"Error reading {py_file}: {e}")

        # Check which dependencies are not imported
        for dep_name in self.dependencies:
            # Common transformations for package names
            possible_names = {
                dep_name,
                dep_name.replace("-", "_"),
                dep_name.replace("_", "-"),
            }

            # Add known mappings
            name_mappings = {
                "pillow": "pil",
                "beautifulsoup4": "bs4",
                "python-dateutil": "dateutil",
                "python-multipart": "multipart",
                "pyyaml": "yaml",
                "msgpack": "msgpack",
            }

            if dep_name in name_mappings:
                possible_names.add(name_mappings[dep_name])

            if not any(name in imports_found for name in possible_names):
                unused.add(dep_name)

        logger.info(f"Found {len(unused)} potentially unused dependencies")
        return unused

    def get_package_info(self, package_name: str) -> dict[str, Any]:
        """Get package information from PyPI."""
        try:
            result = subprocess.run(
                ["uv", "pip", "show", package_name],
                check=False, capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                info = {}
                for line in result.stdout.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        info[key.strip().lower()] = value.strip()
                return info

        except Exception as e:
            logger.debug(f"Error getting info for {package_name}: {e}")

        return {}

    def suggest_optimizations(self) -> dict[str, Any]:
        """Suggest optimization opportunities."""
        logger.info("Analyzing optimization opportunities")
        suggestions = {
            "conflicts": [],
            "duplicates": [],
            "unused": [],
            "pinning_suggestions": [],
            "group_suggestions": [],
        }

        # Version conflicts
        for existing, conflicting in self.conflicts:
            suggestions["conflicts"].append({
                "package": existing.name,
                "versions": [existing.version_spec, conflicting.version_spec],
                "sources": list(existing.source_files | conflicting.source_files),
            })

        # Duplicates across files
        for name, deps in self.duplicates.items():
            if len(deps) > 1:
                suggestions["duplicates"].append({
                    "package": name,
                    "occurrences": len(deps),
                    "sources": list(set().union(*[d.source_files for d in deps])),
                })

        # Unused dependencies
        unused = self.check_for_unused_dependencies()
        for pkg in unused:
            if pkg in self.dependencies:
                dep = self.dependencies[pkg]
                suggestions["unused"].append({
                    "package": pkg,
                    "sources": list(dep.source_files),
                })

        # Version pinning suggestions
        for name, dep in self.dependencies.items():
            if not dep.version_spec or dep.version_spec.startswith(">="):
                suggestions["pinning_suggestions"].append({
                    "package": name,
                    "current": dep.version_spec or "no version",
                    "suggestion": "Consider pinning to specific version for reproducibility",
                })

        # Group organization suggestions
        dev_packages = {
            "pytest", "mypy", "ruff", "black", "flake8", "isort",
            "pre-commit", "jupyter", "ipython", "maturin"
        }

        for name, dep in self.dependencies.items():
            if name in dev_packages and not dep.is_dev:
                suggestions["group_suggestions"].append({
                    "package": name,
                    "suggestion": "Should be in dev dependency group",
                })

        return suggestions

    def generate_optimized_requirements(self) -> tuple[list[str], list[str]]:
        """Generate optimized requirements.txt and pyproject.toml dependencies."""
        logger.info("Generating optimized dependency lists")

        # Sort dependencies
        main_deps = []
        dev_deps = []

        for name in sorted(self.dependencies.keys()):
            dep = self.dependencies[name]

            if dep.is_dev or dep.group == "dev":
                dev_deps.append(str(dep))
            else:
                main_deps.append(str(dep))

        return main_deps, dev_deps

    def save_backup(self) -> None:
        """Create backup of original files."""
        logger.info("Creating backups")

        for filename in ["requirements.txt", "pyproject.toml"]:
            original = self.project_root / filename
            if original.exists():
                backup = self.project_root / f"{filename}.backup"
                try:
                    with open(original, encoding="utf-8") as src:
                        with open(backup, "w", encoding="utf-8") as dst:
                            dst.write(src.read())
                    logger.info(f"Created backup: {backup}")
                except Exception as e:
                    logger.error(f"Failed to create backup for {filename}: {e}")

    def write_optimized_requirements(self, main_deps: list[str]) -> None:
        """Write optimized requirements.txt."""
        req_file = self.project_root / "requirements.txt"

        try:
            with open(req_file, "w", encoding="utf-8") as f:
                f.write("# Optimized requirements.txt\n")
                f.write("# Generated by optimize_requirements.py\n\n")

                # Group by category
                categories = {
                    "Core ML": ["torch", "transformers", "accelerate", "tokenizers",
                               "sentencepiece", "bitsandbytes", "safetensors", "datasets"],
                    "Web Framework": ["fastapi", "uvicorn", "websockets", "aiofiles",
                                     "python-multipart", "starlette", "sse-starlette"],
                    "CLI and UI": ["typer", "rich", "tqdm"],
                    "Data Validation": ["pydantic", "pydantic-settings"],
                    "HTTP Client": ["httpx", "requests", "aiohttp"],
                    "System Utilities": ["psutil", "docker", "prometheus-client"],
                    "Other": [],
                }

                categorized = {cat: [] for cat in categories}

                for dep_str in main_deps:
                    dep_name = dep_str.split("[")[0].split(">=")[0].split("==")[0].lower()
                    placed = False

                    for category, packages in categories.items():
                        if category != "Other" and dep_name in packages:
                            categorized[category].append(dep_str)
                            placed = True
                            break

                    if not placed:
                        categorized["Other"].append(dep_str)

                # Write categorized dependencies
                for category, deps in categorized.items():
                    if deps:
                        f.write(f"# {category}\n")
                        f.writelines(f"{dep}\n" for dep in sorted(deps))
                        f.write("\n")

            logger.info(f"Wrote optimized requirements.txt with {len(main_deps)} dependencies")

        except Exception as e:
            logger.error(f"Failed to write requirements.txt: {e}")

    def run_optimization(self, backup: bool = True, apply: bool = False) -> None:
        """Run the complete optimization process."""
        logger.info("Starting dependency optimization")

        try:
            # Load existing dependencies
            self.load_requirements_txt()
            self.load_pyproject_toml()

            logger.info(f"Loaded {len(self.dependencies)} unique dependencies")

            # Analyze and suggest optimizations
            suggestions = self.suggest_optimizations()

            # Print analysis report
            self.print_analysis_report(suggestions)

            if apply:
                if backup:
                    self.save_backup()

                # Generate optimized files
                main_deps, _dev_deps = self.generate_optimized_requirements()
                self.write_optimized_requirements(main_deps)

                logger.info("Optimization applied successfully!")
            else:
                logger.info("Analysis complete. Use --apply to write optimized files.")

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            sys.exit(1)

    def print_analysis_report(self, suggestions: dict[str, Any]) -> None:
        """Print detailed analysis report."""
        print("\n" + "="*80)
        print("DEPENDENCY OPTIMIZATION REPORT")
        print("="*80)

        print(f"\nTotal dependencies found: {len(self.dependencies)}")

        # Conflicts
        if suggestions["conflicts"]:
            print(f"\n⚠️  VERSION CONFLICTS ({len(suggestions['conflicts'])})")
            print("-" * 40)
            for conflict in suggestions["conflicts"]:
                print(f"  {conflict['package']}: {' vs '.join(conflict['versions'])}")
                print(f"    Sources: {', '.join(conflict['sources'])}")

        # Duplicates
        if suggestions["duplicates"]:
            print(f"\n📋 DUPLICATE DEPENDENCIES ({len(suggestions['duplicates'])})")
            print("-" * 40)
            for dup in suggestions["duplicates"]:
                print(f"  {dup['package']}: {dup['occurrences']} occurrences")
                print(f"    Sources: {', '.join(dup['sources'])}")

        # Unused dependencies
        if suggestions["unused"]:
            print(f"\n🗑️  POTENTIALLY UNUSED ({len(suggestions['unused'])})")
            print("-" * 40)
            for unused in suggestions["unused"]:
                print(f"  {unused['package']}")
                print(f"    Sources: {', '.join(unused['sources'])}")

        # Pinning suggestions
        if suggestions["pinning_suggestions"]:
            print(f"\n📌 VERSION PINNING SUGGESTIONS ({len(suggestions['pinning_suggestions'])})")
            print("-" * 40)
            for pin in suggestions["pinning_suggestions"][:5]:  # Show first 5
                print(f"  {pin['package']}: {pin['current']} -> {pin['suggestion']}")

        # Group suggestions
        if suggestions["group_suggestions"]:
            print(f"\n📦 DEPENDENCY GROUP SUGGESTIONS ({len(suggestions['group_suggestions'])})")
            print("-" * 40)
            for group in suggestions["group_suggestions"]:
                print(f"  {group['package']}: {group['suggestion']}")

        print("\n" + "="*80)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Optimize and deduplicate Python dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_requirements.py                    # Analyze dependencies
  python optimize_requirements.py --apply           # Apply optimizations
  python optimize_requirements.py --no-backup       # Skip backup creation
  python optimize_requirements.py --verbose         # Enable debug logging
        """,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply optimizations (default: analysis only)",
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup files",
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

    # Run optimization
    optimizer = DependencyOptimizer(args.project_root)
    optimizer.run_optimization(
        backup=not args.no_backup,
        apply=args.apply,
    )

if __name__ == "__main__":
    main()
