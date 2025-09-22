#!/usr/bin/env python3
"""
Validate project structure against best practices.

This script analyzes project structure and validates it against Python
packaging standards, modern development practices, and organizational patterns.
"""

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Configure logging
s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("validate_structure.log"),
    ],
)

@dataclass
class StructureIssue:
    """Represents a project structure issue."""

    level: str  # 'error', 'warning', 'suggestion'
    category: str
    path: str
    message: str
    recommendation: str | None = None

    def __str__(self) -> str:
        """String representation of the issue."""
        rec = f" Recommendation: {self.recommendation}" if self.recommendation else ""
        return f"[{self.level.upper()}] {self.category}: {self.path} - {self.message}{rec}"

class ProjectStructureValidator:
    """Validates Python project structure against best practices."""

    def __init__(self, project_root: Path):
        """Initialize the validator."""
        self.project_root = project_root
        self.issues: list[StructureIssue] = []

        # Expected structure patterns
        self.required_files = {
            'README.md': 'Project documentation',
            'pyproject.toml': 'Project configuration (preferred)',
            '.gitignore': 'Git ignore rules',
        }

        self.optional_files = {
            'LICENSE': 'License file',
            'CHANGELOG.md': 'Change documentation',
            'requirements.txt': 'Dependencies (if not using pyproject.toml)',
            'setup.py': 'Legacy setup file',
            'Makefile': 'Build automation',
            '.env.template': 'Environment variables template',
            'docker-compose.yml': 'Docker compose configuration',
            'Dockerfile': 'Docker container definition',
        }

        self.expected_directories = {
            'src/': 'Source code (modern layout)',
            'tests/': 'Test files',
            'docs/': 'Documentation',
            '.github/': 'GitHub workflows and templates',
        }

        self.discouraged_patterns = {
            '__pycache__/': 'Python cache directories should be in .gitignore',
            '*.pyc': 'Compiled Python files should be in .gitignore',
            '.pytest_cache/': 'Pytest cache should be in .gitignore',
            '.mypy_cache/': 'MyPy cache should be in .gitignore',
            'venv/': 'Virtual environments should not be committed',
            '.venv/': 'Virtual environments should not be committed',
            'env/': 'Virtual environments should not be committed',
            'node_modules/': 'Node.js dependencies should not be committed',
        }

        # Common anti-patterns
        self.anti_patterns = {
            'src/src/': 'Nested src directories',
            'package/package/': 'Nested package directories',
            'lib/lib/': 'Nested lib directories',
            'scripts/scripts/': 'Nested scripts directories',
        }

    def scan_project_structure(self) -> dict[str, Any]:
        """Scan and analyze project structure."""
        structure = {
            'files': {},
            'directories': {},
            'python_packages': [],
            'total_files': 0,
            'total_directories': 0,
        }

        # Recursively scan project
        for item in self.project_root.rglob('*'):
            relative_path = str(item.relative_to(self.project_root))

            if item.is_file():
                structure['files'][relative_path] = {
                    'size': item.stat().st_size,
                    'suffix': item.suffix,
                    'name': item.name,
                }
                structure['total_files'] += 1

                # Check for Python packages
                if item.name == '__init__.py':
                    package_dir = str(item.parent.relative_to(self.project_root))
                    structure['python_packages'].append(package_dir)

            elif item.is_dir():
                structure['directories'][relative_path] = {
                    'name': item.name,
                    'depth': len(item.relative_to(self.project_root).parts),
                }
                structure['total_directories'] += 1

        return structure

    def validate_required_files(self, structure: dict[str, Any]) -> None:
        """Validate presence of required files."""
        files = structure['files']

        for required_file, description in self.required_files.items():
            if required_file not in files:
                # Check for alternatives
                alternatives = []
                if required_file == 'README.md':
                    alternatives = ['README.rst', 'README.txt', 'README']
                elif required_file == 'pyproject.toml':
                    alternatives = ['setup.py', 'setup.cfg']

                found_alternative = any(alt in files for alt in alternatives)

                if not found_alternative:
                    self.issues.append(StructureIssue(
                        level='error',
                        category='missing_file',
                        path=required_file,
                        message=f'Missing required file: {description}',
                        recommendation=f'Create {required_file} file'
                    ))
                elif required_file == 'pyproject.toml':
                    self.issues.append(StructureIssue(
                        level='suggestion',
                        category='modernization',
                        path='setup.py',
                        message='Consider migrating to pyproject.toml',
                        recommendation='Modern Python projects use pyproject.toml'
                    ))

    def validate_directory_structure(self, structure: dict[str, Any]) -> None:
        """Validate directory organization."""
        directories = set(structure['directories'].keys())
        files = structure['files']

        # Check for modern src layout vs flat layout
        has_src_layout = 'src' in directories
        has_flat_layout = any(
            f.endswith('.py') and '/' not in f and f != '__init__.py'
            for f in files
        )

        if has_flat_layout and not has_src_layout:
            self.issues.append(StructureIssue(
                level='suggestion',
                category='layout',
                path='.',
                message='Using flat layout instead of src layout',
                recommendation='Consider using src/ layout for better organization'
            ))

        # Check for test directory
        test_dirs = [d for d in directories if 'test' in d.lower()]
        if not test_dirs:
            self.issues.append(StructureIssue(
                level='warning',
                category='testing',
                path='.',
                message='No test directory found',
                recommendation='Create tests/ directory for test files'
            ))
        elif len(test_dirs) > 1:
            self.issues.append(StructureIssue(
                level='warning',
                category='organization',
                path='.',
                message=f'Multiple test directories found: {test_dirs}',
                recommendation='Consolidate tests into single directory'
            ))

        # Check for documentation
        doc_indicators = ['docs', 'doc', 'documentation']
        has_docs = any(d in directories for d in doc_indicators)
        has_readme = any('readme' in f.lower() for f in files)

        if not has_docs and not has_readme:
            self.issues.append(StructureIssue(
                level='error',
                category='documentation',
                path='.',
                message='No documentation found',
                recommendation='Create README.md and consider docs/ directory'
            ))

        # Check for configuration files organization
        config_files = [
            f for f in files
            if any(f.startswith(prefix) for prefix in ['.', 'config', 'pyproject', 'setup'])
        ]

        if len(config_files) > 10:
            self.issues.append(StructureIssue(
                level='suggestion',
                category='organization',
                path='.',
                message=f'Many configuration files in root ({len(config_files)})',
                recommendation='Consider organizing config files into subdirectory'
            ))

    def validate_python_package_structure(self, structure: dict[str, Any]) -> None:
        """Validate Python package organization."""
        packages = structure['python_packages']
        files = structure['files']

        if not packages:
            self.issues.append(StructureIssue(
                level='warning',
                category='packaging',
                path='.',
                message='No Python packages found (no __init__.py files)',
                recommendation='Create __init__.py files to define packages'
            ))
            return

        # Check for proper package hierarchy
        for package in packages:
            package_files = [
                f for f in files
                if f.startswith(package + '/') and f.endswith('.py')
            ]

            if len(package_files) <= 1:  # Only __init__.py
                self.issues.append(StructureIssue(
                    level='suggestion',
                    category='packaging',
                    path=package,
                    message='Package contains only __init__.py',
                    recommendation='Consider if this package is necessary'
                ))

        # Check for circular imports (basic check)
        python_files = [f for f in files if f.endswith('.py')]
        for py_file in python_files[:20]:  # Limit to avoid performance issues
            self._check_imports(py_file)

    def _check_imports(self, file_path: str) -> None:
        """Check for potential import issues."""
        try:
            full_path = self.project_root / file_path
            with open(full_path, encoding='utf-8') as f:
                content = f.read()

            # Check for relative imports without package structure
            if file_path.count('/') == 0 and 'from .' in content:
                self.issues.append(StructureIssue(
                    level='warning',
                    category='imports',
                    path=file_path,
                    message='Relative imports in root-level module',
                    recommendation='Use absolute imports or move to package'
                ))

            # Check for wildcard imports
            if re.search(r'from .* import \*', content):
                self.issues.append(StructureIssue(
                    level='suggestion',
                    category='imports',
                    path=file_path,
                    message='Wildcard imports found',
                    recommendation='Use explicit imports for better code clarity'
                ))

        except Exception as e:
            logger.debug(f"Error checking imports in {file_path}: {e}")

    def validate_file_naming(self, structure: dict[str, Any]) -> None:
        """Validate file and directory naming conventions."""
        files = structure['files']
        directories = structure['directories']

        # Check Python file naming
        python_files = [f for f in files if f.endswith('.py')]
        for py_file in python_files:
            filename = Path(py_file).name

            # Check for invalid characters
            if not re.match(r'^[a-z0-9_]+\.py$', filename):
                if re.search(r'[A-Z]', filename):
                    self.issues.append(StructureIssue(
                        level='suggestion',
                        category='naming',
                        path=py_file,
                        message='Python file uses camelCase/PascalCase',
                        recommendation='Use snake_case for Python files'
                    ))
                elif re.search(r'[^a-zA-Z0-9_.]', filename):
                    self.issues.append(StructureIssue(
                        level='warning',
                        category='naming',
                        path=py_file,
                        message='Python file contains special characters',
                        recommendation='Use only letters, numbers, and underscores'
                    ))

        # Check directory naming
        for directory in directories:
            dir_name = Path(directory).name

            if re.search(r'[A-Z]', dir_name) and not dir_name.startswith('.'):
                self.issues.append(StructureIssue(
                    level='suggestion',
                    category='naming',
                    path=directory,
                    message='Directory uses camelCase/PascalCase',
                    recommendation='Use lowercase with underscores for directories'
                ))

    def validate_file_sizes(self, structure: dict[str, Any]) -> None:
        """Validate file sizes for potential issues."""
        files = structure['files']

        # Large file thresholds
        large_file_mb = 10 * 1024 * 1024  # 10 MB
        huge_file_mb = 100 * 1024 * 1024  # 100 MB

        for file_path, file_info in files.items():
            size = file_info['size']

            if size > huge_file_mb:
                self.issues.append(StructureIssue(
                    level='error',
                    category='file_size',
                    path=file_path,
                    message=f'Very large file ({size / (1024*1024):.1f} MB)',
                    recommendation='Consider Git LFS or external storage'
                ))
            elif size > large_file_mb:
                # Check if it's a common large file type
                if file_info['suffix'] in ['.pkl', '.pt', '.pth', '.h5', '.onnx']:
                    self.issues.append(StructureIssue(
                        level='suggestion',
                        category='file_size',
                        path=file_path,
                        message=f'Large model file ({size / (1024*1024):.1f} MB)',
                        recommendation='Consider Git LFS for model files'
                    ))
                else:
                    self.issues.append(StructureIssue(
                        level='warning',
                        category='file_size',
                        path=file_path,
                        message=f'Large file ({size / (1024*1024):.1f} MB)',
                        recommendation='Review if file should be in repository'
                    ))

    def check_anti_patterns(self, structure: dict[str, Any]) -> None:
        """Check for common anti-patterns."""
        directories = structure['directories']

        # Check for nested directories anti-pattern
        for pattern, description in self.anti_patterns.items():
            if any(pattern in d for d in directories):
                matching_dirs = [d for d in directories if pattern in d]
                for dir_path in matching_dirs:
                    self.issues.append(StructureIssue(
                        level='warning',
                        category='anti_pattern',
                        path=dir_path,
                        message=description,
                        recommendation='Simplify directory structure'
                    ))

        # Check for too deep nesting
        max_depth = max((info['depth'] for info in directories.values()), default=0)
        if max_depth > 6:
            deep_dirs = [
                path for path, info in directories.items()
                if info['depth'] > 5
            ]
            for deep_dir in deep_dirs[:5]:  # Show first 5
                self.issues.append(StructureIssue(
                    level='suggestion',
                    category='organization',
                    path=deep_dir,
                    message=f'Very deep directory nesting (depth {directories[deep_dir]["depth"]})',
                    recommendation='Consider flattening directory structure'
                ))

    def check_git_integration(self) -> None:
        """Check Git-related files and structure."""
        gitignore_path = self.project_root / '.gitignore'

        if not gitignore_path.exists():
            self.issues.append(StructureIssue(
                level='error',
                category='version_control',
                path='.gitignore',
                message='No .gitignore file found',
                recommendation='Create .gitignore to exclude unnecessary files'
            ))
        else:
            try:
                with open(gitignore_path, encoding='utf-8') as f:
                    gitignore_content = f.read()

                # Check for common Python entries
                python_patterns = [
                    '__pycache__', '*.pyc', '*.pyo', '*.pyd',
                    '.Python', 'env/', 'venv/', '.venv/',
                    '.pytest_cache', '.coverage', '.mypy_cache'
                ]

                missing_patterns = []
                for pattern in python_patterns:
                    if pattern not in gitignore_content:
                        missing_patterns.append(pattern)

                if missing_patterns:
                    self.issues.append(StructureIssue(
                        level='suggestion',
                        category='version_control',
                        path='.gitignore',
                        message=f'Missing common Python patterns: {", ".join(missing_patterns[:3])}',
                        recommendation='Add standard Python .gitignore entries'
                    ))

            except Exception as e:
                logger.debug(f"Error reading .gitignore: {e}")

        # Check if .git directory exists
        if not (self.project_root / '.git').exists():
            self.issues.append(StructureIssue(
                level='suggestion',
                category='version_control',
                path='.',
                message='Not a Git repository',
                recommendation='Initialize Git repository with: git init'
            ))

    def check_modern_practices(self, structure: dict[str, Any]) -> None:
        """Check for modern Python development practices."""
        files = structure['files']

        # Check for modern configuration files
        modern_files = {
            'pyproject.toml': 'Modern Python configuration',
            '.pre-commit-config.yaml': 'Pre-commit hooks',
            '.github/workflows/': 'GitHub Actions CI/CD',
            'tox.ini': 'Testing automation',
            'noxfile.py': 'Testing automation (modern alternative to tox)',
        }

        modern_score = 0
        for modern_file in modern_files:
            if modern_file in files or any(modern_file in f for f in files):
                modern_score += 1

        if modern_score < 2:
            self.issues.append(StructureIssue(
                level='suggestion',
                category='modernization',
                path='.',
                message='Few modern development practices detected',
                recommendation='Consider adding pyproject.toml, pre-commit, and CI/CD'
            ))

        # Check for type hints
        python_files = [f for f in files if f.endswith('.py')]
        type_hint_files = 0

        for py_file in python_files[:10]:  # Sample first 10 files
            try:
                full_path = self.project_root / py_file
                with open(full_path, encoding='utf-8') as f:
                    content = f.read()

                if re.search(r':\s*[A-Za-z]|\->', content):
                    type_hint_files += 1
            except Exception:
                continue

        if len(python_files) > 5 and type_hint_files < len(python_files) * 0.3:
            self.issues.append(StructureIssue(
                level='suggestion',
                category='modernization',
                path='.',
                message='Few files use type hints',
                recommendation='Consider adding type hints for better code quality'
            ))

    def generate_structure_report(self) -> dict[str, Any]:
        """Generate comprehensive structure analysis report."""
        structure = self.scan_project_structure()

        # Count issues by category and level
        issue_summary = defaultdict(lambda: defaultdict(int))
        for issue in self.issues:
            issue_summary[issue.level][issue.category] += 1

        # Generate recommendations
        recommendations = []
        error_count = len([i for i in self.issues if i.level == 'error'])
        warning_count = len([i for i in self.issues if i.level == 'warning'])

        if error_count > 0:
            recommendations.append("Fix critical structure errors first")

        if warning_count > 5:
            recommendations.append("Address organizational warnings")

        if 'src' not in structure['directories']:
            recommendations.append("Consider adopting src/ layout for better organization")

        if error_count == 0 and warning_count < 3:
            recommendations.append("Good structure! Consider modernization suggestions")

        report = {
            "timestamp": "now",
            "project_root": str(self.project_root),
            "structure_summary": {
                "total_files": structure['total_files'],
                "total_directories": structure['total_directories'],
                "python_packages": len(structure['python_packages']),
            },
            "issue_summary": dict(issue_summary),
            "total_issues": len(self.issues),
            "issues_by_level": {
                "error": len([i for i in self.issues if i.level == 'error']),
                "warning": len([i for i in self.issues if i.level == 'warning']),
                "suggestion": len([i for i in self.issues if i.level == 'suggestion']),
            },
            "recommendations": recommendations,
            "detailed_issues": [asdict(issue) for issue in self.issues]
        }

        return report

    def run_validation(self) -> bool:
        """Run complete structure validation."""
        logger.info("Starting project structure validation...")

        try:
            structure = self.scan_project_structure()

            # Run all validation checks
            self.validate_required_files(structure)
            self.validate_directory_structure(structure)
            self.validate_python_package_structure(structure)
            self.validate_file_naming(structure)
            self.validate_file_sizes(structure)
            self.check_anti_patterns(structure)
            self.check_git_integration()
            self.check_modern_practices(structure)

            logger.info(f"Structure validation complete. Found {len(self.issues)} issues.")

            # Return True if no errors
            return len([i for i in self.issues if i.level == 'error']) == 0

        except Exception as e:
            logger.error(f"Structure validation failed: {e}")
            return False

    def print_report(self, show_suggestions: bool = True) -> None:
        """Print structure validation report."""
        if not self.issues:
            print("✅ Project structure validation passed - no issues found!")
            return

        print("\n" + "="*80)
        print("PROJECT STRUCTURE VALIDATION REPORT")
        print("="*80)

        # Summary
        error_count = len([i for i in self.issues if i.level == 'error'])
        warning_count = len([i for i in self.issues if i.level == 'warning'])
        suggestion_count = len([i for i in self.issues if i.level == 'suggestion'])

        print(f"\nSummary: {error_count} errors, {warning_count} warnings, {suggestion_count} suggestions")

        # Group issues by level
        for level in ['error', 'warning', 'suggestion']:
            if level == 'suggestion' and not show_suggestions:
                continue

            level_issues = [i for i in self.issues if i.level == level]
            if not level_issues:
                continue

            icon = {'error': '❌', 'warning': '⚠️', 'suggestion': '💡'}[level]
            print(f"\n{icon} {level.upper()}S ({len(level_issues)})")
            print("-" * 40)

            # Group by category
            categories = defaultdict(list)
            for issue in level_issues:
                categories[issue.category].append(issue)

            for category, cat_issues in categories.items():
                print(f"\n  📂 {category.upper()}")
                for issue in cat_issues:
                    print(f"    {issue.path}: {issue.message}")
                    if issue.recommendation:
                        print(f"      💡 {issue.recommendation}")

        print("\n" + "="*80)

    def save_report(self, output_file: Path) -> None:
        """Save structure report to JSON file."""
        report = self.generate_structure_report()

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Structure report saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate project structure against best practices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_structure.py                   # Validate current directory
  python validate_structure.py --project-root /path/to/project
  python validate_structure.py --no-suggestions # Hide suggestions
  python validate_structure.py --output report.json    # Save detailed report
        """,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )

    parser.add_argument(
        "--no-suggestions",
        action="store_true",
        help="Hide suggestion-level messages",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Save detailed report to JSON file",
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
    validator = ProjectStructureValidator(args.project_root)
    success = validator.run_validation()

    # Print report
    validator.print_report(show_suggestions=not args.no_suggestions)

    # Save report if requested
    if args.output:
        validator.save_report(args.output)

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
