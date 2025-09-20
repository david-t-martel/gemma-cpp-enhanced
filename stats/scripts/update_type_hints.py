#!/usr/bin/env python3
"""
Update Python type hints automatically.

This script analyzes Python code and automatically adds/updates type hints
using static analysis, runtime information, and intelligent inference.
"""

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Configure logging
s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("update_type_hints.log"),
    ],
)

@dataclass
class TypeHintSuggestion:
    """Represents a type hint suggestion."""

    file_path: str
    line_number: int
    function_name: str
    parameter_name: str | None
    current_hint: str | None
    suggested_hint: str
    confidence: float  # 0.0 to 1.0
    reason: str

    def __str__(self) -> str:
        """String representation of the suggestion."""
        location = f"{self.file_path}:{self.line_number}"
        if self.parameter_name:
            return f"{location} - {self.function_name}({self.parameter_name}): {self.suggested_hint}"
        else:
            return f"{location} - {self.function_name}() -> {self.suggested_hint}"

class TypeHintUpdater:
    """Updates type hints in Python code."""

    def __init__(self, project_root: Path):
        """Initialize the type hint updater."""
        self.project_root = project_root
        self.suggestions: list[TypeHintSuggestion] = []
        self.type_usage_stats: dict[str, Counter] = defaultdict(Counter)

        # Common type mappings
        self.type_mappings = {
            'str': 'str',
            'string': 'str',
            'int': 'int',
            'integer': 'int',
            'float': 'float',
            'bool': 'bool',
            'boolean': 'bool',
            'list': 'List',
            'dict': 'Dict',
            'tuple': 'Tuple',
            'set': 'Set',
            'none': 'None',
            'any': 'Any',
        }

        # Import statements we might need to add
        self.typing_imports = {
            'List', 'Dict', 'Tuple', 'Set', 'Optional', 'Union', 'Any',
            'Callable', 'Iterator', 'Generator', 'Type', 'ClassVar'
        }

    def get_python_files(self, directory: Path | None = None) -> list[Path]:
        """Get all Python files in the project."""
        if directory is None:
            directory = self.project_root

        python_files = []
        for pattern in ['**/*.py']:
            python_files.extend(directory.glob(pattern))

        # Filter out common non-source files
        exclude_patterns = [
            '__pycache__', '.pytest_cache', '.mypy_cache',
            'venv', '.venv', 'env', '.env',
            'build', 'dist', '.git'
        ]

        filtered_files = []
        for file_path in python_files:
            if not any(pattern in str(file_path) for pattern in exclude_patterns):
                filtered_files.append(file_path)

        return filtered_files

    def parse_file(self, file_path: Path) -> ast.AST | None:
        """Parse a Python file into an AST."""
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
            return ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return None

    def extract_function_info(self, node: ast.FunctionDef) -> dict[str, Any]:
        """Extract information about a function."""
        info = {
            'name': node.name,
            'line_number': node.lineno,
            'parameters': [],
            'return_annotation': None,
            'docstring': ast.get_docstring(node),
            'has_return': False,
            'return_types': set(),
        }

        # Extract parameter information
        for arg in node.args.args:
            param_info = {
                'name': arg.arg,
                'annotation': ast.unparse(arg.annotation) if arg.annotation else None,
                'default': None,
                'inferred_types': set(),
            }
            info['parameters'].append(param_info)

        # Extract return annotation
        if node.returns:
            info['return_annotation'] = ast.unparse(node.returns)

        # Analyze function body for type information
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Return) and stmt.value:
                info['has_return'] = True
                return_type = self.infer_type_from_node(stmt.value)
                if return_type:
                    info['return_types'].add(return_type)

        return info

    def infer_type_from_node(self, node: ast.AST) -> str | None:
        """Infer type from an AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return 'str'
            elif isinstance(node.value, int):
                return 'int'
            elif isinstance(node.value, float):
                return 'float'
            elif isinstance(node.value, bool):
                return 'bool'
            elif node.value is None:
                return 'None'

        elif isinstance(node, ast.List):
            return 'List'
        elif isinstance(node, ast.Dict):
            return 'Dict'
        elif isinstance(node, ast.Tuple):
            return 'Tuple'
        elif isinstance(node, ast.Set):
            return 'Set'

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in ('list', 'dict', 'tuple', 'set'):
                    return func_name.capitalize()
                elif func_name in ('str', 'int', 'float', 'bool'):
                    return func_name
                elif func_name == 'len':
                    return 'int'
                elif func_name in ('min', 'max', 'sum'):
                    return 'Union[int, float]'

        elif isinstance(node, ast.BinOp):
            left_type = self.infer_type_from_node(node.left)
            right_type = self.infer_type_from_node(node.right)

            if left_type == right_type:
                return left_type
            elif left_type in ('int', 'float') and right_type in ('int', 'float'):
                return 'float' if 'float' in (left_type, right_type) else 'int'

        elif isinstance(node, ast.Compare):
            return 'bool'

        return None

    def analyze_docstring_types(self, docstring: str) -> dict[str, str]:
        """Extract type information from docstring."""
        type_info = {}
        if not docstring:
            return type_info

        # Google-style docstrings
        args_section = False
        returns_section = False

        for line in docstring.split('\n'):
            line = line.strip()

            if line.lower().startswith('args:') or line.lower().startswith('arguments:'):
                args_section = True
                returns_section = False
                continue
            elif line.lower().startswith('returns:') or line.lower().startswith('return:'):
                args_section = False
                returns_section = True
                continue
            elif line.lower().startswith(('raises:', 'yields:', 'examples:')):
                args_section = False
                returns_section = False
                continue

            if args_section and ':' in line:
                # Parse parameter type: "param_name (type): description"
                match = re.match(r'(\w+)\s*\(([^)]+)\)\s*:', line)
                if match:
                    param_name, param_type = match.groups()
                    type_info[param_name] = param_type.strip()

            elif returns_section and line:
                # Parse return type
                if line.startswith('(') and ')' in line:
                    return_type = line.split(')')[0][1:].strip()
                    type_info['return'] = return_type

        # Sphinx-style docstrings
        for match in re.finditer(r':param\s+(\w+)\s+(\w+):', docstring):
            param_type, param_name = match.groups()
            type_info[param_name] = param_type

        for match in re.finditer(r':rtype:\s*(.+)', docstring):
            return_type = match.group(1).strip()
            type_info['return'] = return_type

        return type_info

    def run_mypy_analysis(self, file_path: Path) -> dict[str, Any]:
        """Run mypy on a file to get type information."""
        try:
            result = subprocess.run([
                'uv', 'run', 'mypy',
                '--show-error-codes',
                '--no-error-summary',
                str(file_path)
            ], check=False, capture_output=True, text=True, timeout=30)

            mypy_info = {'errors': [], 'suggestions': []}

            for line in result.stdout.split('\n'):
                if 'error:' in line.lower():
                    mypy_info['errors'].append(line.strip())

            return mypy_info

        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug(f"MyPy analysis failed for {file_path}")
            return {'errors': [], 'suggestions': []}

    def generate_type_suggestions(self, file_path: Path) -> list[TypeHintSuggestion]:
        """Generate type hint suggestions for a file."""
        suggestions = []
        tree = self.parse_file(file_path)

        if not tree:
            return suggestions

        # Analyze each function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = self.extract_function_info(node)
                docstring_types = self.analyze_docstring_types(func_info['docstring'])

                # Suggest parameter type hints
                for param in func_info['parameters']:
                    if not param['annotation']:
                        suggested_type = None
                        confidence = 0.0
                        reason = ""

                        # Check docstring for type information
                        if param['name'] in docstring_types:
                            suggested_type = docstring_types[param['name']]
                            confidence = 0.8
                            reason = "from docstring"

                        # Infer from common patterns
                        elif param['name'].endswith('_path') or param['name'] == 'path':
                            suggested_type = 'Union[str, Path]'
                            confidence = 0.7
                            reason = "path parameter pattern"

                        elif param['name'].endswith('_id') or param['name'] == 'id':
                            suggested_type = 'Union[str, int]'
                            confidence = 0.6
                            reason = "ID parameter pattern"

                        elif param['name'] in ('data', 'payload'):
                            suggested_type = 'Dict[str, Any]'
                            confidence = 0.5
                            reason = "data parameter pattern"

                        elif param['name'] in ('items', 'values'):
                            suggested_type = 'List[Any]'
                            confidence = 0.5
                            reason = "list parameter pattern"

                        elif param['name'] in ('config', 'settings'):
                            suggested_type = 'Dict[str, Any]'
                            confidence = 0.5
                            reason = "config parameter pattern"

                        if suggested_type:
                            suggestions.append(TypeHintSuggestion(
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=node.lineno,
                                function_name=func_info['name'],
                                parameter_name=param['name'],
                                current_hint=param['annotation'],
                                suggested_hint=suggested_type,
                                confidence=confidence,
                                reason=reason
                            ))

                # Suggest return type hints
                if not func_info['return_annotation'] and func_info['has_return']:
                    suggested_return_type = None
                    confidence = 0.0
                    reason = ""

                    # Check docstring
                    if 'return' in docstring_types:
                        suggested_return_type = docstring_types['return']
                        confidence = 0.8
                        reason = "from docstring"

                    # Infer from return statements
                    elif func_info['return_types']:
                        return_types = list(func_info['return_types'])
                        if len(return_types) == 1:
                            suggested_return_type = return_types[0]
                            confidence = 0.7
                            reason = "inferred from return statement"
                        elif len(return_types) > 1 and 'None' not in return_types:
                            suggested_return_type = f"Union[{', '.join(sorted(return_types))}]"
                            confidence = 0.6
                            reason = "inferred from multiple return statements"

                    # Common patterns
                    elif func_info['name'].startswith('is_') or func_info['name'].startswith('has_'):
                        suggested_return_type = 'bool'
                        confidence = 0.8
                        reason = "boolean function pattern"

                    elif func_info['name'].startswith('get_') or func_info['name'].startswith('find_'):
                        suggested_return_type = 'Optional[Any]'
                        confidence = 0.5
                        reason = "getter function pattern"

                    elif func_info['name'].startswith('create_') or func_info['name'].startswith('build_'):
                        suggested_return_type = 'Any'
                        confidence = 0.4
                        reason = "creator function pattern"

                    if suggested_return_type:
                        suggestions.append(TypeHintSuggestion(
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=node.lineno,
                            function_name=func_info['name'],
                            parameter_name=None,
                            current_hint=func_info['return_annotation'],
                            suggested_hint=suggested_return_type,
                            confidence=confidence,
                            reason=reason
                        ))

        return suggestions

    def apply_type_hints(self, file_path: Path, suggestions: list[TypeHintSuggestion],
                        min_confidence: float = 0.6) -> bool:
        """Apply type hints to a file."""
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            modified = False
            needed_imports = set()

            # Group suggestions by line number
            suggestions_by_line = defaultdict(list)
            for suggestion in suggestions:
                if suggestion.confidence >= min_confidence:
                    suggestions_by_line[suggestion.line_number].append(suggestion)

            # Apply suggestions
            for line_num, line_suggestions in suggestions_by_line.items():
                if line_num <= len(lines):
                    line = lines[line_num - 1]

                    for suggestion in line_suggestions:
                        # Extract typing imports needed
                        for typing_type in self.typing_imports:
                            if typing_type in suggestion.suggested_hint:
                                needed_imports.add(typing_type)

                        # Apply the type hint
                        if suggestion.parameter_name:
                            # Parameter type hint
                            pattern = rf'\b{re.escape(suggestion.parameter_name)}\b'
                            replacement = f'{suggestion.parameter_name}: {suggestion.suggested_hint}'
                            if re.search(pattern, line):
                                line = re.sub(
                                    rf'\b{re.escape(suggestion.parameter_name)}\b(?!\s*:)',
                                    replacement,
                                    line,
                                    count=1
                                )
                                modified = True
                        # Return type hint
                        elif ')' in line and '->' not in line:
                            line = line.replace(')', f') -> {suggestion.suggested_hint}', 1)
                            modified = True

                    lines[line_num - 1] = line

            # Add typing imports if needed
            if needed_imports:
                import_line = f"from typing import {', '.join(sorted(needed_imports))}"

                # Find where to insert the import
                insert_line = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('from typing import'):
                        # Merge with existing typing import
                        existing_imports = line.split('import')[1].strip().split(', ')
                        all_imports = sorted(set(existing_imports) | needed_imports)
                        lines[i] = f"from typing import {', '.join(all_imports)}"
                        modified = True
                        break
                    elif line.strip().startswith('import ') or line.strip().startswith('from '):
                        insert_line = i + 1
                    elif line.strip() and not line.startswith('#'):
                        break

                if not any('from typing import' in line for line in lines):
                    lines.insert(insert_line, import_line)
                    modified = True

            # Write back if modified
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                logger.info(f"Applied type hints to {file_path}")

            return modified

        except Exception as e:
            logger.error(f"Error applying type hints to {file_path}: {e}")
            return False

    def analyze_project(self) -> None:
        """Analyze the entire project for type hint opportunities."""
        logger.info("Analyzing project for type hint opportunities...")

        python_files = self.get_python_files()
        logger.info(f"Found {len(python_files)} Python files")

        for file_path in python_files:
            try:
                file_suggestions = self.generate_type_suggestions(file_path)
                self.suggestions.extend(file_suggestions)
                logger.debug(f"Generated {len(file_suggestions)} suggestions for {file_path}")
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")

        logger.info(f"Generated {len(self.suggestions)} type hint suggestions")

    def print_report(self, min_confidence: float = 0.5) -> None:
        """Print type hint analysis report."""
        filtered_suggestions = [
            s for s in self.suggestions if s.confidence >= min_confidence
        ]

        print("\n" + "="*80)
        print("TYPE HINT ANALYSIS REPORT")
        print("="*80)

        print(f"\nTotal suggestions: {len(self.suggestions)}")
        print(f"High confidence (>={min_confidence}): {len(filtered_suggestions)}")

        # Group by confidence level
        confidence_groups = {
            'high': [s for s in filtered_suggestions if s.confidence >= 0.8],
            'medium': [s for s in filtered_suggestions if 0.6 <= s.confidence < 0.8],
            'low': [s for s in filtered_suggestions if 0.5 <= s.confidence < 0.6],
        }

        for level, suggestions in confidence_groups.items():
            if suggestions:
                icon = {'high': '🔥', 'medium': '⚡', 'low': '💡'}[level]
                print(f"\n{icon} {level.upper()} CONFIDENCE ({len(suggestions)})")
                print("-" * 40)

                for suggestion in suggestions[:10]:  # Show first 10
                    print(f"  {suggestion} ({suggestion.reason})")

        # Summary by file
        files_with_suggestions = defaultdict(int)
        for suggestion in filtered_suggestions:
            files_with_suggestions[suggestion.file_path] += 1

        if files_with_suggestions:
            print("\n📁 FILES WITH MOST SUGGESTIONS")
            print("-" * 40)
            sorted_files = sorted(
                files_with_suggestions.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for file_path, count in sorted_files[:10]:
                print(f"  {file_path}: {count} suggestions")

        print("\n" + "="*80)

    def run_update(self, apply: bool = False, min_confidence: float = 0.6) -> None:
        """Run the complete type hint update process."""
        logger.info("Starting type hint analysis and updates...")

        try:
            self.analyze_project()
            self.print_report(min_confidence)

            if apply:
                logger.info("Applying type hints...")
                files_modified = 0

                # Group suggestions by file
                suggestions_by_file = defaultdict(list)
                for suggestion in self.suggestions:
                    suggestions_by_file[suggestion.file_path].append(suggestion)

                for file_path_str, file_suggestions in suggestions_by_file.items():
                    file_path = self.project_root / file_path_str
                    if self.apply_type_hints(file_path, file_suggestions, min_confidence):
                        files_modified += 1

                logger.info(f"Modified {files_modified} files")
            else:
                logger.info("Analysis complete. Use --apply to update files.")

        except Exception as e:
            logger.error(f"Type hint update failed: {e}")
            sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update Python type hints automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python update_type_hints.py                    # Analyze and show suggestions
  python update_type_hints.py --apply           # Apply type hints
  python update_type_hints.py --min-confidence 0.8    # Only high-confidence suggestions
  python update_type_hints.py --file src/main.py      # Analyze specific file
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
        help="Apply type hints to files (default: analysis only)",
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold for suggestions (0.0-1.0)",
    )

    parser.add_argument(
        "--file",
        type=Path,
        help="Analyze specific file only",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:

    # Validate arguments
    if not 0.0 <= args.min_confidence <= 1.0:
        logger.error("Minimum confidence must be between 0.0 and 1.0")
        sys.exit(1)

    if not args.project_root.exists():
        logger.error(f"Project root does not exist: {args.project_root}")
        sys.exit(1)

    # Run type hint updates
    updater = TypeHintUpdater(args.project_root)

    if args.file:
        if not args.file.exists():
            logger.error(f"File does not exist: {args.file}")
            sys.exit(1)

        # Analyze single file
        suggestions = updater.generate_type_suggestions(args.file)
        updater.suggestions = suggestions

        if args.apply:
            updater.apply_type_hints(args.file, suggestions, args.min_confidence)
    else:
        # Analyze entire project
        updater.run_update(apply=args.apply, min_confidence=args.min_confidence)

if __name__ == "__main__":
    main()
