#!/usr/bin/env python3
"""
Check ML model file sizes and provide optimization recommendations.

This script analyzes model files in the project, checks their sizes,
and suggests optimization strategies for large models.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import mimetypes
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Add src to path for scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.shared.logging import LogLevel, get_logger, setup_logging

# Setup logging
setup_logging(level=LogLevel.INFO, console=True)
logger = get_logger(__name__)


@dataclass
class ModelFileInfo:
    """Information about a model file."""

    path: str
    name: str
    size_bytes: int
    size_human: str
    file_type: str
    mime_type: str | None
    checksum: str
    last_modified: str
    is_cached: bool = False
    can_optimize: bool = False
    optimization_suggestions: list[str] = None

    def __post_init__(self):
        if self.optimization_suggestions is None:
            self.optimization_suggestions = []


class ModelSizeChecker:
    """Checks ML model file sizes and provides optimization recommendations."""

    def __init__(self, project_root: Path):
        """Initialize the checker with project root."""
        self.project_root = project_root
        self.model_files: list[ModelFileInfo] = []
        self.total_size = 0

        # Common model file extensions
        self.model_extensions = {
            ".pt",
            ".pth",
            ".ckpt",
            ".pkl",
            ".joblib",
            ".h5",
            ".hdf5",
            ".pb",
            ".onnx",
            ".tflite",
            ".bin",
            ".safetensors",
            ".msgpack",
            ".npz",
        }

        # Size thresholds (in bytes)
        self.size_thresholds = {
            "small": 10 * 1024 * 1024,  # 10 MB
            "medium": 100 * 1024 * 1024,  # 100 MB
            "large": 1024 * 1024 * 1024,  # 1 GB
            "huge": 5 * 1024 * 1024 * 1024,  # 5 GB
        }

        # Model-specific directories to scan
        self.model_directories = [
            "models",
            "model",
            "checkpoints",
            "weights",
            "cache",
            "models_cache",
            ".cache",
            "data/models",
            "src/models",
            "assets/models",
        ]

    def format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.debug(f"Failed to calculate checksum for {file_path}: {e}")
            return "unknown"

    def identify_file_type(self, file_path: Path) -> tuple[str, str | None]:
        """Identify the type of model file."""
        extension = file_path.suffix.lower()
        mime_type, _ = mimetypes.guess_type(str(file_path))

        file_type_map = {
            ".pt": "PyTorch Model",
            ".pth": "PyTorch Model",
            ".ckpt": "Checkpoint File",
            ".pkl": "Pickle File",
            ".joblib": "Joblib File",
            ".h5": "HDF5 Model",
            ".hdf5": "HDF5 Model",
            ".pb": "TensorFlow Model",
            ".onnx": "ONNX Model",
            ".tflite": "TensorFlow Lite",
            ".bin": "Binary Model",
            ".safetensors": "SafeTensors Model",
            ".msgpack": "MessagePack File",
            ".npz": "NumPy Archive",
        }

        return file_type_map.get(extension, f"Unknown ({extension})"), mime_type

    def is_model_file(self, file_path: Path) -> bool:
        """Check if file is likely a model file."""
        # Check extension
        if file_path.suffix.lower() in self.model_extensions:
            return True

        # Check file patterns
        name_lower = file_path.name.lower()
        model_patterns = [
            "model",
            "weight",
            "checkpoint",
            "ckpt",
            "pytorch_model",
            "tf_model",
            "config",
        ]

        return any(pattern in name_lower for pattern in model_patterns)

    def scan_directory(self, directory: Path) -> list[Path]:
        """Scan directory for model files."""
        model_files = []

        if not directory.exists():
            return model_files

        try:
            # Recursively find all files
            for file_path in directory.rglob("*"):
                if file_path.is_file() and self.is_model_file(file_path):
                    model_files.append(file_path)
        except PermissionError as e:
            logger.warning(f"Permission denied accessing {directory}: {e}")
        except Exception as e:
            logger.error(f"Error scanning {directory}: {e}")

        return model_files

    def analyze_file(self, file_path: Path) -> ModelFileInfo:
        """Analyze a single model file."""
        try:
            stat = file_path.stat()
            size_bytes = stat.st_size
            size_human = self.format_size(size_bytes)
            file_type, mime_type = self.identify_file_type(file_path)

            # Calculate checksum for files under 100MB to avoid long delays
            checksum = "skipped"
            if size_bytes < 100 * 1024 * 1024:
                checksum = self.calculate_checksum(file_path)

            # Check if file is in cache directory
            is_cached = any(
                cache_dir in file_path.parts for cache_dir in ["cache", ".cache", "__pycache__"]
            )

            # Generate optimization suggestions
            suggestions = self.generate_optimization_suggestions(file_path, size_bytes, file_type)

            model_info = ModelFileInfo(
                path=str(file_path.relative_to(self.project_root)),
                name=file_path.name,
                size_bytes=size_bytes,
                size_human=size_human,
                file_type=file_type,
                mime_type=mime_type,
                checksum=checksum,
                last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                is_cached=is_cached,
                can_optimize=len(suggestions) > 0,
                optimization_suggestions=suggestions,
            )

            return model_info

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return ModelFileInfo(
                path=str(file_path.relative_to(self.project_root)),
                name=file_path.name,
                size_bytes=0,
                size_human="Unknown",
                file_type="Error",
                mime_type=None,
                checksum="error",
                last_modified="unknown",
            )

    def generate_optimization_suggestions(
        self, file_path: Path, size_bytes: int, file_type: str
    ) -> list[str]:
        """Generate optimization suggestions for model files."""
        suggestions = []

        # Size-based suggestions
        if size_bytes > self.size_thresholds["huge"]:
            suggestions.append("Consider model compression or quantization")
            suggestions.append("Use model sharding for distributed loading")
            suggestions.append("Implement lazy loading if possible")

        elif size_bytes > self.size_thresholds["large"]:
            suggestions.append("Consider quantization to reduce model size")
            suggestions.append("Use model compression techniques")

        elif size_bytes > self.size_thresholds["medium"]:
            suggestions.append("Consider using safetensors format for faster loading")

        # File type specific suggestions
        if "PyTorch" in file_type:
            suggestions.extend(
                [
                    "Use torch.save with pickle_protocol=4 for smaller files",
                    "Consider using torch.jit.script for optimized models",
                ]
            )

        elif "Pickle" in file_type:
            suggestions.extend(
                [
                    "Consider switching to safetensors format",
                    "Use protocol=pickle.HIGHEST_PROTOCOL for better compression",
                ]
            )

        elif "HDF5" in file_type:
            suggestions.append("Use compression when saving HDF5 files")

        elif "Binary" in file_type and file_path.suffix == ".bin":
            suggestions.append("Consider converting to safetensors format")

        # Cache-specific suggestions
        if any(cache_dir in file_path.parts for cache_dir in ["cache", ".cache"]):
            suggestions.append("This is a cached file - can be safely deleted if needed")

        # Duplicate detection suggestions
        if size_bytes > self.size_thresholds["medium"]:
            suggestions.append("Check for duplicate models with different names")

        return suggestions

    def find_potential_duplicates(self) -> dict[str, list[ModelFileInfo]]:
        """Find potential duplicate model files."""
        duplicates = {}

        # Group by size and checksum (if available)
        size_groups = {}
        for model in self.model_files:
            if model.size_bytes > 0:
                key = (model.size_bytes, model.checksum)
                if key not in size_groups:
                    size_groups[key] = []
                size_groups[key].append(model)

        # Find groups with multiple files
        for key, models in size_groups.items():
            if len(models) > 1 and key[1] != "unknown" and key[1] != "skipped":
                duplicates[f"size_{key[0]}_checksum_{key[1][:8]}"] = models

        # Also group by name similarity for large files
        name_groups = {}
        for model in self.model_files:
            if model.size_bytes > self.size_thresholds["medium"]:
                # Extract base name without version/timestamp patterns
                base_name = model.name
                for pattern in ["_v1", "_v2", "_final", "_best", "_epoch"]:
                    base_name = base_name.split(pattern)[0]

                if base_name not in name_groups:
                    name_groups[base_name] = []
                name_groups[base_name].append(model)

        # Add name-based duplicates
        for base_name, models in name_groups.items():
            if len(models) > 1:
                duplicates[f"name_{base_name}"] = models

        return duplicates

    def get_size_category(self, size_bytes: int) -> str:
        """Categorize file size."""
        if size_bytes < self.size_thresholds["small"]:
            return "small"
        elif size_bytes < self.size_thresholds["medium"]:
            return "medium"
        elif size_bytes < self.size_thresholds["large"]:
            return "large"
        elif size_bytes < self.size_thresholds["huge"]:
            return "huge"
        else:
            return "massive"

    def scan_models(self) -> None:
        """Scan project for model files."""
        logger.info("Scanning project for model files...")

        # Scan predefined model directories
        for dir_name in self.model_directories:
            directory = self.project_root / dir_name
            if directory.exists():
                logger.info(f"Scanning directory: {directory}")
                found_files = self.scan_directory(directory)
                logger.info(f"Found {len(found_files)} model files in {dir_name}")

                for file_path in found_files:
                    model_info = self.analyze_file(file_path)
                    self.model_files.append(model_info)
                    self.total_size += model_info.size_bytes

        # Also scan root directory for common model files
        logger.info("Scanning root directory for model files...")
        root_files = []
        for pattern in ["*.pt", "*.pth", "*.ckpt", "*.h5", "*.onnx", "*.bin"]:
            root_files.extend(self.project_root.glob(pattern))

        for file_path in root_files:
            if file_path.is_file():
                model_info = self.analyze_file(file_path)
                self.model_files.append(model_info)
                self.total_size += model_info.size_bytes

        logger.info(
            f"Found {len(self.model_files)} model files, "
            f"total size: {self.format_size(self.total_size)}"
        )

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive model size report."""
        if not self.model_files:
            return {
                "summary": {"total_files": 0, "total_size": 0},
                "message": "No model files found in project",
            }

        # Calculate statistics
        sizes = [m.size_bytes for m in self.model_files if m.size_bytes > 0]
        avg_size = sum(sizes) / len(sizes) if sizes else 0

        # Categorize by size
        size_categories = {}
        for model in self.model_files:
            category = self.get_size_category(model.size_bytes)
            if category not in size_categories:
                size_categories[category] = []
            size_categories[category].append(model)

        # Find largest files
        largest_files = sorted(self.model_files, key=lambda x: x.size_bytes, reverse=True)[:10]

        # Find potential duplicates
        duplicates = self.find_potential_duplicates()

        # Count files that can be optimized
        optimizable_files = [m for m in self.model_files if m.can_optimize]

        # Calculate potential savings
        cache_size = sum(m.size_bytes for m in self.model_files if m.is_cached)
        large_files_size = sum(
            m.size_bytes for m in self.model_files if m.size_bytes > self.size_thresholds["large"]
        )

        report = {
            "summary": {
                "total_files": len(self.model_files),
                "total_size": self.total_size,
                "total_size_human": self.format_size(self.total_size),
                "average_size": int(avg_size),
                "average_size_human": self.format_size(int(avg_size)),
                "optimizable_files": len(optimizable_files),
                "cache_size": cache_size,
                "cache_size_human": self.format_size(cache_size),
            },
            "size_categories": {
                category: {
                    "count": len(files),
                    "total_size": sum(f.size_bytes for f in files),
                    "total_size_human": self.format_size(sum(f.size_bytes for f in files)),
                }
                for category, files in size_categories.items()
            },
            "largest_files": [asdict(f) for f in largest_files],
            "duplicates": {key: [asdict(f) for f in files] for key, files in duplicates.items()},
            "optimizable_files": [asdict(f) for f in optimizable_files],
            "recommendations": self.generate_recommendations(
                cache_size, large_files_size, duplicates
            ),
        }

        return report

    def generate_recommendations(
        self, cache_size: int, large_files_size: int, duplicates: dict[str, list[ModelFileInfo]]
    ) -> list[str]:
        """Generate high-level optimization recommendations."""
        recommendations = []

        if cache_size > self.size_thresholds["medium"]:
            recommendations.append(f"Clear cached models to save {self.format_size(cache_size)}")

        if large_files_size > self.size_thresholds["huge"]:
            recommendations.append("Consider model compression/quantization for large models")

        if duplicates:
            total_duplicate_size = sum(
                sum(f.size_bytes for f in files[1:])  # Skip first file in each group
                for files in duplicates.values()
            )
            recommendations.append(
                f"Remove duplicate models to save ~{self.format_size(total_duplicate_size)}"
            )

        if len(self.model_files) > 20:
            recommendations.append("Consider organizing models into subdirectories")

        # Storage recommendations
        if self.total_size > 10 * 1024 * 1024 * 1024:  # 10 GB
            recommendations.extend(
                [
                    "Use Git LFS for large model files",
                    "Consider external model storage (cloud, CDN)",
                    "Implement model versioning strategy",
                ]
            )

        return recommendations

    def print_report(self, detailed: bool = False) -> None:
        """Print model size report."""
        report = self.generate_report()

        if "message" in report:
            print(f"ℹ️  {report['message']}")
            return

        print("\n" + "=" * 80)
        print("MODEL FILE SIZE REPORT")
        print("=" * 80)

        summary = report["summary"]
        print("\n📊 SUMMARY")
        print("-" * 40)
        print(f"Total files: {summary['total_files']}")
        print(f"Total size: {summary['total_size_human']}")
        print(f"Average size: {summary['average_size_human']}")
        print(f"Optimizable files: {summary['optimizable_files']}")
        if summary["cache_size"] > 0:
            print(f"Cache size: {summary['cache_size_human']}")

        # Size categories
        print("\n📏 SIZE CATEGORIES")
        print("-" * 40)
        for category, stats in report["size_categories"].items():
            print(f"{category.capitalize()}: {stats['count']} files ({stats['total_size_human']})")

        # Largest files
        if report["largest_files"]:
            print("\n🔝 LARGEST FILES")
            print("-" * 40)
            for i, file_info in enumerate(report["largest_files"][:5], 1):
                print(
                    f"{i}. {file_info['name']} - {file_info['size_human']} "
                    f"({file_info['file_type']})"
                )

        # Duplicates
        if report["duplicates"]:
            print(f"\n👥 POTENTIAL DUPLICATES ({len(report['duplicates'])} groups)")
            print("-" * 40)
            for group_key, files in list(report["duplicates"].items())[:3]:
                print(f"Group: {group_key}")
                for file_info in files:
                    print(f"  - {file_info['name']} ({file_info['size_human']})")

        # Recommendations
        if report["recommendations"]:
            print("\n💡 RECOMMENDATIONS")
            print("-" * 40)
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"{i}. {rec}")

        # Detailed file list
        if detailed and report["optimizable_files"]:
            print("\n🔧 OPTIMIZABLE FILES")
            print("-" * 40)
            for file_info in report["optimizable_files"][:10]:
                print(f"\n{file_info['name']} ({file_info['size_human']})")
                for suggestion in file_info["optimization_suggestions"][:3]:
                    print(f"  • {suggestion}")

        print("\n" + "=" * 80)

    def save_report(self, output_file: Path) -> None:
        """Save report to JSON file."""
        report = self.generate_report()
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def cleanup_cache(self, dry_run: bool = True) -> list[str]:
        """Clean up cached model files."""
        cache_files = [f for f in self.model_files if f.is_cached]
        removed_files = []

        for cache_file in cache_files:
            file_path = self.project_root / cache_file.path
            if file_path.exists():
                if not dry_run:
                    try:
                        file_path.unlink()
                        logger.info(f"Deleted cache file: {cache_file.path}")
                        removed_files.append(cache_file.path)
                    except Exception as e:
                        logger.error(f"Failed to delete {cache_file.path}: {e}")
                else:
                    logger.info(f"Would delete: {cache_file.path}")
                    removed_files.append(cache_file.path)

        return removed_files


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check ML model file sizes and provide optimization recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_model_size.py                    # Basic scan and report
  python check_model_size.py --detailed         # Detailed optimization suggestions
  python check_model_size.py --output report.json    # Save report to file
  python check_model_size.py --cleanup --dry-run     # Show what cache files would be deleted
        """,
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed optimization suggestions",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Save report to JSON file",
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up cached model files",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting (with --cleanup)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        setup_logging(level=LogLevel.DEBUG, console=True)

    # Validate project root
    if not args.project_root.exists():
        logger.error(f"Project root does not exist: {args.project_root}")
        sys.exit(1)

    # Run analysis
    checker = ModelSizeChecker(args.project_root)
    checker.scan_models()

    # Print report
    checker.print_report(detailed=args.detailed)

    # Save report if requested
    if args.output:
        checker.save_report(args.output)

    # Clean up cache if requested
    if args.cleanup:
        removed = checker.cleanup_cache(dry_run=args.dry_run)
        if removed:
            action = "Would delete" if args.dry_run else "Deleted"
            print(f"\n🧹 CACHE CLEANUP: {action} {len(removed)} files")
            for file_path in removed[:10]:  # Show first 10
                print(f"  - {file_path}")


if __name__ == "__main__":
    main()
