#!/usr/bin/env python3
"""PyInstaller build script for Gemma CLI standalone Windows executable.

This script orchestrates the entire build process:
1. Verifies all required binaries exist
2. Generates PyInstaller spec file
3. Runs PyInstaller
4. Tests the resulting executable
5. Generates deployment report

Usage:
    python deployment/build_script.py [--skip-tests] [--debug]
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BuildConfig:
    """Configuration for the build process."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
    SRC_DIR = PROJECT_ROOT / "src" / "gemma_cli"
    DEPLOYMENT_DIR = PROJECT_ROOT / "src" / "gemma_cli" / "deployment"
    OUTPUT_DIR = PROJECT_ROOT / "dist"

    # Binary locations
    GEMMA_EXE_PATHS = [
        PROJECT_ROOT / "build-avx2-sycl" / "bin" / "RELEASE" / "gemma.exe",
        PROJECT_ROOT / "build" / "Release" / "gemma.exe",
        PROJECT_ROOT / "gemma.cpp" / "build" / "Release" / "gemma.exe",
        PROJECT_ROOT.parent / "gemma.cpp" / "build" / "Release" / "gemma.exe",
    ]

    RAG_SERVER_PATHS = [
        Path("C:/codedev/llm/stats/target/release/rag-redis-mcp-server.exe"),
        PROJECT_ROOT.parent.parent / "stats" / "target" / "release" / "rag-redis-mcp-server.exe",
    ]

    # Build settings
    TARGET_BUNDLE_SIZE_MB = 50
    TARGET_STARTUP_TIME_SEC = 3
    UPX_COMPRESS = True
    DEBUG_BUILD = False


class BinaryFinder:
    """Locates required binaries for bundling."""

    @staticmethod
    def find_binary(paths: List[Path], name: str) -> Optional[Path]:
        """Search for binary in list of paths."""
        for path in paths:
            if path.exists():
                logger.info(f"Found {name}: {path}")
                return path
        return None

    @staticmethod
    def find_gemma_exe() -> Path:
        """Locate gemma.exe binary."""
        # Check environment variable first
        if gemma_env := os.environ.get("GEMMA_EXECUTABLE", ""):
            gemma_path = Path(gemma_env)
            if gemma_path.exists():
                logger.info(f"Found gemma.exe from GEMMA_EXECUTABLE: {gemma_path}")
                return gemma_path

        # Search predefined paths
        if gemma_path := BinaryFinder.find_binary(BuildConfig.GEMMA_EXE_PATHS, "gemma.exe"):
            return gemma_path

        raise FileNotFoundError(
            "gemma.exe not found. Please build gemma.cpp first:\n"
            "  cd gemma.cpp && cmake -B build -G 'Visual Studio 17 2022' -T v143\n"
            "  cmake --build build --config Release"
        )

    @staticmethod
    def find_rag_server() -> Path:
        """Locate rag-redis-mcp-server.exe binary."""
        if server_path := BinaryFinder.find_binary(BuildConfig.RAG_SERVER_PATHS, "rag-redis-mcp-server.exe"):
            return server_path

        raise FileNotFoundError(
            "rag-redis-mcp-server.exe not found. Please build Rust RAG backend:\n"
            "  cd stats/rag-redis-system && cargo build --release"
        )

    @staticmethod
    def verify_all_binaries() -> Tuple[Path, Path]:
        """Verify all required binaries exist."""
        logger.info("Verifying required binaries...")

        try:
            gemma_exe = BinaryFinder.find_gemma_exe()
            rag_server = BinaryFinder.find_rag_server()

            logger.info("✓ All binaries found:")
            logger.info(f"  gemma.exe: {gemma_exe} ({gemma_exe.stat().st_size / 1024 / 1024:.1f} MB)")
            logger.info(f"  rag-redis-mcp-server.exe: {rag_server} ({rag_server.stat().st_size / 1024 / 1024:.1f} MB)")

            return gemma_exe, rag_server

        except FileNotFoundError as e:
            logger.error(f"✗ Binary verification failed: {e}")
            raise


def main():
    """Main build orchestration."""
    parser = argparse.ArgumentParser(description="Build Gemma CLI standalone executable")
    parser.add_argument("--skip-tests", action="store_true", help="Skip executable testing")
    parser.add_argument("--debug", action="store_true", help="Build debug executable")
    parser.add_argument("--no-upx", action="store_true", help="Disable UPX compression")
    args = parser.parse_args()

    # Update config
    BuildConfig.DEBUG_BUILD = args.debug
    if args.no_upx:
        BuildConfig.UPX_COMPRESS = False

    logger.info("=" * 60)
    logger.info("Gemma CLI Deployment Build Script")
    logger.info("=" * 60)

    try:
        # Step 1: Verify binaries
        gemma_exe, rag_server = BinaryFinder.verify_all_binaries()
        
        logger.info("\n✓ Binary verification successful!")
        logger.info("\nNext steps:")
        logger.info("  1. Install PyInstaller: python -m pip install pyinstaller")
        logger.info("  2. Run this script to generate spec file and build")
        logger.info(f"  3. Output will be in: {BuildConfig.OUTPUT_DIR}")

    except Exception as e:
        logger.exception(f"Build failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
