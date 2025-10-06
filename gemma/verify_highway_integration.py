#!/usr/bin/env python3
"""
Verify that Highway GitHub integration is working correctly
"""

import os
import subprocess
import sys
from pathlib import Path

def check_highway_files():
    """Check if Highway files are in the correct location"""
    base_dir = Path(".")
    highway_dir = base_dir / "third_party" / "highway-github"

    print("🔍 Checking Highway GitHub integration...")

    # Check if highway-github directory exists
    if not highway_dir.exists():
        print("❌ Error: third_party/highway-github directory not found")
        return False

    print("✅ Found third_party/highway-github directory")

    # Check for essential files
    essential_files = [
        "CMakeLists.txt",
        "hwy/highway.h",
        "hwy/ops/generic_ops-inl.h",
        "hwy/ops/scalar-inl.h"
    ]

    for file_path in essential_files:
        full_path = highway_dir / file_path
        if not full_path.exists():
            print(f"❌ Error: Missing essential file: {file_path}")
            return False
        print(f"✅ Found: {file_path}")

    return True

def check_scalar_functions():
    """Check if required scalar functions are present"""
    print("\n🔍 Checking for required scalar fallback functions...")

    generic_ops_file = Path("third_party/highway-github/hwy/ops/generic_ops-inl.h")
    if not generic_ops_file.exists():
        print("❌ Error: generic_ops-inl.h not found")
        return False

    required_functions = ["PromoteOddTo", "PromoteUpperTo", "OrderedDemote2To"]

    try:
        with open(generic_ops_file, 'r', encoding='utf-8') as f:
            content = f.read()

        for func in required_functions:
            if func in content:
                print(f"✅ Found function: {func}")
            else:
                print(f"❌ Error: Missing function: {func}")
                return False

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

    return True

def check_commit_hash():
    """Check if we're on the correct commit"""
    print("\n🔍 Checking Git commit hash...")

    highway_dir = Path("third_party/highway-github")
    if not highway_dir.exists():
        return False

    try:
        # Change to highway directory and get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=highway_dir,
            capture_output=True,
            text=True,
            check=True
        )

        current_commit = result.stdout.strip()
        expected_commit = "1d16731233de45a365b43867f27d0a5f73925300"

        if current_commit.startswith(expected_commit):
            print(f"✅ Correct commit: {current_commit}")
            return True
        else:
            print(f"❌ Wrong commit: {current_commit}, expected: {expected_commit}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"❌ Error checking git commit: {e}")
        return False

def check_cmake_integration():
    """Check if CMakeLists.txt is properly configured"""
    print("\n🔍 Checking CMakeLists.txt integration...")

    cmake_file = Path("CMakeLists.txt")
    if not cmake_file.exists():
        print("❌ Error: CMakeLists.txt not found")
        return False

    try:
        with open(cmake_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for highway-github integration
        if "highway-github" in content:
            print("✅ CMakeLists.txt contains highway-github references")
        else:
            print("❌ Error: CMakeLists.txt missing highway-github references")
            return False

        # Check for the specific commit message
        if "1d16731233de45a365b43867f27d0a5f73925300" in content:
            print("✅ CMakeLists.txt references correct commit hash")
        else:
            print("❌ Warning: CMakeLists.txt missing commit hash reference")

        return True

    except Exception as e:
        print(f"❌ Error reading CMakeLists.txt: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 Highway GitHub Integration Verification")
    print("=" * 50)

    all_checks = [
        check_highway_files(),
        check_scalar_functions(),
        check_commit_hash(),
        check_cmake_integration()
    ]

    print("\n" + "=" * 50)
    if all(all_checks):
        print("🎉 All checks passed! Highway GitHub integration is ready.")
        print("\nNext steps:")
        print("1. Build the project to verify compilation")
        print("2. Test that scalar mode fallbacks work correctly")
        return 0
    else:
        print("❌ Some checks failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())