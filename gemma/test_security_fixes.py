#!/usr/bin/env python3
"""Demonstrate security vulnerability fixes in expand_path function."""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("SECURITY VULNERABILITY TEST - Path Traversal and Symlink Escape")
print("=" * 70)

# Test both versions
print("\n1. Testing ORIGINAL expand_path (vulnerable):")
print("-" * 50)

try:
    from src.gemma_cli.config.settings import expand_path as expand_path_original

    # Test 1: Environment variable injection (CVE-style attack)
    print("\nTest 1: Environment Variable Injection Attack")
    os.environ["EVIL_PATH"] = "../../.."
    try:
        result = expand_path_original("$EVIL_PATH/etc/passwd")
        print(f"  ❌ VULNERABLE: Allowed path traversal via env var: {result}")
    except ValueError as e:
        print(f"  ✅ Protected: {str(e).splitlines()[0]}")
    finally:
        del os.environ["EVIL_PATH"]

    # Test 2: Direct path traversal
    print("\nTest 2: Direct Path Traversal Attack")
    try:
        result = expand_path_original("../../../etc/passwd")
        print(f"  ❌ VULNERABLE: Allowed direct traversal: {result}")
    except ValueError as e:
        print(f"  ✅ Protected: {str(e).splitlines()[0]}")

except ImportError as e:
    print(f"  Could not import original expand_path: {e}")

print("\n2. Testing SECURE expand_path_secure (fixed):")
print("-" * 50)

try:
    from src.gemma_cli.config.settings_secure import expand_path_secure

    # Test 1: Environment variable injection
    print("\nTest 1: Environment Variable Injection Attack")
    os.environ["EVIL_PATH"] = "../../.."
    try:
        result = expand_path_secure("$EVIL_PATH/etc/passwd")
        print(f"  ❌ VULNERABLE: Allowed path traversal via env var: {result}")
    except ValueError as e:
        print(f"  ✅ PROTECTED: {str(e).splitlines()[0]}")
    finally:
        del os.environ["EVIL_PATH"]

    # Test 2: Direct path traversal (should be blocked immediately)
    print("\nTest 2: Direct Path Traversal Attack")
    try:
        result = expand_path_secure("../../../etc/passwd")
        print(f"  ❌ VULNERABLE: Allowed direct traversal: {result}")
    except ValueError as e:
        print(f"  ✅ PROTECTED: {str(e).splitlines()[0]}")

    # Test 3: Encoded path traversal
    print("\nTest 3: URL-Encoded Path Traversal Attack")
    try:
        result = expand_path_secure("%2e%2e/%2e%2e/etc/passwd")
        print(f"  ❌ VULNERABLE: Allowed encoded traversal: {result}")
    except ValueError as e:
        print(f"  ✅ PROTECTED: {str(e).splitlines()[0]}")

    # Test 4: Complex nested env var injection
    print("\nTest 4: Nested Environment Variable Injection")
    os.environ["LEVEL1"] = "$LEVEL2/config"
    os.environ["LEVEL2"] = "../.."
    try:
        result = expand_path_secure("$LEVEL1/secret")
        print(f"  ❌ VULNERABLE: Allowed nested env var traversal: {result}")
    except ValueError as e:
        print(f"  ✅ PROTECTED: {str(e).splitlines()[0]}")
    finally:
        if "LEVEL1" in os.environ:
            del os.environ["LEVEL1"]
        if "LEVEL2" in os.environ:
            del os.environ["LEVEL2"]

    # Test 5: Symlink escape test
    print("\nTest 5: Symlink Escape Attack")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        safe_dir = tmp / "safe"
        safe_dir.mkdir()

        outside_dir = tmp / "outside"
        outside_dir.mkdir()
        secret_file = outside_dir / "secret.txt"
        secret_file.write_text("secret data")

        # Create relative symlink that escapes safe directory
        symlink = safe_dir / "escape.txt"
        try:
            symlink.symlink_to("../../outside/secret.txt")

            # Try to access via symlink
            try:
                result = expand_path_secure(str(symlink), allowed_dirs=[safe_dir])
                print(f"  ❌ VULNERABLE: Symlink escape allowed: {result}")
            except ValueError as e:
                print(f"  ✅ PROTECTED: {str(e).splitlines()[0]}")
        except OSError:
            print("  ⚠️  Cannot create symlinks (may need admin privileges)")

    # Test 6: Valid path (should work)
    print("\nTest 6: Valid Path (should be allowed)")
    try:
        home_config = Path.home() / ".gemma_cli" / "config.toml"
        result = expand_path_secure(str(home_config))
        print(f"  ✅ ALLOWED: Valid path accepted: {result}")
    except ValueError as e:
        print(f"  ❌ ERROR: Valid path rejected: {e}")

except ImportError as e:
    print(f"  Could not import secure expand_path: {e}")

print("\n" + "=" * 70)
print("VULNERABILITY SUMMARY:")
print("-" * 70)
print("""
ORIGINAL expand_path vulnerabilities:
1. ❌ Checks ".." AFTER environment variable expansion
   → Allows: export EVIL="../.."; expand_path("$EVIL/etc/passwd")

2. ❌ Only validates absolute symlinks
   → Allows: ln -s ../../etc/shadow safe/link.txt

SECURE expand_path_secure fixes:
1. ✅ Validates BEFORE and AFTER expansion
   → Blocks environment variable injection attacks

2. ✅ Properly resolves ALL symlinks (relative and absolute)
   → Blocks symlink escape attacks

3. ✅ Checks for encoded path traversal (%2e%2e)
   → Blocks URL-encoded attacks

4. ✅ Uses is_relative_to() when available (Python 3.9+)
   → More secure path comparison
""")
print("=" * 70)