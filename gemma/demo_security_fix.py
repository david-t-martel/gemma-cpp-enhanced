#!/usr/bin/env python3
"""Demonstrate the security vulnerability and fix."""

import os
from pathlib import Path

print("\n" + "=" * 70)
print("PATH TRAVERSAL SECURITY VULNERABILITY DEMONSTRATION")
print("=" * 70)

# Simulate the VULNERABLE version (original code)
def expand_path_vulnerable(path_str: str) -> Path:
    """VULNERABLE VERSION - Expands BEFORE checking for '..'"""
    # VULNERABILITY: Expands first
    expanded = os.path.expanduser(path_str)
    expanded = os.path.expandvars(expanded)

    # Then checks - TOO LATE!
    if ".." in path_str:
        raise ValueError(f"Path traversal not allowed: {path_str}")

    return Path(expanded).resolve()

# SECURE version
def expand_path_secure(path_str: str) -> Path:
    """SECURE VERSION - Checks BEFORE and AFTER expansion"""
    # CHECK 1: Before expansion
    if ".." in path_str:
        raise ValueError(f"Path traversal not allowed in input: {path_str}")

    # Expand
    expanded = os.path.expanduser(path_str)
    expanded = os.path.expandvars(expanded)

    # CHECK 2: After expansion (catches env var injection)
    if ".." in expanded:
        raise ValueError(f"Path traversal detected after expansion: {expanded}")

    return Path(expanded).resolve()

print("\n1. ATTACK SCENARIO: Environment Variable Injection")
print("-" * 50)

# Set up malicious environment variable
os.environ["EVIL"] = "../../../etc"

print(f"   Attacker sets: export EVIL='../../../etc'")
print(f"   Attacker uses: expand_path('$EVIL/passwd')")

print("\n   VULNERABLE function result:")
try:
    result = expand_path_vulnerable("$EVIL/passwd")
    print(f"   ❌ ATTACK SUCCESS! Path resolved to: {result}")
except ValueError as e:
    print(f"   ✅ Blocked: {e}")

print("\n   SECURE function result:")
try:
    result = expand_path_secure("$EVIL/passwd")
    print(f"   ❌ ATTACK SUCCESS! Path resolved to: {result}")
except ValueError as e:
    print(f"   ✅ BLOCKED: {e}")

# Clean up
del os.environ["EVIL"]

print("\n2. DIRECT PATH TRAVERSAL")
print("-" * 50)
print("   Attacker uses: expand_path('../../../etc/passwd')")

print("\n   VULNERABLE function result:")
try:
    result = expand_path_vulnerable("../../../etc/passwd")
    print(f"   ❌ ATTACK SUCCESS! Path resolved to: {result}")
except ValueError as e:
    print(f"   ✅ Blocked: {e}")

print("\n   SECURE function result:")
try:
    result = expand_path_secure("../../../etc/passwd")
    print(f"   ❌ ATTACK SUCCESS! Path resolved to: {result}")
except ValueError as e:
    print(f"   ✅ BLOCKED: {e}")

print("\n" + "=" * 70)
print("SUMMARY:")
print("-" * 70)
print("""
The vulnerability exists because the original code:
1. Expands environment variables FIRST (line 412-413)
2. THEN checks for '..' (line 420)

This allows attackers to inject '..' via environment variables,
bypassing the security check.

The fix:
1. Check for '..' BEFORE expansion (prevents direct attacks)
2. Check AGAIN AFTER expansion (catches env var injection)
3. Also check for URL-encoded variations (%2e%2e)
4. Properly validate symlink targets

This is a CRITICAL security issue that could allow attackers
to read any file on the system (e.g., /etc/shadow, SSH keys, etc.)
""")
print("=" * 70)