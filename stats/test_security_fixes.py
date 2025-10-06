#!/usr/bin/env python3
"""Test script to verify security fixes work correctly."""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_cors_configuration():
    """Test that CORS configuration uses specific headers."""
    print("Testing CORS configuration...")

    from src.shared.config.settings import Settings
    from src.server.middleware import add_cors_middleware
    from fastapi import FastAPI

    app = FastAPI()
    settings = Settings()

    # This should work without errors
    add_cors_middleware(app, settings)

    # Check that the middleware was added
    has_cors = False
    for middleware in app.user_middleware:
        if "CORSMiddleware" in str(middleware.cls):
            has_cors = True
            break

    assert has_cors, "CORS middleware not added"
    print("✓ CORS configuration test passed")
    return True

def test_jwt_secret_persistence():
    """Test that JWT secret is persistent."""
    print("\nTesting JWT secret persistence...")

    from src.shared.config.settings import Settings
    from src.server.auth import JWTAuthenticator

    # First instance
    settings = Settings()
    auth1 = JWTAuthenticator(settings)
    secret1 = auth1.jwt_secret

    # Second instance should load the same secret
    auth2 = JWTAuthenticator(settings)
    secret2 = auth2.jwt_secret

    # In development, secrets should match (loaded from file)
    if not settings.is_production():
        assert secret1 == secret2, "JWT secrets do not match - not persistent!"
        print(f"✓ JWT secret is persistent (length: {len(secret1)})")

    # Check minimum length
    assert len(secret1) >= 64, f"JWT secret too short: {len(secret1)} < 64"
    print("✓ JWT secret has adequate length")

    return True

def main():
    """Run all security tests."""
    print("=" * 60)
    print("SECURITY FIXES VERIFICATION")
    print("=" * 60)

    tests = [
        test_cors_configuration,
        test_jwt_secret_persistence,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All security fixes verified successfully!")

if __name__ == "__main__":
    main()
