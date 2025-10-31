#!/usr/bin/env python3
"""Test script for ModelPreset.is_available() and validate() methods."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gemma_cli.config.models import ModelPreset, ValidationResult


def test_is_available():
    """Test is_available() method."""
    print("Testing is_available() method...")

    # Create a test model preset with non-existent files
    preset = ModelPreset(
        name="test-model",
        weights="/nonexistent/weights.sbs",
        tokenizer="/nonexistent/tokenizer.spm",
        format="sfp",
        size_gb=2.5,
        avg_tokens_per_sec=100,
        quality="medium",
        use_case="testing"
    )

    result = preset.is_available()
    print(f"  is_available() with non-existent files: {result}")
    assert result == False, "Should return False for non-existent files"

    print("  ✓ is_available() works correctly")


def test_validate():
    """Test validate() method."""
    print("\nTesting validate() method...")

    # Create a test model preset with non-existent files
    preset = ModelPreset(
        name="test-model",
        weights="/nonexistent/weights.sbs",
        tokenizer="/nonexistent/tokenizer.spm",
        format="sfp",
        size_gb=2.5,
        avg_tokens_per_sec=100,
        quality="medium",
        use_case="testing"
    )

    result = preset.validate()

    # Check result type
    print(f"  Result type: {type(result)}")
    assert isinstance(result, ValidationResult), "Should return ValidationResult"

    # Check fields
    print(f"  is_valid: {result.is_valid}")
    print(f"  errors: {result.errors}")

    assert result.is_valid == False, "Should be invalid for non-existent files"
    assert len(result.errors) == 2, "Should have 2 errors (weights + tokenizer)"
    assert any("Weights" in err for err in result.errors), "Should have weights error"
    assert any("Tokenizer" in err for err in result.errors), "Should have tokenizer error"

    print("  ✓ validate() works correctly")


def test_validate_with_real_files():
    """Test validate() with real model files if they exist."""
    print("\nTesting validate() with real files...")

    # Try with a real model if it exists
    test_paths = [
        ("C:/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs",
         "C:/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/tokenizer.spm"),
    ]

    found_model = False
    for weights, tokenizer in test_paths:
        if Path(weights).exists() and Path(tokenizer).exists():
            found_model = True
            preset = ModelPreset(
                name="test-real-model",
                weights=weights,
                tokenizer=tokenizer,
                format="sfp",
                size_gb=2.5,
                avg_tokens_per_sec=100,
                quality="medium",
                use_case="testing"
            )

            # Test is_available
            available = preset.is_available()
            print(f"  is_available() with real files: {available}")
            assert available == True, "Should be True for existing files"

            # Test validate
            result = preset.validate()
            print(f"  validate() is_valid: {result.is_valid}")
            print(f"  validate() errors: {result.errors}")
            assert result.is_valid == True, "Should be valid for existing files"
            assert len(result.errors) == 0, "Should have no errors"

            print("  ✓ Methods work correctly with real files")
            break

    if not found_model:
        print("  ⚠ No real model files found, skipping real file test")


if __name__ == "__main__":
    try:
        test_is_available()
        test_validate()
        test_validate_with_real_files()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
