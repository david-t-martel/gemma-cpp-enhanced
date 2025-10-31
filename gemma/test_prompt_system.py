"""Standalone test for prompt system."""

import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from gemma_cli.config.prompts import PromptTemplate, PromptManager
    print("OK Import successful")
except Exception as e:
    print(f"FAIL Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def test_basic_template():
    """Test basic template functionality."""
    print("\n=== Testing Basic Template ===")

    # Test simple variable substitution
    content = "Hello {name}, you are {age} years old."
    template = PromptTemplate(content)

    context = {"name": "Alice", "age": 30}
    result = template.render(context)

    expected = "Hello Alice, you are 30 years old."
    assert result == expected, f"Expected: {expected}, Got: {result}"
    print(f"✓ Variable substitution: {result}")


def test_conditional_rendering():
    """Test conditional blocks."""
    print("\n=== Testing Conditional Rendering ===")

    content = """Start
{% if show_middle %}
Middle content
{% endif %}
End"""

    template = PromptTemplate(content)

    # Test with condition true
    result_true = template.render({"show_middle": True})
    assert "Middle content" in result_true
    print("✓ Conditional (true): included")

    # Test with condition false
    result_false = template.render({"show_middle": False})
    assert "Middle content" not in result_false
    print("✓ Conditional (false): excluded")


def test_template_validation():
    """Test template validation."""
    print("\n=== Testing Template Validation ===")

    # Valid template
    valid = "Hello {name}!"
    try:
        PromptTemplate(valid)
        print("✓ Valid template accepted")
    except Exception as e:
        print(f"✗ Valid template rejected: {e}")
        return False

    # Invalid template (unbalanced conditional)
    invalid = "{% if test %}No endif"
    try:
        PromptTemplate(invalid)
        print("✗ Invalid template accepted (should have failed)")
        return False
    except Exception:
        print("✓ Invalid template rejected")

    return True


def test_template_manager():
    """Test template manager."""
    print("\n=== Testing Template Manager ===")

    # Use actual templates directory
    templates_dir = Path("config/prompts")
    if not templates_dir.exists():
        print("✗ Templates directory not found")
        return False

    try:
        manager = PromptManager(templates_dir)
        print("✓ Manager initialized")
    except Exception as e:
        print(f"✗ Manager initialization failed: {e}")
        return False

    # List templates
    templates = manager.list_templates()
    print(f"✓ Found {len(templates)} templates:")
    for t in templates:
        print(f"  - {t['name']}: {t['description']}")

    # Load and render a template
    try:
        template = manager.get_template("default")
        print("✓ Loaded 'default' template")

        context = {
            "assistant_name": "Gemma",
            "model_name": "gemma-2b",
            "date": "2024-01-15",
            "user_name": "TestUser",
            "rag_enabled": True,
            "rag_context": "Test context data",
        }
        result = template.render(context)
        print(f"✓ Rendered template ({len(result)} chars)")

        # Check that variables were substituted
        assert "TestUser" in result, "User name not substituted"
        assert "gemma-2b" in result, "Model name not substituted"
        print("✓ Variables correctly substituted")

    except Exception as e:
        print(f"✗ Template loading/rendering failed: {e}")
        return False

    return True


def test_all_templates():
    """Test all available templates."""
    print("\n=== Testing All Templates ===")

    templates_dir = Path("config/prompts")
    if not templates_dir.exists():
        print("✗ Templates directory not found")
        return False

    manager = PromptManager(templates_dir)
    context = {
        "assistant_name": "Gemma",
        "model_name": "gemma-2b",
        "date": "2024-01-15",
        "user_name": "TestUser",
        "rag_enabled": True,
        "rag_context": "Sample context",
    }

    for template_file in templates_dir.glob("*.md"):
        template_name = template_file.stem
        try:
            template = manager.get_template(template_name)
            result = template.render(context)
            print(f"✓ {template_name}: {len(result)} chars")
        except Exception as e:
            print(f"✗ {template_name}: {e}")
            return False

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Gemma CLI Prompt System Tests")
    print("=" * 60)

    tests = [
        test_basic_template,
        test_conditional_rendering,
        test_template_validation,
        test_template_manager,
        test_all_templates,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result is False:
                failed += 1
            else:
                passed += 1
        except AssertionError as e:
            print(f"✗ Assertion failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
