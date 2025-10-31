"""Tests for prompt template system."""

import pytest
from pathlib import Path
from gemma_cli.config.prompts import (
    PromptTemplate,
    PromptManager,
    TemplateError,
    TemplateValidationError,
    TemplateRenderError,
)


@pytest.fixture
def templates_dir(tmp_path):
    """Create temporary templates directory."""
    templates_dir = tmp_path / "prompts"
    templates_dir.mkdir()
    return templates_dir


@pytest.fixture
def sample_template_content():
    """Sample template content for testing."""
    return """---
name: test
description: Test template
version: 1.0
tags: [test, sample]
---

# Test Template

Hello {user_name}!

{% if show_date %}
Today is {date}.
{% endif %}

## Section 1
Content for section 1.

## Section 2
Content for section 2.

{% if show_footer %}
Footer content.
{% endif %}
"""


@pytest.fixture
def simple_template():
    """Create a simple template for testing."""
    content = "Hello {name}, your age is {age}."
    return PromptTemplate(content)


class TestPromptTemplate:
    """Test PromptTemplate class."""

    def test_init_valid_template(self):
        """Test initializing a valid template."""
        content = "Hello {name}!"
        template = PromptTemplate(content)
        assert template.content == content

    def test_init_invalid_unbalanced_conditionals(self):
        """Test that unbalanced conditionals raise error."""
        content = "{% if condition %}No endif"
        with pytest.raises(TemplateValidationError, match="Unbalanced"):
            PromptTemplate(content)

    def test_init_invalid_nested_conditionals(self):
        """Test that nested conditionals raise error."""
        content = "{% if outer %}{% if inner %}text{% endif %}{% endif %}"
        with pytest.raises(TemplateValidationError, match="Nested"):
            PromptTemplate(content)

    def test_get_variables(self, simple_template):
        """Test extracting variables from template."""
        variables = simple_template.get_variables()
        assert set(variables) == {"name", "age"}

    def test_get_sections(self, sample_template_content):
        """Test extracting section headings."""
        template = PromptTemplate(sample_template_content)
        sections = template.get_sections()
        assert "Test Template" in sections
        assert "Section 1" in sections
        assert "Section 2" in sections

    def test_render_simple(self, simple_template):
        """Test rendering simple template."""
        context = {"name": "Alice", "age": 30}
        result = simple_template.render(context)
        assert result == "Hello Alice, your age is 30."

    def test_render_missing_variable(self, simple_template):
        """Test that missing variables are left unchanged."""
        context = {"name": "Alice"}
        result = simple_template.render(context)
        assert result == "Hello Alice, your age is {age}."

    def test_render_conditional_true(self):
        """Test rendering with true conditional."""
        content = "Start {% if show %}Middle{% endif %} End"
        template = PromptTemplate(content)
        result = template.render({"show": True})
        assert result == "Start Middle End"

    def test_render_conditional_false(self):
        """Test rendering with false conditional."""
        content = "Start {% if show %}Middle{% endif %} End"
        template = PromptTemplate(content)
        result = template.render({"show": False})
        assert result == "Start  End"

    def test_render_conditional_string_true(self):
        """Test conditional with truthy string value."""
        content = "{% if flag %}Shown{% endif %}"
        template = PromptTemplate(content)
        result = template.render({"flag": "yes"})
        assert "Shown" in result

    def test_render_conditional_string_false(self):
        """Test conditional with falsy string value."""
        content = "{% if flag %}Shown{% endif %}"
        template = PromptTemplate(content)
        result = template.render({"flag": "false"})
        assert "Shown" not in result

    def test_render_complex(self, sample_template_content):
        """Test rendering complex template."""
        template = PromptTemplate(sample_template_content)
        context = {
            "user_name": "Bob",
            "date": "2024-01-15",
            "show_date": True,
            "show_footer": False,
        }
        result = template.render(context)

        assert "Hello Bob!" in result
        assert "Today is 2024-01-15." in result
        assert "Footer content" not in result

    def test_extract_section(self, sample_template_content):
        """Test extracting specific section."""
        template = PromptTemplate(sample_template_content)
        section = template.extract_section("Section 1")
        assert section is not None
        assert "Section 1" in section
        assert "Content for section 1" in section

    def test_extract_nonexistent_section(self, sample_template_content):
        """Test extracting non-existent section."""
        template = PromptTemplate(sample_template_content)
        section = template.extract_section("Nonexistent")
        assert section is None

    def test_validate(self, simple_template):
        """Test template validation."""
        assert simple_template.validate() is True

    def test_load_template(self, templates_dir, sample_template_content):
        """Test loading template from file."""
        template_path = templates_dir / "test.md"
        template_path.write_text(sample_template_content)

        template = PromptTemplate.load_template(template_path)
        assert template.content == sample_template_content
        assert template.metadata is not None
        assert template.metadata.name == "test"
        assert "test" in template.metadata.tags

    def test_load_template_file_not_found(self, templates_dir):
        """Test loading non-existent template."""
        with pytest.raises(FileNotFoundError):
            PromptTemplate.load_template(templates_dir / "nonexistent.md")


class TestPromptManager:
    """Test PromptManager class."""

    @pytest.fixture
    def manager(self, templates_dir, sample_template_content):
        """Create prompt manager with sample templates."""
        # Create test templates
        (templates_dir / "default.md").write_text(sample_template_content)
        (templates_dir / "simple.md").write_text("Hello {name}!")
        return PromptManager(templates_dir)

    def test_init(self, templates_dir):
        """Test initializing manager."""
        manager = PromptManager(templates_dir)
        assert manager.templates_dir == templates_dir

    def test_init_dir_not_found(self, tmp_path):
        """Test initializing with non-existent directory."""
        with pytest.raises(FileNotFoundError):
            PromptManager(tmp_path / "nonexistent")

    def test_list_templates(self, manager):
        """Test listing available templates."""
        templates = manager.list_templates()
        assert len(templates) >= 2
        names = [t["name"] for t in templates]
        assert "default" in names
        assert "simple" in names

    def test_get_template(self, manager):
        """Test getting template by name."""
        template = manager.get_template("default")
        assert isinstance(template, PromptTemplate)
        assert template.metadata is not None
        assert template.metadata.name == "test"

    def test_get_template_not_found(self, manager):
        """Test getting non-existent template."""
        with pytest.raises(FileNotFoundError):
            manager.get_template("nonexistent")

    def test_get_template_caching(self, manager):
        """Test that templates are cached."""
        template1 = manager.get_template("default")
        template2 = manager.get_template("default")
        assert template1 is template2  # Same object due to caching

    def test_create_template(self, manager):
        """Test creating new template."""
        content = "New template with {variable}."
        template = manager.create_template("new", content)
        assert isinstance(template, PromptTemplate)

        # Verify file was created
        template_path = manager.templates_dir / "new.md"
        assert template_path.exists()

        # Verify can be loaded
        loaded = manager.get_template("new")
        assert loaded.content == content

    def test_create_template_already_exists(self, manager):
        """Test creating template that already exists."""
        with pytest.raises(FileExistsError):
            manager.create_template("default", "Content")

    def test_create_template_invalid(self, manager):
        """Test creating invalid template."""
        invalid_content = "{% if unbalanced %}"
        with pytest.raises(TemplateValidationError):
            manager.create_template("invalid", invalid_content)

    def test_update_template(self, manager):
        """Test updating existing template."""
        new_content = "Updated content with {var}."
        template = manager.update_template("default", new_content)
        assert template.content == new_content

        # Verify file was updated
        loaded = manager.get_template("default")
        assert loaded.content == new_content

    def test_update_template_not_found(self, manager):
        """Test updating non-existent template."""
        with pytest.raises(FileNotFoundError):
            manager.update_template("nonexistent", "Content")

    def test_update_template_invalid(self, manager):
        """Test updating with invalid content."""
        invalid_content = "{% if unbalanced %}"
        with pytest.raises(TemplateValidationError):
            manager.update_template("default", invalid_content)

    def test_delete_template(self, manager):
        """Test deleting template."""
        # Create a template to delete
        manager.create_template("to_delete", "Content")
        assert (manager.templates_dir / "to_delete.md").exists()

        # Delete it
        manager.delete_template("to_delete")
        assert not (manager.templates_dir / "to_delete.md").exists()

        # Verify it's not in cache
        with pytest.raises(FileNotFoundError):
            manager.get_template("to_delete")

    def test_delete_template_not_found(self, manager):
        """Test deleting non-existent template."""
        with pytest.raises(FileNotFoundError):
            manager.delete_template("nonexistent")

    def test_delete_system_template(self, manager):
        """Test that system templates cannot be deleted."""
        # Create template named "default" or "gemma"
        with pytest.raises(PermissionError):
            manager.delete_template("default")

    def test_set_active_template(self, manager):
        """Test setting active template."""
        manager.set_active_template("default")
        assert manager._active_template == "default"

    def test_set_active_template_not_found(self, manager):
        """Test setting non-existent template as active."""
        with pytest.raises(FileNotFoundError):
            manager.set_active_template("nonexistent")

    def test_get_active_template(self, manager):
        """Test getting active template."""
        manager.set_active_template("default")
        template = manager.get_active_template()
        assert isinstance(template, PromptTemplate)

    def test_get_active_template_none_set(self, manager):
        """Test getting active template when none is set."""
        template = manager.get_active_template()
        assert template is None

    def test_render_active(self, manager):
        """Test rendering active template."""
        manager.set_active_template("simple")
        result = manager.render_active({"name": "Charlie"})
        assert result == "Hello Charlie!"

    def test_render_active_no_template_set(self, manager):
        """Test rendering when no active template."""
        with pytest.raises(ValueError, match="No active template"):
            manager.render_active({"name": "Test"})

    def test_clear_cache(self, manager):
        """Test clearing template cache."""
        # Load a template to populate cache
        manager.get_template("default")
        assert len(manager._cache) > 0

        # Clear cache
        manager.clear_cache()
        assert len(manager._cache) == 0


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self, templates_dir):
        """Test complete template workflow."""
        # Create manager
        manager = PromptManager(templates_dir)

        # Create template
        content = """Hello {user}!
{% if show_help %}
Need help? Just ask!
{% endif %}
"""
        manager.create_template("welcome", content)

        # List templates
        templates = manager.list_templates()
        assert any(t["name"] == "welcome" for t in templates)

        # Set as active
        manager.set_active_template("welcome")

        # Render
        context = {"user": "David", "show_help": True}
        result = manager.render_active(context)
        assert "Hello David!" in result
        assert "Need help?" in result

        # Update template
        new_content = "Welcome back, {user}!"
        manager.update_template("welcome", new_content)

        # Render updated
        result = manager.render_active(context)
        assert "Welcome back, David!" in result

        # Delete template
        manager.delete_template("welcome")
        assert not (templates_dir / "welcome.md").exists()

    def test_real_templates_load(self):
        """Test loading actual templates from config/prompts."""
        # This assumes the templates exist in the project
        templates_dir = Path("config/prompts")
        if not templates_dir.exists():
            pytest.skip("Templates directory not found")

        manager = PromptManager(templates_dir)

        # Test loading each template
        for template_name in ["default", "coding", "creative", "technical", "concise"]:
            template_path = templates_dir / f"{template_name}.md"
            if template_path.exists():
                template = manager.get_template(template_name)
                assert isinstance(template, PromptTemplate)

                # Test rendering with sample context
                context = {
                    "assistant_name": "Gemma",
                    "model_name": "gemma-2b",
                    "date": "2024-01-15",
                    "user_name": "TestUser",
                    "rag_enabled": True,
                    "rag_context": "Test context",
                }
                result = template.render(context)
                assert len(result) > 0
                assert "TestUser" in result or "{user_name}" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
