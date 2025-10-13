"""System prompt template management for gemma-cli.

Provides template loading, rendering, and management with
variable substitution and conditional logic.

This module implements a flexible template system for managing system prompts
with YAML frontmatter metadata, variable substitution, and conditional blocks.
Templates are stored as Markdown files with metadata headers.

Example template format:
    ---
    name: default
    version: 1.0
    description: Default system prompt
    author: Gemma CLI
    tags: [general, balanced]
    variables: [model_name, context_length]
    ---

    You are {model_name}, a helpful AI assistant.
    Your context window is {context_length} tokens.

    {% if enable_rag %}
    You have access to RAG context for enhanced responses.
    {% endif %}

Classes:
    PromptMetadata: Metadata model for prompt templates.
    PromptTemplate: Individual prompt template with rendering.
    PromptManager: Manager for all prompt templates.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator
from rich.console import Console
from rich.table import Table

console = Console()


class PromptMetadata(BaseModel):
    """Metadata for a prompt template.

    Attributes:
        name: Unique identifier for the template.
        version: Version string (e.g., "1.0", "2.1.3").
        description: Human-readable description of template purpose.
        author: Optional author name or organization.
        tags: List of tags for categorization.
        variables: List of variable names used in template.
        created: ISO 8601 timestamp of creation.
        modified: ISO 8601 timestamp of last modification.
    """

    name: str = Field(..., min_length=1, max_length=100)
    version: str = Field(..., pattern=r'^\d+\.\d+(\.\d+)?$')
    description: str = Field(..., min_length=1, max_length=500)
    author: Optional[str] = Field(None, max_length=100)
    tags: List[str] = Field(default_factory=list)
    variables: List[str] = Field(default_factory=list)
    created: Optional[str] = None
    modified: Optional[str] = None

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate template name format.

        Args:
            v: Name to validate.

        Returns:
            Validated name.

        Raises:
            ValueError: If name contains invalid characters.
        """
        if not re.match(r'^[a-z0-9_-]+$', v):
            raise ValueError(
                'Template name must contain only lowercase letters, '
                'numbers, hyphens, and underscores'
            )
        return v

    @field_validator('tags', 'variables')
    @classmethod
    def validate_list_items(cls, v: List[str]) -> List[str]:
        """Validate list items are non-empty strings.

        Args:
            v: List to validate.

        Returns:
            Validated list.

        Raises:
            ValueError: If list contains empty strings.
        """
        for item in v:
            if not item or not isinstance(item, str):
                raise ValueError('List items must be non-empty strings')
        return v


class PromptTemplate:
    """Individual prompt template with rendering capabilities.

    Handles loading templates from files, parsing YAML frontmatter,
    and rendering templates with variable substitution and conditionals.

    Attributes:
        file_path: Path to template file.
        metadata: Template metadata from frontmatter.
        content: Raw template content (without frontmatter).
    """

    # Pattern for YAML frontmatter
    FRONTMATTER_PATTERN = re.compile(
        r'^---\s*\n(.*?)\n---\s*\n',
        re.DOTALL | re.MULTILINE
    )

    # Pattern for variable substitution {variable_name}
    VARIABLE_PATTERN = re.compile(r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}')

    # Pattern for conditional blocks {% if condition %}...{% endif %}
    CONDITIONAL_PATTERN = re.compile(
        r'\{%\s*if\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*%\}(.*?)\{%\s*endif\s*%\}',
        re.DOTALL
    )

    def __init__(self, file_path: Path):
        """Initialize template from file.

        Args:
            file_path: Path to template file.

        Raises:
            FileNotFoundError: If template file doesn't exist.
            ValueError: If template format is invalid.
        """
        self.file_path = file_path
        self.metadata: Optional[PromptMetadata] = None
        self.content: str = ""

        if not file_path.exists():
            raise FileNotFoundError(f"Template file not found: {file_path}")

        self.load_template()

    def load_template(self) -> None:
        """Load and parse template file.

        Parses YAML frontmatter for metadata and extracts content.
        Auto-detects variables used in template.

        Raises:
            ValueError: If template format is invalid.
            yaml.YAMLError: If YAML parsing fails.
        """
        try:
            raw_content = self.file_path.read_text(encoding='utf-8')
        except Exception as e:
            raise ValueError(f"Failed to read template file: {e}")

        # Parse frontmatter
        match = self.FRONTMATTER_PATTERN.match(raw_content)
        if not match:
            raise ValueError(
                f"Template missing YAML frontmatter: {self.file_path.name}"
            )

        frontmatter_text = match.group(1)
        self.content = raw_content[match.end():].strip()

        # Parse YAML metadata
        try:
            metadata_dict = yaml.safe_load(frontmatter_text)
            if not isinstance(metadata_dict, dict):
                raise ValueError("Frontmatter must be a YAML dictionary")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in frontmatter: {e}")

        # Auto-detect variables if not specified
        if 'variables' not in metadata_dict or not metadata_dict['variables']:
            metadata_dict['variables'] = self._extract_variables()

        # Add timestamps if missing
        if 'created' not in metadata_dict:
            metadata_dict['created'] = datetime.now().isoformat()
        if 'modified' not in metadata_dict:
            metadata_dict['modified'] = datetime.now().isoformat()

        # Validate metadata
        try:
            self.metadata = PromptMetadata(**metadata_dict)
        except ValidationError as e:
            raise ValueError(f"Invalid metadata: {e}")

    def _extract_variables(self) -> List[str]:
        """Extract variable names from template content.

        Returns:
            List of unique variable names found in template.
        """
        variables: Set[str] = set()

        # Extract from variable substitutions
        for match in self.VARIABLE_PATTERN.finditer(self.content):
            variables.add(match.group(1))

        # Extract from conditionals
        for match in self.CONDITIONAL_PATTERN.finditer(self.content):
            variables.add(match.group(1))

        return sorted(list(variables))

    def render(self, context: Dict[str, Any]) -> str:
        """Render template with provided context.

        Performs variable substitution and evaluates conditional blocks.

        Args:
            context: Dictionary of variable values.

        Returns:
            Rendered template string.

        Raises:
            ValueError: If required variables are missing.
        """
        if not self.metadata:
            raise ValueError("Template not loaded")

        # Validate required variables
        missing_vars = set(self.metadata.variables) - set(context.keys())
        if missing_vars:
            raise ValueError(
                f"Missing required variables: {', '.join(sorted(missing_vars))}"
            )

        rendered = self.content

        # Process conditional blocks first
        rendered = self._render_conditionals(rendered, context)

        # Perform variable substitution
        rendered = self._render_variables(rendered, context)

        return rendered.strip()

    def _render_conditionals(self, content: str, context: Dict[str, Any]) -> str:
        """Render conditional blocks based on context.

        Args:
            content: Template content to process.
            context: Variable context.

        Returns:
            Content with conditionals evaluated.
        """
        def replace_conditional(match: re.Match) -> str:
            condition = match.group(1)
            block_content = match.group(2)

            # Evaluate condition (simple truthiness check)
            condition_value = context.get(condition, False)
            if condition_value:
                return block_content.strip()
            return ""

        return self.CONDITIONAL_PATTERN.sub(replace_conditional, content)

    def _render_variables(self, content: str, context: Dict[str, Any]) -> str:
        """Substitute variables in content.

        Args:
            content: Template content to process.
            context: Variable context.

        Returns:
            Content with variables substituted.
        """
        def replace_variable(match: re.Match) -> str:
            var_name = match.group(1)
            value = context.get(var_name, '')
            return str(value)

        return self.VARIABLE_PATTERN.sub(replace_variable, content)

    def validate(self) -> bool:
        """Validate template syntax and metadata.

        Returns:
            True if template is valid.

        Raises:
            ValueError: If template has validation errors.
        """
        if not self.metadata:
            raise ValueError("Template not loaded")

        # Check for unmatched conditional blocks
        if_count = len(re.findall(r'\{%\s*if\s+', self.content))
        endif_count = len(re.findall(r'\{%\s*endif\s*%\}', self.content))
        if if_count != endif_count:
            raise ValueError(
                f"Unmatched conditional blocks: {if_count} if, {endif_count} endif"
            )

        # Check for invalid syntax patterns
        invalid_patterns = [
            (r'\{%(?!.*%\})', 'Unclosed template tag'),
            (r'\{(?![%{])[^}]*$', 'Unclosed variable substitution'),
        ]

        for pattern, error_msg in invalid_patterns:
            if re.search(pattern, self.content):
                raise ValueError(error_msg)

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Get template metadata as dictionary.

        Returns:
            Metadata dictionary.

        Raises:
            ValueError: If template not loaded.
        """
        if not self.metadata:
            raise ValueError("Template not loaded")

        return self.metadata.model_dump()


class PromptManager:
    """Manager for prompt templates.

    Handles loading, listing, creating, updating, and deleting templates.
    Manages active template selection and provides rendering utilities.

    Attributes:
        templates_dir: Directory containing template files.
        templates: Dictionary of loaded templates by name.
        active_template_name: Name of currently active template.
    """

    DEFAULT_TEMPLATE = """---
name: default
version: 1.0
description: Default system prompt for Gemma models
author: Gemma CLI
tags: [general, balanced]
variables: [model_name, context_length]
---

You are {model_name}, a helpful AI assistant.
Your context window is {context_length} tokens.

Please provide clear, accurate, and helpful responses.
"""

    def __init__(self, templates_dir: Path, active_template: str = "default"):
        """Initialize prompt manager.

        Args:
            templates_dir: Directory containing template files.
            active_template: Name of default active template.
        """
        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, PromptTemplate] = {}
        self.active_template_name = active_template

        # Create templates directory if it doesn't exist
        self.templates_dir.mkdir(parents=True, exist_ok=True)

        # Create default template if no templates exist
        if not list(self.templates_dir.glob('*.md')):
            self._create_default_template()

        # Load all templates
        self._load_all_templates()

    def _create_default_template(self) -> None:
        """Create default template file."""
        default_path = self.templates_dir / 'default.md'
        default_path.write_text(self.DEFAULT_TEMPLATE, encoding='utf-8')

    def _load_all_templates(self) -> None:
        """Load all template files from templates directory."""
        self.templates.clear()

        for template_file in self.templates_dir.glob('*.md'):
            try:
                template = PromptTemplate(template_file)
                if template.metadata:
                    self.templates[template.metadata.name] = template
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Failed to load template {template_file.name}: {e}[/yellow]"
                )

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates.

        Returns:
            List of template metadata dictionaries.
        """
        return [
            {
                'name': name,
                'is_active': name == self.active_template_name,
                **template.get_metadata()
            }
            for name, template in self.templates.items()
        ]

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name.

        Args:
            name: Template name.

        Returns:
            PromptTemplate if found, None otherwise.
        """
        return self.templates.get(name)

    def get_active_template(self) -> Optional[PromptTemplate]:
        """Get currently active template.

        Returns:
            Active PromptTemplate if set, None otherwise.
        """
        return self.get_template(self.active_template_name)

    def set_active_template(self, name: str) -> None:
        """Set active template by name.

        Args:
            name: Template name to activate.

        Raises:
            ValueError: If template doesn't exist.
        """
        if name not in self.templates:
            raise ValueError(f"Template not found: {name}")

        self.active_template_name = name
        console.print(f"[green]Active template set to: {name}[/green]")

    def create_template(
        self,
        name: str,
        content: str,
        metadata: Dict[str, Any],
        force: bool = False
    ) -> PromptTemplate:
        """Create new template.

        Args:
            name: Template name.
            content: Template content (without frontmatter).
            metadata: Template metadata dictionary.
            force: Overwrite if template exists.

        Returns:
            Created PromptTemplate.

        Raises:
            ValueError: If template exists and force=False.
        """
        template_path = self.templates_dir / f'{name}.md'

        if template_path.exists() and not force:
            raise ValueError(
                f"Template already exists: {name}. Use force=True to overwrite."
            )

        # Ensure required metadata fields
        if 'name' not in metadata:
            metadata['name'] = name
        if 'created' not in metadata:
            metadata['created'] = datetime.now().isoformat()
        metadata['modified'] = datetime.now().isoformat()

        # Validate metadata
        try:
            PromptMetadata(**metadata)
        except ValidationError as e:
            raise ValueError(f"Invalid metadata: {e}")

        # Build template file content
        yaml_frontmatter = yaml.dump(metadata, sort_keys=False)
        full_content = f"---\n{yaml_frontmatter}---\n\n{content.strip()}\n"

        # Write template file
        template_path.write_text(full_content, encoding='utf-8')

        # Load and register template
        template = PromptTemplate(template_path)
        self.templates[name] = template

        console.print(f"[green]Created template: {name}[/green]")
        return template

    def update_template(
        self,
        name: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PromptTemplate:
        """Update existing template.

        Args:
            name: Template name.
            content: New template content (if updating).
            metadata: New metadata fields (if updating).

        Returns:
            Updated PromptTemplate.

        Raises:
            ValueError: If template doesn't exist.
        """
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template not found: {name}")

        # Get current metadata
        current_metadata = template.get_metadata()
        current_content = template.content

        # Update metadata
        if metadata:
            current_metadata.update(metadata)
        current_metadata['modified'] = datetime.now().isoformat()

        # Update content
        if content is not None:
            current_content = content

        # Recreate template
        template_path = self.templates_dir / f'{name}.md'
        yaml_frontmatter = yaml.dump(current_metadata, sort_keys=False)
        full_content = f"---\n{yaml_frontmatter}---\n\n{current_content.strip()}\n"
        template_path.write_text(full_content, encoding='utf-8')

        # Reload template
        self.templates[name] = PromptTemplate(template_path)

        console.print(f"[green]Updated template: {name}[/green]")
        return self.templates[name]

    def delete_template(self, name: str, confirm: bool = False) -> None:
        """Delete template.

        Args:
            name: Template name.
            confirm: Confirmation flag to prevent accidental deletion.

        Raises:
            ValueError: If template doesn't exist or is active.
            RuntimeError: If confirmation not provided.
        """
        if not confirm:
            raise RuntimeError(
                "Template deletion requires confirmation. Set confirm=True."
            )

        if name not in self.templates:
            raise ValueError(f"Template not found: {name}")

        if name == self.active_template_name:
            raise ValueError(
                "Cannot delete active template. Set different active template first."
            )

        # Delete file
        template_path = self.templates_dir / f'{name}.md'
        if template_path.exists():
            template_path.unlink()

        # Remove from registry
        del self.templates[name]

        console.print(f"[green]Deleted template: {name}[/green]")

    def render_active(self, context: Dict[str, Any]) -> str:
        """Render active template with context.

        Args:
            context: Variable context for rendering.

        Returns:
            Rendered template string.

        Raises:
            ValueError: If no active template or rendering fails.
        """
        template = self.get_active_template()
        if not template:
            raise ValueError(f"Active template not found: {self.active_template_name}")

        try:
            return template.render(context)
        except Exception as e:
            raise ValueError(f"Failed to render template: {e}")

    def display_templates_table(self) -> None:
        """Display templates in a Rich table."""
        table = Table(title="Available Prompt Templates")

        table.add_column("Active", style="cyan", width=8)
        table.add_column("Name", style="magenta")
        table.add_column("Version", style="green")
        table.add_column("Description", style="white")
        table.add_column("Tags", style="yellow")
        table.add_column("Variables", style="blue")

        for template_info in self.list_templates():
            is_active = "✓" if template_info['is_active'] else ""
            name = template_info['name']
            version = template_info['version']
            description = template_info['description']
            tags = ', '.join(template_info.get('tags', []))
            variables = ', '.join(template_info.get('variables', []))

            table.add_row(
                is_active,
                name,
                version,
                description,
                tags,
                variables
            )

        console.print(table)

    def reload_templates(self) -> None:
        """Reload all templates from disk."""
        self._load_all_templates()
        console.print("[green]Templates reloaded[/green]")
