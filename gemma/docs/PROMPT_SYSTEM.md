# Gemma CLI Prompt Template System

The prompt template system provides a flexible, powerful way to manage system prompts for the Gemma CLI with support for variable interpolation, conditional rendering, and multiple preset templates.

## Features

- **Variable Substitution**: Dynamic content with `{variable_name}` syntax
- **Conditional Blocks**: Show/hide content with `{% if condition %}...{% endif %}`
- **Section Extraction**: Extract specific sections from templates
- **Multiple Templates**: Pre-built templates for different use cases
- **Metadata Support**: YAML frontmatter for template information
- **Validation**: Automatic syntax validation and error checking
- **Caching**: Efficient template loading and caching

## Quick Start

### Basic Usage

```python
from pathlib import Path
from gemma_cli.config.prompts import PromptTemplate, PromptManager

# Initialize manager
manager = PromptManager(Path("config/prompts"))

# List available templates
templates = manager.list_templates()
for template in templates:
    print(f"{template['name']}: {template['description']}")

# Load and render a template
template = manager.get_template("coding")
context = {
    "assistant_name": "Gemma",
    "model_name": "gemma-2b",
    "date": "2024-01-15",
    "user_name": "Alice",
    "rag_enabled": True,
    "rag_context": "Previous discussion about Python..."
}
rendered = template.render(context)
```

### Working with Templates

```python
# Create a new template
content = """
Hello {user_name}!

{% if show_help %}
Here's how to use the system...
{% endif %}
"""
manager.create_template("welcome", content)

# Update existing template
manager.update_template("welcome", "Updated content")

# Set active template
manager.set_active_template("coding")

# Render active template
result = manager.render_active(context)

# Delete custom template
manager.delete_template("welcome")
```

## Template Syntax

### Variables

Use curly braces for variable substitution:

```markdown
Hello {user_name}!
Today is {date}.
You're using {model_name}.
```

Variables not provided in the context are left unchanged:

```python
template.render({"user_name": "Bob"})
# Output: "Hello Bob!\nToday is {date}.\n..."
```

### Conditional Blocks

Show/hide content based on boolean conditions:

```markdown
{% if rag_enabled %}
## Context Integration
You have access to relevant memories...
{% endif %}
```

Conditional evaluation:
- `True`, `"yes"`, `"true"`, `"1"`, non-empty strings → True
- `False`, `"false"`, `"no"`, `"0"`, `""`, `None` → False

### Metadata (YAML Frontmatter)

Add metadata to templates:

```markdown
---
name: my_template
description: Description of what this template does
version: 1.0
author: your-name
tags: [coding, technical, advanced]
---

# Template Content

Rest of your template...
```

### Sections

Organize content with markdown headings:

```markdown
# Main Title

## Section 1
Content for section 1...

## Section 2
Content for section 2...
```

Extract specific sections:

```python
template = PromptTemplate(content)
section = template.extract_section("Section 1")
# Returns content from "## Section 1" to next heading
```

## Available Templates

### default.md
**Description**: Balanced general-purpose assistant
**Best for**: Standard conversations, general assistance
**Variables**: `assistant_name`, `model_name`, `date`, `user_name`, `rag_enabled`, `rag_context`

### coding.md
**Description**: Code generation and programming assistance
**Best for**: Software development, debugging, code review
**Variables**: Same as default, optimized for technical content

### creative.md
**Description**: Creative writing and storytelling
**Best for**: Fiction writing, brainstorming, character development
**Variables**: Same as default, encourages imagination

### technical.md
**Description**: Technical documentation specialist
**Best for**: Writing docs, API references, tutorials
**Variables**: Same as default, focuses on clarity and structure

### concise.md
**Description**: Brief, direct responses
**Best for**: Quick answers, when verbosity is not desired
**Variables**: Same as default, minimizes elaboration

### GEMMA.md
**Description**: Original Gemma personality
**Best for**: Standard usage with comprehensive guidelines
**Variables**: Context-aware, no explicit variables

## Template Development

### Creating Custom Templates

```python
# Minimal template
simple_template = """
You are {assistant_name}, configured for {purpose}.

## Guidelines
{guidelines}
"""

# With metadata
full_template = """---
name: custom
description: My custom template
version: 1.0
tags: [custom, specialized]
---

# {template_title}

Your content here...

{% if enable_feature %}
## Feature Section
Feature-specific content...
{% endif %}
"""

manager.create_template("custom", full_template)
```

### Template Best Practices

1. **Clear variable names**: Use descriptive names like `user_name` not `u`
2. **Provide defaults**: Handle missing variables gracefully
3. **Logical structure**: Organize with clear sections
4. **Conditional clarity**: Use conditionals for optional sections
5. **Document metadata**: Include description and tags
6. **Test rendering**: Verify with different contexts

### Validation

Templates are automatically validated on load/create:

```python
# Valid template
valid = PromptTemplate("Hello {name}!")

# Invalid - unbalanced conditional
try:
    invalid = PromptTemplate("{% if test %}No endif")
except TemplateValidationError as e:
    print(f"Validation failed: {e}")
```

Common validation errors:
- Unbalanced `{% if %}` / `{% endif %}` blocks
- Nested conditionals (not supported)
- Invalid variable names (must be alphanumeric + underscore)

## Integration with Settings

The prompt system integrates with `SystemConfig`:

```python
from gemma_cli.config.settings import Settings

# Load settings
settings = Settings()

# Get prompt file path
prompt_file = settings.system.prompt_file  # "config/prompts/GEMMA.md"

# Check RAG settings
rag_enabled = settings.system.enable_rag_context
max_rag_tokens = settings.system.max_rag_context_tokens
```

## API Reference

### PromptTemplate

```python
class PromptTemplate:
    def __init__(self, content: str, metadata: Optional[TemplateMetadata] = None)

    @classmethod
    def load_template(cls, path: Path) -> "PromptTemplate"

    def render(self, context: Dict[str, Any]) -> str
    def get_variables(self) -> List[str]
    def get_sections(self) -> List[str]
    def extract_section(self, section_name: str) -> Optional[str]
    def validate(self) -> bool
```

### PromptManager

```python
class PromptManager:
    def __init__(self, templates_dir: Path)

    def list_templates(self) -> List[Dict[str, Any]]
    def get_template(self, name: str) -> PromptTemplate
    def create_template(self, name: str, content: str) -> PromptTemplate
    def update_template(self, name: str, content: str) -> PromptTemplate
    def delete_template(self, name: str) -> None

    def get_active_template(self) -> Optional[PromptTemplate]
    def set_active_template(self, name: str) -> None
    def render_active(self, context: Dict[str, Any]) -> str

    def clear_cache(self) -> None
```

### Exceptions

```python
class TemplateError(Exception)
class TemplateValidationError(TemplateError)
class TemplateRenderError(TemplateError)
```

## Examples

### Example 1: Dynamic Personality

```python
template = PromptTemplate("""
You are {assistant_name}, a {personality_type} assistant.

{% if formal_mode %}
Your responses should be professional and formal.
{% endif %}

{% if casual_mode %}
Feel free to be friendly and conversational.
{% endif %}
""")

# Formal mode
result = template.render({
    "assistant_name": "Gemma",
    "personality_type": "business",
    "formal_mode": True,
    "casual_mode": False
})
```

### Example 2: Context-Aware Responses

```python
manager = PromptManager(Path("config/prompts"))
manager.set_active_template("coding")

# With RAG context
context = {
    "assistant_name": "Gemma",
    "model_name": "gemma-2b",
    "user_name": "Developer",
    "rag_enabled": True,
    "rag_context": """
    [Previous discussions:
    1. User is working on Python backend with FastAPI
    2. Prefers async/await over threading
    3. Uses PostgreSQL database
    ]
    """
}

prompt = manager.render_active(context)
# Prompt now includes RAG context and coding-specific guidelines
```

### Example 3: Template Switching

```python
manager = PromptManager(Path("config/prompts"))

# Start with creative template for brainstorming
manager.set_active_template("creative")
brainstorm_prompt = manager.render_active({...})

# Switch to technical template for documentation
manager.set_active_template("technical")
docs_prompt = manager.render_active({...})

# Switch to concise for quick Q&A
manager.set_active_template("concise")
qa_prompt = manager.render_active({...})
```

## Testing

Run the included test suite:

```bash
cd C:/codedev/llm/gemma
python test_prompt_system.py
```

Or with pytest:

```bash
pytest tests/test_prompts.py -v
```

## Performance Considerations

- **Template caching**: Templates are cached after first load
- **Clear cache**: Call `manager.clear_cache()` if templates change on disk
- **Render performance**: Variable substitution is fast; conditional evaluation is simple
- **Memory usage**: Minimal; templates are text-based

## Future Enhancements

Potential additions:
- Loop constructs: `{% for item in list %}`
- Filters: `{variable|uppercase}`
- Template inheritance: `{% extends "base.md" %}`
- Macros/functions: `{% macro greeting(name) %}`
- External template loading from URLs

## Troubleshooting

### Template not found
```
FileNotFoundError: Template file not found: config/prompts/mytemplate.md
```
**Solution**: Ensure template file exists with `.md` extension

### Unbalanced conditionals
```
TemplateValidationError: Unbalanced conditional blocks
```
**Solution**: Check every `{% if %}` has matching `{% endif %}`

### Variables not substituted
Check variable names match exactly (case-sensitive):
```python
# Wrong
template.render({"username": "Alice"})

# Correct
template.render({"user_name": "Alice"})
```

### System template cannot be deleted
```
PermissionError: Cannot delete system template: default
```
**Solution**: System templates (`default`, `gemma`) are protected; create custom variants instead

---

For more information, see the source code documentation in `src/gemma_cli/config/prompts.py`.
