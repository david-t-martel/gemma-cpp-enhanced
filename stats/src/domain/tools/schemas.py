"""Tool schemas and validation for the LLM agent framework."""

import re
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from uuid import UUID

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator


class ToolType(str, Enum):
    """Tool type enumeration."""

    BUILTIN = "builtin"
    CUSTOM = "custom"
    MCP = "mcp"
    COMPOSITE = "composite"


class ToolCategory(str, Enum):
    """Tool category enumeration."""

    FILE_SYSTEM = "file_system"
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
    DATA_PROCESSING = "data_processing"
    COMMUNICATION = "communication"
    SYSTEM = "system"
    DATABASE = "database"
    SECURITY = "security"
    MONITORING = "monitoring"
    CUSTOM = "custom"


class SecurityLevel(BaseModel):
    """Security level configuration."""

    level: str = Field(..., pattern="^(minimal|standard|strict|maximum)$")
    sandbox_required: bool = True
    network_access: bool = False
    file_system_access: bool = False
    process_isolation: bool = True
    resource_limits: dict[str, Any] = Field(default_factory=dict)


class ResourceLimits(BaseModel):
    """Resource limits for tool execution."""

    max_memory_mb: int | None = 512
    max_cpu_percent: float | None = 50.0
    max_execution_time_seconds: int | None = 30
    max_file_size_mb: int | None = 100
    max_network_requests: int | None = 10
    max_subprocess_count: int | None = 1


class ValidationRule(BaseModel):
    """Parameter validation rule."""

    type: str
    constraint: str
    message: str

    @field_validator("type")
    @classmethod
    def validate_rule_type(cls, v):
        valid_types = ["regex", "range", "length", "custom"]
        if v not in valid_types:
            raise ValueError(f"Rule type must be one of {valid_types}")
        return v


class ParameterSchema(BaseModel):
    """Enhanced parameter schema with validation rules."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    format: str | None = None
    validation_rules: list[ValidationRule] = Field(default_factory=list)
    sensitive: bool = False  # Mark parameter as containing sensitive data
    examples: list[Any] = Field(default_factory=list)

    @field_validator("type")
    @classmethod
    def validate_parameter_type(cls, v):
        valid_types = [
            "string",
            "integer",
            "number",
            "boolean",
            "array",
            "object",
            "null",
            "any",
            "file",
            "url",
            "email",
            "uuid",
            "datetime",
            "json",
            "base64",
            "code",
        ]
        if v not in valid_types:
            raise ValueError(f"Parameter type must be one of {valid_types}")
        return v

    @field_validator("pattern")
    @classmethod
    def validate_regex_pattern(cls, v):
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
        return v

    def validate_value(self, value: Any) -> bool:
        """Validate a value against this parameter schema."""
        # Basic type validation
        if not self._validate_type(value):
            return False

        # Enum validation
        if self.enum and value not in self.enum:
            return False

        # Range validation
        if self.minimum is not None and isinstance(value, (int, float)) and value < self.minimum:
            return False
        if self.maximum is not None and isinstance(value, (int, float)) and value > self.maximum:
            return False

        # Length validation
        if hasattr(value, "__len__"):
            length = len(value)
            if self.min_length is not None and length < self.min_length:
                return False
            if self.max_length is not None and length > self.max_length:
                return False

        # Pattern validation
        if self.pattern and isinstance(value, str) and not re.match(self.pattern, value):
            return False

        # Custom validation rules
        return all(self._apply_validation_rule(rule, value) for rule in self.validation_rules)

    def _validate_type(self, value: Any) -> bool:
        """Validate basic type."""
        type_validators = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
            "null": lambda v: v is None,
            "any": lambda v: True,
            "file": lambda v: isinstance(v, str),  # File path
            "url": lambda v: isinstance(v, str) and self._is_valid_url(v),
            "email": lambda v: isinstance(v, str) and self._is_valid_email(v),
            "uuid": lambda v: self._is_valid_uuid(v),
            "datetime": lambda v: isinstance(v, (str, datetime)),
            "json": lambda v: self._is_valid_json(v),
            "base64": lambda v: isinstance(v, str) and self._is_valid_base64(v),
            "code": lambda v: isinstance(v, str),
        }

        validator_func = type_validators.get(self.type)
        return validator_func(value) if validator_func else False

    def _apply_validation_rule(self, rule: ValidationRule, value: Any) -> bool:
        """Apply custom validation rule."""
        if rule.type == "regex":
            return isinstance(value, str) and bool(re.match(rule.constraint, value))
        elif rule.type == "range":
            # Parse range constraint like "0-100"
            try:
                min_val, max_val = map(float, rule.constraint.split("-"))
                return isinstance(value, (int, float)) and min_val <= value <= max_val
            except ValueError:
                return False
        elif rule.type == "length":
            # Parse length constraint like "5-20"
            try:
                if "-" in rule.constraint:
                    min_len, max_len = map(int, rule.constraint.split("-"))
                    return min_len <= len(value) <= max_len
                else:
                    exact_len = int(rule.constraint)
                    return len(value) == exact_len
            except (ValueError, TypeError):
                return False
        elif rule.type == "custom":
            # For custom rules, we'd need to implement a safe eval mechanism
            # For now, just return True
            return True

        return True

    @staticmethod
    def _is_valid_url(value: str) -> bool:
        """Basic URL validation."""
        url_pattern = re.compile(
            r"^https?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
            r"localhost|"  # localhost...
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )
        return bool(url_pattern.match(value))

    @staticmethod
    def _is_valid_email(value: str) -> bool:
        """Basic email validation."""
        email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        return bool(email_pattern.match(value))

    @staticmethod
    def _is_valid_uuid(value: Any) -> bool:
        """UUID validation."""
        try:
            if isinstance(value, str):
                UUID(value)
                return True
            elif isinstance(value, UUID):
                return True
            return False
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _is_valid_json(value: Any) -> bool:
        """JSON validation."""
        if isinstance(value, (dict, list)):
            return True
        if isinstance(value, str):
            try:
                import json

                json.loads(value)
                return True
            except (ValueError, TypeError):
                return False
        return False

    @staticmethod
    def _is_valid_base64(value: str) -> bool:
        """Base64 validation."""
        try:
            import base64

            base64.b64decode(value, validate=True)
            return True
        except Exception:
            return False


class ReturnSchema(BaseModel):
    """Schema for tool return values."""

    type: str
    description: str
    properties: dict[str, ParameterSchema] | None = None
    examples: list[Any] = Field(default_factory=list)


class ToolMetadata(BaseModel):
    """Tool metadata information."""

    version: str = "1.0.0"
    author: str | None = None
    license: str | None = None
    homepage: str | None = None
    repository: str | None = None
    documentation: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)


class ToolExample(BaseModel):
    """Tool usage example."""

    name: str
    description: str
    parameters: dict[str, Any]
    expected_result: Any | None = None
    notes: str | None = None


class EnhancedToolSchema(BaseModel):
    """Enhanced tool schema with comprehensive validation and metadata."""

    name: str = Field(..., pattern=r"^[a-zA-Z][a-zA-Z0-9_-]*$")
    display_name: str | None = None
    description: str = Field(..., min_length=10, max_length=500)
    long_description: str | None = None
    category: str
    type: str
    parameters: list[ParameterSchema] = Field(default_factory=list)
    returns: ReturnSchema | None = None
    security: SecurityLevel = Field(default_factory=lambda: SecurityLevel(level="standard"))
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    metadata: ToolMetadata = Field(default_factory=ToolMetadata)
    examples: list[ToolExample] = Field(default_factory=list)
    deprecated: bool = False
    deprecation_message: str | None = None
    async_capable: bool = True
    stateful: bool = False
    idempotent: bool = False
    cacheable: bool = True
    cache_ttl: int | None = None  # Cache TTL in seconds

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if len(v) < 3 or len(v) > 64:
            raise ValueError("Tool name must be between 3 and 64 characters")
        return v

    @model_validator(mode="after")
    def validate_deprecation(self):
        if self.deprecated and not self.deprecation_message:
            raise ValueError("Deprecation message is required for deprecated tools")
        return self

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format for function calling."""
        properties = {}
        required = []

        for param in self.parameters:
            prop_def = {"type": param.type, "description": param.description}

            # Add constraints
            if param.enum:
                prop_def["enum"] = param.enum
            if param.minimum is not None:
                prop_def["minimum"] = param.minimum
            if param.maximum is not None:
                prop_def["maximum"] = param.maximum
            if param.min_length is not None:
                prop_def["minLength"] = param.min_length
            if param.max_length is not None:
                prop_def["maxLength"] = param.max_length
            if param.pattern:
                prop_def["pattern"] = param.pattern
            if param.format:
                prop_def["format"] = param.format
            if param.default is not None:
                prop_def["default"] = param.default
            if param.examples:
                prop_def["examples"] = param.examples

            properties[param.name] = prop_def

            if param.required:
                required.append(param.name)

        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }

        # Add metadata for AI models that support it
        if self.display_name:
            schema["display_name"] = self.display_name
        if self.deprecated:
            schema["deprecated"] = True
            schema["deprecation_message"] = self.deprecation_message

        return schema

    def validate_parameters(self, parameters: dict[str, Any]) -> list[str]:
        """Validate parameters and return list of error messages."""
        errors = []

        # Check for required parameters
        required_params = {p.name for p in self.parameters if p.required}
        provided_params = set(parameters.keys())
        missing_params = required_params - provided_params

        if missing_params:
            errors.append(f"Missing required parameters: {', '.join(missing_params)}")

        # Check for unknown parameters
        valid_params = {p.name for p in self.parameters}
        unknown_params = provided_params - valid_params

        if unknown_params:
            errors.append(f"Unknown parameters: {', '.join(unknown_params)}")

        # Validate each parameter
        param_schemas = {p.name: p for p in self.parameters}
        for param_name, value in parameters.items():
            if param_name in param_schemas:
                param_schema = param_schemas[param_name]
                if not param_schema.validate_value(value):
                    errors.append(f"Invalid value for parameter '{param_name}': {value}")

        return errors

    def get_parameter_schema(self, name: str) -> ParameterSchema | None:
        """Get parameter schema by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None

    def is_secure_for_environment(self, environment: str) -> bool:
        """Check if tool is secure for given environment."""
        environment_requirements = {
            "production": ["standard", "strict", "maximum"],
            "staging": ["minimal", "standard", "strict", "maximum"],
            "development": ["minimal", "standard", "strict", "maximum"],
            "local": ["minimal", "standard", "strict", "maximum"],
        }

        required_levels = environment_requirements.get(environment, ["maximum"])
        return self.security.level in required_levels


class ToolSchemaRegistry(BaseModel):
    """Registry for tool schemas."""

    schemas: dict[str, EnhancedToolSchema] = Field(default_factory=dict)
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def add_schema(self, schema: EnhancedToolSchema) -> None:
        """Add a schema to the registry."""
        self.schemas[schema.name] = schema
        self.updated_at = datetime.utcnow()

    def remove_schema(self, name: str) -> bool:
        """Remove a schema from the registry."""
        if name in self.schemas:
            del self.schemas[name]
            self.updated_at = datetime.utcnow()
            return True
        return False

    def get_schema(self, name: str) -> EnhancedToolSchema | None:
        """Get a schema by name."""
        return self.schemas.get(name)

    def list_schemas(
        self, category: str | None = None, deprecated: bool | None = None
    ) -> list[EnhancedToolSchema]:
        """List schemas with optional filtering."""
        schemas = list(self.schemas.values())

        if category is not None:
            schemas = [s for s in schemas if s.category == category]

        if deprecated is not None:
            schemas = [s for s in schemas if s.deprecated == deprecated]

        return schemas

    def validate_all_schemas(self) -> dict[str, list[str]]:
        """Validate all schemas and return any errors."""
        errors = {}
        for name, schema in self.schemas.items():
            schema_errors = []
            try:
                # Validate the schema itself
                EnhancedToolSchema(**schema.dict())
            except Exception as e:
                schema_errors.append(f"Schema validation error: {e!s}")

            if schema_errors:
                errors[name] = schema_errors

        return errors

    def to_json_schemas(self) -> list[dict[str, Any]]:
        """Convert all schemas to JSON Schema format."""
        return [schema.to_json_schema() for schema in self.schemas.values()]
