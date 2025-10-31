"""
Onboarding system for gemma-cli first-run setup.

This module provides a comprehensive interactive wizard for first-time users
to configure gemma-cli, validate their environment, and learn basic usage.
"""

from .checks import (
    check_model_files,
    check_redis_connection,
    check_system_requirements,
    display_health_check_results,
)
from .templates import TEMPLATES, customize_template, get_template
from .tutorial import InteractiveTutorial
from .wizard import OnboardingWizard

__all__ = [
    "OnboardingWizard",
    "InteractiveTutorial",
    "TEMPLATES",
    "get_template",
    "customize_template",
    "check_system_requirements",
    "check_redis_connection",
    "check_model_files",
    "display_health_check_results",
]
