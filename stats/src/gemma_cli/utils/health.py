"""Health check utilities."""

from pathlib import Path
from typing import Any, Dict


async def run_health_check(ctx_obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Run comprehensive health check.

    Args:
        ctx_obj: Click context object

    Returns:
        Dictionary with health check results
    """
    results = {}

    # Check configuration
    try:
        from gemma_cli.utils.config import load_config, validate_config

        config_path = ctx_obj["CONFIG_PATH"]
        config = load_config(config_path)
        errors, warnings = validate_config(config_path)

        results["configuration"] = {
            "healthy": len(errors) == 0,
            "details": f"{len(warnings)} warnings, {len(errors)} errors" if errors or warnings else "Valid",
        }
    except Exception as e:
        results["configuration"] = {
            "healthy": False,
            "details": str(e),
        }

    # Check model availability
    try:
        model_path = Path(config.get("model", {}).get("model_path", ""))
        results["model"] = {
            "healthy": model_path.exists(),
            "details": f"Found at {model_path}" if model_path.exists() else f"Not found: {model_path}",
        }
    except Exception as e:
        results["model"] = {
            "healthy": False,
            "details": str(e),
        }

    # Check memory system
    try:
        from gemma_cli.rag.adapter import HybridRAGManager

        rag = HybridRAGManager()
        if await rag.initialize():
            backend = rag.get_active_backend()
            results["memory_system"] = {
                "healthy": True,
                "details": f"Active backend: {backend.value if backend else 'None'}",
            }
            await rag.close()
        else:
            results["memory_system"] = {
                "healthy": False,
                "details": "Failed to initialize",
            }
    except Exception as e:
        results["memory_system"] = {
            "healthy": False,
            "details": str(e),
        }

    # Check MCP (placeholder)
    results["mcp_servers"] = {
        "healthy": True,
        "details": "Not yet implemented",
    }

    return results
