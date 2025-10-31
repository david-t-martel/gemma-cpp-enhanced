"""System information utilities."""

import platform
import sys
from typing import Any, Dict


def get_system_info() -> Dict[str, Dict[str, Any]]:
    """Get comprehensive system information.

    Returns:
        Dictionary with system information organized by category
    """
    import psutil

    return {
        "platform": {
            "OS": platform.system(),
            "Release": platform.release(),
            "Version": platform.version(),
            "Machine": platform.machine(),
            "Processor": platform.processor(),
        },
        "python": {
            "Version": sys.version.split()[0],
            "Implementation": platform.python_implementation(),
            "Executable": sys.executable,
        },
        "memory": {
            "Total": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
            "Available": f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
            "Used %": f"{psutil.virtual_memory().percent}%",
        },
        "cpu": {
            "Physical Cores": psutil.cpu_count(logical=False),
            "Logical Cores": psutil.cpu_count(logical=True),
            "Usage %": f"{psutil.cpu_percent(interval=1)}%",
        },
    }
