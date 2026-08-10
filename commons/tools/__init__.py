"""Common tools for agentic execution.

This package provides:
- PermissionManager: User permission handling for tool execution
- PlatformDetector: Cross-platform shell detection and configuration
"""

from .permission_manager import PermissionManager, PermissionMode, RiskLevel
from .platform_detector import PlatformDetector, ShellConfig

__all__ = [
    'PermissionManager',
    'PermissionMode',
    'RiskLevel',
    'PlatformDetector',
    'ShellConfig',
]
