"""Platform Detector for cross-platform shell execution.

Automatically detects the operating system and configures the appropriate
shell for command execution.
"""

import platform
import shutil
import subprocess
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ShellConfig:
    """Configuration for a shell executable."""
    name: str
    executable: str
    args_prefix: str  # '-Command' for PowerShell, '-c' for bash/zsh
    platform: str


class PlatformDetector:
    """Detects platform and configures appropriate shell.
    
    This module provides cross-platform shell detection and configuration
    for the agentic tool execution system.
    """
    
    # Shell configurations
    SHELL_CONFIGS = {
        'windows': ShellConfig(
            name='powershell',
            executable='powershell.exe',
            args_prefix='-Command',
            platform='windows'
        ),
        'windows_pwsh': ShellConfig(
            name='pwsh',
            executable='pwsh.exe',
            args_prefix='-Command',
            platform='windows'
        ),
        'linux': ShellConfig(
            name='bash',
            executable='/bin/bash',
            args_prefix='-c',
            platform='linux'
        ),
        'darwin_bash': ShellConfig(
            name='bash',
            executable='/bin/bash',
            args_prefix='-c',
            platform='darwin'
        ),
        'darwin_zsh': ShellConfig(
            name='zsh',
            executable='/bin/zsh',
            args_prefix='-c',
            platform='darwin'
        ),
    }
    
    @staticmethod
    def detect_platform() -> str:
        """Detect the current platform.
        
        Returns:
            'windows', 'linux', or 'darwin'
        """
        system = platform.system().lower()
        if system == 'windows':
            return 'windows'
        elif system == 'linux':
            return 'linux'
        elif system == 'darwin':
            return 'darwin'
        else:
            logger.warning(f"Unknown platform: {system}, defaulting to linux")
            return 'linux'
    
    @staticmethod
    def get_default_shell() -> str:
        """Get the default shell for the current platform.
        
        Returns:
            Shell name: 'powershell', 'pwsh', 'bash', or 'zsh'
        """
        current_platform = PlatformDetector.detect_platform()
        
        if current_platform == 'windows':
            # Try pwsh first (PowerShell Core), then powershell
            if PlatformDetector.is_command_available('pwsh'):
                return 'pwsh'
            return 'powershell'
        elif current_platform == 'darwin':
            # Try zsh first (modern Mac default), then bash
            if PlatformDetector.is_command_available('zsh'):
                return 'zsh'
            return 'bash'
        else:
            return 'bash'
    
    @staticmethod
    def get_shell_executable(shell: str) -> str:
        """Get the full path to a shell executable.
        
        Args:
            shell: Shell name ('powershell', 'pwsh', 'bash', 'zsh')
            
        Returns:
            Full path to the executable
        """
        shell_lower = shell.lower()
        
        # Check predefined configs
        for config in PlatformDetector.SHELL_CONFIGS.values():
            if config.name == shell_lower:
                # Verify the executable exists
                executable_path = shutil.which(config.executable)
                if executable_path:
                    return executable_path
                # Fallback to just the name
                return config.executable
        
        # Try to find the executable
        executable_path = shutil.which(shell)
        if executable_path:
            return executable_path
        
        # Return the shell name as fallback
        logger.warning(f"Could not find executable for shell: {shell}")
        return shell
    
    @staticmethod
    def get_shell_args(shell: str) -> list[str]:
        """Get the arguments for executing a command in the given shell.
        
        Args:
            shell: Shell name ('powershell', 'pwsh', 'bash', 'zsh')
            
        Returns:
            List of arguments to pass before the command
        """
        shell_lower = shell.lower()
        
        if shell_lower in ('powershell', 'pwsh'):
            return ['-Command']
        elif shell_lower in ('bash', 'zsh', 'sh'):
            return ['-c']
        else:
            # Default to bash-style
            return ['-c']
    
    @staticmethod
    def get_shell_config(shell: Optional[str] = None) -> ShellConfig:
        """Get the complete shell configuration.
        
        Args:
            shell: Shell name (auto-detect if None)
            
        Returns:
            ShellConfig with all shell details
        """
        if shell is None:
            shell = PlatformDetector.get_default_shell()
        
        shell_lower = shell.lower()
        current_platform = PlatformDetector.detect_platform()
        
        # Find matching config (prefer current platform)
        for config in PlatformDetector.SHELL_CONFIGS.values():
            if config.name == shell_lower and config.platform == current_platform:
                return config
        # Fall back to any matching name
        for config in PlatformDetector.SHELL_CONFIGS.values():
            if config.name == shell_lower:
                return config
        
        # Create a default config if not found
        if shell_lower in ('powershell', 'pwsh'):
            executable = 'pwsh.exe' if shell_lower == 'pwsh' else 'powershell.exe'
            return ShellConfig(
                name=shell_lower,
                executable=executable,
                args_prefix='-Command',
                platform=current_platform
            )
        else:
            return ShellConfig(
                name=shell_lower,
                executable=f'/bin/{shell_lower}',
                args_prefix='-c',
                platform=current_platform
            )
    
    @staticmethod
    def is_command_available(command: str) -> bool:
        """Check if a command is available on the system.
        
        Args:
            command: Command name to check
            
        Returns:
            True if command is available, False otherwise
        """
        try:
            # Use 'where' on Windows, 'which' on Unix
            if PlatformDetector.detect_platform() == 'windows':
                result = subprocess.run(
                    ['where', command],
                    capture_output=True,
                    timeout=5
                )
            else:
                result = subprocess.run(
                    ['which', command],
                    capture_output=True,
                    timeout=5
                )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
    
    @staticmethod
    def get_platform_info() -> dict:
        """Get detailed platform information.
        
        Returns:
            Dictionary with platform details
        """
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'detected_platform': PlatformDetector.detect_platform(),
            'default_shell': PlatformDetector.get_default_shell(),
        }
